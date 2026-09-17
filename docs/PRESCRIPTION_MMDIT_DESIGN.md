# Prescription-Conditioned MM-DiT 设计说明

## 要解决的真实问题

现有系统的 SFT→DPO→GRPO 模型能够输出带诊断、动作、理由和置信度的结构化增强处方，
但 Demo 最后仍由简单谱减处理波形。因而“LLM 规划是否真的改善增强结果”没有被验证。
本扩展不改变处方规划逻辑，而是加入一个可训练、可消融的生成式处方执行器。

## 三流架构

```text
clean latent + noise at time t ── target patch embed ─┐
degraded speech latent ───────── observed patch embed ├─ joint attention MM-DiT ─ velocity
JSON prescription ────────────── field/value encoder ┘
```

- 目标流：训练时由 clean complex-STFT latent 与高斯噪声线性插值得到；推理从噪声开始。
- 观测流：退化语音的 complex-STFT latent，全程作为条件，不添加扩散噪声。
- 处方流：将 `noise_type`、`reverb`、`band_limited`、`confidence` 和每个 action 映射成
  可审计的 `(field_id, category_id, numeric_value)` token，不使用 oracle 标签填充预测处方。
- 每种流有独立的 QKV、输出投影、LayerNorm 和 MLP；Q/K/V 在 token 维拼接后执行联合注意力。
- timestep 通过 target 流的 AdaLN shift/scale/gate 注入。

默认连续 latent 是可逆的复数 STFT，而不是临时指定一个未经验证的 Audio VAE。接口被隔离在
`codec.py`，后续若引入冻结 Audio VAE，只需保持 `[B,C,F,T]` encode/decode 契约；正式对比
必须把 codec 重建误差单独报告，不能把 codec 损失算成 MM-DiT 失败。
复数 STFT 默认乘以 16 后送入 flow、解码时再除回去，用来避免原始 latent 方差远小于单位高斯
导致训练尺度失衡；预检会用真实 batch 验证 latent 标准差，超出安全范围即阻止正式长跑。

## Rectified Flow

训练采样：

```text
z_t = (1 - t) z_clean + t ε
v*  = ε - z_clean
loss = MSE(vθ, v*) + λ L1(z_t - t vθ, z_clean)
```

推理从 `t=1` 的高斯噪声出发，用 4/8/16 步 Euler 更新到 `t=0`。训练中随机丢弃处方 token，
从而支持 classifier-free guidance 和“没有处方”的核心消融。

## 安全边界

生成结果必须依次通过：有限值、峰值、能量比例、改变量和 planner confidence 门禁。任一失败
严格返回原音频。长音频按固定窗口分块生成并 overlap-add，避免 token 数随时长平方增长。
安全门会造成一部分结果等于输入，因此报告中必须同时给出 fallback rate，不能只报告通过后的
精选样本。

## 五臂核心实验

1. 现有谱减。
2. 官方预训练 DeepFilterNet3 强基线。
3. MM-DiT 无处方。
4. MM-DiT + Oracle 处方。
5. MM-DiT + LLM 预测处方。
6. MM-DiT + shuffled/wrong 处方作为反事实控制。

严格说是六个系统；“五臂因果实验”指四个 MM-DiT/处方臂加一个经典执行器家族，
DeepFilterNet3 作为额外强基线单列。关键证据不是模型超过谱减，而是：

```text
Oracle ≳ predicted > no-prescription > shuffled/wrong
```

如果四个 MM-DiT 臂几乎相同，说明模型忽略处方，LLM 只是装饰；此时不能写
“LLM-guided enhancement”。

## 数据防泄漏

- `lse.mmdit_pair.v1` 强制保存 clean/noisy 路径、speaker_id、split、oracle/predicted 处方和来源。
- 预检默认要求 speaker-disjoint；重复 speaker 跨 split 会 BLOCKED。
- planner 预测输入只包含波形测量特征，不包含 degradation 标签、oracle 处方或 sample-id 条件。
- Oracle 处方仅用于上界臂；正式主结果必须使用 held-out LLM predicted prescription。
- 所有阈值在 test 之前用 validation 锁定。

## 单卡策略

- 12 GB：256 hidden、6 blocks、batch 1、gradient accumulation 16。
- RTX 4090 24 GB：384 hidden、8 blocks、batch 2、gradient accumulation 8。
- BF16、gradient checkpointing、2 秒 crop、约 1096 个 joint-attention tokens。
- 单卡不宣称 ZeRO 的分布式收益；本实验不需要为了简历标签强行启用 ZeRO-3。

## 失败标准

- runtime smoke 不能前向/反向、出现 NaN、checkpoint 无法重载：禁止正式训练。
- 32 条小样本不能明显过拟合：优先排查数据对齐/损失/模型，而不是加步数。
- Oracle 不优于 shuffled：处方通道无效，停止“LLM-guided”主张。
- 仅谱指标提升但 CER、speaker similarity 或主观质量恶化：不能宣称整体增强。
- fallback rate 过高：先校准生成尺度和安全门，不隐藏失败样本。
