# Prescription MM-DiT v6：吸收 StepAudio 3 思路后的工程设计

> 状态：本地代码、AutoDL 12k 正式训练与 200 条完整评测已完成。实测结论见
> `MMDIT_V6_AUTODL_EXECUTION_REPORT_20260917.md`。
> 本文描述“为什么这样改、代码如何工作、怎样证明有效”，不把 smoke 当成模型效果。

## 1. 这次不是改题，而是补齐原项目最薄弱的环节

项目仍然解决同一个问题：给定嘈杂语音和结构化增强处方，MM-DiT 在复数 STFT
空间预测从 noisy latent 到 clean latent 的 rectified-flow 速度场。此次没有把模型改成
TTS，也没有照搬 StepAudio 的 RVQ 自回归生成器。

真正要解决的是五个工程硬伤：

1. 新条件分支随机初始化时可能先破坏输入，缺少严格的 no-op 起点；
2. 仅优化 latent MSE/L1，不能直接约束语义、频谱和听感；
3. 所有损失从第一步同时加入，容易发生目标干扰；
4. 处方是全局标签，无法表达“哪一段该处理、哪一段必须保持”；
5. Planner 是否调用 LLM 没有成本意识，偏好训练也可能退化成文本自评分。

## 2. 从 StepAudio 3 吸收了什么

以下借鉴来自三个公开技术报告：

- [StepAudio 3 Gen](https://arxiv.org/abs/2609.12945)：12.5 Hz、16 层 RVQ、
  RVQ Adaptor、干扰感知渐进训练和共享语义—声学表示；
- [StepAudio 3 Music](https://arxiv.org/abs/2609.16034)：先生成 ABC-CoT 显式结构计划，
  再由 flow-matching DiT 渲染连续音频 latent，并用 DPO 对齐最终体验；
- [StepAudio 3 Realtime](https://arxiv.org/abs/2609.14005)：连续
  listen-converse-think-act、边说边想和异步工具执行，核心是按难度分配推理成本。

对应到本项目，不是复制大模型，而是提炼六条可验证原则。

| StepAudio 3 原则 | 本项目 v6 落地 | 为什么适用 |
| --- | --- | --- |
| 新音频能力不能破坏已有能力 | 零速度 identity initialization | 训练开始时增强器严格等价于“不处理” |
| 语义与声学表示都要保留 | WavLM 语义损失 + MR-STFT 损失 | 同时保护字词内容和多尺度频谱 |
| 多目标要渐进引入 | 三阶段 curriculum | 降低 flow、频谱、语义和安全目标的梯度冲突 |
| 先规划、再渲染 | 时间化 EnhanceScript → MM-DiT | 让条件具有时间边界和 preserve 约束 |
| 简单问题不必深度推理 | direct / LLM / abstain 三路路由 | 节省 LLM 成本并提高拒答可审计性 |
| 对齐最终结果而非漂亮文本 | 执行增强后再构建 preference pair | 防止 Planner 学会“说得像对的”而音频变差 |

### 明确没有采用的部分

- 没有将 MM-DiT 换成 16-codebook RVQ 自回归模型；那会改变项目主线和数据需求；
- 没有把音乐的 ABC 符号直接用于语音；只吸收“显式中间计划”的思想；
- 没有声称 StepAudio 3 的榜单结果能迁移到本项目；所有收益必须由本项目消融验证。

## 3. 端到端数据流

```text
noisy waveform
  ├─ acoustic router ── label + confidence
  │       ├─ 高置信支持类 → direct template
  │       ├─ 中置信支持类 → LLM Planner
  │       └─ 低置信/不支持 → abstain / hold
  │
  └─ global prescription
          ↓
    EnhanceScript
    [time range, diagnosis, action, strength, confidence, preserve flags]
          ↓ tokenization
    prescription tokens + observed STFT + flow target
          ↓
    Prescription-conditioned MM-DiT
          ↓ predicted velocity / estimated clean latent
          ↓ inverse STFT
    enhanced waveform
          ↓
    waveform safety + SI-SDR/PESQ/STOI + CER + speaker similarity
          ├─ serving gate: accept or fall back
          └─ offline reward: build chosen/rejected Planner pairs
```

## 4. 六项核心实现

### 4.1 严格 no-op 初始化

Rectified flow 使用 observed-source：

```text
x_t = (1 - t) * noisy_latent + t * clean_latent
target_velocity = clean_latent - noisy_latent
```

推理从 noisy latent 出发。如果网络初始速度场严格为零，ODE 在第一个更新前保持输入，
因此初始行为是“什么也不做”，而不是随机污染语音。

实现位于 `lse_v2/mmdit/model.py`：

- 每个 joint-attention block 的 AdaLN modulation 末层权重和偏置置零；
- 最终 velocity head 置零；
- `identity_init` 默认仍为 `false`，旧配置和旧 checkpoint 行为不变；
- v6 配置显式打开该开关。

这不是把整个网络永远冻结。第一步先更新输出头；输出头离开零点后，梯度继续进入
target stream 和条件门。20-step canary 会检查梯度、loss 和 checkpoint 是否正常。

### 4.2 多分辨率 STFT 损失

单一 latent L1 不能同时覆盖短时瞬态和较长谐波。v6 对 256/512/1024 三个 FFT
分辨率计算：

```text
L_mrstft = mean_r [ |||S_hat|-|S|||_F / (|||S|||_F + eps)
                    + L1(log(|S_hat|+eps), log(|S|+eps)) ]
```

第一项关心整体谱形，第二项防止低能量细节被忽略。损失直接作用于由
`estimated_x0` 解码出的波形，因此梯度能回到 MM-DiT。

### 4.3 冻结 WavLM 的语义保持损失

WavLM 仅作为 teacher，不更新参数。增强语音与干净参考分别送入 teacher，比较中间层
frame representation 的 cosine distance：

```text
L_sem = mean(1 - cosine(H_enhanced, stop_grad(H_clean)))
```

关键边界：

- teacher 参数全部 `requires_grad=False`；
- clean 分支处于 `no_grad`；
- enhanced 分支不能 `detach`，否则语义损失无法训练增强器；
- 默认每 4 个 micro-step 计算一次，控制显存和吞吐开销；
- AutoDL 上先执行独立 teacher probe，确认下载、hidden state、反传和峰值显存。

### 4.4 干扰感知 curriculum

正式配置不是从第一步叠满所有目标：

| 阶段 | Step | 主要目的 | MR-STFT | Semantic | 条件 dropout / mismatch |
| --- | ---: | --- | ---: | ---: | --- |
| oracle_alignment | 1–4000 | 先学稳定的 noisy→clean 主映射 | 0.05 | 0 | 0 / 0 |
| safety_alignment | 4001–9000 | 引入条件鲁棒性和内容保持 | 0.10 | 0.03 | 0.15 / 0.5 |
| joint_cooldown | 9001–12000 | 小学习率联合收敛 | 0.10 | 0.05 | 0.15 / 0.5 |

总损失：

```text
L_total = L_flow + lambda_stft * L_mrstft + lambda_sem * L_sem
```

每个 checkpoint 记录当前阶段、完整 curriculum 与辅助损失配置，恢复训练时不会丢失谱系。
由于各阶段的总损失权重不同，`checkpoint-best` 只在当前阶段内部比较；进入新阶段时
重新建立 best，避免用第一阶段较小但不可比的目标值压住后续模型。

### 4.5 时间化 EnhanceScript

旧处方只说“这段有 pink noise，做 spectral subtraction”，没有时间信息。新 schema：

```json
{
  "schema_version": "lse.enhance_script.v1",
  "duration_ms": 2000,
  "preserve": {
    "speech_content": true,
    "speaker_identity": true,
    "prosody": true
  },
  "segments": [
    {
      "start_ms": 0,
      "end_ms": 640,
      "diagnosis": "pink",
      "action": "spectral_subtraction",
      "strength": 0.35,
      "confidence": 0.92
    }
  ]
}
```

合约会拒绝负时间、倒序、越界、重叠、非法 strength/confidence。为了兼容现有数据，
`enhance_script.py` 可以把全局处方安全地 bootstrap 成一个覆盖全段的 segment。它不假装
拥有帧级标注；真正的时间局部收益需要后续构造局部噪声数据进行消融。

### 4.6 Adaptive Planning 与 outcome-grounded preference

Planner 路由是确定性策略，不把最终安全选择交给 LLM：

```text
unsupported label                 → abstain
confidence < minimum threshold    → abstain
confidence >= direct threshold    → direct template
otherwise                         → LLM Planner
```

偏好样本也不使用“文本是否好看”的分数。每个候选处方必须先真正执行增强，再记录：

- SI-SDR improvement；
- SNR improvement；
- PESQ/STOI improvement；
- CER before/after；
- speaker similarity；
- artifact penalty；
- waveform gate 是否通过、action 是否受支持。

如果 waveform 不安全、action 不支持，或 CER 严重恶化，reward 直接为 -1。每个样本仅在
最好与最坏候选的真实 reward gap 足够大时生成 DPO pair，避免制造无信息偏好。

## 5. 代码地图

| 文件 | 作用 |
| --- | --- |
| `lse_v2/mmdit/model.py` | identity initialization |
| `lse_v2/mmdit/flow.py` | 暴露可微 `estimated_x0` |
| `lse_v2/mmdit/losses.py` | MR-STFT、WavLM semantic teacher |
| `lse_v2/mmdit/curriculum.py` | 阶段选择、flow override、loss/LR 权重 |
| `lse_v2/mmdit/contracts.py` | EnhanceScript 验证与 token 化 |
| `lse_v2/mmdit/enhance_script.py` | 旧 manifest 的兼容升级 |
| `lse_v2/mmdit/adaptive_planner.py` | direct / LLM / abstain 路由 |
| `lse_v2/mmdit/enhancement_reward.py` | 结果驱动 reward 与 pair 选择 |
| `lse_v2/mmdit/planner_preference_data.py` | 构建可审计偏好数据集 |
| `lse_v2/mmdit/train.py` | curriculum 与辅助损失正式接入训练循环 |
| `lse_v2/mmdit/preflight.py` | identity、curriculum、运行时门禁 |

## 6. 需要 AutoDL 回答的科学问题

本地 smoke 只能证明接口通，不能证明模型好。正式实验至少要回答：

1. identity init 是否降低前 100 step 的破坏性输出和 clean-input 退化？
2. MR-STFT 是否提升 PESQ/LSD，却没有恶化 SI-SDR 和 CER？
3. semantic loss 是否真正降低 CER regression，而非只让 WavLM feature 更像？
4. EnhanceScript 相比相同全局标签，是否只在时间局部噪声数据上有收益？
5. oracle、shuffled、no-prescription、predicted 四臂是否产生合理顺序？
6. adaptive route 是否减少 LLM 调用，同时保持任务 reward 和安全率？

建议消融：

| Arm | identity | MR-STFT | semantic | EnhanceScript | 目的 |
| --- | --- | --- | --- | --- | --- |
| A | 否 | 否 | 否 | 否 | v5 基线 |
| B | 是 | 否 | 否 | 否 | 隔离安全初始化 |
| C | 是 | 是 | 否 | 否 | 隔离频谱目标 |
| D | 是 | 是 | 是 | 否 | 隔离语义保持 |
| E | 是 | 是 | 是 | 是 | 完整 v6 |

在 E 没有显著胜过 C/D 前，不能声称时间化规划有效；在 predicted arm 没有胜过
no-prescription 前，不能声称 Planner 对增强有因果贡献。

## 7. 当前证据边界

已完成：代码、合约、配置、preflight、WavLM teacher probe、20-step 全目标 canary、
12,000-step 正式训练、200 条六臂评测、配对 bootstrap 和 160 项全量 pytest。

正式结果支持“处方条件被真实利用”：在 160 条真实退化样本上，oracle 与 predicted
相对无处方分别提升 `+0.258 dB` 与 `+0.216 dB`，95% bootstrap CI 均大于 0；shuffled
处方更差。尚未完成：identity、MR-STFT、semantic 的独立训练消融，以及 Planner
preference 的多候选真实执行。PESQ/STOI 仍未超过 DeepFilterNet3，不能声称 SOTA。
