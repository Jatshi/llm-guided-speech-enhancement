# Prescription-MM-DiT 上机前就绪报告

日期：2026-09-16  
结论：**本地代码与数据契约准备完成；真实 GPU 训练、DeepFilterNet3 实测和质量指标仍为 NOT_MEASURED。**

## 1. 已完成的工程链路

- 三流 Prescription-Conditioned MM-DiT：加噪 clean latent、noisy observation latent、结构化处方 token。
- Rectified-flow 训练与 Euler/CFG 采样。
- 可逆 complex-STFT codec，latent 缩放、反缩放与真实 batch 尺度门禁。
- 单卡 BF16、gradient checkpointing、梯度累积、best/last checkpoint、断点续训。
- 训练/验证随机数隔离，验证集 loss 可重复；梯度累积日志记录真实 micro-batch 均值。
- 长音频分块、overlap-add、逐块数值安全门和原音回退记录。
- 谱减、DeepFilterNet3、无处方、oracle、shuffled、predicted 六系统评测。
- smoke → 200-step canary → formal 三层付费门禁。

## 2. 本地可复核证据

| 检查 | 结果 |
|---|---|
| 完整 pytest | 58 passed |
| Python compileall | passed |
| Ruff check / format | passed |
| 5 个 AutoDL shell 脚本语法 | passed |
| 合成 pair | 10 条；6/2/2 speaker-disjoint |
| 微型模型 | 120,514 参数；2 步；累积 2 micro-batch/step |
| runtime forward | input/output `[1,2,65,126]`，finite |
| latent 尺度 | std 1.026；`latent_scale_sane=true` |
| checkpoint | best/last 可保存并由评测重载 |
| 评测产物 | 5 个启用臂 × 2 条 = 10 行；报告/CSV 已生成 |
| 长音频路径 | 0.5 秒输入按 0.1 秒窗切成 7 块，overlap-add 完成 |

两步随机初始化模型的增强分数很差，这是预期的 wiring smoke，不是可发布结果。数值门禁只能发现
NaN、爆音、能量和改变量异常，不能把未训练模型识别成“感知质量良好”；checkpoint 是否可用必须由
canary 与 held-out paired metrics 决定。

## 3. 真实数据已离线对齐

本地使用现有 EXP2 产物做了不读波形的契约演练：

| 项目 | 数量 |
|---|---:|
| 总 pair | 4,975 |
| train / validation / test | 4,000 / 500 / 475 |
| speaker | 141 |
| speaker 跨 split 泄漏 | 0 |
| 连续音频 planner test 预测 | 475 / 475 |
| 无效预测 / 未使用预测 | 0 / 0 |

对应输入：

```text
results/exp2/aligned.jsonl
results/exp2/RAW/2a/predictions.jsonl
```

本地移动硬盘未找到这些 manifest 所引用的真实 WAV；路径仍指向
`/root/autodl-tmp/lse-v4-data/...`。因此远端连接后的第一件事不是训练，而是核验持久化数据盘上
这些文件是否仍存在。不存在则先恢复数据，正式 preflight 会因 `check_files=true` 阻止烧卡。

空数据盘的可复现替代路径已经提供：`scripts/autodl_mmdit_public_data.sh` 从 OpenSLR 下载并校验
LibriSpeech dev-clean，生成独立命名的 1,600/200/200 speaker-disjoint 五退化配对基准。它不复刻
EXP2，也不会把新实验伪装成旧实验续跑。

## 4. 拿到 AutoDL 后的固定顺序

```bash
export LSE_PROJECT_DIR=/root/autodl-tmp/llm-guided-speech-enhancement
export LSE_MMDIT_VENV=/root/autodl-tmp/lse-mmdit-env
export MMDIT_SOURCE_MANIFEST="$LSE_PROJECT_DIR/results/exp2/aligned.jsonl"
export MMDIT_PREDICTIONS="$LSE_PROJECT_DIR/results/exp2/RAW/2a/predictions.jsonl"
export MMDIT_CONFIG=configs/mmdit_4090.json

cd "$LSE_PROJECT_DIR"
bash scripts/autodl_mmdit_bootstrap.sh
bash scripts/autodl_mmdit_prepare.sh
bash scripts/autodl_mmdit_run.sh
```

`autodl_mmdit_run.sh` 会先跑 GPU smoke，再跑 200-step canary；canary 少于 50 steps、loss 非有限，或
末段 10-window 均值未比初段至少下降 2% 时
直接停止，不进入 10k-step 正式训练。正式预检还要求 DeepFilterNet3 已安装、全部音频存在、split
完整、speaker 无泄漏、latent 尺度合理、参数/token 预算和 CUDA 前向全部通过。

## 5. 尚不能提前声称的内容

- 未测量 SI-SDR、PESQ、STOI 或对 DeepFilterNet3 的优劣。
- 未证明 `predicted > none > shuffled`；若不成立，必须停止“LLM-guided”因果主张。
- DNSMOS、Whisper CER、speaker similarity 仍缺冻结评测器或 transcript/reference 元数据，不能用空值
  冒充结果。
- 单张 4090 不宣称 ZeRO-3 的多卡并行收益；本执行器当前不需要 DeepSpeed。
