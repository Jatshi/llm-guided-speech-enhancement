# Prescription-MM-DiT AutoDL 完整执行报告（2026-09-17）

## 1. 结论先行

本轮不是 smoke test。已完成以下闭环：

- LibriSpeech dev-clean 公开语料物化：2,000 对音频，train/validation/test 为 1,600/200/200，且说话人不交叉；
- 51,717,122 参数 Prescription-MM-DiT 正式训练 10,000 步；
- 200 条独立测试样本、6 个对照臂、共 1,200 行最终评测；
- Qwen2.5-1.5B LoRA 规划器重新对齐、声学路由器训练、验证集阈值校准；
- 低置信及动作不支持条件的显式拒答/回退机制；
- 最终模型、预测、逐样本指标、日志与校验哈希均已保存。

最终可成立的结论是：**这是一个安全选择性增强系统，而不是全条件通吃的增强模型。** 最终 predicted 分支在 200 条测试集上获得：

| 指标 | Safety-gated predicted | 谱减基线 | DeepFilterNet3 |
|---|---:|---:|---:|
| SI-SDR improvement | **+0.114 dB** | -3.821 dB | -23.724 dB |
| SNR improvement | **+0.132 dB** | -4.393 dB | -23.850 dB |
| PESQ | **2.027** | 2.009 | 2.405 |
| STOI | **0.881** | 0.868 | 0.891 |
| 回退率 | 76.5% | 0% | 0% |

DeepFilterNet3 在非 clean 条件上的平均 SI-SDRi 为 +0.374 dB，仍强于本系统的 +0.142 dB；因此不能声称 MM-DiT 已全面超过强增强基线。最终系统的优势是：它能识别能力边界，对 clean、reverb、telephone 和低置信样本拒绝盲目处理，从而避免灾难性过处理。

## 2. 运行环境

- AutoDL GPU：NVIDIA GeForce RTX 4080 SUPER，32 GB 显存；
- Torch：2.1.2+cu118；
- Transformers：4.48.3；
- TRL：0.16.1；
- PEFT：0.14.0；
- datasets：3.2.0；
- 正式 MM-DiT 训练峰值显存约 10.8 GB；
- Qwen2.5-1.5B LoRA SFT 显存约 24.2 GB，GPU 利用率约 82%–100%。

基础模型 `Qwen/Qwen2.5-1.5B-Instruct` 整文件为 3,087,467,144 字节。Hugging Face API 给出的 LFS SHA-256 为：

```text
dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
```

本地文件校验一致。曾误把 Xet bridge URL 中的对象路径 `46ea...` 当成文件 SHA；通过仓库 blobs API 与 27 个跨区间字节比较定位并纠正，模型文件实际没有损坏。

## 3. 数据与泄漏控制

- 数据源：LibriSpeech dev-clean；
- 总样本：2,000；
- 五种条件各 400：clean、white、pink、reverb、telephone；
- 划分：train 1,600，validation 200，test 200；
- 说话人：train/validation/test 分别 25/4/4，无交叉；
- 测试集不参与 MM-DiT、LoRA 或声学路由器拟合；
- 规划器输入仅包含从 noisy waveform 测得的声学特征，不含数据集退化标签或 sample id；
- 声学路由器阈值 0.70 只由 validation 选择：validation 覆盖率 33%，接受样本诊断准确率 90.9%。

## 4. 版本演进与失败诊断

### 4.1 Gaussian-source MM-DiT

最初从高斯噪声直接生成完整 STFT。10,000 步训练和 200 条评测均跑完，但 oracle 没有稳定优于 shuffled，且安全回退率混淆了比较。根因是任务过难：模型既要恢复语音内容，又要学习处方条件，容易忽略处方。

### 4.2 Observed-source residual flow

改为从观测 noisy STFT 出发学习 residual flow：

- 正确处方：noisy → clean；
- 无处方/处方 dropout：noisy → noisy，学习安全 no-op；
- 错配处方：增加 mismatch no-op loss；
- 评测同时保存 gate 前后的 raw 指标。

正式训练结果：

- 10,000 步；
- 用时 1,803.97 秒；
- best validation：0.0735645；
- checkpoint-best 位于 step 10,000；
- 训练状态：COMPLETED。

该版本学会了稳定 no-op，并在 pink 条件表现出处方敏感性：oracle 为 +0.959 dB，shuffled 为 -0.038 dB；但全条件能力仍不足。

### 4.3 旧 Hugging Face LoRA 的问题

旧 adapter 可被 Qwen2.5-1.5B 正常加载，但：

- 原生严格合法 JSON：0/200；
- 最小尾括号修复后：195/200；
- 195 条全部预测为 white、band_limited=true，reverb 缺失；
- 说明发生明显模式坍缩，且训练契约与当前执行契约不一致。

修复器只补齐末尾未闭合的 `}`/`]`，不增加字段、不移动值、不修改语义，并单独记录 `strict/repaired/invalid`，因此不会把修复后的输出冒充模型原生正确输出。

### 4.4 重新训练 LoRA

使用当前 1,600/200 的 train/validation 数据重新 SFT：

- 五类完全均衡；
- test 消费数为 0；
- 300 步，约 5 分钟；
- 200/200 测试输出均为原生严格合法 JSON；
- 但纯 LLM 诊断准确率仅 29.5%。

随后尝试 completion-only loss，仅对 assistant JSON 计算损失：

- 格式仍为 200/200；
- 语义准确率反而降至 20.5%；
- 只会选择 white/pink。

因此问题不是单纯的 loss mask，而是小型通用 LLM 不擅长直接从少量连续统计量完成稳定声学分类。

### 4.5 可分性验证与混合规划器

使用完全相同的 5 个实测特征训练 ExtraTrees：

- 只用 1,600 条 train 拟合；
- validation 准确率 73%；
- test 准确率 79%；
- reverb 36/40；
- telephone 38/40；
- pink 37/40；
- white 32/40；
- clean 15/40，主要与 reverb 混淆。

这说明特征具有较强可分性，主要瓶颈是 LLM 的数值映射。最终混合规划器采用：

- ExtraTrees：给出诊断和类别概率；
- LLM：保留动作参数和解释；
- 每条样本记录原始 LLM 处方、路由器概率、融合结果和 safety decision；
- 不隐藏 override。

### 4.6 置信度门控与动作能力门控

仅使用 0.70 置信度门控时，总 SI-SDRi 从 -21.327 dB 改善到 -4.358 dB，但仍有少量高置信 clean 被处理。

最终加入动作能力门控：当前可执行动作只有 `spectral_subtraction`，因此仅支持 white/pink；其余条件的正确策略是：

- clean：保持原音；
- reverb：需要 dereverb/WPE 类动作，当前版本拒绝；
- telephone：需要带宽扩展，当前版本拒绝；
- 低于 0.70：证据不足，拒绝。

该策略不是按测试结果挑样本，而是由动作语义和 validation 置信度共同确定。

## 5. 最终逐条件结果

| 条件 | Predicted SI-SDRi | Oracle SI-SDRi | Shuffled SI-SDRi | Predicted fallback |
|---|---:|---:|---:|---:|
| clean | **0.000** | -96.646 | -96.924 | 100% |
| white | -0.003 | +0.002 | -0.068 | 40.0% |
| pink | **+0.573** | +0.959 | -0.038 | 42.5% |
| reverb | **0.000** | +0.012 | 0.000 | 100% |
| telephone | **0.000** | -0.425 | 0.000 | 100% |

非 clean 平均 SI-SDRi：

- safety-gated predicted：+0.142 dB；
- oracle：+0.137 dB；
- shuffled：-0.027 dB；
- no prescription：+0.006 dB；
- 谱减：-1.124 dB；
- DeepFilterNet3：+0.374 dB。

predicted 略高于 oracle 不是模型超过 oracle，而是安全门控回退掉了 oracle 会伤害的 telephone 等样本。这个结果应解释为“系统级策略优于无条件执行 oracle 动作”，不能解释为“预测比真实标签更准”。

## 6. 可以与不可以写进简历的表述

可以写：

> 构建基于 observed-source rectified flow 的处方条件 MM-DiT，在 2,000 对公开语音数据上完成 10k-step 训练与 200 条六臂对照；设计声学路由、验证集置信度校准及动作能力门控，使系统在证据不足或动作不支持时安全回退，最终取得 +0.114 dB 整体 SI-SDRi，避免 clean/reverb/telephone 的灾难性过处理。

可以写：

> 发现并修复旧 LoRA 的 JSON 截断、契约错配与模式坍缩；重新构建无测试泄漏的 SFT 数据，达到 200/200 原生合法 JSON，并通过可解释声学路由器将诊断准确率从纯 LLM 的 29.5% 提升到 79%。

不可以写：

- “全面超过 DeepFilterNet3”；
- “所有条件均实现语音增强”；
- “纯 LLM 规划器达到 79%”；
- “低回退率实时通用增强系统”；
- “oracle 条件下显著领先所有基线”。

## 7. 关键产物与 SHA-256

```text
d8e1677d15b3ad6b53b87a1a08c3b95c4dc1c028dfa237267646d5d7b8306c21  evaluation_report.json
406287dc3c6ec6e81a26f2c02ed29e99de6cbe681e0e0ce4210f07758ea19c48  planner_predictions.hybrid-safe.manifest.json
777a4ca76efe2d00ad66a74e5cf198c018f7639951b93c04630a6c6bed51b27f  extra_trees_safe.joblib
868bf0918a649f3e7b0feb7638dfface83dd2abcbd5274510823db4cb43832e7  planner SFT adapter_model.safetensors
```

远端关键路径：

```text
/root/autodl-tmp/llm-guided-speech-enhancement/outputs/mmdit-32gb-residual/checkpoint-best.pt
/root/autodl-tmp/llm-guided-speech-enhancement/outputs/mmdit-planner-sft/final
/root/autodl-tmp/llm-guided-speech-enhancement/outputs/mmdit-planner-router/extra_trees_safe.joblib
/root/autodl-tmp/llm-guided-speech-enhancement/data/mmdit/planner_predictions.hybrid-safe.jsonl
/root/autodl-tmp/llm-guided-speech-enhancement/results/mmdit-32gb-residual-hybrid-safe
```

## 8. 复现命令

```bash
python -m lse_v2.mmdit.planner_dataset \
  --pairs data/mmdit/pairs.jsonl \
  --output-dir data/mmdit/planner_sft \
  --workers 8

python -m lse_v2.training \
  --config configs/mmdit_planner_sft_32gb.json \
  --stage sft --resume never

python -m lse_v2.mmdit.planner_export \
  --pairs data/mmdit/pairs.jsonl \
  --output data/mmdit/planner_predictions.sft.jsonl \
  --base-model /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
  --adapter outputs/mmdit-planner-sft/final \
  --split test --batch-size 8 --max-new-tokens 256

python -m lse_v2.mmdit.planner_router \
  --pairs data/mmdit/pairs.jsonl \
  --llm-predictions data/mmdit/planner_predictions.sft.jsonl \
  --output data/mmdit/planner_predictions.hybrid-safe.jsonl \
  --model-output outputs/mmdit-planner-router/extra_trees_safe.joblib \
  --workers 8 \
  --supported-diagnosis white \
  --supported-diagnosis pink

python -m lse_v2.mmdit.attach_predictions \
  --pairs data/mmdit/pairs.jsonl \
  --predictions data/mmdit/planner_predictions.hybrid-safe.jsonl \
  --output data/mmdit/pairs.hybrid_safe_predictions.jsonl

python -m lse_v2.mmdit.evaluate \
  --config configs/mmdit_32gb_residual_hybrid_safe.json \
  --checkpoint outputs/mmdit-32gb-residual/checkpoint-best.pt
```

## 9. 下一步最有价值的改进

1. 增加可执行 dereverb 与 bandwidth-extension action head，而不是仅靠 spectral subtraction。
2. 增加调制谱、频带能量比、混响衰减和语音活动区间统计，重点解决 clean/reverb 混淆。
3. 在独立 calibration split 上校准 ExtraTrees 概率，而不是只使用分类概率原值。
4. 对 clean 使用 waveform 差异、DNSMOS/PESQ 和绝对失真指标；不要用近乎无穷大的 clean SI-SDR 作为唯一汇总依据。
5. 将 selective risk、coverage、误执行率作为一等指标，而不是只报告被接受样本的质量。
6. 若继续保留 LLM，建议让 LLM 负责策略解释和多动作编排，不再承担底层连续声学分类。

