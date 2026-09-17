# Prescription-MM-DiT AutoDL 运行手册

## 0. 当前证据状态

本地已经完成：合成配对数据生成、静态/运行时预检、CPU 前向与反向、2 步训练、checkpoint
保存重载、六系统评测和安全回退。它们只证明代码接线完整，不是语音增强质量结果。

## 1. 环境

```bash
export LSE_PROJECT_DIR=/root/autodl-tmp/llm-guided-speech-enhancement
export LSE_MMDIT_VENV=/root/autodl-tmp/lse-mmdit-env
cd "$LSE_PROJECT_DIR"
bash scripts/autodl_mmdit_bootstrap.sh
```

安装失败时不要随意升级 torch/transformers；保存完整 pip 错误后修固定版本。

## 2. 准备真实配对数据

输入 manifest 必须同时有真实 noisy 和 clean 路径，不能把 clean 当 noisy proxy：

```bash
export MMDIT_SOURCE_MANIFEST=/root/autodl-tmp/datasets/lse_paired_manifest.jsonl
```

若沿用本项目已经对齐的 EXP2 数据与连续音频 planner 预测，使用：

```bash
export MMDIT_SOURCE_MANIFEST="$LSE_PROJECT_DIR/results/exp2/aligned.jsonl"
export MMDIT_PREDICTIONS="$LSE_PROJECT_DIR/results/exp2/RAW/2a/predictions.jsonl"
```

该组合已在本地验证为 4,975 条 pair、141 位 speaker、speaker leakage=0，且 475 条 test 均有
预测处方；但 WAV 只存在于原 AutoDL 数据路径，换机器后仍必须由 `--check-files` 重新核验。

已有 planner 预测时：

```bash
export MMDIT_PREDICTIONS=/root/autodl-tmp/lse-planner/predictions.jsonl
bash scripts/autodl_mmdit_prepare.sh
```

或者直接用已训练 LoRA 生成不含 oracle 标签的预测：

```bash
export MMDIT_BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
export MMDIT_PLANNER_ADAPTER=/root/autodl-tmp/lse-planner/grpo-final
bash scripts/autodl_mmdit_prepare.sh
```

内置导出器默认由准备脚本仅生成 `test` split 的预测：训练和验证继续使用 oracle 条件，正式测试同时
比较无处方、预测处方、oracle 处方和 shuffled 处方。这样不会为不参与最终指标的 1,800 条样本额外
支付大模型推理时间。

这里的内置导出器只适用于“文本 prompt → JSON”的 CausalLM LoRA。若现有 planner 是带音频 projector
的 native-audio checkpoint，必须先走它自身的推理入口导出 `sample_id + prediction` JSONL，再通过
`MMDIT_PREDICTIONS` 接入；脚本不会假装加载一个结构不兼容的 adapter。

若没有预测处方，训练仍可使用 Oracle 条件，但 predicted 主实验会明确标为 BLOCKED。

若原数据盘为空，可改用公开、可重建的 LibriSpeech dev-clean speaker-disjoint 基准：

```bash
bash scripts/autodl_mmdit_public_data.sh
```

脚本先校验 OpenSLR 官方压缩包 MD5，再生成 1,600/200/200 条、4 秒、16kHz 的 clean/noisy
配对；五种退化为 clean、white、pink、reverb、telephone，各 split 内严格均衡。该基准与原 EXP2
不是同一数据集，结果必须单独命名，不能冒充原数据的续跑。

## 3. 必跑 smoke

```bash
bash scripts/autodl_mmdit_smoke.sh 2>&1 | tee results/mmdit-smoke-v2/run.log
```

通过条件：preflight READY、训练退出码 0、`checkpoint-best.pt` 可加载、评测 CSV 每个计划臂都有
逐样本记录。smoke 的指标没有科研意义。

## 4. 选择显存配置

```bash
# 12 GB（3080 Ti 等）
export MMDIT_CONFIG=configs/mmdit_12gb.json

# 24 GB RTX 4090
export MMDIT_CONFIG=configs/mmdit_4090.json

# 32 GB RTX 4080 SUPER / V100（保持有效 batch=16，但取消梯度检查点以提高吞吐）
export MMDIT_CONFIG=configs/mmdit_32gb.json
```

正式预检：

```bash
python -m lse_v2.mmdit.preflight \
  --config "$MMDIT_CONFIG" \
  --output results/mmdit-preflight/formal.json \
  --runtime-smoke
```

任一 gate 为 false 都不得开始付费长跑。特别检查 speaker leakage、DeepFilterNet 包、参数预算、
joint token 数、`latent_scale_sane` 和 runtime 峰值显存。复数 STFT latent 默认乘以 16，使训练目标
与单位高斯流的数量级接近；真实数据若不落在预检范围会直接 BLOCKED，而不是烧卡试错。

## 5. 正式训练与恢复

```bash
bash scripts/autodl_mmdit_run.sh
```

脚本发现 `checkpoint-last.pt` 会自动恢复。关键产物：

```text
outputs/mmdit-4090/checkpoint-last.pt
outputs/mmdit-4090/checkpoint-best.pt
outputs/mmdit-4090/train_metrics.jsonl
outputs/mmdit-4090/training_report.json
results/mmdit-4090/per_sample.csv
results/mmdit-4090/summary.csv
results/mmdit-4090/evaluation_report.json
```

## 6. 单文件推理

```bash
python -m lse_v2.mmdit.infer \
  --checkpoint outputs/mmdit-4090/checkpoint-best.pt \
  --input demo/noisy.wav \
  --prescription demo/prescription.json \
  --output demo/enhanced.wav \
  --chunk-seconds 2.0 --overlap-seconds 0.25
```

旁边会生成 JSON，记录每个 chunk 是否触发原音回退。

## 7. 停止条件

- 第一次 GPU smoke 不应超过约 10 分钟；超过即检查死锁、attention token 和 I/O。
- 任意 NaN/Inf、CUDA OOM 或 loss 连续异常上升，停止长跑并保留日志。
- 正式质量结论必须等 test 六系统逐样本 CSV 完整后再写。
- 评测结束关闭服务/进程并用 `nvidia-smi` 确认显存释放。
