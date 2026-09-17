# Prescription MM-DiT v6 AutoDL 运行手册

> 目标：先用分钟级链路排除工程错误，再决定是否进入 12,000-step 正式训练。
> 默认项目路径 `/root/autodl-tmp/llm-guided-speech-enhancement`，可用环境变量覆盖。

## 1. 上机前已经准备好的入口

| 层级 | 命令 | 验证内容 |
| --- | --- | --- |
| 轻量 smoke | `bash scripts/autodl_mmdit_v6_smoke.sh` | 小模型、假数据、2 step、checkpoint、评测 |
| 数据/教师准备 | `bash scripts/autodl_mmdit_v6_prepare.sh` | 真实 manifest、WavLM、语义梯度、正式 preflight |
| 全目标 canary | `bash scripts/autodl_mmdit_v6_canary.sh` | 51.7M MM-DiT + WavLM + 三类 loss，20 step |
| 正式训练 | `bash scripts/autodl_mmdit_v6_run.sh` | 顺序执行前三层，通过后才跑 12k 与完整评测 |

## 2. 开机后的第一轮只做环境核查

```bash
export LSE_PROJECT_DIR=/root/autodl-tmp/llm-guided-speech-enhancement
export LSE_MMDIT_VENV=/root/autodl-tmp/lse-mmdit-env
cd "$LSE_PROJECT_DIR"
nvidia-smi
df -h /root/autodl-tmp
git status --short
```

不要直接开 12k。先确认旧的实测 manifest 仍在：

```bash
test -s data/mmdit/pairs.hybrid_safe_predictions.jsonl
wc -l data/mmdit/pairs.hybrid_safe_predictions.jsonl
```

如果路径不同，只设置变量，不复制或覆盖原始数据：

```bash
export MMDIT_V6_SOURCE=/absolute/path/to/pairs.hybrid_safe_predictions.jsonl
```

## 3. 推荐执行顺序

### Gate A：本地逻辑在 Linux/CUDA 环境仍成立

```bash
bash scripts/autodl_mmdit_v6_smoke.sh
```

通过标准：

- `results/mmdit-v6-smoke/preflight.json` 为 `READY`；
- `outputs/mmdit-v6-smoke/training_report.json` 为 `COMPLETED`；
- `results/mmdit-v6-smoke/evaluation_report.json` 为 `MEASURED`。

这里的指标没有质量意义，只证明代码能从数据走到评测。

### Gate B：真实数据与语义 teacher 可用

```bash
bash scripts/autodl_mmdit_v6_prepare.sh
```

该脚本会：

1. 复用 AutoDL 已有 CUDA PyTorch，仅补齐锁定的音频、指标、测试与 Transformers 依赖；
2. 跑 v6 单测与静态检查；
3. 为真实 manifest 增加 EnhanceScript；
4. 下载 `microsoft/wavlm-base-plus`；
5. 验证 WavLM 参数冻结、增强波形输入仍有梯度；
6. 对正式模型做 GPU runtime preflight。

必须检查：

```bash
cat results/mmdit-v6-stepaudio3/semantic_teacher_probe.json
cat results/mmdit-v6-stepaudio3/preflight.json
```

teacher probe 的 `status` 必须为 `PASSED`，`trainable_teacher_parameters` 必须为 0，
`input_gradient_l2` 必须为有限非负数。preflight 所有 gate 必须为 `true`。

### Gate C：20-step 全目标 canary

```bash
bash scripts/autodl_mmdit_v6_canary.sh
```

观察：

```bash
tail -f outputs/mmdit-v6-canary/canary_train.log
watch -n 2 nvidia-smi
```

停止条件：

- CUDA OOM；
- loss、grad norm 或 teacher loss 出现 NaN/Inf；
- 第 5 step 前没有 checkpoint；
- identity 初始化后长期只有 output head 有梯度；
- 20 step 时间异常长，说明语义 teacher 的频率或 batch 需要调整。

canary 失败时，不进入正式训练。优先按以下顺序降显存：

1. `batch_size: 12 → 6 → 3`，并用 gradient accumulation 保持有效 batch；
2. `gradient_checkpointing: false → true`；
3. `semantic.every_steps: 2 → 4`；
4. 再考虑 gradient accumulation 保持有效 batch。

不要首先删除 semantic loss，否则 canary 就没有验证最重要的新链路。

### Gate D：12k 正式训练与评测

前三个 gate 都通过后：

```bash
bash scripts/autodl_mmdit_v6_run.sh
```

脚本会自动从 `checkpoint-last.pt` 恢复。正式输出：

```text
outputs/mmdit-v6-stepaudio3/
  checkpoint-last.pt
  checkpoint-best.pt
  train_metrics.jsonl
  training_report.json
  formal_train.log

results/mmdit-v6-stepaudio3/
  preflight.json
  semantic_teacher_probe.json
  evaluation_report.json
  per_sample.csv
  summary.csv
```

## 4. 训练中需要记录的证据

至少每个阶段记录：

- step、phase、总 loss、base flow loss；
- velocity MSE、x0 L1、mismatch no-op loss；
- MR-STFT loss、semantic loss；
- learning rate、grad norm、耗时；
- `nvidia-smi` 峰值显存；
- git commit、config SHA256、manifest SHA256；
- checkpoint step 和恢复事件。

不能只抄终端最后一行。训练日志与 report 都要同步回本地。

## 5. 正式结果判定

最低判定不是“loss 降了”，而是同一留出集上的五臂比较：

1. spectral subtraction；
2. DeepFilterNet3；
3. MM-DiT no prescription；
4. MM-DiT oracle prescription；
5. MM-DiT shuffled prescription；
6. 有真实预测时再加 predicted prescription。

重点检查：

- oracle 应优于 shuffled/no-prescription，才说明处方被真正利用；
- predicted 应优于 no-prescription，才说明 Planner 有现实价值；
- raw 与 safety-gated 指标必须同时保留，避免 fallback 掩盖坏输出；
- CER、speaker similarity 与 waveform safety 不能因 SI-SDR 提升而被牺牲；
- smoke/canary 结果不得写成正式效果。

## 6. 一键命令与人工检查点

如果希望我在开机后接管执行，先只同步代码，不要手动运行正式训练。最安全的人工节点是：

```bash
bash scripts/autodl_mmdit_v6_smoke.sh
bash scripts/autodl_mmdit_v6_prepare.sh
bash scripts/autodl_mmdit_v6_canary.sh
```

我核对三层结果和显存后，再执行正式训练。这样即使有依赖、数据或显存问题，损失的是
几分钟而不是几个小时。
