# Run Manifest — LSE 2.0

状态：**AutoDL全量SFT已完成；conservative DPO运行中；GRPO与四阶段矩阵排队自动执行。**

## 固定环境

| 项目 | 值 |
| --- | --- |
| Python | 3.10或3.11 |
| GPU | RTX 4090 24GB |
| PyTorch | 2.5.1 + CUDA 12.1 wheel |
| Transformers | 4.48.3 |
| TRL | 0.16.1 |
| PEFT | 0.14.0 |
| DeepSpeed | 0.16.3 |
| 基座 | Qwen/Qwen2.5-1.5B-Instruct |
| Seed | 42 |
| 配置 | `configs/autodl_4090.json` |

## 本地已验证命令

```bash
python -m compileall -q lse_v2 src
python -m lse_v2.data_cli build \
  --manifest examples/audio_manifest.smoke.jsonl \
  --output-dir outputs/smoke/data \
  --seed 42
python -m lse_v2.pipeline --config configs/smoke.json --dry-run
python -m pytest  # 35 passed
python -m ruff check .
python -m ruff format --check .
python scripts/distributed_contract_smoke.py --world-size 2
```

## AutoDL实际运行入口

```bash
export PORTFOLIO_V2_MODE=full
export AUDIO_MANIFEST=/root/autodl-tmp/datasets/audio_manifest.v2.jsonl
bash scripts/autodl_v2_bootstrap.sh
bash scripts/autodl_v2_run.sh
```

## 预期产物路径

| 产物 | 路径 |
| --- | --- |
| SFT adapter | `outputs/v2/sft/final/` |
| DPO adapter | `outputs/v2/dpo/final/` |
| GRPO adapter | `outputs/v2/grpo/final/` |
| 三阶段状态 | `outputs/v2/<stage>/stage_manifest.json` |
| 总管线状态 | `outputs/v2/pipeline_status.json` |
| 模型预测 | `outputs/v2/evaluation/predictions.jsonl` |
| 奖励与消融报告 | `outputs/v2/evaluation/reward_report.json` |

## GPU结果（严禁预填）

| 字段 | 真实值 |
| --- | --- |
| 开始/结束时间 | SFT：2026-07-29 04:39:34–06:09:01 UTC；总管线仍在运行 |
| Git commit | 本地特性分支尚未发布；最终发布时填写 |
| 数据清单SHA256 | 最终归档后从dataset manifest填写 |
| SFT峰值显存/耗时 | 峰值显存待最终日志审计；耗时5,243.8s |
| DPO峰值显存/耗时 | 运行中 |
| GRPO峰值显存/耗时 | 待运行 |
| SFT train/held-out loss | 0.09630 / 0.08168 |
| SFT held-out token accuracy | 0.96517 |
| 最终总奖励 | GRPO后填写 |
| JSON合法率 | 四阶段矩阵后填写 |
| 诊断一致性 | 四阶段矩阵后填写 |
| 过处理违规率 | 四阶段矩阵后填写 |
| 失败案例数 | 四阶段矩阵后填写 |

GitHub 草稿 PR 已建立；Python 3.10、3.11、3.12 三组 CI 均通过。PR 在最终
GRPO adapter、四阶段原始预测、模型卡和 Hugging Face 链接提交前保持 draft。

## DeepSpeed对照（待真实GPU运行）

原始记录模板：
`docs/deepspeed_comparison_template.csv`

| Profile | World size | 峰值显存 | tokens/s | samples/s | 证据状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| 无DeepSpeed | 1 | 待运行 | 待运行 | 待运行 | pending |
| ZeRO-2 | 1 | 待最终日志审计 | 未单独记录 | SFT 21.74 | SFT verified；cDPO running |
| ZeRO-3 CPU offload | 1 | 待运行 | 待运行 | 待运行 | pending |

`world_size=1`时ZeRO-2没有跨GPU分片收益。ZeRO-3 CPU offload仅作为显存换吞吐方案，
不能在实测前声称更优。CPU多进程contract smoke也不是多GPU证据。

smoke中产生的参考答案奖励报告只用于验证奖励实现，不能抄作模型结果。
