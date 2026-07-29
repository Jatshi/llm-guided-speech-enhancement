# Run Manifest — LSE 2.0

状态：**AutoDL 全量 SFT、conservative DPO、300 步 GRPO 与同一 200 条留出集四阶段矩阵均已完成。**

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
python -m pytest  # 45 passed
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
| 开始/结束时间 | SFT：2026-07-29 04:39:34–06:09:01 UTC；cDPO：06:44:30–09:45:25 UTC；GRPO 稳定续训：10:51:06–12:02:17 UTC |
| Git commit | 发布提交在最终 GitHub 合并后记录；训练代码谱系保留在特性分支 |
| 四阶段样本ID SHA256 | `50eaa2c3c59d1c5441757517fd9f9bc059f6b943ed55802b0e7fd82df8c75588` |
| SFT耗时/吞吐 | 5,243.8s / 21.74 samples/s |
| DPO耗时/吞吐 | 10,633.30s / 10.721 samples/s |
| GRPO峰值显存/耗时 | 24,067 / 24,564 MiB；4,247.43s（从 checkpoint-50 稳定续训） |
| SFT train/held-out loss | 0.09630 / 0.08168 |
| SFT held-out token accuracy | 0.96517 |
| cDPO train/held-out loss | 0.324913 / 0.324529 |
| cDPO held-out pair accuracy / margin | 1.000000 / 2.188927 |
| GRPO训练平均奖励 | 0.946019（300 步；范围 0.796875–0.993750） |
| 四阶段总分 | Base 0.222188；SFT 0.964000；cDPO 0.964000；GRPO 0.964000 |
| 四阶段JSON合法率 | Base 0.355000；SFT/cDPO/GRPO 均为 1.000000 |
| 四阶段诊断分 | Base 0.222188；SFT/cDPO/GRPO 均为 1.000000 |
| 四阶段一致性分 | Base 0.267500；SFT/cDPO/GRPO 均为 0.820000 |
| 四阶段过处理约束分 | Base 0.355000；SFT/cDPO/GRPO 均为 1.000000 |

GitHub 草稿 PR 已建立；Python 3.10、3.11、3.12 三组 CI 均通过。最终
GRPO adapter 与四阶段原始预测已完成；Hugging Face 上传、最终提交和 PR 合并由
发布门禁完成。

## DeepSpeed 真实证据与边界

原始记录模板：
`docs/deepspeed_comparison_template.csv`

| Profile | World size | 峰值显存 | tokens/s | samples/s | 证据状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| 无DeepSpeed | 1 | 未运行 | 未运行 | 未运行 | 不虚构未执行对照 |
| ZeRO-2 | 1 | GRPO NVML峰值24,067 MiB | 未单独记录 | SFT 21.74；cDPO 10.721；GRPO 2.260 | checkpoint-300 optimizer/model state均验证stage 2 |
| ZeRO-3 CPU offload | 1 | 未运行 | 未运行 | 未运行 | 单卡不宣称跨GPU分片收益 |

`world_size=1`时ZeRO-2没有跨GPU分片收益。ZeRO-3 CPU offload仅作为显存换吞吐方案，
不能在实测前声称更优。CPU多进程contract smoke也不是多GPU证据。

smoke中产生的参考答案奖励报告只用于验证奖励实现，不能抄作模型结果。
