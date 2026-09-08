# LLM-Guided Speech Enhancement 4.1 最终实测结果

## 结论

v4.1 已在 RTX 3080 Ti 12GB 上完成 GRPO 修复的完整真实链路。训练前 canary、300-step
QLoRA + DeepSpeed ZeRO-2 训练、1,416 条测试推理、DSP 泛化评测和 SFT 对照晋升门禁
全部成功。`grpo_acceptance.json` 为 `passed`，因此本版本可以发布 recovered GRPO，
不再是 smoke 或“等待 GPU”的候选状态。

## 关键结果

| 阶段 | 指标 | 结果 |
| --- | --- | ---: |
| 正式 canary | 有效 JSON | 48/48 |
| 正式 canary | 非饱和 group | 12/12 |
| GRPO 训练 | optimizer steps / groups | 300 / 1,200 |
| GRPO 训练 | sampled valid JSON rate | 0.991458 |
| GRPO 训练 | non-saturated group rate | 0.998333 |
| GRPO 训练 | longest saturated run | 1 |
| GRPO 训练 | mean reward | 0.964207 |
| GRPO 训练 | mean anchor loss | 0.114845 |
| GRPO 训练 | elapsed | 14,526.04s |
| 测试推理 | successful / failed | 1,416 / 0 |
| 测试推理 | mean / P50 / P95 latency | 786.15 / 784.31 / 873.03ms |
| 晋升门禁 | SFT mean reward | 0.953821 |
| 晋升门禁 | recovered GRPO mean reward | **0.976320** |
| 晋升门禁 | 两侧 valid JSON / placeholder | 1.0 / 0.0 |
| 晋升门禁 | status | **passed** |

## 客观 DSP 泛化评测

测试集为 1,416 条 LibriSpeech train-clean-100 语音与物化噪声退化。系统先生成处方，
执行候选 DSP，再用客观指标决定接受或回滚；回滚样本输出原始输入，避免负增益传播。

| 指标 | 全集均值 |
| --- | ---: |
| 接受率 / 回滚率 | 68.15% / 31.85% |
| SI-SDR gain | **+0.35468 dB** |
| PESQ gain | **+0.04056** |
| STOI gain | **+0.00260** |

所有 SNR 分桶的三项平均增益均为正。最困难的 `<5dB` 分桶 SI-SDR gain 为
+0.41457dB；`10–20dB` 分桶接受率最高，为 70.14%。

## GRPO 奖励坍缩是怎样修复的

v4.0 从已损坏数值语法的 DPO adapter 启动，严格 JSON reward 使同组候选全部为 0，
advantage 因而归零。v4.1 先改为从已验证 SFT 启动，并加入小上限的 JSON-progress shaping、
SFT anchor、训练前 canary 和连续饱和 fail-fast。

第一次全量 canary 又发现更细的第二层问题：输出已经合法，但 reward 只判断参数是否在
边界内，导致合法参数大量同分。最终修复利用合成退化自带的 `expected_response`，对同
动作的衰减、增益、频率和 Q 值按相对误差连续计分。这里不使用 LLM 裁判，reward 可以
离线复算。修复后 canary 非饱和率由 16.67% 提升为 100%，训练期保持 99.83%。

## 真实踩坑与解决方式

1. **DPO 污染 GRPO 起点**：显式支持 `grpo.input_stage=sft`，并修复执行器忽略 lineage
   配置的问题。
2. **12GB 显存 OOM**：保持每组 4 个候选，缩短上下文/生成上限，启用 expandable
   segments 与梯度检查点，而不是牺牲 GRPO 的组内比较能力。
3. **ZeRO-2 重复规约断言**：将梯度检查点改为 non-reentrant，避免同一参数在重入反向
   中被 DeepSpeed 重复标记完成。
4. **合法 JSON 仍同分**：没有降低门禁，而是把二值边界奖励改成边界 + 连续目标校准。
5. **推理看起来比训练慢**：测试阶段逐条生成并执行 1,416 条处方，随后还对每条候选
   运行 SI-SDR/PESQ/STOI 与回滚审计；它处理的是完整数据集，而不是少量 optimizer step。

## 可审计产物

远端运行根目录：`outputs/native_v4_1_grpo_12gb/`。

- `grpo/canary_report.json`：48 个训练前生成及分组方差；
- `grpo/grpo_diagnostics.jsonl`：1,200 个训练 group 的 reward 诊断；
- `grpo/stage_manifest.json`：训练聚合指标；
- `grpo/final/`：LoRA adapter、audio projector 与 artifact manifest；
- `test_predictions/predictions.jsonl`：1,416 条真实模型输出；
- `test_predictions/prediction_report.json`：成功率、延迟和吞吐；
- `generalization/benchmark_report.json`：总体与 SNR 分桶的客观增益；
- `grpo_acceptance.json`：SFT 与 recovered GRPO 的最终晋升判定；
- `run_manifest.json`、`environment.freeze.txt`：运行配置、版本与环境。

## 声明边界

这些数字证明完整链路在固定测试集上可复现，并证明原 reward collapse 已消失。它们不证明
对任意真实录音、任意语言或主观听感都显著优于 SFT。公开表述应使用“通过预注册晋升
门禁、可验证 reward 提升、所有 SNR 分桶客观均值为正”，不应写成未经听测支持的
“全面达到 SOTA”。
