# Prescription MM-DiT v6 AutoDL 正式执行报告

日期：2026-09-17
状态：训练与完整留出集评测已完成；结论为“核心条件机制成立，但工程效果仍需继续提高”。

## 1. 一句话结论

v6 在 RTX 4080 SUPER 32GB 上完成了 WavLM 教师探针、20-step canary、12,000-step
三阶段训练和 200 条测试样本的六臂评测。对 160 条真实退化样本，oracle 处方相对无处方
提升 `+0.258 dB`，预测处方相对无处方提升 `+0.216 dB`，配对 bootstrap 的 95% 置信区间
都严格大于 0；shuffled 处方则更差。因此“MM-DiT 确实利用了处方”这一核心因果主张有
实测支持。另一方面，绝对增益仍小，Planner 对 reverb/telephone 的 action 还未真正覆盖，
预测臂回退率高，PESQ/STOI 也没有超过 DeepFilterNet3，不能宣传成 SOTA 或完整落地。

## 2. 执行环境与数据

| 项目 | 实测值 |
| --- | --- |
| GPU | NVIDIA RTX 4080 SUPER，32,760 MiB |
| Python / PyTorch | 3.10.8 / 2.1.2+cu118 |
| 正式模型参数 | 51,717,122 |
| 冻结教师 | `microsoft/wavlm-base-plus`，layer 9 |
| 数据 | 2,000 对：train 1,600 / val 200 / test 200 |
| 测试构成 | clean、white、pink、reverb、telephone 各 40 条 |
| 正式步数 | 12,000 |
| 最佳 checkpoint | step 11,750，`joint_cooldown` |

WavLM 不能直接从 Hugging Face 主站下载，最终通过 `hf-mirror.com` 完整缓存后改用离线模式。
教师探针结果为 `PASSED`：教师可训练参数为 0，增强波形输入梯度 L2 为 `2.8898`，说明
teacher 冻结但语义损失仍能向 MM-DiT 反传。

## 3. 训练链路

| 阶段 | Step | 作用 |
| --- | ---: | --- |
| oracle_alignment | 1–4,000 | 先学习稳定的 noisy→clean 映射 |
| safety_alignment | 4,001–9,000 | 加入条件 dropout、mismatch no-op、MR-STFT 和语义约束 |
| joint_cooldown | 9,001–12,000 | 低学习率联合收敛 |

训练报告状态为 `COMPLETED`，最佳验证损失为 `0.232174`。报告记录的 1,502 秒是恢复后的
计时；加上第一次运行到 4,500 步的部分，总训练墙钟时间约 34 分钟。清洗后的
`train_metrics.jsonl` 恰好包含 step 1–12,000，每步一条。

## 4. 运行中发现并修复的工程问题

### 4.1 跨 curriculum 阶段比较 validation loss

原实现用同一个全局 best 比较三个阶段，而三个阶段的 loss 权重不同，数值不可直接比较。
这会让第一阶段的小数值长期锁住 `checkpoint-best.pt`。训练在 step 4,500 checkpoint 处暂停，
修复为“阶段内 best”：进入新阶段时重新建立 best，再从原 optimizer/scheduler 状态恢复。
最终最佳 checkpoint 正确落在 `joint_cooldown` 的 step 11,750。

### 4.2 恢复训练后 JSONL 出现重复 step

暂停时内存中的训练已超过最近 checkpoint，但 checkpoint 只保存到 step 4,500。恢复后，
旧日志中 checkpoint 之后的 171 行与新运行重复。模型参数没有问题，但日志统计不再一一对应。
现已加入恢复前自动裁剪和按 step 去重逻辑，并把本次日志清洗为严格 12,000 行。

## 5. 为什么原始总均值看起来“全部失败”

200 条测试样本中有 40 条 clean identity 样本，其 noisy 与 clean 基本相同，输入 SI-SDR
约为 143 dB。模型只要做极小改动，SI-SDR improvement 就可能记为约 `-100 dB`。因此把
clean 安全测试和真实退化任务直接算术平均，会让 160 条退化样本上的正增益被完全淹没。

评测器现已原生写出输入 SI-SDR/SNR、退化类型、mean/median、`corrupted_only` 和每类退化
的分组结果。clean 仍被保留，因为它回答“模型会不会破坏本来就干净的语音”；只是不能再
与增强任务混成一个均值来解释。

## 6. 160 条真实退化样本结果

下表的 SI-SDR 是安全门控前的 raw 模型输出，避免 fallback 掩盖模型能力。

| Arm | raw SI-SDR 增益 / dB | STOI | PESQ | fallback |
| --- | ---: | ---: | ---: | ---: |
| DeepFilterNet3 | +0.374 | 0.8719 | 1.9886 | 0% |
| MM-DiT 无处方 | +0.005 | 0.8578 | 1.3683 | 0% |
| MM-DiT oracle 处方 | **+0.263** | 0.8647 | 1.3149 | 0% |
| MM-DiT predicted 处方 | **+0.221** | 0.8578 | 1.3684 | 70.6% |
| MM-DiT shuffled 处方 | -0.114 | 0.8575 | 1.3599 | 0% |
| Spectral subtraction | -1.124 | 0.8408 | 1.3459 | 0% |

predicted arm 的 raw 增益为 `+0.221 dB`，经过现有 confidence gate 后反而只剩
`+0.138 dB`。gate 在不少实际有小幅正收益的样本上选择了回退，说明下一版应做按退化
类型校准的风险—收益门控。

## 7. 配对统计与 v5 对比

对同一批 160 条样本做 20,000 次配对 bootstrap：

| 对比 | 平均差 / dB | 胜率 | 95% CI |
| --- | ---: | ---: | --- |
| oracle − 无处方 | +0.258 | 64.4% | [+0.089, +0.424] |
| oracle − shuffled | +0.376 | 75.6% | [+0.195, +0.552] |
| predicted − 无处方 | +0.216 | 68.1% | [+0.142, +0.294] |
| DeepFilterNet3 − oracle | +0.111 | 62.5% | [-0.537, +0.726] |

与同数据上的 v5 比较：

- oracle 增益提高 `+0.126 dB`，95% CI `[+0.060, +0.194]`；
- predicted 提高 `+0.120 dB`，但 CI `[-0.037, +0.288]` 穿过 0；
- 无处方基本不变，说明提升来自条件利用而不是整体模型漂移；
- shuffled 比 v5 更差 `-0.087 dB`，结合 oracle 变好，扩大了正确与错误条件的因果间隔。

## 8. Planner 与安全门控的真实边界

Planner 的退化类型命中 158/200（79%），但 200 条 predicted prescription 的 action 当前
全部是 `spectral_subtraction`。对 reverb 和 telephone，虽然 diagnosis 多数正确，却没有
真正输出 dereverb / bandwidth-extension action，confidence 也接近 0，因此全部或几乎全部
被 gate 回退。当前 predicted arm 只能证明“预测条件对 white/pink 有帮助”，不能声称已经
完成全退化类型的自适应处理。

## 9. 简历表述边界

可以写：

> 构建 51.7M 参数的处方条件 MM-DiT，引入零速度初始化、三阶段 curriculum、MR-STFT 与
> 冻结 WavLM 语义约束；在 160 条退化语音配对评测中，oracle / predicted 处方相对无处方
> 分别获得 +0.258 / +0.216 dB 配对 SI-SDR 增益，并通过 shuffled-condition 因果对照验证
> 模型真实利用结构化条件。

不能写“全面超过 DeepFilterNet3”“所有退化类型均已支持”或“SOTA”。安全门控后的全体
正均值也不能被描述为模型 raw 质量提升，因为其中包含大量 fallback。

## 10. 下一步优先级

1. 给 Planner 增加真实的 dereverb 与 bandwidth-extension action，而不是只改 diagnosis；
2. 用每种退化的 held-out calibration set 学习 gate，优化期望收益而非单一 confidence 阈值；
3. 在 clean 样本上增加严格 identity regularization，报告改变量而非不稳定的超高 SI-SDR；
4. 以 DeepFilterNet3 为 teacher 或 residual front-end，重点追赶 PESQ/STOI；
5. 再做 identity / MR-STFT / semantic 三项消融，确认 v6 的 `+0.126 dB` oracle 改善来自哪里。

## 11. 证据文件

```text
results/mmdit-v6-stepaudio3/
  evaluation_report.json
  diagnostic_summary.json
  per_sample.csv
  summary.csv
  semantic_teacher_probe.json
  preflight.json
  formal_evaluation_v2.log

outputs/mmdit-v6-stepaudio3/
  checkpoint-best.pt
  checkpoint-last.pt
  train_metrics.jsonl
  training_report.json
  formal_train.log
```
