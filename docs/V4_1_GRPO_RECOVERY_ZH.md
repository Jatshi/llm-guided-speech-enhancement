# 4.1 GRPO 坍缩修复与 AutoDL 执行说明

## 结论先说

v4.0 的 GRPO 不是“效果一般”，而是没有获得有效的 group-relative 学习信号：它从已经输出非法数值占位符的 DPO adapter 启动，严格 JSON reward 让同组所有候选都得到 0 分，归一化 advantage 随之全部归零。300 个 optimizer steps、每步 4 次梯度累积，共 1,200 个微组全部饱和；训练 loss 中只剩 KL 项，最终 1,416 条测试输出全部无法解析。

4.1 保留原产品逻辑，只修复训练协议。修复版已经在 RTX 3080 Ti 12GB 上完成 300-step
真实训练、1,416 条测试推理、DSP 泛化评测与晋升门禁，最终状态为 `passed`。

## 七个实质性修复

### 1. 从 SFT 而不是 DPO 启动

v4.0 的 SFT 在留出集上达到 1,416/1,416 JSON 有效，DPO 则已经破坏数值语法。因此 4.1 的 reference adapter、policy adapter 和 audio projector 都从 SFT final 克隆。DPO 结果仍保留为负证据，但不再成为 GRPO 的输入。

代码同时修复了一个隐藏的 lineage 问题：旧执行器虽然配置里有 `input_stage`，实际运行却总把“上一个阶段”当输入。4.1 会按配置显式解析依赖，`grpo.input_stage=sft` 不再只是写在 JSON 里却不生效。

### 2. 给不完整 JSON 一个很小的方向信号

旧 reward 对任何解析失败输出都返回五项全零。4.1 对“以 `{` 开始、已经出现必要字段、但在末尾截断”的 JSON 前缀提供最多 0.15 的 format 分；五项平均后总分最多 0.03，永远压不过有效 JSON。

纯文本垃圾、超长文本和包含 `?` 占位符的输出仍得 0 分。它不是放宽最终协议：DSP 边界与最终评测依然只接受严格 JSON。这个 shaping reward 只用于让早期探索知道“更接近闭合 JSON”的方向。

### 3. SFT anchor 防止格式遗忘

每个 GRPO 微组除策略损失和 reference KL 外，还对该样本的 chosen response 计算一次交叉熵：

`L = L_GRPO + 0.05 × L_anchor`

这使偶发 reward tie 不会把更新退化为纯 KL，也持续提醒模型输出完整、可执行的处方格式。anchor 权重很小，GRPO 的相对偏好仍是主要优化目标；它的职责是保护已经学会的结构能力。

### 4. optimizer step 之前先做真实生成 canary

正式训练前，模型必须在真实音频 prompt 上采样多组候选。默认门槛：

- JSON 有效率至少 0.80；
- 非饱和组比例至少 0.20；
- canary 必须写出所有 reward 分量、组内方差和最多 24 条原始样本。

不满足时在第 0 个 optimizer step 前终止，产物写入 `canary_report.json`，避免再次花数小时跑一条没有 policy signal 的曲线。

### 5. 训练中持续监控 reward collapse

每个 group 记录 mean/std/span、JSON 有效率、五项 reward 和是否饱和，保存到 `grpo_diagnostics.jsonl`。若连续 16 组 reward 完全相同，流程写出 `collapse_report.json` 后中止。这个阈值允许任务本身出现若干满分 tie，但不允许长时间伪训练。

stage manifest 新增：`valid_json_rate`、`non_saturated_group_rate`、`mean_anchor_loss`、`mean_policy_loss`、`mean_kl`、最大连续饱和组数和完整 canary 摘要。

### 6. GRPO 不能因为“最后训练”就自动晋升

正式训练结束后，用 greedy decoding 对完整 1,416 条 test split 重新推理，并和已验证 SFT 在完全相同的样本上比较。最低门槛是：

- recovered GRPO JSON 有效率至少 0.99；
- `?` 占位符比例必须为 0；
- 平均可验证 reward 相比 SFT 的下降不超过 0.005；
- 泛化评测必须完整产出。

任一失败时保留模型和诊断，但 `grpo_acceptance.json` 标记 failed，不替换 SFT、不发布“GRPO 提升”结论。

### 7. 用连续参数校准把“合法”reward 变成“更优”reward

初版 4.1 在 48/48 输出均为有效 JSON 的情况下，仍只有 2/12 个 canary group 出现奖励
差异。根因是原参数分量只检查边界：例如 8dB 和 12dB 的降噪量只要都合法，就得到同分，
GRPO 看不到哪一个更接近已知退化。

最终版在 `parameter_bounds` 分量内部加入连续校准：按动作类型对齐候选与合成数据的
`expected_response`，对 `reduction_db`、`gain_db`、频率、Q 值等数值按相对偏差连续计分，
再与原边界合法性分数等权融合。这个目标由数据生成过程确定，不需要 LLM judge；缺少
真值的外部数据保持原评分行为。修复后正式 canary 的 12/12 个组均非饱和。

## 为什么先短训再正式训练

单纯的第 0 步 canary 只能证明“初始策略有可用样本和 reward 方差”，不能证明梯度更新后仍稳定。因此一键脚本先执行独立的 12-step recovery：8 组预训练 canary 通过后，再进行 12 个 optimizer steps，并检查训练期 JSON 有效率、非饱和组比例和 anchor loss。只有这一层也通过，才开始从原始 SFT 重新启动 300-step 正式实验。

短训与正式实验使用不同输出目录，避免 canary checkpoint 污染正式结果。

## AutoDL 执行

项目、v4.0 输出、模型目录和虚拟环境恢复到原路径后运行：

```bash
cd /root/autodl-tmp/lse-v4
export LSE_GRPO_RECOVERY_MODE=all
bash scripts/autodl_v4_1_grpo_recovery.sh
```

可分步执行：

```bash
LSE_GRPO_RECOVERY_MODE=preflight bash scripts/autodl_v4_1_grpo_recovery.sh
LSE_GRPO_RECOVERY_MODE=canary bash scripts/autodl_v4_1_grpo_recovery.sh
LSE_GRPO_RECOVERY_MODE=full bash scripts/autodl_v4_1_grpo_recovery.sh
```

脚本复用：

- `outputs/native_v4/sft/final`；
- `outputs/native_v4/audio_embedding_cache`；
- `outputs/native_v4/sft_test_predictions/predictions.jsonl`；
- 本地基础模型目录。

所以不需要重跑 1,500-step SFT 和 800-step DPO，也不需要重新编码 20,000 条音频。若缓存缺失，程序会按音频 SHA-256 判断并补算，而不是静默使用错误缓存。

## 需要重点观察的文件

| 时机 | 文件 | 判断 |
| --- | --- | --- |
| 训练前 | `outputs/native_v4_1_grpo_canary/grpo/canary_report.json` | status 必须 passed |
| 短训后 | `.../grpo/stage_manifest.json` | JSON 与非饱和率达标 |
| 正式训练中 | `outputs/native_v4_1_grpo/grpo/grpo_diagnostics.jsonl` | reward 有跨度、无长串饱和 |
| 异常时 | `.../grpo/collapse_report.json` | 出现即代表主动止损 |
| 正式训练后 | `.../test_predictions/prediction_report.json` | successful 接近/达到 1,416 |
| 最终 | `.../grpo_acceptance.json` | status 必须 passed |

## 本地已经验证了什么

- 原始 101 个测试在修改前全部通过；
- 新增 collapse recovery、lineage、anchor、recovery runner、acceptance gate 测试；
- 完整 113 个测试与 Ruff 检查通过；
- 两份 AutoDL 配置可解析，recovery 路径和参数正确；
- 用 v4.0 的真实 1,416 条预测做负对照，新门禁正确拒绝旧 GRPO：SFT JSON 有效率 1.0、平均 reward 0.97214；旧 GRPO 分别为 0 和 0。

## 最终实测结论与边界

GPU canary、300-step QLoRA/DeepSpeed 训练、完整测试推理与客观 DSP 泛化评测已经完成。
训练期间 1,200 个 group 中只有 2 个饱和，非饱和率 99.83%；留出集有效 JSON 为 100%，
平均可验证 reward 从 0.953821 提升到 0.976320。该结果证明 reward collapse 已被修复并
产生了稳定 group-relative 信号，但不等价于对所有真实噪声场景都显著提升；数据仍以
LibriSpeech 与物化合成退化为主，进一步结论需要真实录音、多语言和人工听测。

完整数字见 [`V4_1_RESULTS_ZH.md`](V4_1_RESULTS_ZH.md)。
