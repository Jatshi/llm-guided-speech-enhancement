# Prescription MM-DiT v6 本地准备清单

更新时间：2026-09-17

## 已完成

- 建立分支 `feat/stepaudio3-v6-prep`，没有提交或推送；
- 加入严格 zero-velocity / no-op 初始化，且默认关闭以兼容旧版本；
- 将可微 `estimated_x0` 从 flow loss 暴露给波形级辅助目标；
- 实现 multi-resolution STFT loss；
- 实现冻结 WavLM teacher 的 semantic consistency loss；
- 实现三阶段 curriculum、每阶段 flow/loss/LR override；
- 定义并验证时间化 EnhanceScript，支持旧 manifest 自动升级；
- 实现 direct-template / LLM / abstain Adaptive Planning 决策；
- 实现基于真实增强结果的 reward 和 preference pair builder；
- 将全部新模块接入正式 train、checkpoint、report 和 preflight；
- 新增 smoke、20-step canary、32GB 正式配置；
- 新增 AutoDL smoke / prepare / canary / full-run 四个脚本；
- 本地完成 2-step 数据→训练→checkpoint→五臂评测闭环；
- 全仓库 pytest 160 项通过；本次修改文件的 Ruff 与 compileall 通过。

## 本地实测证据

| 检查 | 结果 | 边界 |
| --- | --- | --- |
| v6 单元测试 | 21 项通过 | 逻辑与合约 |
| 全量 pytest | 160 项通过 | 回归兼容 |
| preflight | READY，11 个 gate 全 true | CPU smoke，不是 GPU 质量 |
| smoke 参数量 | 120,514 | 仅接口模型 |
| 正式 MM-DiT 参数量 | 51,717,122 | 不含冻结 WavLM |
| smoke 训练 | 2/2 step 完成 | 不代表收敛 |
| smoke 评测 | 2 条、5 个 MM-DiT/DSP arm、10 行 | 不代表效果 |

## AutoDL 已完成（2026-09-17）

- 已恢复 2,000 条真实 paired manifest，并完成 1,600/200/200 划分核查；
- WavLM teacher probe 通过，冻结参数为 0 且输入梯度可回传；
- 20-step 51.7M 参数全目标 canary 完成；
- 12,000-step 三阶段正式训练完成，最佳 checkpoint 为 step 11,750；
- 200 条测试样本、六个 arm、共 1,200 行完整评测完成；
- 评测已增加 clean safety slice、corrupted-only 与按退化类型分组；
- 正式结果与局限见 `docs/MMDIT_V6_AUTODL_EXECUTION_REPORT_20260917.md`。

仍未完成：identity / MR-STFT / semantic 的独立训练消融，以及执行多个 Planner 候选后生成
真实 preference 数据。这两项不能由本次单一 full-v6 run 冒充。

## 文件索引

### 新增核心代码

```text
lse_v2/mmdit/adaptive_planner.py
lse_v2/mmdit/curriculum.py
lse_v2/mmdit/enhance_script.py
lse_v2/mmdit/enhancement_reward.py
lse_v2/mmdit/losses.py
lse_v2/mmdit/planner_preference_data.py
```

### 修改核心代码

```text
lse_v2/mmdit/model.py
lse_v2/mmdit/flow.py
lse_v2/mmdit/contracts.py
lse_v2/mmdit/config.py
lse_v2/mmdit/preflight.py
lse_v2/mmdit/train.py
```

### 配置与入口

```text
configs/mmdit_v6_stepaudio3_smoke.json
configs/mmdit_v6_stepaudio3_canary.json
configs/mmdit_v6_stepaudio3_32gb.json
scripts/probe_mmdit_v6_teacher.py
scripts/autodl_mmdit_v6_smoke.sh
scripts/autodl_mmdit_v6_prepare.sh
scripts/autodl_mmdit_v6_canary.sh
scripts/autodl_mmdit_v6_run.sh
```

### 文档与测试

```text
tests/test_mmdit_v6_stepaudio3.py
docs/MMDIT_V6_STEPAUDIO3_DESIGN_ZH.md
docs/MMDIT_V6_AUTODL_RUNBOOK_ZH.md
docs/MMDIT_V6_LOCAL_PREPARATION_INVENTORY_ZH.md
```

## 已知但未越权修改的仓库状态

仓库在本轮开始前已有若干用户未跟踪文件，包括 EXP2 相关 Python 文件、4.0/4.1 文档和
MM-DiT HTML 学习手册。本轮没有删除、覆盖或格式化这些文件。repo-wide Ruff 会报告其中
两个 EXP2 文件的既有长行；v6 修改范围本身 Ruff 通过。
