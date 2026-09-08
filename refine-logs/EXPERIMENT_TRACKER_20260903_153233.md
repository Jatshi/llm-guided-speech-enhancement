# GRPO v4.1 实验跟踪表

| ID | 阶段 | 目标 | 状态 | 证据 |
|---|---|---|---|---|
| L01 | 本地基线 | 运行现有测试并记录基线 | complete | 修改前 101 tests passed |
| L02 | TDD | 新增 collapse recovery 失败测试 | complete | 六类失败按预期复现后修复 |
| L03 | 实现 | reward、anchor、gate、lineage、recovery runner | complete | 代码、两份配置与一键脚本已生成 |
| L04 | 本地验收 | pytest + Ruff + dry-run | complete | 113 tests passed；Ruff passed；两份配置 dry-run passed |
| L05 | 负对照 | 新门禁拒绝 v4.0 collapsed GRPO | complete | SFT JSON/reward=1.0/0.97214；旧 GRPO=0/0，status=failed |
| G01 | GPU canary | 真实 SFT adapter 生成门禁 | blocked-autoDL | 需要用户开启 AutoDL |
| G02 | GPU 短训 | 12-step 策略信号与格式稳定性 | blocked-G01 | 一键脚本已就绪 |
| G03 | GPU 正式训练 | 300-step GRPO recovery | blocked-G02 | 一键脚本已就绪 |
| E01 | 最终评测 | 1,416 条推理、泛化与 SFT 对照门禁 | blocked-G03 | 自动验收代码已就绪 |

下一动作：用户开启 AutoDL 后，先同步本分支并运行 `LSE_GRPO_RECOVERY_MODE=all bash scripts/autodl_v4_1_grpo_recovery.sh`。任何 canary 或短训门禁失败都会在正式训练前停止。
