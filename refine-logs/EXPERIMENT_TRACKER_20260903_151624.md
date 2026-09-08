# GRPO v4.1 实验跟踪表

| ID | 阶段 | 目标 | 状态 | 证据 |
|---|---|---|---|---|
| L01 | 本地基线 | 运行现有测试并记录基线 | pending | 待运行 |
| L02 | TDD | 新增 collapse recovery 失败测试 | pending | 待运行 |
| L03 | 实现 | reward、anchor、gate、lineage、recovery runner | pending | 待实现 |
| L04 | 本地验收 | pytest + Ruff + dry-run | pending | 待运行 |
| G01 | GPU canary | 真实 SFT adapter 生成门禁 | blocked-autoDL | 等待用户开机 |
| G02 | GPU 短训 | 验证 policy signal、checkpoint 和恢复 | blocked-G01 | 依赖 G01 |
| G03 | GPU 正式训练 | 完整 GRPO recovery | blocked-G02 | 依赖 G02 |
| E01 | 最终评测 | 全量测试集与泛化评测 | blocked-G03 | 依赖 G03 |

