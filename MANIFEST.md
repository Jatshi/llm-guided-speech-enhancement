# Research Output Manifest

> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-09-03 15:16 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260903_151624.md | implementation | permanent GRPO recovery experiment plan |
| 2026-09-03 15:16 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | latest experiment plan pointer |
| 2026-09-03 15:16 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260903_151624.md | implementation | initial recovery tracker |
| 2026-09-03 15:32 | /light-backend-coding | lse_v2/grpo_control.py | implementation | reward diagnostics and canary gate |
| 2026-09-03 15:32 | /light-backend-coding | lse_v2/grpo_recovery.py | implementation | recovery-only training entrypoint |
| 2026-09-03 15:32 | /light-backend-coding | lse_v2/grpo_acceptance.py | implementation | held-out promotion gate |
| 2026-09-03 15:32 | /light-backend-coding | configs/native_audio_autodl_grpo_recovery_canary.json | implementation | 12-step GPU canary profile |
| 2026-09-03 15:32 | /light-backend-coding | configs/native_audio_autodl_grpo_recovery_32gb.json | implementation | 300-step recovery profile |
| 2026-09-03 15:32 | /light-backend-coding | scripts/autodl_v4_1_grpo_recovery.sh | implementation | fail-closed AutoDL runner |
| 2026-09-03 15:32 | /light-backend-coding | validation/v4_0_negative_control_acceptance.json | implementation | old collapsed GRPO rejection evidence |
| 2026-09-03 15:32 | /light-backend-coding | docs/V4_1_GRPO_RECOVERY_ZH.md | implementation | Chinese recovery and run guide |
| 2026-09-03 15:32 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260903_153233.md | implementation | local-complete tracker snapshot |
| 2026-09-03 15:32 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest tracker pointer |
| 2026-09-03 15:32 | /light-backend-coding | scripts/ensure_v4_models.py | implementation | idempotent base-model preparation |
| 2026-09-03 15:32 | /light-backend-coding | tests/test_ensure_v4_models.py | implementation | model artifact completeness test |
| 2026-09-08 20:50 | /run-experiment | configs/native_audio_autodl_grpo_recovery_canary_12gb.json | implementation | verified 12GB 12-step GRPO canary profile |
| 2026-09-08 20:50 | /run-experiment | configs/native_audio_autodl_grpo_recovery_12gb.json | implementation | 12GB 300-step recovery profile |
| 2026-09-08 20:50 | /run-experiment | scripts/autodl_v4_1_session.sh | implementation | detached runner with durable exit code and allocator config |
| 2026-09-17 12:00 | /light-backend-coding | lse_v2/mmdit/ | implementation | prescription-conditioned MM-DiT, observed-source flow, safety gate, planner export and acoustic router |
| 2026-09-17 12:00 | /light-backend-coding | configs/mmdit_32gb_residual_hybrid_safe.json | implementation | final safety-gated 32GB evaluation configuration |
| 2026-09-17 12:00 | /light-backend-coding | configs/mmdit_planner_sft_32gb.json | implementation | Qwen2.5-1.5B planner SFT configuration |
| 2026-09-17 12:00 | /light-backend-coding | scripts/autodl_mmdit_run.sh | implementation | full AutoDL MM-DiT preparation, training and evaluation entrypoint |
| 2026-09-17 12:00 | /run-experiment | validation/v5_mmdit/evaluation_report.json | evaluation | measured 200-sample, six-arm final evaluation |
| 2026-09-17 12:00 | /run-experiment | validation/v5_mmdit/per_sample.csv | evaluation | 1,200-row auditable per-sample evaluation table |
| 2026-09-17 12:00 | /analyze-results | docs/MMDIT_AUTODL_EXECUTION_REPORT_2026-09-17.md | analysis | complete training, failure analysis, claim boundary and hashes |
| 2026-09-17 12:00 | /light-backend-coding | MODEL_CARD_V5_MMDIT.md | release | v5 component, evaluation, safety and limitation card |
| 2026-09-17 12:00 | /light-backend-coding | docs/V5_RELEASE_NOTES_ZH.md | release | Chinese v5 release notes and lessons learned |
