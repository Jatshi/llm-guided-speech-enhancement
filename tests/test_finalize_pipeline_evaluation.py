from __future__ import annotations

from pathlib import Path

from scripts.finalize_pipeline_evaluation import completed_pipeline_state


def test_completed_pipeline_state_clears_failure_and_records_fresh_evaluation() -> None:
    state = completed_pipeline_state(
        previous={
            "status": "failed",
            "failed_stage": "grpo",
            "error": "post-training evaluation retained training memory",
            "stages": {"sft": {"status": "complete"}},
        },
        config_path=Path("/repo/config.json"),
        stage_manifest=Path("/repo/grpo/stage_manifest.json"),
        evaluation_report=Path("/repo/evaluation/reward_report.json"),
        predictions=Path("/repo/evaluation/predictions.jsonl"),
    )

    assert state["status"] == "complete"
    assert "failed_stage" not in state
    assert "error" not in state
    assert state["stages"]["sft"]["status"] == "complete"
    assert state["stages"]["grpo"]["status"] == "complete"
    assert state["evaluation"]["status"] == "model_predictions_scored"
    assert "fresh process" in state["evaluation"]["note"]
