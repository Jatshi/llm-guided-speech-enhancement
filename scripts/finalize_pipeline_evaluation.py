from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lse_v2.config import load_config, resolve_path
from lse_v2.evaluation import evaluate_rewards
from lse_v2.inference import generate_predictions
from lse_v2.io import utc_now, write_json_atomic


def completed_pipeline_state(
    *,
    previous: dict[str, Any],
    config_path: Path,
    stage_manifest: Path,
    evaluation_report: Path,
    predictions: Path,
) -> dict[str, Any]:
    state = dict(previous)
    state.update(
        {
            "schema_version": "lse.pipeline_run.v2",
            "config": str(config_path),
            "dry_run": False,
            "status": "complete",
            "finished_at": utc_now(),
        }
    )
    state.pop("failed_stage", None)
    state.pop("error", None)
    stages = dict(state.get("stages") or {})
    stages["grpo"] = {"status": "complete", "manifest": str(stage_manifest)}
    state["stages"] = stages
    state["evaluation"] = {
        "status": "model_predictions_scored",
        "report": str(evaluation_report),
        "predictions": str(predictions),
        "note": (
            "Scores are computed from saved final-GRPO model predictions. "
            "Evaluation ran in a fresh process after training memory was released."
        ),
    }
    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize the fixed held-out evaluation in a fresh GPU process"
    )
    parser.add_argument("--config", default="configs/autodl_4090.json")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    stage_dir = resolve_path(config, config["training"]["stages"]["grpo"]["output_dir"])
    adapter = stage_dir / "final"
    stage_manifest = stage_dir / "stage_manifest.json"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"final GRPO adapter not found: {adapter}")
    manifest = json.loads(stage_manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != "grpo" or manifest.get("status") != "complete":
        raise RuntimeError("GRPO stage manifest is not complete")

    dataset = resolve_path(config, config["data"]["grpo_eval"])
    predictions = resolve_path(config, config["evaluation"]["predictions"])
    report_path = resolve_path(config, config["evaluation"]["report"])
    expected = int(config["evaluation"]["max_samples"])
    generated = generate_predictions(
        config_path,
        dataset,
        adapter,
        predictions,
        max_samples=expected,
    )
    if generated != expected:
        raise RuntimeError(f"expected {expected} predictions, generated {generated}")
    report = evaluate_rewards(dataset, predictions_path=predictions)
    if report.get("num_samples") != expected:
        raise RuntimeError(
            f"expected {expected} evaluated predictions, got {report.get('num_samples')}"
        )
    write_json_atomic(report_path, report)

    status_path = resolve_path(
        config, config["training"].get("pipeline_status", "outputs/v2/pipeline_status.json")
    )
    previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    state = completed_pipeline_state(
        previous=previous,
        config_path=config_path,
        stage_manifest=stage_manifest,
        evaluation_report=report_path,
        predictions=predictions,
    )
    write_json_atomic(status_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
