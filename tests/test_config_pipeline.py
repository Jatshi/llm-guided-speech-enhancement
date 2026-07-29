from __future__ import annotations

import json
from pathlib import Path

import pytest

from lse_v2.config import ConfigError, find_latest_checkpoint, load_config
from lse_v2.contracts import build_alignment_datasets
from lse_v2.evaluation import evaluate_rewards
from lse_v2.io import read_jsonl, write_jsonl
from lse_v2.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[1]


def test_latest_checkpoint_is_numeric(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-bad").mkdir()
    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-100"


def test_reference_evaluation_is_machine_readable(tmp_path: Path) -> None:
    build_alignment_datasets(REPO / "examples" / "audio_manifest.smoke.jsonl", tmp_path / "data")
    report = evaluate_rewards(tmp_path / "data" / "grpo" / "eval.jsonl")
    assert report["schema_version"] == "lse.reward_report.v2"
    assert report["num_samples"] == 2
    assert report["metrics"]["valid_json_rate"] == 1.0
    assert report["missing_predictions_filled_with_reference"] == 2
    assert set(report["reward_ablations"]) == {
        "without_format",
        "without_diagnosis",
        "without_parameter_bounds",
        "without_consistency",
        "without_overprocessing",
    }
    assert "not a model benchmark" in report["note"]


def test_prediction_evaluation_never_backfills_missing_rows(tmp_path: Path) -> None:
    build_alignment_datasets(REPO / "examples" / "audio_manifest.smoke.jsonl", tmp_path / "data")
    dataset = tmp_path / "data" / "grpo" / "eval.jsonl"
    rows = read_jsonl(dataset)
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "sample_id": rows[0]["sample_id"],
                "response": rows[0]["reward_context"]["expected_response"],
            }
        ],
    )
    report = evaluate_rewards(dataset, predictions_path=predictions)
    assert report["dataset_samples"] == 2
    assert report["num_samples"] == 1
    assert report["prediction_coverage"] == 0.5
    assert report["missing_predictions_skipped"] == 1
    assert report["missing_predictions_filled_with_reference"] == 0
    assert report["metrics"]["valid_json_rate"] == 1.0
    assert "never filled with references" in report["note"]


def test_prediction_evaluation_reports_invalid_json_rate(tmp_path: Path) -> None:
    build_alignment_datasets(REPO / "examples" / "audio_manifest.smoke.jsonl", tmp_path / "data")
    dataset = tmp_path / "data" / "grpo" / "eval.jsonl"
    rows = read_jsonl(dataset)
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {"sample_id": rows[0]["sample_id"], "response": "{invalid"},
            {
                "sample_id": rows[1]["sample_id"],
                "response": rows[1]["reward_context"]["expected_response"],
            },
        ],
    )

    report = evaluate_rewards(dataset, predictions_path=predictions)

    assert report["num_samples"] == 2
    assert report["metrics"]["valid_json_rate"] == 0.5


def test_pipeline_dry_run_validates_every_stage(tmp_path: Path) -> None:
    build_alignment_datasets(REPO / "examples" / "audio_manifest.smoke.jsonl", tmp_path / "data")
    config = json.loads((REPO / "configs" / "smoke.json").read_text(encoding="utf-8"))
    config["project"]["root"] = str(tmp_path)
    for stage in ("sft", "dpo", "grpo"):
        config["data"][f"{stage}_train"] = f"data/{stage}/train.jsonl"
        config["data"][f"{stage}_eval"] = f"data/{stage}/eval.jsonl"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    state = run_pipeline(config_path, dry_run=True)

    assert state["status"] == "validated"
    assert set(state["stages"]) == {"sft", "dpo", "grpo"}
    assert all(item["status"] == "validated" for item in state["stages"].values())
    assert Path(config["project"]["root"], "outputs/smoke/reward_report.json").is_file()


def test_reference_configs_are_valid() -> None:
    autodl = load_config(REPO / "configs" / "autodl_4090.json")
    assert autodl["project"]["seed"] == 42
    assert autodl["training"]["stages"]["dpo"]["label_smoothing"] == 0.1
    assert load_config(REPO / "configs" / "smoke.json")["project"]["seed"] == 42


def test_dpo_label_smoothing_rejects_invalid_probability(tmp_path: Path) -> None:
    config = json.loads((REPO / "configs" / "smoke.json").read_text(encoding="utf-8"))
    config["training"]["stages"]["dpo"]["label_smoothing"] = 0.5
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="label_smoothing"):
        load_config(path)
