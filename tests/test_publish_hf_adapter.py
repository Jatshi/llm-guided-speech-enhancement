from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.publish_hf_adapter import validate_release


def release_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    artifact_dir = tmp_path / "final"
    artifact_dir.mkdir()
    (artifact_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    manifest_path = tmp_path / "stage_manifest.json"
    manifest_path.write_text(
        json.dumps({"stage": "grpo", "status": "complete"}),
        encoding="utf-8",
    )
    model_card_path = tmp_path / "MODEL_CARD.md"
    model_card_path.write_text("---\nlicense: mit\n---\n# Final model\n", encoding="utf-8")
    evaluation_path = tmp_path / "stage_matrix.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "records": 200,
                "selection": {
                    "method": "uniform_without_replacement",
                    "seed": 42,
                    "sample_ids_sha256": "a" * 64,
                },
                "stages": {"base": {}, "sft": {}, "dpo": {}, "grpo": {}},
            }
        ),
        encoding="utf-8",
    )
    for stage in ("base", "sft", "dpo", "grpo"):
        (tmp_path / f"{stage}_predictions.jsonl").write_text(
            "\n".join(
                json.dumps({"sample_id": str(index), "response": "{}"}) for index in range(200)
            )
            + "\n",
            encoding="utf-8",
        )
        (tmp_path / f"{stage}_reward_report.json").write_text(
            json.dumps({"num_samples": 200}),
            encoding="utf-8",
        )
    return artifact_dir, manifest_path, model_card_path, evaluation_path


def test_validate_release_accepts_completed_grpo(tmp_path: Path) -> None:
    artifact_dir, manifest_path, model_card_path, evaluation_path = release_fixture(tmp_path)
    summary = validate_release(
        artifact_dir,
        manifest_path,
        model_card_path,
        evaluation_path,
    )
    assert summary["stage"] == "grpo"
    assert summary["status"] == "complete"


def test_validate_release_rejects_incomplete_stage(tmp_path: Path) -> None:
    artifact_dir, manifest_path, model_card_path, evaluation_path = release_fixture(tmp_path)
    manifest_path.write_text(
        json.dumps({"stage": "grpo", "status": "running"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completed GRPO"):
        validate_release(artifact_dir, manifest_path, model_card_path, evaluation_path)


def test_validate_release_rejects_pending_model_card(tmp_path: Path) -> None:
    artifact_dir, manifest_path, model_card_path, evaluation_path = release_fixture(tmp_path)
    model_card_path.write_text(
        "---\nlicense: mit\n---\n<!-- FINAL_EVAL_TABLE -->\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pending final-evaluation"):
        validate_release(artifact_dir, manifest_path, model_card_path, evaluation_path)


def test_validate_release_rejects_incomplete_stage_matrix(tmp_path: Path) -> None:
    artifact_dir, manifest_path, model_card_path, evaluation_path = release_fixture(tmp_path)
    evaluation_path.write_text(
        json.dumps({"status": "running", "stages": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stage-matrix evaluation must be complete"):
        validate_release(artifact_dir, manifest_path, model_card_path, evaluation_path)


def test_validate_release_rejects_truncated_stage_predictions(tmp_path: Path) -> None:
    artifact_dir, manifest_path, model_card_path, evaluation_path = release_fixture(tmp_path)
    (tmp_path / "grpo_predictions.jsonl").write_text(
        json.dumps({"sample_id": "only-one", "response": "{}"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="grpo predictions must contain exactly 200"):
        validate_release(artifact_dir, manifest_path, model_card_path, evaluation_path)
