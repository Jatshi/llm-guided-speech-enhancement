from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.evaluate_stage_matrix import main, stage_candidates


def test_stage_candidates_are_chained_final_adapters() -> None:
    config = {
        "_config_path": str((Path.cwd() / "configs" / "test.json").resolve()),
        "_config_dir": str((Path.cwd() / "configs").resolve()),
        "project": {"root": ".."},
        "training": {
            "stages": {
                "sft": {"output_dir": "outputs/sft"},
                "dpo": {"output_dir": "outputs/dpo"},
                "grpo": {"output_dir": "outputs/grpo"},
            }
        },
    }
    candidates = stage_candidates(config)
    assert [name for name, _ in candidates] == ["base", "sft", "dpo", "grpo"]
    assert candidates[0][1] is None
    assert all(path is None or path.name == "final" for _, path in candidates)


def test_stage_matrix_rejects_empty_holdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "empty.jsonl"
    dataset_path.write_text("", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_stage_matrix",
            "--config",
            str(repo_root / "configs" / "autodl_4090.json"),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(ValueError, match="dataset is empty"):
        main()
