from __future__ import annotations

import json
from pathlib import Path

from lse_v2.io import read_jsonl, write_jsonl
from lse_v2.mmdit.planner_dataset import build_planner_sft_dataset


def _pair(sample_id: str, split: str, noise_type: str) -> dict[str, object]:
    return {
        "schema_version": "lse.mmdit_pair.v1",
        "sample_id": sample_id,
        "speaker_id": "speaker",
        "split": split,
        "audio": {
            "noisy_path": f"{sample_id}-noisy.wav",
            "clean_path": f"{sample_id}-clean.wav",
            "sample_rate": 16000,
        },
        "oracle_prescription": {
            "diagnosis": {
                "noise_type": noise_type,
                "reverb": noise_type == "reverb",
                "band_limited": noise_type == "telephone",
            },
            "actions": [{"type": "spectral_subtraction", "reduction_db": 8.0}],
            "rationale": "Synthetic target.",
            "confidence": 0.9,
        },
        "predicted_prescription": None,
        "provenance": {},
    }


def test_planner_sft_excludes_test_and_hides_oracle_labels(tmp_path: Path, monkeypatch) -> None:
    pairs = tmp_path / "pairs.jsonl"
    write_jsonl(
        pairs,
        [
            _pair("train", "train", "white"),
            _pair("validation", "validation", "pink"),
            _pair("test", "test", "telephone"),
        ],
    )
    monkeypatch.setattr(
        "lse_v2.mmdit.planner_dataset.validate_pair_record", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "lse_v2.mmdit.planner_dataset.build_planner_prompt",
        lambda *_args, **_kwargs: 'AUXILIARY_EVIDENCE={"rms": 0.1}',
    )

    report = build_planner_sft_dataset(pairs, tmp_path / "out", workers=1)

    assert report["train_records"] == 1
    assert report["validation_records"] == 1
    assert report["test_records_consumed"] == 0
    train = read_jsonl(tmp_path / "out" / "train.jsonl")[0]
    assert "white" not in train["messages"][1]["content"]
    assert json.loads(train["messages"][2]["content"])["diagnosis"]["noise_type"] == "white"
