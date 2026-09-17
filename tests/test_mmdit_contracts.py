import json
from pathlib import Path

import pytest

from lse_v2.mmdit.attach_predictions import attach_predictions
from lse_v2.mmdit.contracts import (
    MMDIT_PAIR_SCHEMA,
    PairContractError,
    prescription_tokens,
    validate_pair_record,
)
from lse_v2.mmdit.prepare import prepare_pairs


def _pair() -> dict:
    return {
        "schema_version": MMDIT_PAIR_SCHEMA,
        "sample_id": "speaker-a-001",
        "speaker_id": "speaker-a",
        "split": "train",
        "audio": {
            "noisy_path": "/tmp/noisy.wav",
            "clean_path": "/tmp/clean.wav",
            "sample_rate": 16000,
        },
        "oracle_prescription": {
            "diagnosis": {"noise_type": "babble", "reverb": False},
            "actions": [
                {
                    "type": "spectral_subtraction",
                    "reduction_db": 8.0,
                    "low_hz": 80,
                    "high_hz": 7600,
                }
            ],
            "confidence": 0.8,
        },
        "predicted_prescription": None,
        "provenance": {"dataset": "unit-test"},
    }


def test_pair_contract_and_prescription_tokens_are_deterministic():
    row = _pair()
    validate_pair_record(row)
    first = prescription_tokens(row["oracle_prescription"], max_tokens=8)
    second = prescription_tokens(row["oracle_prescription"], max_tokens=8)
    assert first == second
    assert len(first) == 8
    assert first[0][0] != 0
    assert first[-1] == (0, 0, 0.0)


def test_pair_contract_rejects_missing_clean_reference():
    row = _pair()
    row["audio"]["clean_path"] = ""
    with pytest.raises(PairContractError, match="clean_path"):
        validate_pair_record(row)


def test_prepare_pairs_supports_existing_audio_manifest(tmp_path: Path):
    source = {
        "schema_version": "lse.audio_manifest.v2",
        "sample_id": "S001-0001",
        "speaker_id": "S001",
        "split": "train",
        "audio": {
            "noisy_path": "/data/noisy.wav",
            "clean_path": "/data/clean.wav",
            "sample_rate": 16000,
        },
        "acoustics": {"noise_type": "white"},
        "target": _pair()["oracle_prescription"],
        "provenance": {"dataset": "fixture"},
    }
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps(source) + "\n", encoding="utf-8")
    output = tmp_path / "pairs.jsonl"
    report = prepare_pairs(manifest, output, check_files=False)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert report["records"] == 1
    assert row["schema_version"] == MMDIT_PAIR_SCHEMA
    assert row["speaker_id"] == "S001"


def test_attach_predictions_accepts_existing_raw_response_schema(tmp_path: Path):
    pair = _pair()
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "sample_id": pair["sample_id"],
                "raw_response": json.dumps(pair["oracle_prescription"]),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "attached.jsonl"
    report = attach_predictions(pairs, predictions, output)
    attached = json.loads(output.read_text(encoding="utf-8"))
    assert report["attached"] == 1
    assert report["invalid_predictions"] == 0
    assert attached["predicted_prescription"] == pair["oracle_prescription"]
