from __future__ import annotations

import json
from pathlib import Path

import pytest

from lse_v2.contracts import (
    AUDIO_SCHEMA,
    ContractError,
    build_alignment_datasets,
    manifest_from_legacy,
    validate_alignment_record,
    validate_audio_record,
)
from lse_v2.io import read_jsonl


def sample_record(sample_id: str = "sample-1", split: str = "train") -> dict:
    return {
        "schema_version": AUDIO_SCHEMA,
        "sample_id": sample_id,
        "split": split,
        "audio": {
            "noisy_path": "/not-required-in-contract.wav",
            "sample_rate": 16000,
            "duration_seconds": 2.0,
        },
        "acoustics": {
            "noise_type": "white",
            "snr_db": 10.0,
            "features": {"spectral_flatness": 0.4},
        },
        "target": {
            "diagnosis": {"noise_type": "white", "reverb": False},
            "actions": [
                {
                    "type": "spectral_subtraction",
                    "reduction_db": 10.0,
                    "low_hz": 80,
                    "high_hz": 7600,
                }
            ],
            "rationale": "Measured evidence supports conservative suppression.",
            "confidence": 0.8,
        },
    }


def test_audio_contract_rejects_unsafe_sample_rate() -> None:
    record = sample_record()
    record["audio"]["sample_rate"] = 100
    with pytest.raises(ContractError, match="sample_rate"):
        validate_audio_record(record)


def test_builds_all_three_alignment_contracts(tmp_path: Path) -> None:
    records = [sample_record("train", "train"), sample_record("eval", "eval")]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    output = tmp_path / "bundle"
    report = build_alignment_datasets(manifest, output)

    assert report["counts"]["sft"] == {"train": 1, "eval": 1, "test": 0}
    for stage, schema in (
        ("sft", "lse.sft.v2"),
        ("dpo", "lse.dpo.v2"),
        ("grpo", "lse.grpo.v2"),
    ):
        row = read_jsonl(output / stage / "train.jsonl")[0]
        assert row["schema_version"] == schema
        validate_alignment_record(row)


def test_duplicate_sample_ids_are_rejected(tmp_path: Path) -> None:
    record = sample_record()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="Duplicate"):
        build_alignment_datasets(manifest, tmp_path / "output")


def test_legacy_metadata_without_split_is_migrated_then_split(tmp_path: Path) -> None:
    legacy = tmp_path / "metadata.json"
    legacy.write_text(
        json.dumps(
            [
                {
                    "id": "legacy-1",
                    "clean_path": "/clean/proxy.wav",
                    "audio_path": None,
                    "degradation_config": {
                        "noise_type": "pink",
                        "snr_db": 9.0,
                    },
                    "provenance": {
                        "dataset": "AISHELL-1",
                        "source_md5": "2f494334227864a8a8fec932999db9d8",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    assert manifest_from_legacy(legacy, manifest) == 1
    row = read_jsonl(manifest)[0]
    assert row["split"] == "unassigned"
    assert row["audio"]["source_role"] == "clean_proxy_for_synthetic_degradation"
    assert row["provenance"]["dataset"] == "AISHELL-1"
    assert row["provenance"]["source_md5"] == "2f494334227864a8a8fec932999db9d8"
    build_alignment_datasets(manifest, tmp_path / "bundle", eval_ratio=0.0)
    assert len(read_jsonl(tmp_path / "bundle" / "sft" / "train.jsonl")) == 1
