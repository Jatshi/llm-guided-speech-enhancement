from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lse_v2.materialization import MaterializationConfig, materialize_dataset


def test_materializer_writes_real_noisy_audio_and_prevents_source_leakage(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    sample_rate = 16000
    clean_rows = []
    for speaker_index in range(8):
        speaker = f"speaker-{speaker_index}"
        for utterance_index in range(2):
            path = tmp_path / f"{speaker}-{utterance_index}.wav"
            time = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate
            waveform = 0.1 * np.sin(2 * np.pi * (180 + 20 * speaker_index) * time)
            soundfile.write(path, waveform, sample_rate)
            clean_rows.append(
                {
                    "source_id": f"{speaker}-{utterance_index}",
                    "speaker_id": speaker,
                    "audio_path": str(path),
                    "dataset": "unit-clean",
                    "license": "generated",
                }
            )
    clean_manifest = tmp_path / "clean.jsonl"
    clean_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in clean_rows), encoding="utf-8"
    )

    report = materialize_dataset(
        MaterializationConfig(
            clean_manifest=clean_manifest,
            output_dir=tmp_path / "materialized",
            variants_per_source=2,
            max_sources=16,
            seed=23,
            eval_ratio=0.25,
            test_ratio=0.25,
        )
    )

    rows = [json.loads(line) for line in Path(report.manifest).read_text().splitlines()]
    assert len(rows) == 32
    assert all(Path(row["audio"]["noisy_path"]).is_file() for row in rows)
    assert all(Path(row["audio"]["clean_path"]).is_file() for row in rows)
    speaker_splits: dict[str, set[str]] = {}
    for row in rows:
        speaker_splits.setdefault(row["provenance"]["speaker_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in speaker_splits.values())
    assert set(report.split_counts) <= {"train", "eval", "test"}
    assert report.materialized_audio_records == 32


def test_materializer_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    audio = tmp_path / "clean.wav"
    soundfile.write(audio, np.linspace(-0.1, 0.1, 1600, dtype=np.float32), 16000)
    manifest = tmp_path / "clean.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "source_id": "one",
                "speaker_id": "speaker-one",
                "audio_path": str(audio),
                "dataset": "generated",
                "license": "generated",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first = materialize_dataset(
        MaterializationConfig(manifest, tmp_path / "first", seed=5, eval_ratio=0, test_ratio=0)
    )
    second = materialize_dataset(
        MaterializationConfig(manifest, tmp_path / "second", seed=5, eval_ratio=0, test_ratio=0)
    )
    first_row = json.loads(Path(first.manifest).read_text().splitlines()[0])
    second_row = json.loads(Path(second.manifest).read_text().splitlines()[0])
    assert first_row["sample_id"] == second_row["sample_id"]
    assert first_row["acoustics"] == second_row["acoustics"]
