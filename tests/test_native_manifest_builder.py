from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_build_native_manifest_preserves_audio_and_source_split(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    from lse_v2.native_data import build_native_manifest

    audio = tmp_path / "noisy.wav"
    soundfile.write(audio, np.zeros(1600, dtype=np.float32), 16000)
    source = tmp_path / "audio_manifest.jsonl"
    row = {
        "schema_version": "lse.audio_manifest.v2",
        "sample_id": "sample-one",
        "split": "train",
        "audio": {
            "noisy_path": str(audio),
            "clean_path": str(audio),
            "sample_rate": 16000,
            "source_role": "materialized_noisy_audio",
        },
        "acoustics": {
            "noise_type": "cafe",
            "snr_db": 8.0,
            "reverb_rt60": None,
            "bandlimit_hz": None,
            "features": {},
        },
        "target": {
            "diagnosis": {"noise_type": "cafe"},
            "actions": [{"type": "spectral_subtraction", "reduction_db": 9}],
            "rationale": "Measured cafe noise requires conservative suppression.",
            "confidence": 0.8,
        },
        "provenance": {"source_id": "speaker-17", "speaker_id": "speaker-17"},
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "native.jsonl"

    report = build_native_manifest(source, output)
    native = json.loads(output.read_text(encoding="utf-8"))

    assert report["records"] == 1
    assert native["schema_version"] == "lse.native_audio.v1"
    assert native["audio_path"] == str(audio.resolve())
    assert native["source_id"] == "speaker-17"
    assert "cafe" not in native["prompt_text"]
    assert "privileged_degradation_labels_in_prompt" in native["prompt_text"]
    assert json.loads(native["chosen_text"])["confidence"] == 0.8
    assert native["reward_context"]["snr_db"] == 8.0


def test_native_manifest_refuses_clean_proxy(tmp_path: Path) -> None:
    from lse_v2.native_data import build_native_manifest

    source = tmp_path / "audio_manifest.jsonl"
    source.write_text(
        json.dumps(
            {
                "schema_version": "lse.audio_manifest.v2",
                "sample_id": "proxy",
                "split": "train",
                "audio": {
                    "noisy_path": "missing.wav",
                    "sample_rate": 16000,
                    "source_role": "clean_proxy_for_synthetic_degradation",
                },
                "acoustics": {"noise_type": "white", "features": {}},
                "target": {
                    "diagnosis": {},
                    "actions": [{"type": "dc_remove"}],
                    "rationale": "x",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="materialized noisy audio"):
        build_native_manifest(source, tmp_path / "native.jsonl")
