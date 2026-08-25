from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lse_v2.end_to_end import EnhancementRequest, run_enhancement


def test_end_to_end_writes_waveform_and_audit_report(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    clean = 0.2 * np.sin(2 * np.pi * 220 * time)
    noisy = clean + 0.04 * np.sin(2 * np.pi * 1000 * time)
    noisy_path = tmp_path / "noisy.wav"
    clean_path = tmp_path / "clean.wav"
    soundfile.write(noisy_path, noisy, sample_rate)
    soundfile.write(clean_path, clean, sample_rate)
    prescription = json.dumps(
        {
            "diagnosis": {"noise_type": "tonal", "reverb": False},
            "actions": [
                {
                    "type": "notch",
                    "low_hz": 950,
                    "high_hz": 1050,
                    "q": 8,
                    "reduction_db": 18,
                }
            ],
            "rationale": "Remove the measured narrow-band interference.",
            "confidence": 0.9,
        }
    )

    result = run_enhancement(
        EnhancementRequest(
            audio_path=noisy_path,
            prescription=prescription,
            output_dir=tmp_path / "run",
            clean_reference=clean_path,
            minimum_si_sdr_gain_db=0.1,
        )
    )

    assert result.decision == "accept"
    assert result.output_audio.is_file()
    assert result.audit_report.is_file()
    report = json.loads(result.audit_report.read_text(encoding="utf-8"))
    assert report["schema_version"] == "lse.enhancement_run.v1"
    assert report["metrics"]["deltas"]["si_sdr"] > 0
    assert report["prescription_sha256"]


def test_end_to_end_rolls_back_when_required_gain_is_not_met(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    audio = np.linspace(-0.1, 0.1, 1600, dtype=np.float32)
    source = tmp_path / "source.wav"
    soundfile.write(source, audio, 16000)
    prescription = json.dumps(
        {
            "diagnosis": {},
            "actions": [{"type": "gain", "gain_db": 3}],
            "rationale": "test rollback",
            "confidence": 0.5,
        }
    )

    result = run_enhancement(
        EnhancementRequest(
            audio_path=source,
            prescription=prescription,
            output_dir=tmp_path / "rollback",
            clean_reference=source,
            minimum_si_sdr_gain_db=10.0,
        )
    )

    assert result.decision == "rollback"
    restored, _ = soundfile.read(result.output_audio, dtype="float32")
    original, _ = soundfile.read(source, dtype="float32")
    np.testing.assert_allclose(restored, original, atol=1e-6)
