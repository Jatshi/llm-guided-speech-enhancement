from __future__ import annotations

import json

import numpy as np
import pytest

from lse_v2.dsp import DSPAction, DSPPlan, ProductionDSPExecutor, plan_from_prescription


def _tone(frequency: float, sample_rate: int = 16000, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
    return np.sin(2 * np.pi * frequency * time).astype(np.float32)


def _band_energy(audio: np.ndarray, sample_rate: int, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    frequencies = np.fft.rfftfreq(audio.size, 1 / sample_rate)
    return float(spectrum[(frequencies >= low) & (frequencies <= high)].sum())


def test_notch_action_reduces_only_target_band() -> None:
    pytest.importorskip("scipy")
    sample_rate = 16000
    source = 0.5 * _tone(300, sample_rate) + 0.5 * _tone(1000, sample_rate)
    plan = DSPPlan(
        actions=[DSPAction(kind="notch", low_hz=950, high_hz=1050, q=8, reduction_db=18)]
    )

    enhanced = ProductionDSPExecutor().execute(source, sample_rate, plan)

    target_ratio = _band_energy(enhanced, sample_rate, 950, 1050) / _band_energy(
        source, sample_rate, 950, 1050
    )
    speech_ratio = _band_energy(enhanced, sample_rate, 250, 350) / _band_energy(
        source, sample_rate, 250, 350
    )
    assert target_ratio < 0.15
    assert speech_ratio > 0.65


def test_prescription_parser_accepts_existing_action_vocabulary() -> None:
    payload = {
        "diagnosis": {"noise_type": "hvac", "reverb": True},
        "actions": [
            {
                "type": "notch",
                "low_hz": 200,
                "high_hz": 500,
                "q": 4,
                "reduction_db": 10,
            },
            {"type": "dereverb", "reduction_db": 4},
            {"type": "equalizer", "low_hz": 2000, "high_hz": 4000, "gain_db": 2},
        ],
        "rationale": "Treat measured HVAC lines and reverberation conservatively.",
        "confidence": 0.8,
    }

    plan = plan_from_prescription(json.dumps(payload))

    assert [action.kind for action in plan.actions] == ["notch", "dereverb", "equalizer"]


@pytest.mark.parametrize(
    "action",
    [
        {"type": "notch", "low_hz": 500, "high_hz": 200},
        {"type": "highpass", "low_hz": 900},
        {"type": "equalizer", "low_hz": 1000, "high_hz": 2000, "gain_db": 30},
        {"type": "invented_filter"},
        {"type": "gain", "gain_db": -3, "unvalidated_parameter": 1},
    ],
)
def test_unsafe_or_unknown_actions_fail_closed(action: dict[str, object]) -> None:
    payload = {
        "diagnosis": {},
        "actions": [action],
        "rationale": "test",
        "confidence": 0.5,
    }
    with pytest.raises(ValueError):
        plan_from_prescription(json.dumps(payload))


def test_dereverb_and_spectral_subtraction_keep_shape_and_finite_values() -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(7)
    source = np.clip(_tone(220) + 0.08 * rng.standard_normal(16000), -1, 1).astype(np.float32)
    plan = DSPPlan(
        actions=[
            DSPAction(kind="dereverb", reduction_db=4),
            DSPAction(
                kind="spectral_subtraction",
                reduction_db=6,
                low_hz=80,
                high_hz=7600,
            ),
        ]
    )

    enhanced = ProductionDSPExecutor().execute(source, 16000, plan)

    assert enhanced.shape == source.shape
    assert enhanced.dtype == np.float32
    assert np.all(np.isfinite(enhanced))
    assert np.max(np.abs(enhanced)) <= 1.0


def test_stft_istft_round_trip_preserves_waveform() -> None:
    pytest.importorskip("scipy")
    source = (0.6 * _tone(220, seconds=0.73) + 0.2 * _tone(1700, seconds=0.73)).astype(np.float32)
    executor = ProductionDSPExecutor()

    _frequencies, _times, spectrum = executor._stft(source, 16000)
    reconstructed = executor._istft(spectrum, 16000, source.size)

    assert reconstructed.shape == source.shape
    assert np.max(np.abs(reconstructed - source)) < 1e-4
