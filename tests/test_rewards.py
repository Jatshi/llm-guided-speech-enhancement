from __future__ import annotations

import json
import math

import pytest

from lse_v2.rewards import grpo_reward, score_prescription


def good_response() -> str:
    return json.dumps(
        {
            "diagnosis": {
                "noise_type": "white",
                "reverb": False,
                "band_limited": False,
            },
            "actions": [
                {
                    "type": "spectral_subtraction",
                    "reduction_db": 10.0,
                    "low_hz": 80,
                    "high_hz": 7600,
                }
            ],
            "rationale": "Measured SNR supports conservative suppression.",
            "confidence": 0.8,
        }
    )


def test_valid_evidence_consistent_response_scores_one() -> None:
    score = score_prescription(
        good_response(),
        {"noise_type": "white", "snr_db": 10.0, "reverb_rt60": None},
    )
    assert score.total == pytest.approx(1.0)
    assert score.valid_json
    assert not score.violations


def test_overprocessing_is_penalized() -> None:
    payload = json.loads(good_response())
    payload["actions"][0]["reduction_db"] = 36.0
    score = score_prescription(
        json.dumps(payload),
        {"noise_type": "white", "snr_db": 22.0},
    )
    assert score.total < 0.8
    assert any("over" in item or "range" in item for item in score.violations)


def test_invalid_json_has_no_format_or_parameter_reward() -> None:
    score = score_prescription("not-json", {"noise_type": "white"})
    assert score.format == 0
    assert score.parameter_bounds == 0
    assert not score.valid_json


def test_grpo_reward_accepts_conversational_completions() -> None:
    scores = grpo_reward(
        [[{"role": "assistant", "content": good_response()}]],
        reward_context=[{"noise_type": "white", "snr_db": 10.0}],
    )
    assert scores == [1.0]


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [
        ("low_hz", []),
        ("low_hz", {"unexpected": 80}),
        ("high_hz", "7600"),
        ("reduction_db", [10.0]),
        ("gain_db", True),
        ("q", None),
    ],
)
def test_structured_or_non_numeric_action_parameters_never_crash_reward(
    field: str, malformed_value: object
) -> None:
    payload = json.loads(good_response())
    payload["actions"][0]["type"] = "highpass"
    payload["actions"][0][field] = malformed_value

    score = score_prescription(
        json.dumps(payload),
        {"noise_type": "white", "snr_db": 10.0},
    )

    assert 0.0 <= score.total <= 1.0
    assert score.valid_json
    assert any(field in violation for violation in score.violations)


def test_non_finite_action_parameters_are_penalized_without_non_finite_reward() -> None:
    payload = json.loads(good_response())
    payload["actions"][0]["reduction_db"] = float("inf")

    score = score_prescription(
        json.dumps(payload),
        {"noise_type": "white", "snr_db": 10.0},
    )

    assert math.isfinite(score.total)
    assert "action_0_reduction_db_out_of_range" in score.violations


def test_unknown_ablation_component_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        score_prescription(good_response(), {}, weights={"invented": 1.0})
