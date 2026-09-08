from __future__ import annotations

import json

import pytest


@pytest.mark.parametrize("duplicate_manifest", [False, True])
def test_acceptance_rejects_duplicate_ids(duplicate_manifest: bool) -> None:
    from lse_v2.grpo_acceptance import evaluate_recovery_predictions

    manifest = [{"sample_id": "a", "reward_context": {"noise_type": "white"}}]
    predictions = [{"sample_id": "a", "raw_response": _response(12)}]
    with pytest.raises(ValueError, match="unique"):
        evaluate_recovery_predictions(
            manifest * (2 if duplicate_manifest else 1),
            predictions,
            predictions * (1 if duplicate_manifest else 2),
        )


def test_acceptance_rejects_empty_manifest() -> None:
    from lse_v2.grpo_acceptance import evaluate_recovery_predictions

    with pytest.raises(ValueError, match="non-empty"):
        evaluate_recovery_predictions([], [], [])


def _response(reduction_db: float) -> str:
    return json.dumps(
        {
            "diagnosis": {"noise_type": "white", "reverb": False},
            "actions": [{"type": "denoise", "reduction_db": reduction_db}],
            "rationale": "measured",
            "confidence": 0.8,
        }
    )


def test_acceptance_gate_promotes_valid_non_regressing_grpo() -> None:
    from lse_v2.grpo_acceptance import evaluate_recovery_predictions

    manifest = [{"sample_id": "a", "reward_context": {"noise_type": "white", "snr_db": 10}}]
    baseline = [{"sample_id": "a", "raw_response": _response(12), "status": "ok"}]
    candidate = [{"sample_id": "a", "raw_response": _response(10), "status": "ok"}]

    report = evaluate_recovery_predictions(manifest, baseline, candidate)

    assert report["status"] == "passed"
    assert report["candidate"]["valid_json_rate"] == 1
    assert not report["failed_checks"]


def test_acceptance_gate_rejects_invalid_or_placeholder_grpo() -> None:
    from lse_v2.grpo_acceptance import evaluate_recovery_predictions

    manifest = [{"sample_id": "a", "reward_context": {"noise_type": "white", "snr_db": 10}}]
    baseline = [{"sample_id": "a", "raw_response": _response(12), "status": "ok"}]
    candidate = [
        {
            "sample_id": "a",
            "raw_response": '{"actions":[{"reduction_db":?}]}',
            "status": "failed",
        }
    ]

    report = evaluate_recovery_predictions(manifest, baseline, candidate)

    assert report["status"] == "failed"
    assert "valid_json_rate" in report["failed_checks"]
    assert "placeholder_rate" in report["failed_checks"]
