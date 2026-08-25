from __future__ import annotations

from lse_v2.listening_test import aggregate_listening_responses


def test_listening_aggregation_decodes_blind_preference_and_scores() -> None:
    key = {
        "trial-1": {"candidate_side": "A"},
        "trial-2": {"candidate_side": "B"},
    }
    responses = [
        {
            "trial_id": "trial-1",
            "listener_id": "l1",
            "preference": "A",
            "a_naturalness": 4,
            "b_naturalness": 2,
        },
        {
            "trial_id": "trial-2",
            "listener_id": "l1",
            "preference": "tie",
            "a_naturalness": 3,
            "b_naturalness": 3,
        },
    ]

    report = aggregate_listening_responses(responses, key)

    assert report["responses"] == 2
    assert report["listeners"] == 1
    assert report["candidate_preferred_rate"] == 0.5
    assert report["tie_rate"] == 0.5
    assert report["naturalness"]["candidate_mean"] == 3.5
    assert report["naturalness"]["source_mean"] == 2.5
