from __future__ import annotations

import json

from lse_v2.mmdit.planner_export import parse_planner_prediction


def _prescription() -> dict[str, object]:
    return {
        "actions": [
            {
                "high_hz": 7600,
                "low_hz": 80,
                "reduction_db": 12.0,
                "type": "spectral_subtraction",
            }
        ],
        "confidence": 0.75,
        "diagnosis": {"band_limited": True, "noise_type": "white"},
        "rationale": "Measured evidence supports conservative suppression.",
    }


def test_planner_prediction_accepts_strict_json() -> None:
    expected = _prescription()
    parsed, status = parse_planner_prediction(json.dumps(expected))
    assert parsed == expected
    assert status == "strict"


def test_planner_prediction_only_repairs_trailing_container_closures() -> None:
    expected = _prescription()
    truncated = json.dumps(expected)[:-1]
    parsed, status = parse_planner_prediction(truncated)
    assert parsed == expected
    assert status == "repaired"


def test_planner_prediction_does_not_repair_semantics_or_strings() -> None:
    parsed, status = parse_planner_prediction('{"diagnosis": {"noise_type": "white')
    assert parsed is None
    assert status == "invalid"


def test_planner_prediction_rejects_mismatched_containers() -> None:
    parsed, status = parse_planner_prediction('{"diagnosis": []}')
    assert parsed is None
    assert status == "invalid"
