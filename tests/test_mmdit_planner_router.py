from __future__ import annotations

from lse_v2.mmdit.planner_router import fuse_router_diagnosis


def test_router_overrides_diagnosis_but_retains_llm_actions() -> None:
    llm = {
        "diagnosis": {
            "noise_type": "white",
            "reverb": False,
            "band_limited": False,
        },
        "actions": [{"type": "spectral_subtraction", "reduction_db": 8.0}],
        "rationale": "LLM action rationale.",
        "confidence": 0.9,
    }
    hybrid = fuse_router_diagnosis(llm, "telephone", 0.73)
    assert hybrid["diagnosis"] == {
        "noise_type": "telephone",
        "reverb": False,
        "band_limited": True,
    }
    assert hybrid["actions"] == llm["actions"]
    assert hybrid["rationale"] == llm["rationale"]
    assert hybrid["confidence"] == 0.73
    assert llm["diagnosis"]["noise_type"] == "white"
