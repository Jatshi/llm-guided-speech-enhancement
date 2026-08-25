from __future__ import annotations

import json

import pytest

from lse_v2.service import RuleBasedPlanner, _invoke_planner, authorize_token


def test_rule_based_fallback_emits_executable_bounded_json() -> None:
    planner = RuleBasedPlanner()
    text = planner.plan({"noise_type": "cafe", "snr_db": 7.0})
    payload = json.loads(text)

    assert payload["actions"]
    assert payload["actions"][0]["reduction_db"] <= 18
    assert 0 <= payload["confidence"] <= 1


def test_token_authorization_fails_closed_when_configured() -> None:
    authorize_token("Bearer secret", "secret")
    with pytest.raises(PermissionError):
        authorize_token("Bearer wrong", "secret")
    with pytest.raises(PermissionError):
        authorize_token(None, "secret")


def test_token_authorization_is_optional_on_loopback() -> None:
    authorize_token(None, None)


def test_service_prefers_native_audio_method_when_available(tmp_path) -> None:
    class Native:
        def plan_audio(self, audio_path, prompt_text):
            assert audio_path == tmp_path / "audio.wav"
            assert "Observable evidence" in prompt_text
            return "native"

    result = _invoke_planner(Native(), tmp_path / "audio.wav", {"snr_db": 4})

    assert result == "native"
