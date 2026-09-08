from __future__ import annotations

import json
from pathlib import Path

import pytest


def _config_payload(tmp_path: Path) -> dict:
    return {
        "manifest": "data/native.jsonl",
        "output_dir": "outputs/recovery",
        "whisper_model": "whisper",
        "language_model": "llm",
        "prefix_tokens": 8,
        "stages": {
            "sft": {"max_steps": 1},
            "dpo": {"max_steps": 1, "input_stage": "sft"},
            "grpo": {"max_steps": 1, "input_stage": "sft", "num_generations": 2},
        },
        "recovery": {
            "source_stage_dir": "artifacts/sft/final",
            "audio_cache_dir": "cache/audio",
        },
    }


def test_recovery_config_resolves_source_and_cache_relative_to_config(tmp_path: Path) -> None:
    from lse_v2.grpo_recovery import load_grpo_recovery_config

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config_payload(tmp_path)), encoding="utf-8")

    recovery = load_grpo_recovery_config(config_path)

    assert recovery.source_stage_dir == (tmp_path / "artifacts/sft/final").resolve()
    assert recovery.audio_cache_dir == (tmp_path / "cache/audio").resolve()
    assert recovery.native.stages["grpo"]["input_stage"] == "sft"


def test_recovery_requires_sft_source_stage(tmp_path: Path) -> None:
    from lse_v2.grpo_recovery import load_grpo_recovery_config

    payload = _config_payload(tmp_path)
    payload["stages"]["grpo"]["input_stage"] = "dpo"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SFT"):
        load_grpo_recovery_config(config_path)


def test_validate_source_stage_requires_real_adapter_and_projector(tmp_path: Path) -> None:
    from lse_v2.grpo_recovery import validate_source_stage

    source = tmp_path / "sft" / "final"
    source.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        validate_source_stage(source)

    adapter = source / "adapter" / "sft"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (source / "audio_projector.pt").write_bytes(b"projector")
    (source / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": "lse.native_stage_artifacts.v1", "stage": "sft"}),
        encoding="utf-8",
    )

    report = validate_source_stage(source)

    assert report["stage"] == "sft"
    assert report["adapter"] == str(adapter.resolve())
