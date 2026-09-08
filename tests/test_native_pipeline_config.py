from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lse_v2.native_pipeline import dry_run_native_pipeline, load_native_pipeline_config


def test_native_pipeline_dry_run_validates_real_audio_and_stage_chain(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    audio = tmp_path / "audio.wav"
    soundfile.write(audio, np.zeros(1600, dtype=np.float32), 16000)
    manifest = tmp_path / "native.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "lse.native_audio.v1",
                "sample_id": "one",
                "split": "train",
                "audio_path": str(audio),
                "prompt_text": "Inspect the waveform.",
                "chosen_text": json.dumps(
                    {
                        "diagnosis": {},
                        "actions": [{"type": "dc_remove"}],
                        "rationale": "x",
                        "confidence": 0.5,
                    }
                ),
                "rejected_text": json.dumps(
                    {
                        "diagnosis": {},
                        "actions": [{"type": "gain", "gain_db": 30}],
                        "rationale": "x",
                        "confidence": 0.9,
                    }
                ),
                "reward_context": {"noise_type": "unknown"},
                "source_id": "speaker-one",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest),
                "output_dir": str(tmp_path / "outputs"),
                "whisper_model": "openai/whisper-small",
                "language_model": "Qwen/Qwen2.5-1.5B-Instruct",
                "prefix_tokens": 8,
                "stages": {
                    "sft": {"max_steps": 10},
                    "dpo": {"max_steps": 10, "input_stage": "sft"},
                    "grpo": {"max_steps": 10, "input_stage": "dpo", "num_generations": 2},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_native_pipeline_config(config_path)
    report = dry_run_native_pipeline(config)

    assert report["status"] == "validated"
    assert report["records"] == 1
    assert report["materialized_audio_records"] == 1
    assert list(report["stages"]) == ["sft", "dpo", "grpo"]


def test_grpo_can_bootstrap_from_sft_and_lineage_resolver_honors_it(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    audio = tmp_path / "audio.wav"
    soundfile.write(audio, np.zeros(1600, dtype=np.float32), 16000)
    manifest = tmp_path / "native.jsonl"
    row = {
        "schema_version": "lse.native_audio.v1",
        "sample_id": "one",
        "split": "train",
        "audio_path": str(audio),
        "prompt_text": "Inspect the waveform.",
        "chosen_text": json.dumps(
            {
                "diagnosis": {},
                "actions": [{"type": "dc_remove"}],
                "rationale": "x",
                "confidence": 0.5,
            }
        ),
        "rejected_text": json.dumps(
            {
                "diagnosis": {},
                "actions": [{"type": "gain", "gain_db": 30}],
                "rationale": "x",
                "confidence": 0.9,
            }
        ),
        "reward_context": {"noise_type": "unknown"},
        "source_id": "speaker-one",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest),
                "output_dir": str(tmp_path / "outputs"),
                "whisper_model": "openai/whisper-small",
                "language_model": "Qwen/Qwen2.5-1.5B-Instruct",
                "prefix_tokens": 8,
                "stages": {
                    "sft": {"max_steps": 10},
                    "dpo": {"max_steps": 10, "input_stage": "sft"},
                    "grpo": {
                        "max_steps": 10,
                        "input_stage": "sft",
                        "num_generations": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_native_pipeline_config(config_path)
    from lse_v2.native_training_pipeline import resolve_stage_input_dir

    sft = tmp_path / "outputs" / "sft" / "final"
    dpo = tmp_path / "outputs" / "dpo" / "final"
    assert resolve_stage_input_dir(config, "grpo", {"sft": sft, "dpo": dpo}) == sft


def test_grpo_rejects_unknown_input_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "manifest": "missing.jsonl",
                "output_dir": "outputs",
                "whisper_model": "whisper",
                "language_model": "llm",
                "prefix_tokens": 8,
                "stages": {
                    "sft": {"max_steps": 1},
                    "dpo": {"max_steps": 1, "input_stage": "sft"},
                    "grpo": {"max_steps": 1, "input_stage": "base", "num_generations": 2},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="grpo.input_stage"):
        load_native_pipeline_config(config_path)
