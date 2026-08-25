from __future__ import annotations

import json
from pathlib import Path

import pytest


class _ToyTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [min(ord(char) % 8 + 1, 8) for char in text]


def test_build_lm_example_masks_prompt_but_trains_response() -> None:
    from lse_v2.native_training_pipeline import build_lm_example

    example = build_lm_example(_ToyTokenizer(), "prompt", "ok", max_tokens=32)

    assert len(example["input_ids"]) == len(example["labels"])
    assert example["labels"][: len("prompt")] == [-100] * len("prompt")
    assert example["labels"][-1] == _ToyTokenizer.eos_token_id
    assert any(value != -100 for value in example["labels"])


def test_grpo_surrogate_has_gradient_and_kl_penalty() -> None:
    torch = pytest.importorskip("torch")
    from lse_v2.native_training_pipeline import grpo_surrogate_loss

    policy = torch.tensor([[-1.0, -0.5]], requires_grad=True)
    old = policy.detach().clone()
    reference = torch.tensor([[-1.2, -0.8]])
    advantages = torch.tensor([[-1.0, 1.0]])

    loss, stats = grpo_surrogate_loss(
        policy,
        old,
        reference,
        advantages,
        beta=0.04,
        clip_epsilon=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert policy.grad is not None and torch.count_nonzero(policy.grad) > 0
    assert stats["kl"] >= 0


def test_stage_checkpoint_pruning_keeps_latest_numeric_steps(tmp_path: Path) -> None:
    from lse_v2.native_training_pipeline import _prune_stage_checkpoints

    for step in (50, 100, 900, 1000):
        (tmp_path / f"checkpoint-{step}").mkdir()

    _prune_stage_checkpoints(tmp_path, keep=2)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint-1000", "checkpoint-900"]


def test_completed_stage_report_requires_manifest_and_final_artifacts(tmp_path: Path) -> None:
    from lse_v2.native_training_pipeline import _completed_stage_report

    stage_dir = tmp_path / "sft"
    final_dir = stage_dir / "final"
    final_dir.mkdir(parents=True)
    report = {
        "schema_version": "lse.native_stage_run.v1",
        "stage": "sft",
        "status": "completed",
        "steps": 1500,
        "output": str(final_dir),
    }
    (stage_dir / "stage_manifest.json").write_text(json.dumps(report), encoding="utf-8")

    assert _completed_stage_report(tmp_path, "sft", 1500) is None
    adapter_dir = final_dir / "adapter" / "sft"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (final_dir / "audio_projector.pt").write_text("{}", encoding="utf-8")
    artifact_manifest = {
        "schema_version": "lse.native_stage_artifacts.v1",
        "stage": "sft",
        "adapter": str(adapter_dir),
    }
    (final_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest), encoding="utf-8"
    )
    assert _completed_stage_report(tmp_path, "sft", 1500) == report


def test_completed_stage_report_rejects_incomplete_stage(tmp_path: Path) -> None:
    from lse_v2.native_training_pipeline import _completed_stage_report

    stage_dir = tmp_path / "dpo"
    final_dir = stage_dir / "final"
    final_dir.mkdir(parents=True)
    adapter_dir = final_dir / "adapter" / "dpo"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (final_dir / "audio_projector.pt").write_text("{}", encoding="utf-8")
    artifact_manifest = {
        "schema_version": "lse.native_stage_artifacts.v1",
        "stage": "dpo",
        "adapter": str(adapter_dir),
    }
    (final_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest), encoding="utf-8"
    )
    report = {
        "schema_version": "lse.native_stage_run.v1",
        "stage": "dpo",
        "status": "completed",
        "steps": 99,
        "output": str(final_dir),
    }
    (stage_dir / "stage_manifest.json").write_text(json.dumps(report), encoding="utf-8")

    assert _completed_stage_report(tmp_path, "dpo", 100) is None


def test_accelerator_checkpoint_uses_non_strict_module_load_for_deepspeed(
    tmp_path: Path,
) -> None:
    from lse_v2.native_training_pipeline import _load_accelerator_checkpoint

    class _State:
        deepspeed_plugin = object()

    class _Accelerator:
        state = _State()
        call = None

        def load_state(self, path: str, **kwargs) -> None:
            self.call = (path, kwargs)

    accelerator = _Accelerator()
    checkpoint = tmp_path / "checkpoint-100"
    _load_accelerator_checkpoint(accelerator, checkpoint)

    assert accelerator.call == (str(checkpoint), {"load_module_strict": False})
