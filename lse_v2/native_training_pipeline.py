"""Real waveform -> frozen Whisper -> projector -> LoRA SFT/cDPO/GRPO training.

This module intentionally owns the complete native-audio chain instead of routing
audio through metadata-only prompts.  Whisper states are cached once because the
encoder is frozen; every alignment stage then trains a compact projector together
with the active LoRA adapter.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .audio_conditioning import AudioConditioningConfig, AudioPrefixProjector
from .config import find_latest_checkpoint, set_global_seed
from .grpo_control import evaluate_canary_gate, summarize_reward_group
from .io import git_commit, read_jsonl, utc_now, write_json_atomic, write_jsonl
from .native_alignment import (
    conservative_dpo_loss,
    masked_sequence_log_probability,
    normalized_group_advantages,
)
from .native_audio_training import load_audio
from .native_pipeline import NATIVE_STAGES, NativePipelineConfig, load_native_records
from .rewards import score_prescription


def build_lm_example(
    tokenizer: Any, prompt: str, response: str, max_tokens: int
) -> dict[str, list[int]]:
    """Tokenize one response while masking every prompt token from the loss."""

    if max_tokens < 4:
        raise ValueError("max_tokens must be at least 4")
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))
    eos = tokenizer.eos_token_id
    if eos is not None and (not response_ids or response_ids[-1] != eos):
        response_ids.append(eos)
    # Preserve at least one supervised response token when a prompt is very long.
    response_ids = response_ids[: max(1, min(len(response_ids), max_tokens // 2))]
    prompt_ids = prompt_ids[-max(1, max_tokens - len(response_ids)) :]
    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def grpo_surrogate_loss(
    policy_logps,
    old_logps,
    reference_logps,
    advantages,
    *,
    beta: float,
    clip_epsilon: float,
):
    """Clipped group-relative objective with a non-negative KL approximation."""

    import torch

    if beta < 0 or not 0 < clip_epsilon < 1:
        raise ValueError("beta must be non-negative and clip_epsilon must be in (0, 1)")
    ratio = torch.exp(policy_logps - old_logps)
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    policy_term = -torch.minimum(ratio * advantages, clipped * advantages).mean()
    log_ratio = policy_logps - reference_logps
    kl = (torch.exp(-log_ratio) + log_ratio - 1).mean()
    loss = policy_term + beta * kl
    return loss, {
        "policy_loss": float(policy_term.detach().cpu()),
        "kl": float(kl.detach().cpu()),
    }


def combine_grpo_and_anchor_loss(
    grpo_loss,
    anchor_loss,
    *,
    anchor_weight: float,
    group_saturated: bool,
):
    """Add a supervised format anchor so a tied group cannot become KL-only training."""

    if anchor_weight < 0:
        raise ValueError("anchor_weight must be non-negative")
    loss = grpo_loss + anchor_weight * anchor_loss
    return loss, {
        "anchor_loss": float(anchor_loss.detach().cpu()),
        "anchor_weight": anchor_weight,
        "group_saturated": bool(group_saturated),
    }


def resolve_stage_input_dir(
    config: NativePipelineConfig,
    stage: str,
    stage_outputs: dict[str, Path],
) -> Path | None:
    input_stage = config.stages[stage].get("input_stage")
    if input_stage is None:
        return None
    try:
        return stage_outputs[str(input_stage)]
    except KeyError as exc:
        raise RuntimeError(f"{stage} requires unavailable input stage {input_stage!r}") from exc


def _sample_grpo_group(
    model: Any,
    tokenizer: Any,
    batch: dict[str, Any],
    row: dict[str, Any],
    params: dict[str, Any],
) -> tuple[list[str], list[Any]]:
    """Sample and deterministically score one prompt group."""

    import torch

    generations = int(params.get("num_generations", 4))
    model.eval()
    with torch.no_grad():
        generated = model.generate_with_audio(
            encoded_audio=batch["encoded_audio"],
            audio_frame_mask=batch["audio_frame_mask"],
            audio_present=batch["audio_present"],
            input_ids=batch["input_ids"],
            token_attention_mask=batch["token_attention_mask"],
            do_sample=True,
            use_cache=True,
            temperature=float(params.get("temperature", 1.0)),
            top_p=float(params.get("top_p", 0.95)),
            max_new_tokens=int(params.get("max_new_tokens", 192)),
            num_return_sequences=generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    completions = [text.strip() for text in decoded]
    breakdowns = [score_prescription(text, row["reward_context"]) for text in completions]
    return completions, breakdowns


def _run_grpo_canary(
    *,
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    loader: Any,
    params: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Probe real generations before the first optimizer update and fail closed."""

    groups = int(params.get("canary_groups", 8))
    if groups <= 0:
        raise ValueError("grpo.canary_groups must be positive")
    iterator = iter(loader)
    scored_groups: list[list[Any]] = []
    samples: list[dict[str, Any]] = []
    unwrapped = accelerator.unwrap_model(model)
    for group_index in range(groups):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        rows = batch.pop("rows")
        batch = _to_device(batch, accelerator.device)
        completions, breakdowns = _sample_grpo_group(unwrapped, tokenizer, batch, rows[0], params)
        scored_groups.append(breakdowns)
        print(
            f"grpo_canary group={group_index + 1}/{groups} "
            f"summary={summarize_reward_group(breakdowns).to_dict()}",
            flush=True,
        )
        for completion, breakdown in zip(completions, breakdowns, strict=True):
            if len(samples) >= 24:
                break
            samples.append(
                {
                    "group": group_index,
                    "completion": completion,
                    "reward": breakdown.to_dict(),
                }
            )
    report = evaluate_canary_gate(
        scored_groups,
        min_valid_json_rate=float(params.get("min_canary_valid_json_rate", 0.8)),
        min_non_saturated_group_rate=float(params.get("min_canary_non_saturated_group_rate", 0.2)),
    )
    report["samples"] = samples
    if accelerator.is_main_process:
        write_json_atomic(output_dir / "canary_report.json", report)
    accelerator.wait_for_everyone()
    if report["status"] != "passed":
        checks = ", ".join(report["failed_checks"])
        raise RuntimeError(
            "GRPO canary failed before any optimizer step: "
            f"{checks}; inspect {output_dir / 'canary_report.json'}"
        )
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "accelerate", "deepspeed"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _gpu_description() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def precompute_audio_embeddings(
    records: list[dict[str, Any]],
    *,
    whisper_model: str,
    cache_dir: Path,
    batch_size: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Encode every real waveform once with a frozen Whisper encoder."""

    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel

    if not torch.cuda.is_available():
        raise RuntimeError("audio embedding precomputation requires CUDA")
    if batch_size <= 0:
        raise ValueError("embedding batch_size must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.jsonl"
    existing = {
        row["sample_id"]: row
        for row in (
            [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
            if index_path.is_file()
            else []
        )
    }
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    extractor = WhisperFeatureExtractor.from_pretrained(whisper_model, return_attention_mask=True)
    whisper = WhisperModel.from_pretrained(whisper_model, torch_dtype=dtype).encoder.cuda().eval()
    whisper.requires_grad_(False)
    cached: list[dict[str, Any] | None] = [None] * len(records)
    pending: list[tuple[int, dict[str, Any], str, Path]] = []
    reused = 0
    started = time.perf_counter()
    for index, record in enumerate(records):
        audio_path = Path(record["audio_path"])
        audio_sha = _sha256_file(audio_path)
        safe_name = hashlib.sha256(record["sample_id"].encode()).hexdigest()[:24]
        target = cache_dir / f"{safe_name}.npz"
        old = existing.get(record["sample_id"])
        if (
            target.is_file()
            and old
            and old.get("audio_sha256") == audio_sha
            and old.get("whisper_model") == whisper_model
        ):
            reused += 1
            cached[index] = {
                **record,
                "embedding_path": str(target.resolve()),
                "audio_sha256": audio_sha,
                "whisper_model": whisper_model,
            }
        else:
            pending.append((index, record, audio_sha, target))

    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset : offset + batch_size]
        waveforms = [
            load_audio(Path(record["audio_path"]), extractor.sampling_rate)
            for _, record, _, _ in chunk
        ]
        inputs = extractor(
            waveforms,
            sampling_rate=extractor.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = inputs.input_features.cuda().to(dtype=dtype)
        with torch.inference_mode():
            hidden_batch = whisper(features).last_hidden_state.float().cpu().numpy()
        input_masks = inputs.attention_mask.numpy().astype(np.float32)
        for (index, record, audio_sha, target), hidden, input_mask in zip(
            chunk, hidden_batch, input_masks, strict=True
        ):
            frame_count = max(1, round(float(input_mask.sum()) * hidden.shape[0] / input_mask.size))
            # Whisper pads every input to its 30-second context window.  Keeping
            # those padded encoder frames multiplies the cache size and batch
            # collation cost without carrying any signal, so persist only the
            # frames supported by the real waveform.
            hidden = hidden[:frame_count]
            frame_mask = np.ones(frame_count, dtype=np.uint8)
            np.savez_compressed(
                target,
                hidden=hidden.astype(np.float16),
                frame_mask=frame_mask,
            )
            cached[index] = {
                **record,
                "embedding_path": str(target.resolve()),
                "audio_sha256": audio_sha,
                "whisper_model": whisper_model,
            }
        if (offset // batch_size + 1) % 16 == 0:
            write_jsonl(index_path, [row for row in cached if row is not None])
            print(f"audio_cache encoded={offset + len(chunk)}/{len(pending)}", flush=True)
    completed_cache = [row for row in cached if row is not None]
    if len(completed_cache) != len(records):
        raise RuntimeError("audio embedding cache did not cover every record")
    write_jsonl(index_path, completed_cache)
    report = {
        "schema_version": "lse.audio_embedding_cache.v1",
        "created_at": utc_now(),
        "whisper_model": whisper_model,
        "records": len(completed_cache),
        "reused": reused,
        "encoded": len(completed_cache) - reused,
        "batch_size": batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "index": str(index_path),
    }
    write_json_atomic(cache_dir / "cache_manifest.json", report)
    del whisper
    torch.cuda.empty_cache()
    return completed_cache, report


class AudioConditionedPolicy(__import__("torch").nn.Module):
    """Causal LM whose prefix comes from frozen-Whisper frame embeddings."""

    def __init__(
        self,
        language_model: Any,
        *,
        encoder_dim: int,
        llm_dim: int,
        prefix_tokens: int,
        reference_projector: Any | None = None,
        policy_adapter: str = "default",
        reference_adapter: str | None = None,
    ) -> None:
        import torch

        super().__init__()
        self.language_model = language_model
        self.projector = AudioPrefixProjector(
            AudioConditioningConfig(
                encoder_dim=encoder_dim,
                llm_dim=llm_dim,
                prefix_tokens=prefix_tokens,
            )
        )
        self.reference_projector = reference_projector
        if self.reference_projector is not None:
            self.reference_projector.requires_grad_(False)
        self.missing_audio_prefix = torch.nn.Parameter(torch.zeros(1, prefix_tokens, llm_dim))
        self.policy_adapter = policy_adapter
        self.reference_adapter = reference_adapter

    def _select_adapter(self, name: str | None) -> None:
        if name and hasattr(self.language_model, "set_adapter"):
            self.language_model.set_adapter(name)

    def _prefix(self, encoded_audio, audio_frame_mask, adapter_name, audio_present):
        projector = (
            self.reference_projector
            if adapter_name == self.reference_adapter and self.reference_projector is not None
            else self.projector
        )
        projector_dtype = next(projector.parameters()).dtype
        encoded_audio = encoded_audio.to(dtype=projector_dtype)
        prefix = projector(encoded_audio, audio_frame_mask)
        if audio_present is not None:
            missing = self.missing_audio_prefix.to(dtype=prefix.dtype).expand(
                prefix.shape[0], -1, -1
            )
            selector = audio_present.to(dtype=prefix.dtype).view(-1, 1, 1)
            prefix = selector * prefix + (1 - selector) * missing
        return prefix

    def forward(
        self,
        *,
        encoded_audio,
        audio_frame_mask,
        input_ids,
        token_attention_mask,
        labels=None,
        audio_present=None,
        adapter_name: str | None = None,
    ):
        import torch

        active = adapter_name or self.policy_adapter
        self._select_adapter(active)
        prefix = self._prefix(encoded_audio, audio_frame_mask, active, audio_present)
        token_embeds = self.language_model.get_input_embeddings()(input_ids)
        prefix = prefix.to(dtype=token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        prefix_attention = token_attention_mask.new_ones(prefix.shape[:2])
        attention_mask = torch.cat([prefix_attention, token_attention_mask], dim=1)
        full_labels = None
        if labels is not None:
            prefix_labels = labels.new_full(prefix.shape[:2], -100)
            full_labels = torch.cat([prefix_labels, labels], dim=1)
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=full_labels,
            use_cache=False,
        )
        output.full_labels = full_labels
        return output

    def generate_with_audio(
        self,
        *,
        encoded_audio,
        audio_frame_mask,
        input_ids,
        token_attention_mask,
        audio_present=None,
        **generation_kwargs,
    ):
        import torch

        self._select_adapter(self.policy_adapter)
        prefix = self._prefix(encoded_audio, audio_frame_mask, self.policy_adapter, audio_present)
        token_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix.to(token_embeds.dtype), token_embeds], dim=1)
        attention_mask = torch.cat(
            [token_attention_mask.new_ones(prefix.shape[:2]), token_attention_mask], dim=1
        )
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )


def _pad_examples(tokenizer: Any, examples: list[dict[str, list[int]]]) -> dict[str, Any]:
    import torch

    width = max(len(item["input_ids"]) for item in examples)
    pad_id = tokenizer.pad_token_id
    ids, masks, labels = [], [], []
    for item in examples:
        pad = width - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * pad)
        masks.append([1] * len(item["input_ids"]) + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "token_attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _audio_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    arrays = [np.load(row["embedding_path"], allow_pickle=False) for row in rows]
    max_frames = max(item["hidden"].shape[0] for item in arrays)
    width = arrays[0]["hidden"].shape[1]
    hidden = np.zeros((len(rows), max_frames, width), dtype=np.float16)
    mask = np.zeros((len(rows), max_frames), dtype=np.uint8)
    for index, item in enumerate(arrays):
        frames = item["hidden"].shape[0]
        hidden[index, :frames] = item["hidden"]
        mask[index, :frames] = item["frame_mask"]
        item.close()
    return {
        "encoded_audio": torch.from_numpy(hidden),
        "audio_frame_mask": torch.from_numpy(mask),
    }


class NativeCollator:
    def __init__(self, tokenizer: Any, stage: str, max_tokens: int, audio_dropout: float) -> None:
        self.tokenizer = tokenizer
        self.stage = stage
        self.max_tokens = max_tokens
        self.audio_dropout = audio_dropout

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch: dict[str, Any] = {**_audio_batch(rows), "rows": rows}
        batch["audio_present"] = (
            torch.rand(len(rows)) >= self.audio_dropout
            if self.stage == "sft"
            else torch.ones(len(rows))
        )
        if self.stage == "sft":
            examples = [
                build_lm_example(
                    self.tokenizer, row["prompt_text"], row["chosen_text"], self.max_tokens
                )
                for row in rows
            ]
            batch.update(_pad_examples(self.tokenizer, examples))
        elif self.stage == "dpo":
            for name, field in (("chosen", "chosen_text"), ("rejected", "rejected_text")):
                examples = [
                    build_lm_example(
                        self.tokenizer, row["prompt_text"], row[field], self.max_tokens
                    )
                    for row in rows
                ]
                batch[name] = _pad_examples(self.tokenizer, examples)
        else:
            prompts = [
                {
                    "input_ids": list(
                        self.tokenizer.encode(row["prompt_text"], add_special_tokens=True)
                    )[-self.max_tokens :],
                    "labels": [],
                }
                for row in rows
            ]
            # Prompt-only generation does not use labels.
            batch.update(_pad_examples(self.tokenizer, prompts))
            batch.pop("labels")
        return batch


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    def move(value: Any) -> Any:
        if hasattr(value, "to"):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    return {key: move(value) for key, value in batch.items()}


def _kbit_training_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(params.get("gradient_checkpointing", True))
    return {
        "use_gradient_checkpointing": enabled,
        # Reentrant checkpointing can invoke hooks for a shared LoRA parameter
        # twice. DeepSpeed ZeRO-2 rejects that second reduction.
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if enabled else None,
    }


def _load_language_model(config: NativePipelineConfig, stage: str, input_dir: Path | None):
    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    params = config.stages[stage]
    qlora = bool(params.get("qlora", True))
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "trust_remote_code": True,
    }
    if qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    base = AutoModelForCausalLM.from_pretrained(config.language_model, **model_kwargs)
    base.config.use_cache = False
    if qlora:
        base = prepare_model_for_kbit_training(base, **_kbit_training_kwargs(params))
    lora = LoraConfig(
        r=int(params.get("lora_r", 16)),
        lora_alpha=int(params.get("lora_alpha", 32)),
        lora_dropout=float(params.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=params.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    if stage == "sft":
        policy_adapter, reference_adapter = "sft", None
        model = get_peft_model(base, lora, adapter_name=policy_adapter)
    else:
        if input_dir is None:
            raise ValueError(f"{stage} requires an input stage")
        adapter_dir = _saved_adapter_path(input_dir)
        reference_adapter, policy_adapter = "reference", stage
        model = PeftModel.from_pretrained(
            base, adapter_dir, adapter_name=reference_adapter, is_trainable=False
        )
        model.load_adapter(adapter_dir, adapter_name=policy_adapter, is_trainable=True)
        model.set_adapter(policy_adapter)
    tokenizer = AutoTokenizer.from_pretrained(config.language_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, policy_adapter, reference_adapter


def _accelerator_for(config: NativePipelineConfig, stage: str):
    from accelerate import Accelerator, DeepSpeedPlugin

    params = config.stages[stage]
    deepspeed = params.get("deepspeed")
    plugin = None
    if deepspeed:
        ds_path = Path(str(deepspeed)).expanduser()
        if not ds_path.is_absolute():
            ds_path = (config.config_path.parent / ds_path).resolve()
        if not ds_path.is_file():
            raise FileNotFoundError(ds_path)
        # DeepSpeed otherwise attempts MPI discovery even for a direct,
        # single-GPU Python launch.  Declare the one-rank torch.distributed
        # environment explicitly so AutoDL does not need an MPI runtime.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", os.environ["LOCAL_RANK"])
        os.environ.setdefault("WORLD_SIZE", "1")
        plugin = DeepSpeedPlugin(hf_ds_config=str(ds_path))
    mixed_precision = str(params.get("mixed_precision", "bf16"))
    return Accelerator(
        gradient_accumulation_steps=int(params.get("gradient_accumulation_steps", 1)),
        mixed_precision=mixed_precision,
        deepspeed_plugin=plugin,
    )


def _saved_adapter_path(stage_output: Path) -> Path:
    """Resolve PEFT's root or named-adapter save layout."""

    root = stage_output / "adapter"
    if (root / "adapter_config.json").is_file():
        return root
    candidates = sorted(root.glob("*/adapter_config.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one saved adapter below {root}")
    return candidates[0].parent


def _prune_stage_checkpoints(output_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    for stale in checkpoints[:-keep]:
        shutil.rmtree(stale)


def _save_stage(accelerator: Any, model: Any, tokenizer: Any, output_dir: Path, stage: str) -> None:
    import torch

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped.language_model.save_pretrained(
        output_dir / "adapter",
        selected_adapters=[unwrapped.policy_adapter],
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output_dir / "adapter")
    torch.save(
        {
            "projector": unwrapped.projector.state_dict(),
            "missing_audio_prefix": unwrapped.missing_audio_prefix.detach().cpu(),
            "stage": stage,
        },
        output_dir / "audio_projector.pt",
    )
    adapter_path = _saved_adapter_path(output_dir)
    write_json_atomic(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": "lse.native_stage_artifacts.v1",
            "stage": stage,
            "adapter": str(adapter_path),
            "audio_projector": str(output_dir / "audio_projector.pt"),
            "tokenizer": str(output_dir / "adapter"),
        },
    )


def _stage_model(
    config: NativePipelineConfig,
    stage: str,
    encoder_dim: int,
    input_dir: Path | None,
):
    import copy

    import torch

    tokenizer, language_model, policy_adapter, reference_adapter = _load_language_model(
        config, stage, input_dir
    )
    llm_dim = int(language_model.config.hidden_size)
    reference_projector = None
    projector_state = None
    missing_state = None
    if input_dir is not None:
        checkpoint = torch.load(
            input_dir / "audio_projector.pt", map_location="cpu", weights_only=True
        )
        projector_state = checkpoint["projector"]
        missing_state = checkpoint["missing_audio_prefix"]
        reference_projector = AudioPrefixProjector(
            AudioConditioningConfig(
                encoder_dim=encoder_dim,
                llm_dim=llm_dim,
                prefix_tokens=config.prefix_tokens,
            )
        )
        reference_projector.load_state_dict(projector_state)
        reference_projector = copy.deepcopy(reference_projector).eval()
    model = AudioConditionedPolicy(
        language_model,
        encoder_dim=encoder_dim,
        llm_dim=llm_dim,
        prefix_tokens=config.prefix_tokens,
        reference_projector=reference_projector,
        policy_adapter=policy_adapter,
        reference_adapter=reference_adapter,
    )
    if projector_state is not None:
        model.projector.load_state_dict(projector_state)
        model.missing_audio_prefix.data.copy_(missing_state)
    return tokenizer, model


def _forward_logp(model: Any, batch: dict[str, Any], response: dict[str, Any], adapter: str):
    output = model(
        encoded_audio=batch["encoded_audio"],
        audio_frame_mask=batch["audio_frame_mask"],
        audio_present=batch["audio_present"],
        input_ids=response["input_ids"],
        token_attention_mask=response["token_attention_mask"],
        labels=response["labels"],
        adapter_name=adapter,
    )
    return masked_sequence_log_probability(output.logits, output.full_labels)


def _load_accelerator_checkpoint(accelerator: Any, checkpoint: Path) -> None:
    """Restore a checkpoint while tolerating regenerated QLoRA quantization metadata."""

    kwargs: dict[str, Any] = {}
    if getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None) is not None:
        # DeepSpeed serializes bitsandbytes NF4 metadata alongside the module state.
        # Recreated 4-bit modules regenerate that immutable metadata, so strict loading
        # would reject harmless extra ``absmax``/``quant_map`` keys after a cold restart.
        kwargs["load_module_strict"] = False
    accelerator.load_state(str(checkpoint), **kwargs)


def _run_stage(
    config: NativePipelineConfig,
    stage: str,
    records: list[dict[str, Any]],
    encoder_dim: int,
    input_dir: Path | None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    params = config.stages[stage]
    accelerator = _accelerator_for(config, stage)
    tokenizer, model = _stage_model(config, stage, encoder_dim, input_dir)
    # The quantized LM is loaded directly on its CUDA rank.  Move only the
    # non-quantized trainable audio modules; calling ``model.to`` would be invalid
    # for bitsandbytes 4-bit weights.
    projector_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.projector.to(device=accelerator.device, dtype=projector_dtype)
    if model.reference_projector is not None:
        model.reference_projector.to(device=accelerator.device, dtype=projector_dtype)
    model.missing_audio_prefix.data = model.missing_audio_prefix.data.to(
        device=accelerator.device, dtype=projector_dtype
    )
    if hasattr(model.language_model, "hf_device_map"):
        model.hf_device_map = model.language_model.hf_device_map
    train_rows = [row for row in records if row["split"] == "train"]
    if not train_rows:
        raise ValueError("native manifest contains no train records")
    batch_size = 1 if stage == "grpo" else int(params.get("batch_size", 1))
    collator = NativeCollator(
        tokenizer,
        stage,
        int(params.get("max_tokens", 768)),
        float(params.get("audio_dropout", 0.1)),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train_rows,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=int(params.get("num_workers", 0)),
        generator=generator,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(params.get("learning_rate", 2e-4)),
        weight_decay=float(params.get("weight_decay", 0.01)),
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    output_dir = config.output_dir / stage
    resume = find_latest_checkpoint(output_dir)
    if resume is not None:
        _load_accelerator_checkpoint(accelerator, resume)
    completed = 0
    if resume is not None:
        state_path = resume / "trainer_state.json"
        if state_path.is_file():
            completed = int(json.loads(state_path.read_text(encoding="utf-8"))["step"])
    max_steps = int(params["max_steps"])
    losses: list[float] = []
    reward_values: list[float] = []
    anchor_losses: list[float] = []
    policy_losses: list[float] = []
    kl_values: list[float] = []
    valid_json_generations = 0
    total_generations = 0
    saturated_groups = 0
    total_groups = 0
    consecutive_saturated_groups = 0
    max_observed_consecutive_saturated_groups = 0
    diagnostics: list[dict[str, Any]] = []
    diagnostics_path = output_dir / "grpo_diagnostics.jsonl"
    if stage == "grpo" and resume is not None and diagnostics_path.is_file():
        diagnostics = read_jsonl(diagnostics_path)
        for item in diagnostics:
            rewards = item.get("rewards", [])
            reward_values.extend(float(reward.get("total", 0.0)) for reward in rewards)
            total_generations += len(rewards)
            valid_json_generations += sum(bool(reward.get("valid_json")) for reward in rewards)
            total_groups += 1
            saturated = bool(item.get("saturated"))
            saturated_groups += int(saturated)
            consecutive_saturated_groups = consecutive_saturated_groups + 1 if saturated else 0
            max_observed_consecutive_saturated_groups = max(
                max_observed_consecutive_saturated_groups,
                consecutive_saturated_groups,
            )
            training = item.get("training", {})
            if training.get("anchor_loss") is not None:
                anchor_losses.append(float(training["anchor_loss"]))
            if training.get("policy_loss") is not None:
                policy_losses.append(float(training["policy_loss"]))
            if training.get("kl") is not None:
                kl_values.append(float(training["kl"]))
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    canary_report = None
    if stage == "grpo":
        canary_report = _run_grpo_canary(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            loader=loader,
            params=params,
            output_dir=output_dir,
        )
    iterator = iter(loader)
    while completed < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        rows = batch.pop("rows")
        batch = _to_device(batch, accelerator.device)
        with accelerator.accumulate(model):
            if stage == "sft":
                output = model(
                    encoded_audio=batch["encoded_audio"],
                    audio_frame_mask=batch["audio_frame_mask"],
                    audio_present=batch["audio_present"],
                    input_ids=batch["input_ids"],
                    token_attention_mask=batch["token_attention_mask"],
                    labels=batch["labels"],
                )
                loss = output.loss
            elif stage == "dpo":
                unwrapped = accelerator.unwrap_model(model)
                policy_name = unwrapped.policy_adapter
                reference_name = unwrapped.reference_adapter
                policy_chosen = _forward_logp(model, batch, batch["chosen"], policy_name)
                policy_rejected = _forward_logp(model, batch, batch["rejected"], policy_name)
                with torch.no_grad():
                    reference_chosen = _forward_logp(model, batch, batch["chosen"], reference_name)
                    reference_rejected = _forward_logp(
                        model, batch, batch["rejected"], reference_name
                    )
                # PEFT adapter switching also toggles leaf trainability. Reactivate
                # policy before backward so its LoRA gradients are accumulated.
                unwrapped.language_model.set_adapter(policy_name)
                loss = conservative_dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    reference_chosen,
                    reference_rejected,
                    beta=float(params.get("beta", 0.1)),
                    label_smoothing=float(params.get("label_smoothing", 0.1)),
                )
            else:
                unwrapped = accelerator.unwrap_model(model)
                generations = int(params.get("num_generations", 4))
                completions, breakdowns = _sample_grpo_group(
                    unwrapped, tokenizer, batch, rows[0], params
                )
                rewards = torch.tensor(
                    [item.total for item in breakdowns],
                    dtype=torch.float32,
                    device=accelerator.device,
                ).view(1, -1)
                reward_values.extend(rewards.flatten().tolist())
                advantages = normalized_group_advantages(rewards)
                group_summary = summarize_reward_group(breakdowns)
                group_saturated = group_summary.saturated
                total_groups += 1
                total_generations += len(breakdowns)
                valid_json_generations += sum(item.valid_json for item in breakdowns)
                saturated_groups += int(group_saturated)
                consecutive_saturated_groups = (
                    consecutive_saturated_groups + 1 if group_saturated else 0
                )
                max_observed_consecutive_saturated_groups = max(
                    max_observed_consecutive_saturated_groups,
                    consecutive_saturated_groups,
                )
                diagnostics.append(
                    {
                        "group": total_groups,
                        **group_summary.to_dict(),
                        "rewards": [item.to_dict() for item in breakdowns],
                    }
                )
                max_consecutive = int(params.get("max_consecutive_saturated_groups", 16))
                if consecutive_saturated_groups >= max_consecutive:
                    collapse_report = {
                        "schema_version": "lse.grpo_collapse.v1",
                        "status": "aborted",
                        "reason": "consecutive_saturated_groups",
                        "groups": total_groups,
                        "consecutive_saturated_groups": consecutive_saturated_groups,
                        "limit": max_consecutive,
                        "valid_json_rate": valid_json_generations / total_generations,
                    }
                    if accelerator.is_main_process:
                        write_json_atomic(output_dir / "collapse_report.json", collapse_report)
                        write_jsonl(output_dir / "grpo_diagnostics.jsonl", diagnostics)
                    raise RuntimeError(
                        "GRPO reward remained tied for "
                        f"{consecutive_saturated_groups} consecutive groups; aborting instead "
                        "of performing KL-only updates"
                    )
                repeated = [rows[0] for _ in completions]
                audio_repeat = _to_device(_audio_batch(repeated), accelerator.device)
                response_batch = _to_device(
                    _pad_examples(
                        tokenizer,
                        [
                            build_lm_example(
                                tokenizer,
                                rows[0]["prompt_text"],
                                completion,
                                int(params.get("max_tokens", 768)),
                            )
                            for completion in completions
                        ],
                    ),
                    accelerator.device,
                )
                group_batch = {
                    **audio_repeat,
                    "audio_present": torch.ones(generations, device=accelerator.device),
                }
                unwrapped.train()
                policy_logps = _forward_logp(
                    model, group_batch, response_batch, unwrapped.policy_adapter
                ).view(1, -1)
                with torch.no_grad():
                    reference_logps = _forward_logp(
                        model, group_batch, response_batch, unwrapped.reference_adapter
                    ).view(1, -1)
                unwrapped.language_model.set_adapter(unwrapped.policy_adapter)
                grpo_loss, grpo_stats = grpo_surrogate_loss(
                    policy_logps,
                    policy_logps.detach(),
                    reference_logps,
                    advantages,
                    beta=float(params.get("beta", 0.04)),
                    clip_epsilon=float(params.get("clip_epsilon", 0.2)),
                )
                anchor_response = _to_device(
                    _pad_examples(
                        tokenizer,
                        [
                            build_lm_example(
                                tokenizer,
                                rows[0]["prompt_text"],
                                rows[0]["chosen_text"],
                                int(params.get("max_tokens", 768)),
                            )
                        ],
                    ),
                    accelerator.device,
                )
                anchor_output = model(
                    encoded_audio=batch["encoded_audio"],
                    audio_frame_mask=batch["audio_frame_mask"],
                    audio_present=batch["audio_present"],
                    input_ids=anchor_response["input_ids"],
                    token_attention_mask=anchor_response["token_attention_mask"],
                    labels=anchor_response["labels"],
                    adapter_name=unwrapped.policy_adapter,
                )
                loss, anchor_stats = combine_grpo_and_anchor_loss(
                    grpo_loss,
                    anchor_output.loss,
                    anchor_weight=float(params.get("sft_anchor_weight", 0.05)),
                    group_saturated=group_saturated,
                )
                anchor_losses.append(anchor_stats["anchor_loss"])
                policy_losses.append(grpo_stats["policy_loss"])
                kl_values.append(grpo_stats["kl"])
                diagnostics[-1]["training"] = {
                    "anchor_loss": anchor_stats["anchor_loss"],
                    "anchor_weight": anchor_stats["anchor_weight"],
                    "policy_loss": grpo_stats["policy_loss"],
                    "kl": grpo_stats["kl"],
                }
            if not torch.isfinite(loss):
                raise RuntimeError(f"{stage} produced non-finite loss")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable, float(params.get("max_grad_norm", 1.0)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step = accelerator.sync_gradients
        if not optimizer_step:
            continue
        completed += 1
        losses.append(float(loss.detach().cpu()))
        if stage == "grpo" and accelerator.is_main_process:
            print(
                f"grpo step={completed}/{max_steps} loss={losses[-1]:.6f} "
                f"valid_json={valid_json_generations / total_generations:.3f} "
                f"non_saturated={(total_groups - saturated_groups) / total_groups:.3f}",
                flush=True,
            )
        save_steps = int(params.get("save_steps", 50))
        if completed % save_steps == 0 and completed < max_steps:
            checkpoint = output_dir / f"checkpoint-{completed}"
            accelerator.save_state(str(checkpoint))
            if accelerator.is_main_process:
                write_json_atomic(checkpoint / "trainer_state.json", {"step": completed})
                _prune_stage_checkpoints(output_dir, keep=int(params.get("save_total_limit", 2)))
                if stage == "grpo":
                    write_jsonl(diagnostics_path, diagnostics)
    final_dir = output_dir / "final"
    _save_stage(accelerator, model, tokenizer, final_dir, stage)
    report = {
        "schema_version": "lse.native_stage_run.v1",
        "stage": stage,
        "status": "completed",
        "started_from": str(input_dir) if input_dir else config.language_model,
        "steps": completed,
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "loss_mean": sum(losses) / len(losses) if losses else None,
        "mean_reward": sum(reward_values) / len(reward_values) if reward_values else None,
        "saturated_groups": saturated_groups if stage == "grpo" else None,
        "total_groups": total_groups if stage == "grpo" else None,
        "non_saturated_group_rate": (
            (total_groups - saturated_groups) / total_groups
            if stage == "grpo" and total_groups
            else None
        ),
        "valid_json_rate": (
            valid_json_generations / total_generations
            if stage == "grpo" and total_generations
            else None
        ),
        "max_consecutive_saturated_groups": (
            max_observed_consecutive_saturated_groups if stage == "grpo" else None
        ),
        "mean_anchor_loss": (sum(anchor_losses) / len(anchor_losses) if anchor_losses else None),
        "mean_policy_loss": (sum(policy_losses) / len(policy_losses) if policy_losses else None),
        "mean_kl": sum(kl_values) / len(kl_values) if kl_values else None,
        "canary": canary_report,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(final_dir),
    }
    if accelerator.is_main_process:
        write_json_atomic(output_dir / "stage_manifest.json", report)
        if stage == "grpo":
            write_jsonl(diagnostics_path, diagnostics)
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    from deepspeed.comm import comm as deepspeed_comm
    from deepspeed.utils import groups as deepspeed_groups

    if deepspeed_comm.cdb is not None and deepspeed_comm.cdb.is_initialized():
        deepspeed_comm.destroy_process_group()
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    deepspeed_comm.cdb = None
    deepspeed_groups._WORLD_GROUP = None
    deepspeed_groups._DATA_PARALLEL_GROUP = None
    deepspeed_groups._ZERO_PARAM_INTRA_PARALLEL_GROUP = None
    deepspeed_groups.mesh_device = None
    deepspeed_groups.mpu = None
    deepspeed_groups._EXPERT_PARALLEL_GROUP.clear()
    deepspeed_groups._EXPERT_DATA_PARALLEL_GROUP.clear()
    deepspeed_groups._ALL_TO_ALL_GROUP.clear()
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _completed_stage_report(
    output_dir: Path,
    stage: str,
    expected_steps: int,
) -> dict[str, Any] | None:
    """Return a validated completed-stage report, otherwise force a safe resume."""

    manifest_path = output_dir / stage / "stage_manifest.json"
    final_dir = output_dir / stage / "final"
    artifact_manifest_path = final_dir / "artifact_manifest.json"
    if not (
        manifest_path.is_file()
        and artifact_manifest_path.is_file()
        and (final_dir / "audio_projector.pt").is_file()
    ):
        return None
    try:
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
        adapter_dir = Path(str(artifact_manifest["adapter"])).resolve()
        completed_steps = int(report.get("steps", -1))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        report.get("schema_version") != "lse.native_stage_run.v1"
        or report.get("stage") != stage
        or report.get("status") != "completed"
        or completed_steps < expected_steps
        or Path(str(report.get("output", ""))).resolve() != final_dir.resolve()
        or artifact_manifest.get("schema_version") != "lse.native_stage_artifacts.v1"
        or artifact_manifest.get("stage") != stage
        or not adapter_dir.is_relative_to(final_dir.resolve())
        or not (adapter_dir / "adapter_config.json").is_file()
    ):
        return None
    return report


def run_native_training_pipeline(config: NativePipelineConfig) -> dict[str, Any]:
    """Execute cache -> SFT -> cDPO -> GRPO with stage lineage and manifests."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("native training requires a CUDA GPU")
    set_global_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_native_records(config.manifest)
    records, cache_report = precompute_audio_embeddings(
        records,
        whisper_model=config.whisper_model,
        cache_dir=config.output_dir / "audio_embedding_cache",
        batch_size=int(os.environ.get("LSE_EMBEDDING_BATCH_SIZE", "16")),
    )
    first = np.load(records[0]["embedding_path"], allow_pickle=False)
    encoder_dim = int(first["hidden"].shape[1])
    first.close()
    stages: dict[str, Any] = {}
    stage_outputs: dict[str, Path] = {}
    for stage in NATIVE_STAGES:
        final_dir = config.output_dir / stage / "final"
        input_dir = resolve_stage_input_dir(config, stage, stage_outputs)
        completed_report = _completed_stage_report(
            config.output_dir,
            stage,
            int(config.stages[stage]["max_steps"]),
        )
        if completed_report is not None:
            stages[stage] = completed_report
            stage_outputs[stage] = final_dir
            continue
        stages[stage] = _run_stage(config, stage, records, encoder_dim, input_dir)
        stage_outputs[stage] = final_dir
    report = {
        "schema_version": "lse.native_pipeline_run.v1",
        "status": "completed",
        "completed_at": utc_now(),
        "config": str(config.config_path),
        "git_commit": git_commit(config.config_path.parent),
        "manifest": str(config.manifest),
        "records": len(records),
        "physical_audio_records": len(records),
        "audio_cache": cache_report,
        "stages": stages,
        "models": {"audio_encoder": config.whisper_model, "language_model": config.language_model},
        "versions": _versions(),
        "gpu": _gpu_description(),
        "gpu_claim": True,
        "final_adapter": str(config.output_dir / "grpo" / "final"),
        "command": " ".join(sys.argv),
    }
    write_json_atomic(config.output_dir / "run_manifest.json", report)
    return report
