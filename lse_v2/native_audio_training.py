"""Train a native audio-prefix projector against a frozen causal language model."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class NativeAudioRecord:
    sample_id: str
    audio_path: Path
    target_text: str


def load_native_audio_manifest(path: str | Path) -> list[NativeAudioRecord]:
    source = Path(path)
    rows: list[NativeAudioRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        sample_id = str(payload.get("sample_id", ""))
        raw_audio = str(payload.get("audio_path", ""))
        target_text = str(payload.get("target_text", ""))
        if not sample_id or not raw_audio or not target_text:
            raise ValueError(f"line {line_number} requires sample_id, audio_path, target_text")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        audio_path = Path(raw_audio)
        if not audio_path.is_absolute():
            audio_path = (source.parent / audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        rows.append(
            NativeAudioRecord(
                sample_id=sample_id,
                audio_path=audio_path,
                target_text=target_text,
            )
        )
    if not rows:
        raise ValueError("native audio manifest is empty")
    return rows


def load_audio(path: Path, target_sample_rate: int) -> np.ndarray:
    import soundfile as sf  # type: ignore[import-untyped]

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError(f"invalid waveform: {path}")
    if sample_rate != target_sample_rate:
        duration = waveform.size / sample_rate
        source_times = np.linspace(0.0, duration, waveform.size, endpoint=False)
        target_size = max(1, round(duration * target_sample_rate))
        target_times = np.linspace(0.0, duration, target_size, endpoint=False)
        waveform = np.interp(target_times, source_times, waveform).astype(np.float32)
    return np.asarray(waveform, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class NativeAudioTrainConfig:
    manifest: Path
    output_dir: Path
    whisper_model: str = "openai/whisper-small"
    language_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    epochs: int = 1
    learning_rate: float = 2e-4
    pooling_stride: int = 16
    max_target_tokens: int = 192
    seed: int = 17

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0 or self.pooling_stride <= 0:
            raise ValueError("epochs, learning_rate, and pooling_stride must be positive")


def train_native_audio_projector(config: NativeAudioTrainConfig) -> dict[str, Any]:
    import torch
    from torch import nn
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        WhisperFeatureExtractor,
        WhisperModel,
    )

    from lse_v2.audio_conditioning import AudioConditioningConfig, WhisperAudioProjector

    if not torch.cuda.is_available():
        raise RuntimeError("native audio training requires CUDA")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    records = load_native_audio_manifest(config.manifest)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.whisper_model)
    whisper = WhisperModel.from_pretrained(config.whisper_model, torch_dtype=dtype).to(device)
    language_model = AutoModelForCausalLM.from_pretrained(
        config.language_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.language_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    whisper.requires_grad_(False).eval()
    language_model.requires_grad_(False).eval()
    encoder_dim = int(whisper.config.d_model)
    llm_dim = int(language_model.config.hidden_size)
    conditioner = WhisperAudioProjector(
        whisper.encoder,
        AudioConditioningConfig(
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
            pooling_stride=config.pooling_stride,
        ),
    ).to(device=device, dtype=dtype)
    conditioner.train()
    optimizer = torch.optim.AdamW(conditioner.projector.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for _epoch in range(config.epochs):
        for record in records:
            waveform = load_audio(record.audio_path, feature_extractor.sampling_rate)
            features = feature_extractor(
                waveform,
                sampling_rate=feature_extractor.sampling_rate,
                return_tensors="pt",
            ).input_features.to(device=device, dtype=dtype)
            tokenized = tokenizer(
                record.target_text,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_target_tokens,
            )
            token_ids = tokenized.input_ids.to(device)
            prefix = conditioner(features)
            token_embeddings = language_model.get_input_embeddings()(token_ids)
            inputs_embeds = torch.cat([prefix, token_embeddings], dim=1)
            prefix_labels = torch.full(
                (token_ids.shape[0], prefix.shape[1]),
                -100,
                dtype=token_ids.dtype,
                device=device,
            )
            labels = torch.cat([prefix_labels, token_ids], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = output.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("native audio loss is missing or non-finite")
            loss.backward()
            nn.utils.clip_grad_norm_(conditioner.projector.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.output_dir / "audio_projector.pt"
    torch.save(conditioner.projector.state_dict(), checkpoint)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "lse.native_audio_train.v3",
        "status": "completed",
        "config": {
            **asdict(config),
            "manifest": str(config.manifest),
            "output_dir": str(config.output_dir),
        },
        "examples": len(records),
        "optimizer_steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "all_losses_finite": all(math.isfinite(item) for item in losses),
        "elapsed_seconds": elapsed,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "checkpoint": str(checkpoint),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "torch": torch.__version__,
    }
    (config.output_dir / "run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
