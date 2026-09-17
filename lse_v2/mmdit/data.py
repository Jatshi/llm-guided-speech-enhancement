"""Paired waveform dataset with deterministic aligned cropping."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from lse_v2.io import read_jsonl

from .contracts import prescription_tokens, validate_pair_record


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for MM-DiT audio loading") from exc
    waveform, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    value = torch.from_numpy(waveform).mean(dim=1)
    if not value.numel():
        raise ValueError(f"empty audio file: {path}")
    return value, int(sample_rate)


def _resample(value: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return value
    length = max(1, round(value.numel() * target_rate / source_rate))
    return F.interpolate(
        value.view(1, 1, -1), size=length, mode="linear", align_corners=False
    ).view(-1)


def _aligned_segment(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    length: int,
    *,
    sample_id: str,
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    available = min(noisy.numel(), clean.numel())
    noisy, clean = noisy[:available], clean[:available]
    if available >= length:
        digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}".encode()).digest()
        start = int.from_bytes(digest[:8], "big") % (available - length + 1)
        return noisy[start : start + length], clean[start : start + length]
    return F.pad(noisy, (0, length - available)), F.pad(clean, (0, length - available))


class PairedEnhancementDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str,
        sample_rate: int,
        segment_seconds: float,
        prescription_token_count: int,
        prescription_source: str = "oracle",
        seed: int = 42,
        check_files: bool = True,
    ) -> None:
        records = read_jsonl(manifest)
        self.records = [record for record in records if record.get("split") == split]
        if not self.records:
            raise ValueError(f"no records for split={split!r}")
        for record in self.records:
            validate_pair_record(record, check_files=check_files)
        if prescription_source not in {"oracle", "predicted", "none"}:
            raise ValueError("prescription_source must be oracle/predicted/none")
        self.sample_rate = sample_rate
        self.segment_samples = round(sample_rate * segment_seconds)
        self.prescription_token_count = prescription_token_count
        self.prescription_source = prescription_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        noisy, noisy_rate = _load_audio(record["audio"]["noisy_path"])
        clean, clean_rate = _load_audio(record["audio"]["clean_path"])
        noisy = _resample(noisy, noisy_rate, self.sample_rate)
        clean = _resample(clean, clean_rate, self.sample_rate)
        noisy, clean = _aligned_segment(
            noisy,
            clean,
            self.segment_samples,
            sample_id=record["sample_id"],
            seed=self.seed,
            epoch=self.epoch,
        )
        joint_peak = max(float(noisy.abs().max()), float(clean.abs().max()), 1.0)
        noisy, clean = noisy / joint_peak, clean / joint_peak
        prescription = None
        if self.prescription_source == "oracle":
            prescription = record["oracle_prescription"]
        elif self.prescription_source == "predicted":
            prescription = record.get("predicted_prescription")
        tokens = prescription_tokens(prescription, max_tokens=self.prescription_token_count)
        fields, categories, values = zip(*tokens, strict=True)
        return {
            "sample_id": record["sample_id"],
            "noisy": noisy,
            "clean": clean,
            "fields": torch.tensor(fields, dtype=torch.long),
            "categories": torch.tensor(categories, dtype=torch.long),
            "values": torch.tensor(values, dtype=torch.float32),
            "planner_confidence": float((prescription or {}).get("confidence", 0.0)),
            "record": record,
        }


def collate_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [item["sample_id"] for item in items],
        "noisy": torch.stack([item["noisy"] for item in items]),
        "clean": torch.stack([item["clean"] for item in items]),
        "fields": torch.stack([item["fields"] for item in items]),
        "categories": torch.stack([item["categories"] for item in items]),
        "values": torch.stack([item["values"] for item in items]),
        "planner_confidence": torch.tensor(
            [item["planner_confidence"] for item in items], dtype=torch.float32
        ),
        "record": [item["record"] for item in items],
    }


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "noisy": batch["noisy"].to(device, non_blocking=True),
        "clean": batch["clean"].to(device, non_blocking=True),
        "fields": batch["fields"].to(device, non_blocking=True),
        "categories": batch["categories"].to(device, non_blocking=True),
        "values": batch["values"].to(device, non_blocking=True),
    }
