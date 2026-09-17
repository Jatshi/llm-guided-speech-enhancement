"""Checkpoint format shared by training, evaluation, and inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .flow import RectifiedFlow, RectifiedFlowConfig
from .model import MMDiTConfig, PrescriptionConditionedMMDiT


def atomic_torch_save(payload: dict[str, Any], destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_model_checkpoint(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> tuple[PrescriptionConditionedMMDiT, RectifiedFlow, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != "lse.mmdit.checkpoint.v1":
        raise ValueError("unsupported MM-DiT checkpoint schema")
    model = PrescriptionConditionedMMDiT(MMDiTConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    flow = RectifiedFlow(RectifiedFlowConfig(**payload["flow_config"]))
    return model, flow, payload
