#!/usr/bin/env python3
"""Download and verify the frozen semantic teacher including input gradients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lse_v2.mmdit.losses import load_frozen_semantic_teacher, semantic_consistency_loss


def probe(model_name: str, output: Path, *, seconds: float = 1.0) -> dict[str, object]:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_frozen_semantic_teacher(model_name, device)
    samples = max(400, round(16_000 * seconds))
    timeline = torch.arange(samples, device=device, dtype=torch.float32) / 16_000
    reference = (0.1 * torch.sin(2 * torch.pi * 220 * timeline)).unsqueeze(0)
    estimate = (reference + 0.005 * torch.randn_like(reference)).detach().requires_grad_()
    loss = semantic_consistency_loss(teacher, estimate, reference, layer=9)
    loss.backward()
    gradient = estimate.grad
    gradient_norm = float(torch.linalg.vector_norm(gradient)) if gradient is not None else None
    passed = bool(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient_norm is not None
        and gradient_norm > 0
    )
    report: dict[str, object] = {
        "schema_version": "lse.mmdit.semantic_teacher_probe.v1",
        "status": "PASSED" if passed else "FAILED",
        "model": model_name,
        "device": str(device),
        "seconds": seconds,
        "loss": float(loss.detach()),
        "input_gradient_l2": gradient_norm,
        "trainable_teacher_parameters": sum(
            parameter.numel() for parameter in teacher.parameters() if parameter.requires_grad
        ),
        "cuda_peak_memory_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    report = probe(args.model, args.output, seconds=args.seconds)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
