"""Dependency-light objective metrics for paired enhancement evaluation."""

from __future__ import annotations

import math

import torch

from .codec import ComplexSTFTCodec


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    estimate = estimate.double().flatten()
    reference = reference.double().flatten()
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    scale = torch.dot(estimate, reference) / (torch.dot(reference, reference) + 1e-12)
    target = scale * reference
    noise = estimate - target
    return float(10 * torch.log10(target.square().sum() / (noise.square().sum() + 1e-12)))


def snr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = reference.double().square().sum()
    denominator = (estimate.double() - reference.double()).square().sum()
    return float(10 * torch.log10(numerator / (denominator + 1e-12)))


def log_spectral_distance(
    estimate: torch.Tensor, reference: torch.Tensor, codec: ComplexSTFTCodec
) -> float:
    estimate_spectrum = codec.encode(estimate)
    reference_spectrum = codec.encode(reference)
    estimate_magnitude = estimate_spectrum.square().sum(dim=1).sqrt().clamp_min(1e-7)
    reference_magnitude = reference_spectrum.square().sum(dim=1).sqrt().clamp_min(1e-7)
    delta = 20 * (estimate_magnitude.log10() - reference_magnitude.log10())
    return float(delta.square().mean(dim=1).sqrt().mean())


def optional_pesq_stoi(
    estimate: torch.Tensor, reference: torch.Tensor, sample_rate: int
) -> dict[str, float | None]:
    estimate_np = estimate.detach().cpu().flatten().numpy()
    reference_np = reference.detach().cpu().flatten().numpy()
    result: dict[str, float | None] = {"pesq": None, "stoi": None}
    try:
        from pesq import pesq

        mode = "wb" if sample_rate == 16000 else "nb"
        result["pesq"] = float(pesq(sample_rate, reference_np, estimate_np, mode))
    except (ImportError, ValueError, RuntimeError):
        pass
    try:
        from pystoi import stoi

        result["stoi"] = float(stoi(reference_np, estimate_np, sample_rate, extended=False))
    except (ImportError, ValueError, RuntimeError):
        pass
    return result


def finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None
