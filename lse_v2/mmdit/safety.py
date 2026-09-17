"""No-op fallback gate for generated enhancement waveforms."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class EnhancementGateConfig:
    min_planner_confidence: float = 0.35
    max_peak: float = 0.999
    min_energy_ratio: float = 0.2
    max_energy_ratio: float = 5.0
    max_change_rms_ratio: float = 3.0


class EnhancementGate:
    def __init__(self, config: EnhancementGateConfig | None = None) -> None:
        self.config = config or EnhancementGateConfig()

    def select(
        self,
        original: torch.Tensor,
        candidate: torch.Tensor,
        *,
        planner_confidence: float,
    ) -> tuple[torch.Tensor, dict]:
        if original.shape != candidate.shape:
            raise ValueError("original and candidate shapes must match")
        reasons: list[str] = []
        if not torch.isfinite(candidate).all():
            reasons.append("non_finite")
        peak = float(candidate.abs().max()) if candidate.numel() else float("inf")
        original_rms = float(original.square().mean().sqrt()) + 1e-8
        candidate_rms = float(candidate.square().mean().sqrt())
        energy_ratio = (candidate_rms / original_rms) ** 2
        change_ratio = float((candidate - original).square().mean().sqrt()) / original_rms
        if planner_confidence < self.config.min_planner_confidence:
            reasons.append("low_planner_confidence")
        if peak > self.config.max_peak:
            reasons.append("peak_limit")
        if not self.config.min_energy_ratio <= energy_ratio <= self.config.max_energy_ratio:
            reasons.append("energy_ratio")
        if change_ratio > self.config.max_change_rms_ratio:
            reasons.append("excessive_change")
        used_fallback = bool(reasons)
        report = {
            "used_fallback": used_fallback,
            "reasons": reasons,
            "peak": peak,
            "energy_ratio": energy_ratio,
            "change_rms_ratio": change_ratio,
            "planner_confidence": planner_confidence,
            "gate_config": asdict(self.config),
        }
        return (original if used_fallback else candidate), report
