"""Prescription-conditioned multimodal diffusion transformer for speech enhancement.

The model stack is imported lazily so contract and planner utilities remain usable in
lightweight environments that intentionally do not install PyTorch.
"""

from __future__ import annotations

from typing import Any

from .contracts import MMDIT_PAIR_SCHEMA

__all__ = [
    "MMDIT_PAIR_SCHEMA",
    "MMDiTConfig",
    "PrescriptionConditionedMMDiT",
    "RectifiedFlow",
    "RectifiedFlowConfig",
]


def __getattr__(name: str) -> Any:
    if name in {"RectifiedFlow", "RectifiedFlowConfig"}:
        from .flow import RectifiedFlow, RectifiedFlowConfig

        return {"RectifiedFlow": RectifiedFlow, "RectifiedFlowConfig": RectifiedFlowConfig}[name]
    if name in {"MMDiTConfig", "PrescriptionConditionedMMDiT"}:
        from .model import MMDiTConfig, PrescriptionConditionedMMDiT

        return {
            "MMDiTConfig": MMDiTConfig,
            "PrescriptionConditionedMMDiT": PrescriptionConditionedMMDiT,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
