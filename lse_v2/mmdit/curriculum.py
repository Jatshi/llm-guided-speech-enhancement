"""Validated progressive schedule for interference-aware MM-DiT training."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .flow import RectifiedFlowConfig


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    start_step: int
    until_step: int
    flow_overrides: dict[str, float]
    mrstft_weight: float
    semantic_weight: float
    lr_scale: float

    def flow_config(self, base: RectifiedFlowConfig) -> RectifiedFlowConfig:
        allowed = set(base.__dataclass_fields__)
        unknown = set(self.flow_overrides).difference(allowed)
        if unknown:
            raise ValueError(f"unknown flow overrides: {sorted(unknown)}")
        return replace(base, **self.flow_overrides)


@dataclass(frozen=True)
class CurriculumSchedule:
    phases: tuple[CurriculumPhase, ...]

    @classmethod
    def from_config(
        cls, rows: list[dict[str, Any]] | None, *, max_steps: int
    ) -> CurriculumSchedule:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not rows:
            return cls(
                (
                    CurriculumPhase(
                        name="joint",
                        start_step=1,
                        until_step=max_steps,
                        flow_overrides={},
                        mrstft_weight=0.0,
                        semantic_weight=0.0,
                        lr_scale=1.0,
                    ),
                )
            )
        phases = []
        start = 1
        for index, row in enumerate(rows):
            name = str(row.get("name", "")).strip()
            if not name:
                raise ValueError(f"curriculum phase {index} requires a name")
            until = int(row.get("until_step", 0))
            if until < start:
                raise ValueError("curriculum until_step values must be strictly increasing")
            weights = row.get("loss_weights", {})
            flow = row.get("flow", {})
            if not isinstance(weights, dict) or not isinstance(flow, dict):
                raise ValueError("curriculum flow and loss_weights must be objects")
            mrstft = float(weights.get("mrstft", 0.0))
            semantic = float(weights.get("semantic", 0.0))
            lr_scale = float(row.get("lr_scale", 1.0))
            if min(mrstft, semantic) < 0 or lr_scale <= 0:
                raise ValueError("curriculum weights must be non-negative and lr_scale positive")
            phases.append(
                CurriculumPhase(
                    name=name,
                    start_step=start,
                    until_step=until,
                    flow_overrides={key: float(value) for key, value in flow.items()},
                    mrstft_weight=mrstft,
                    semantic_weight=semantic,
                    lr_scale=lr_scale,
                )
            )
            start = until + 1
        if phases[-1].until_step < max_steps:
            raise ValueError("curriculum must cover training.max_steps")
        return cls(tuple(phases))

    def phase_for_step(self, step: int) -> CurriculumPhase:
        if step < 1:
            raise ValueError("step is one-based")
        for phase in self.phases:
            if phase.start_step <= step <= phase.until_step:
                return phase
        return self.phases[-1]

    @property
    def needs_semantic_teacher(self) -> bool:
        return any(phase.semantic_weight > 0 for phase in self.phases)
