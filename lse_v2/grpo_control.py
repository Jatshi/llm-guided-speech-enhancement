"""Fail-closed diagnostics and canary gates for native-audio GRPO."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .rewards import RewardBreakdown


@dataclass(frozen=True, slots=True)
class RewardGroupSummary:
    generations: int
    mean_reward: float
    reward_std: float
    reward_span: float
    valid_json_rate: float
    saturated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_reward_group(
    breakdowns: Sequence[RewardBreakdown], *, epsilon: float = 1e-6
) -> RewardGroupSummary:
    if len(breakdowns) < 2:
        raise ValueError("a GRPO group requires at least two generations")
    totals = [float(item.total) for item in breakdowns]
    span = max(totals) - min(totals)
    return RewardGroupSummary(
        generations=len(totals),
        mean_reward=statistics.fmean(totals),
        reward_std=statistics.pstdev(totals),
        reward_span=span,
        valid_json_rate=sum(item.valid_json for item in breakdowns) / len(totals),
        saturated=span <= epsilon,
    )


def evaluate_canary_gate(
    groups: Sequence[Sequence[RewardBreakdown]],
    *,
    min_valid_json_rate: float,
    min_non_saturated_group_rate: float,
) -> dict[str, Any]:
    if not groups:
        raise ValueError("canary requires at least one reward group")
    for name, value in (
        ("min_valid_json_rate", min_valid_json_rate),
        ("min_non_saturated_group_rate", min_non_saturated_group_rate),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    summaries = [summarize_reward_group(group) for group in groups]
    flat = [item for group in groups for item in group]
    valid_json_rate = sum(item.valid_json for item in flat) / len(flat)
    non_saturated_rate = sum(not item.saturated for item in summaries) / len(summaries)
    failed_checks: list[str] = []
    if valid_json_rate < min_valid_json_rate:
        failed_checks.append("valid_json_rate")
    if non_saturated_rate < min_non_saturated_group_rate:
        failed_checks.append("non_saturated_group_rate")
    component_names = (
        "format",
        "diagnosis",
        "parameter_bounds",
        "consistency",
        "overprocessing",
        "total",
    )
    return {
        "schema_version": "lse.grpo_canary.v1",
        "status": "failed" if failed_checks else "passed",
        "groups": len(groups),
        "generations": len(flat),
        "valid_json_rate": valid_json_rate,
        "non_saturated_group_rate": non_saturated_rate,
        "mean_components": {
            name: statistics.fmean(float(getattr(item, name)) for item in flat)
            for name in component_names
        },
        "failed_checks": failed_checks,
        "thresholds": {
            "min_valid_json_rate": min_valid_json_rate,
            "min_non_saturated_group_rate": min_non_saturated_group_rate,
        },
        "group_summaries": [item.to_dict() for item in summaries],
    }
