"""Route easy, ambiguous, and unsupported degradations before invoking an LLM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptivePlanningConfig:
    direct_confidence: float = 0.85
    minimum_action_confidence: float = 0.35
    supported_diagnoses: tuple[str, ...] = ("white", "pink")

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_action_confidence <= self.direct_confidence <= 1:
            raise ValueError("require 0 <= minimum <= direct_confidence <= 1")
        if not self.supported_diagnoses:
            raise ValueError("supported_diagnoses must not be empty")


@dataclass(frozen=True)
class PlanningDecision:
    route: str
    reason: str
    label: str
    confidence: float

    @property
    def should_call_llm(self) -> bool:
        return self.route == "llm_planner"


def choose_planning_route(
    label: str, confidence: float, config: AdaptivePlanningConfig | None = None
) -> PlanningDecision:
    config = config or AdaptivePlanningConfig()
    normalized = str(label).strip().lower()
    confidence = float(max(0.0, min(1.0, confidence)))
    if normalized not in set(config.supported_diagnoses):
        return PlanningDecision("abstain", "unsupported_action", normalized, confidence)
    if confidence < config.minimum_action_confidence:
        return PlanningDecision("abstain", "insufficient_evidence", normalized, confidence)
    if confidence >= config.direct_confidence:
        return PlanningDecision("direct_template", "easy_high_confidence", normalized, confidence)
    return PlanningDecision("llm_planner", "ambiguous_supported_case", normalized, confidence)


def planning_benefit_label(
    direct_reward: float, planned_reward: float, *, margin: float = 0.05
) -> bool:
    if margin < 0:
        raise ValueError("margin must be non-negative")
    return float(planned_reward) - float(direct_reward) > margin
