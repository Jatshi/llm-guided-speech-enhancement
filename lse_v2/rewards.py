"""Verifiable reward components for GRPO and offline ablation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

DEFAULT_WEIGHTS = {
    "format": 1.0,
    "diagnosis": 1.0,
    "parameter_bounds": 1.0,
    "consistency": 1.0,
    "overprocessing": 1.0,
}


@dataclass(frozen=True)
class RewardBreakdown:
    format: float
    diagnosis: float
    parameter_bounds: float
    consistency: float
    overprocessing: float
    total: float
    valid_json: bool
    violations: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "diagnosis": self.diagnosis,
            "parameter_bounds": self.parameter_bounds,
            "consistency": self.consistency,
            "overprocessing": self.overprocessing,
            "total": self.total,
            "valid_json": self.valid_json,
            "violations": list(self.violations),
        }


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        return "".join(completion_text(item) for item in completion)
    return str(completion)


def parse_prescription(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_reward(payload: dict[str, Any] | None, violations: list[str]) -> float:
    if payload is None:
        violations.append("invalid_json")
        return 0.0
    required = {"diagnosis", "actions", "rationale", "confidence"}
    missing = required.difference(payload)
    if missing:
        violations.append("missing_keys:" + ",".join(sorted(missing)))
        return max(0.0, 1.0 - 0.25 * len(missing))
    if not isinstance(payload["diagnosis"], dict):
        violations.append("diagnosis_not_object")
        return 0.5
    if not isinstance(payload["actions"], list) or not payload["actions"]:
        violations.append("actions_empty_or_invalid")
        return 0.5
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        violations.append("rationale_empty")
        return 0.75
    confidence = payload["confidence"]
    if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
        violations.append("confidence_out_of_range")
        return 0.75
    return 1.0


def _diagnosis_reward(
    payload: dict[str, Any] | None, context: dict[str, Any], violations: list[str]
) -> float:
    if payload is None or not isinstance(payload.get("diagnosis"), dict):
        return 0.0
    expected = str(context.get("noise_type", "")).strip().lower()
    predicted = str(payload["diagnosis"].get("noise_type", "")).strip().lower()
    if not expected:
        return 0.5
    if predicted == expected:
        score = 1.0
    elif expected in predicted or predicted in expected:
        score = 0.75
    else:
        violations.append(f"noise_type_mismatch:{predicted}!={expected}")
        score = 0.0
    expected_reverb = context.get("reverb_rt60")
    if expected_reverb is not None:
        predicted_reverb = bool(payload["diagnosis"].get("reverb"))
        if bool(expected_reverb) != predicted_reverb:
            violations.append("reverb_mismatch")
            score *= 0.75
    return score


def _parameter_bounds_reward(payload: dict[str, Any] | None, violations: list[str]) -> float:
    if payload is None or not isinstance(payload.get("actions"), list):
        return 0.0
    checks = 0
    passed = 0
    for index, action in enumerate(payload["actions"]):
        if not isinstance(action, dict):
            violations.append(f"action_{index}_not_object")
            checks += 1
            continue
        bounds = {
            "reduction_db": (0.0, 24.0),
            "gain_db": (-24.0, 6.0),
            "q": (0.5, 20.0),
            "low_hz": (0.0, 20000.0),
            "high_hz": (20.0, 24000.0),
        }
        for key, (low, high) in bounds.items():
            if key not in action:
                continue
            checks += 1
            value = action[key]
            if isinstance(value, int | float) and math.isfinite(value) and low <= value <= high:
                passed += 1
            else:
                violations.append(f"action_{index}_{key}_out_of_range")
        if "low_hz" in action and "high_hz" in action:
            checks += 1
            if action["low_hz"] < action["high_hz"]:
                passed += 1
            else:
                violations.append(f"action_{index}_frequency_order")
    if checks == 0:
        violations.append("no_executable_parameters")
        return 0.0
    return passed / checks


def _consistency_reward(
    payload: dict[str, Any] | None, context: dict[str, Any], violations: list[str]
) -> float:
    if payload is None or not isinstance(payload.get("actions"), list):
        return 0.0
    score = 1.0
    noise_type = str(context.get("noise_type", "")).lower()
    action_types = {
        str(action.get("type", "")).lower()
        for action in payload["actions"]
        if isinstance(action, dict)
    }
    if noise_type in {"white", "pink", "cafe", "hvac"} and not action_types:
        violations.append("missing_action_type")
        score -= 0.5
    if context.get("reverb_rt60") and not any(
        action in {"dereverb", "wpe", "de_reverb"} for action in action_types
    ):
        violations.append("reverb_without_dereverb_action")
        score -= 0.25
    if context.get("bandlimit_hz") and not any(
        action in {"bandwidth_extension", "equalizer", "eq"} for action in action_types
    ):
        violations.append("bandlimit_without_restoration_action")
        score -= 0.25
    for index, action in enumerate(payload["actions"]):
        if not isinstance(action, dict):
            continue
        if action.get("type") == "highpass" and action.get("low_hz", 0) > 300:
            violations.append(f"action_{index}_highpass_conflicts_with_speech")
            score -= 0.25
    return max(0.0, score)


def _overprocessing_reward(
    payload: dict[str, Any] | None, context: dict[str, Any], violations: list[str]
) -> float:
    if payload is None or not isinstance(payload.get("actions"), list):
        return 0.0
    penalty = 0.0
    total_reduction = 0.0
    for index, action in enumerate(payload["actions"]):
        if not isinstance(action, dict):
            continue
        reduction = action.get("reduction_db", 0.0)
        if isinstance(reduction, int | float):
            total_reduction += max(0.0, float(reduction))
            if reduction > 18:
                penalty += min(0.5, (reduction - 18) / 24)
                violations.append(f"action_{index}_over_suppression")
        gain = action.get("gain_db")
        if isinstance(gain, int | float) and abs(gain) > 18:
            penalty += 0.25
            violations.append(f"action_{index}_extreme_gain")
        if action.get("type") == "highpass" and action.get("low_hz", 0) > 180:
            penalty += 0.25
            violations.append(f"action_{index}_highpass_overprocessing")
    if total_reduction > 30:
        penalty += min(0.5, (total_reduction - 30) / 30)
        violations.append("cumulative_over_suppression")
    snr = context.get("snr_db")
    if isinstance(snr, int | float) and snr >= 20 and total_reduction > 12:
        penalty += 0.25
        violations.append("clean_signal_overprocessed")
    return max(0.0, 1.0 - penalty)


def score_prescription(
    text: str,
    context: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> RewardBreakdown:
    payload = parse_prescription(text)
    violations: list[str] = []
    components = {
        "format": _format_reward(payload, violations),
        "diagnosis": _diagnosis_reward(payload, context, violations),
        "parameter_bounds": _parameter_bounds_reward(payload, violations),
        "consistency": _consistency_reward(payload, context, violations),
        "overprocessing": _overprocessing_reward(payload, context, violations),
    }
    effective = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights).difference(effective)
        if unknown:
            raise ValueError(f"Unknown reward components: {sorted(unknown)}")
        effective.update(weights)
    denominator = sum(max(0.0, value) for value in effective.values())
    if denominator <= 0:
        raise ValueError("At least one reward weight must be positive")
    total = sum(components[key] * max(0.0, effective[key]) for key in components) / denominator
    return RewardBreakdown(
        **components,
        total=round(float(total), 6),
        valid_json=payload is not None,
        violations=tuple(dict.fromkeys(violations)),
    )


def grpo_reward(
    completions: list[Any],
    reward_context: list[dict[str, Any]] | dict[str, Any] | None = None,
    **_: Any,
) -> list[float]:
    """TRL-compatible reward function; every reward is locally verifiable."""
    if reward_context is None:
        contexts = [{} for _ in completions]
    elif isinstance(reward_context, dict):
        contexts = [reward_context for _ in completions]
    else:
        contexts = reward_context
    if len(contexts) != len(completions):
        raise ValueError("reward_context and completions must have the same length")
    return [
        score_prescription(completion_text(completion), context).total
        for completion, context in zip(completions, contexts, strict=True)
    ]
