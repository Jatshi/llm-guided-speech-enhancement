"""Outcome-grounded rewards and preference pairs for Planner post-training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnhancementRewardConfig:
    si_sdr_scale: float = 5.0
    snr_scale: float = 5.0
    pesq_scale: float = 1.0
    stoi_scale: float = 0.2
    maximum_cer_increase: float = 0.5
    content_penalty_rate: float = 3.0
    speaker_floor: float = 0.5


def _number(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _bounded_tanh(value: float, scale: float) -> float:
    if scale <= 0:
        raise ValueError("reward scales must be positive")
    return math.tanh(value / scale)


def score_enhancement_outcome(
    metrics: dict[str, Any], config: EnhancementRewardConfig | None = None
) -> float:
    """Score actual enhanced audio, with hard protection for content and safety.

    The reward intentionally cannot be rescued by a high spectral score after a
    severe ASR regression or failed waveform gate.  This prevents the Planner
    from learning aggressive prescriptions that sound cleaner but change words.
    """

    config = config or EnhancementRewardConfig()
    if not bool(metrics.get("waveform_safe", False)) or bool(
        metrics.get("unsupported_action", False)
    ):
        return -1.0
    cer_before = _number(metrics, "cer_before")
    cer_after = _number(metrics, "cer_after")
    cer_increase = max(0.0, cer_after - cer_before)
    if cer_increase > config.maximum_cer_increase:
        return -1.0
    speaker = _number(metrics, "speaker_similarity", 1.0)
    speaker_term = max(-1.0, min(1.0, (speaker - config.speaker_floor) / 0.5))
    acoustic = (
        0.35 * _bounded_tanh(_number(metrics, "si_sdr_improvement"), config.si_sdr_scale)
        + 0.15 * _bounded_tanh(_number(metrics, "snr_improvement"), config.snr_scale)
        + 0.15 * _bounded_tanh(_number(metrics, "pesq_improvement"), config.pesq_scale)
        + 0.10 * _bounded_tanh(_number(metrics, "stoi_improvement"), config.stoi_scale)
        + 0.15 * speaker_term
    )
    content_factor = math.exp(-config.content_penalty_rate * cer_increase)
    artifact_penalty = max(0.0, _number(metrics, "artifact_penalty"))
    reward = acoustic * content_factor - 0.10 * artifact_penalty
    return round(max(-1.0, min(1.0, reward)), 6)


def select_preference_pair(
    candidates: list[dict[str, Any]], *, min_gap: float = 0.05
) -> dict[str, Any] | None:
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")
    if len(candidates) < 2:
        return None
    ordered = sorted(candidates, key=lambda row: _number(row, "reward"))
    rejected, chosen = ordered[0], ordered[-1]
    gap = _number(chosen, "reward") - _number(rejected, "reward")
    if gap < min_gap:
        return None
    return {
        "chosen": chosen,
        "rejected": rejected,
        "reward_gap": gap,
        "selection": "largest_verified_reward_gap",
    }
