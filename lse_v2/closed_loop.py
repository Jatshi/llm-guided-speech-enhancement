"""Execute, re-measure, and safely accept or roll back enhancement plans."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatAudio: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class EnhancementAction:
    kind: Literal["gain", "dc_remove", "spectral_gate"]
    value: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("action value must be finite")
        if self.kind == "gain" and not 0.0 <= self.value <= 4.0:
            raise ValueError("gain must be in [0, 4]")
        if self.kind == "spectral_gate" and not 0.0 <= self.value <= 1.0:
            raise ValueError("spectral gate quantile must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class EnhancementPlan:
    actions: list[EnhancementAction]

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("enhancement plan requires at least one action")
        if len(self.actions) > 8:
            raise ValueError("enhancement plan exceeds the eight-action safety limit")


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    si_sdr: float | None = None
    stoi: float | None = None
    pesq: float | None = None
    dnsmos: float | None = None
    wer: float | None = None
    speaker_similarity: float | None = None

    def utility(self) -> float:
        """Combine available metrics after normalizing their direction and range."""

        terms: list[tuple[float, float]] = []
        if self.si_sdr is not None:
            terms.append((0.25, 1.0 / (1.0 + math.exp(-self.si_sdr / 6.0))))
        if self.stoi is not None:
            terms.append((0.20, min(1.0, max(0.0, self.stoi))))
        if self.pesq is not None:
            terms.append((0.15, min(1.0, max(0.0, (self.pesq - 1.0) / 3.5))))
        if self.dnsmos is not None:
            terms.append((0.15, min(1.0, max(0.0, (self.dnsmos - 1.0) / 4.0))))
        if self.wer is not None:
            terms.append((0.15, 1.0 - min(1.0, max(0.0, self.wer))))
        if self.speaker_similarity is not None:
            terms.append((0.10, min(1.0, max(0.0, self.speaker_similarity))))
        if not terms:
            raise ValueError("at least one objective metric is required")
        weight = sum(item[0] for item in terms)
        return sum(item_weight * value for item_weight, value in terms) / weight


class EnhancementExecutor(Protocol):
    def execute(self, audio: FloatAudio, sample_rate: int, plan: EnhancementPlan) -> FloatAudio: ...


class NumpyDSPExecutor:
    """Dependency-light DSP executor used by tests and the local CPU baseline."""

    def execute(self, audio: FloatAudio, sample_rate: int, plan: EnhancementPlan) -> FloatAudio:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        output = np.asarray(audio, dtype=np.float32).copy()
        if output.ndim != 1 or output.size == 0:
            raise ValueError("audio must be a non-empty mono waveform")
        for action in plan.actions:
            if action.kind == "gain":
                output = np.clip(output * action.value, -1.0, 1.0).astype(np.float32)
            elif action.kind == "dc_remove":
                output = (output - np.mean(output)).astype(np.float32)
            elif action.kind == "spectral_gate":
                spectrum = np.fft.rfft(output)
                magnitude = np.abs(spectrum)
                threshold = float(np.quantile(magnitude, action.value))
                spectrum[magnitude < threshold] = 0
                output = np.fft.irfft(spectrum, n=output.size).astype(np.float32)
                output = np.clip(output, -1.0, 1.0)
        return output


@dataclass(frozen=True, slots=True)
class ClosedLoopResult:
    audio: FloatAudio
    candidate_audio: FloatAudio
    before: MetricSnapshot
    after: MetricSnapshot
    utility_gain: float
    decision: Literal["accept", "rollback"]
    attempts: list[EnhancementAttempt]


@dataclass(frozen=True, slots=True)
class EnhancementAttempt:
    plan: EnhancementPlan
    metrics: MetricSnapshot
    utility_gain: float
    decision: Literal["accept", "revise", "rollback"]


class ClosedLoopEnhancer:
    def __init__(
        self,
        *,
        executor: EnhancementExecutor,
        measure: Callable[[FloatAudio, int], MetricSnapshot],
        reviser: Callable[
            [EnhancementPlan, MetricSnapshot, MetricSnapshot], EnhancementPlan | None
        ]
        | None = None,
        max_revisions: int = 0,
        min_utility_gain: float = 0.0,
    ) -> None:
        if max_revisions < 0:
            raise ValueError("max_revisions cannot be negative")
        self.executor = executor
        self.measure = measure
        self.reviser = reviser
        self.max_revisions = max_revisions
        self.min_utility_gain = min_utility_gain

    def run(
        self,
        audio: FloatAudio,
        *,
        sample_rate: int,
        plan: EnhancementPlan,
    ) -> ClosedLoopResult:
        source = np.asarray(audio, dtype=np.float32).copy()
        before = self.measure(source, sample_rate)
        current_plan = plan
        attempts: list[EnhancementAttempt] = []
        candidate = source
        after = before
        gain = 0.0
        accepted = False
        for attempt_index in range(self.max_revisions + 1):
            candidate = self.executor.execute(source, sample_rate, current_plan)
            after = self.measure(candidate, sample_rate)
            gain = after.utility() - before.utility()
            if gain >= self.min_utility_gain:
                accepted = True
                attempts.append(
                    EnhancementAttempt(
                        plan=current_plan,
                        metrics=after,
                        utility_gain=gain,
                        decision="accept",
                    )
                )
                break
            revised = None
            if self.reviser is not None and attempt_index < self.max_revisions:
                revised = self.reviser(current_plan, before, after)
            if revised is None:
                attempts.append(
                    EnhancementAttempt(
                        plan=current_plan,
                        metrics=after,
                        utility_gain=gain,
                        decision="rollback",
                    )
                )
                break
            attempts.append(
                EnhancementAttempt(
                    plan=current_plan,
                    metrics=after,
                    utility_gain=gain,
                    decision="revise",
                )
            )
            current_plan = revised
        return ClosedLoopResult(
            audio=candidate.copy() if accepted else source,
            candidate_audio=candidate,
            before=before,
            after=after,
            utility_gain=gain,
            decision="accept" if accepted else "rollback",
            attempts=attempts,
        )
