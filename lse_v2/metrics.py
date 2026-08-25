"""Objective waveform metrics with explicit unavailable semantics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatAudio = NDArray[np.float32]
MetricBackend = Callable[[FloatAudio, int, FloatAudio | None], float]


@dataclass(frozen=True, slots=True)
class MetricValue:
    value: float | None
    available: bool
    reason: str | None = None

    @classmethod
    def measured(cls, value: float) -> MetricValue:
        if not math.isfinite(value):
            raise ValueError("metric value must be finite")
        return cls(float(value), True, None)

    @classmethod
    def unavailable(cls, reason: str) -> MetricValue:
        return cls(None, False, reason)

    def to_dict(self) -> dict[str, float | bool | str | None]:
        return {"value": self.value, "available": self.available, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class MetricReport:
    values: dict[str, MetricValue]

    def to_dict(self) -> dict[str, dict[str, float | bool | str | None]]:
        return {name: value.to_dict() for name, value in self.values.items()}


@dataclass(frozen=True, slots=True)
class MetricComparison:
    before: MetricReport
    after: MetricReport
    deltas: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "deltas": self.deltas,
        }


def _aligned(estimate: FloatAudio, reference: FloatAudio) -> tuple[np.ndarray, np.ndarray]:
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    length = min(estimate.size, reference.size)
    if length < 2:
        raise ValueError("metric waveforms must contain at least two aligned samples")
    estimate, reference = estimate[:length], reference[:length]
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(reference)):
        raise ValueError("metric waveforms must be finite")
    return estimate, reference


def si_sdr(estimate: FloatAudio, reference: FloatAudio, epsilon: float = 1e-8) -> float:
    """Scale-invariant SDR in dB using zero-mean, time-aligned waveforms."""

    estimate64, reference64 = _aligned(estimate, reference)
    estimate64 -= np.mean(estimate64)
    reference64 -= np.mean(reference64)
    # Explicit reductions avoid a Windows MKL ``ddot`` process abort observed
    # after PyTorch and SciPy were loaded in the same test process.
    reference_energy = float(np.sum(reference64 * reference64, dtype=np.float64))
    if reference_energy <= epsilon:
        raise ValueError("SI-SDR reference energy is zero")
    projection = float(np.sum(estimate64 * reference64, dtype=np.float64))
    target = projection / (reference_energy + epsilon) * reference64
    noise = estimate64 - target
    target_energy = float(np.sum(target * target, dtype=np.float64))
    noise_energy = float(np.sum(noise * noise, dtype=np.float64))
    ratio = (target_energy + epsilon) / (noise_energy + epsilon)
    return float(10 * np.log10(ratio))


def _stoi_backend(estimate: FloatAudio, sample_rate: int, reference: FloatAudio | None) -> float:
    if reference is None:
        raise ValueError("clean reference required")
    from pystoi import stoi  # type: ignore[import-untyped]

    estimate64, reference64 = _aligned(estimate, reference)
    return float(stoi(reference64, estimate64, sample_rate, extended=False))


def _pesq_backend(estimate: FloatAudio, sample_rate: int, reference: FloatAudio | None) -> float:
    if reference is None:
        raise ValueError("clean reference required")
    if sample_rate not in {8_000, 16_000}:
        raise ValueError("PESQ supports only 8 kHz or 16 kHz")
    from pesq import pesq  # type: ignore[import-untyped]

    estimate64, reference64 = _aligned(estimate, reference)
    mode = "wb" if sample_rate == 16_000 else "nb"
    return float(pesq(sample_rate, reference64, estimate64, mode))


class ObjectiveMetricSuite:
    def __init__(self, *, optional_backends: dict[str, MetricBackend] | None = None) -> None:
        self.optional_backends = dict(optional_backends or {})

    @classmethod
    def with_installed_backends(cls) -> ObjectiveMetricSuite:
        backends: dict[str, MetricBackend] = {}
        try:
            import pystoi  # noqa: F401

            backends["stoi"] = _stoi_backend
        except ImportError:
            pass
        try:
            import pesq  # noqa: F401

            backends["pesq"] = _pesq_backend
        except ImportError:
            pass
        return cls(optional_backends=backends)

    def measure(
        self,
        audio: FloatAudio,
        sample_rate: int,
        *,
        clean_reference: FloatAudio | None = None,
    ) -> MetricReport:
        values: dict[str, MetricValue] = {}
        if clean_reference is None:
            values["si_sdr"] = MetricValue.unavailable("clean_reference_required")
        else:
            try:
                values["si_sdr"] = MetricValue.measured(si_sdr(audio, clean_reference))
            except (ValueError, FloatingPointError) as exc:
                values["si_sdr"] = MetricValue.unavailable(f"{type(exc).__name__}: {exc}")
        for name in ("stoi", "pesq", "dnsmos", "wer", "speaker_similarity"):
            backend = self.optional_backends.get(name)
            if backend is None:
                values[name] = MetricValue.unavailable("backend_not_configured")
                continue
            try:
                values[name] = MetricValue.measured(
                    backend(np.asarray(audio, dtype=np.float32), sample_rate, clean_reference)
                )
            except Exception as exc:  # optional third-party metric boundary
                values[name] = MetricValue.unavailable(f"{type(exc).__name__}: {exc}")
        return MetricReport(values)

    def compare(
        self,
        before: FloatAudio,
        after: FloatAudio,
        sample_rate: int,
        *,
        clean_reference: FloatAudio | None = None,
    ) -> MetricComparison:
        before_report = self.measure(before, sample_rate, clean_reference=clean_reference)
        after_report = self.measure(after, sample_rate, clean_reference=clean_reference)
        deltas: dict[str, float] = {}
        lower_is_better = {"wer"}
        for name, before_value in before_report.values.items():
            after_value = after_report.values.get(name)
            if (
                before_value.available
                and after_value is not None
                and after_value.available
                and before_value.value is not None
                and after_value.value is not None
            ):
                delta = after_value.value - before_value.value
                deltas[name] = -delta if name in lower_is_better else delta
        return MetricComparison(before_report, after_report, deltas)

    def compare_frames(
        self,
        before: FloatAudio,
        after: FloatAudio,
        sample_rate: int,
        *,
        clean_reference: FloatAudio,
        frame_seconds: float = 0.5,
    ) -> dict[str, object]:
        """Report per-frame gain distributions while preserving unavailable counts."""

        frame_samples = round(frame_seconds * sample_rate)
        if frame_samples < 2:
            raise ValueError("frame_seconds is too short")
        length = min(before.size, after.size, clean_reference.size)
        frames = length // frame_samples
        if frames == 0:
            raise ValueError("waveforms are shorter than one metric frame")
        metric_names = ("si_sdr", "stoi", "pesq", "dnsmos", "wer", "speaker_similarity")
        gains: dict[str, list[float]] = {name: [] for name in metric_names}
        for index in range(frames):
            start, end = index * frame_samples, (index + 1) * frame_samples
            comparison = self.compare(
                before[start:end],
                after[start:end],
                sample_rate,
                clean_reference=clean_reference[start:end],
            )
            for name, value in comparison.deltas.items():
                gains[name].append(value)
        report: dict[str, object] = {"frames": frames, "frame_seconds": frame_seconds}
        suffix = {"si_sdr": "si_sdr_gain_db"}
        for name, values in gains.items():
            array = np.asarray(values, dtype=np.float64)
            report[suffix.get(name, f"{name}_gain")] = {
                "available_frames": len(values),
                "unavailable_frames": frames - len(values),
                "mean": float(np.mean(array)) if values else None,
                "p10": float(np.quantile(array, 0.10)) if values else None,
                "p50": float(np.quantile(array, 0.50)) if values else None,
                "p90": float(np.quantile(array, 0.90)) if values else None,
            }
        return report
