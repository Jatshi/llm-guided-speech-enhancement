"""Validated, executable DSP actions for production speech-enhancement runs.

The language model proposes a prescription; this module owns the safety boundary.
Unknown actions, non-finite parameters, and speech-destructive filter settings fail
before touching a waveform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .rewards import parse_prescription

FloatAudio = NDArray[np.float32]

SUPPORTED_ACTIONS = {
    "bandstop",
    "bandwidth_extension",
    "dc_remove",
    "dereverb",
    "equalizer",
    "gain",
    "highpass",
    "limiter",
    "lowpass",
    "notch",
    "spectral_gate",
    "spectral_subtraction",
}
ALIASES = {"eq": "equalizer", "de_reverb": "dereverb", "wpe": "dereverb"}


def _finite(value: Any, name: str, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class DSPAction:
    kind: str
    reduction_db: float | None = None
    gain_db: float | None = None
    low_hz: float | None = None
    high_hz: float | None = None
    q: float | None = None
    gate_quantile: float | None = None
    peak: float | None = None

    def __post_init__(self) -> None:
        canonical = ALIASES.get(self.kind, self.kind)
        object.__setattr__(self, "kind", canonical)
        if canonical not in SUPPORTED_ACTIONS:
            raise ValueError(f"unsupported DSP action: {canonical}")
        for name in (
            "reduction_db",
            "gain_db",
            "low_hz",
            "high_hz",
            "q",
            "gate_quantile",
            "peak",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.reduction_db is not None and not 0 <= self.reduction_db <= 24:
            raise ValueError("reduction_db must be in [0, 24]")
        if self.gain_db is not None and not -24 <= self.gain_db <= 6:
            raise ValueError("gain_db must be in [-24, 6]")
        if self.q is not None and not 0.1 <= self.q <= 30:
            raise ValueError("q must be in [0.1, 30]")
        if self.gate_quantile is not None and not 0 <= self.gate_quantile <= 0.95:
            raise ValueError("gate_quantile must be in [0, 0.95]")
        if self.peak is not None and not 0.5 <= self.peak <= 0.99:
            raise ValueError("peak must be in [0.5, 0.99]")
        for name in ("low_hz", "high_hz"):
            value = getattr(self, name)
            if value is not None and not 0 < value <= 48_000:
                raise ValueError(f"{name} must be in (0, 48000]")
        if self.low_hz is not None and self.high_hz is not None and self.low_hz >= self.high_hz:
            raise ValueError("low_hz must be lower than high_hz")
        if canonical == "highpass" and (self.low_hz is None or self.low_hz > 300):
            raise ValueError("highpass requires low_hz <= 300 to preserve speech")
        if canonical == "lowpass" and (self.high_hz is None or self.high_hz < 2_000):
            raise ValueError("lowpass requires high_hz >= 2000 to preserve speech")
        if canonical in {"notch", "bandstop", "equalizer"}:
            if self.low_hz is None or self.high_hz is None:
                raise ValueError(f"{canonical} requires low_hz and high_hz")


@dataclass(frozen=True, slots=True)
class DSPPlan:
    actions: list[DSPAction]

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("DSP plan requires at least one action")
        if len(self.actions) > 8:
            raise ValueError("DSP plan exceeds the eight-action safety limit")


def plan_from_prescription(text: str) -> DSPPlan:
    payload = parse_prescription(text)
    if payload is None:
        raise ValueError("prescription must be a JSON object")
    required = {"diagnosis", "actions", "rationale", "confidence"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"prescription missing keys: {sorted(missing)}")
    confidence = _finite(payload.get("confidence"), "confidence")
    if confidence is None or not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("actions must be a non-empty list")
    actions: list[DSPAction] = []
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, dict):
            raise ValueError(f"action {index} must be an object")
        kind = raw.get("type")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"action {index} requires type")
        allowed = {
            "type",
            "reduction_db",
            "gain_db",
            "low_hz",
            "high_hz",
            "q",
            "gate_quantile",
            "peak",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"action {index} contains unknown parameters: {unknown}")
        actions.append(
            DSPAction(
                kind=kind,
                reduction_db=raw.get("reduction_db"),
                gain_db=raw.get("gain_db"),
                low_hz=raw.get("low_hz"),
                high_hz=raw.get("high_hz"),
                q=raw.get("q"),
                gate_quantile=raw.get("gate_quantile"),
                peak=raw.get("peak"),
            )
        )
    return DSPPlan(actions)


class ProductionDSPExecutor:
    """Execute the checked-in prescription vocabulary with bounded SciPy DSP."""

    def execute(self, audio: FloatAudio, sample_rate: int, plan: DSPPlan) -> FloatAudio:
        if sample_rate < 8_000 or sample_rate > 192_000:
            raise ValueError("sample_rate must be in [8000, 192000]")
        output = np.asarray(audio, dtype=np.float32).copy()
        if output.ndim != 1 or output.size == 0 or not np.all(np.isfinite(output)):
            raise ValueError("audio must be a finite non-empty mono waveform")
        for action in plan.actions:
            self._validate_nyquist(action, sample_rate)
            output = self._execute_action(output, sample_rate, action)
            if output.shape != audio.shape or not np.all(np.isfinite(output)):
                raise RuntimeError(f"DSP action {action.kind} produced an invalid waveform")
            output = np.clip(output, -1.0, 1.0).astype(np.float32, copy=False)
        return output

    @staticmethod
    def _validate_nyquist(action: DSPAction, sample_rate: int) -> None:
        nyquist = sample_rate / 2
        for name in ("low_hz", "high_hz"):
            value = getattr(action, name)
            if value is not None and value >= nyquist:
                raise ValueError(f"{name} must be below Nyquist ({nyquist:g} Hz)")

    def _execute_action(self, audio: FloatAudio, sample_rate: int, action: DSPAction) -> FloatAudio:
        if action.kind == "dc_remove":
            return (audio - float(np.mean(audio))).astype(np.float32)
        if action.kind == "gain":
            gain = 10 ** ((action.gain_db or 0.0) / 20)
            return (audio * gain).astype(np.float32)
        if action.kind == "limiter":
            peak = action.peak or 0.95
            return peak * np.tanh(audio * 1.5).astype(np.float32) / np.tanh(1.5)
        if action.kind in {"highpass", "lowpass"}:
            cutoff = action.low_hz if action.kind == "highpass" else action.high_hz
            return self._butter(audio, sample_rate, action.kind, float(cutoff))
        if action.kind in {"notch", "bandstop"}:
            return self._band_reject(audio, sample_rate, action)
        if action.kind == "equalizer":
            center = (float(action.low_hz) + float(action.high_hz)) / 2
            q = action.q or center / max(float(action.high_hz) - float(action.low_hz), 1.0)
            return self._peaking_eq(audio, sample_rate, center, q, action.gain_db or 0.0)
        if action.kind == "bandwidth_extension":
            # Conservative high-shelf restoration; it does not hallucinate missing harmonics.
            low = action.low_hz or min(3_400.0, sample_rate * 0.30)
            high = action.high_hz or min(7_000.0, sample_rate * 0.45)
            center = (low + high) / 2
            return self._peaking_eq(audio, sample_rate, center, 0.7, action.gain_db or 2.0)
        if action.kind == "spectral_subtraction":
            return self._spectral_subtraction(audio, sample_rate, action)
        if action.kind == "spectral_gate":
            return self._spectral_gate(audio, sample_rate, action.gate_quantile or 0.25)
        if action.kind == "dereverb":
            return self._dereverb(audio, sample_rate, action.reduction_db or 4.0)
        raise AssertionError(f"unhandled supported action: {action.kind}")

    @staticmethod
    def _safe_filter(sos: np.ndarray, audio: FloatAudio) -> FloatAudio:
        from scipy.signal import sosfilt, sosfiltfilt

        try:
            return np.asarray(sosfiltfilt(sos, audio), dtype=np.float32)
        except ValueError:
            return np.asarray(sosfilt(sos, audio), dtype=np.float32)

    def _butter(self, audio: FloatAudio, sample_rate: int, kind: str, cutoff: float) -> FloatAudio:
        from scipy.signal import butter

        sos = butter(4, cutoff, btype=kind, fs=sample_rate, output="sos")
        return self._safe_filter(sos, audio)

    def _band_reject(self, audio: FloatAudio, sample_rate: int, action: DSPAction) -> FloatAudio:
        from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

        low, high = float(action.low_hz), float(action.high_hz)
        center = (low + high) / 2
        if action.kind == "notch" or high - low < center * 0.35:
            quality = action.q or max(0.1, center / max(high - low, 1.0))
            b, a = iirnotch(center, quality, fs=sample_rate)
            sos = tf2sos(b, a)
        else:
            sos = butter(4, [low, high], btype="bandstop", fs=sample_rate, output="sos")
        try:
            rejected = sosfiltfilt(sos, audio)
        except ValueError:
            rejected = self._safe_filter(sos, audio)
        mix = 1.0 - 10 ** (-(action.reduction_db or 12.0) / 20)
        return np.asarray((1 - mix) * audio + mix * rejected, dtype=np.float32)

    @staticmethod
    def _peaking_eq(
        audio: FloatAudio, sample_rate: int, center_hz: float, q: float, gain_db: float
    ) -> FloatAudio:
        from scipy.signal import filtfilt, lfilter

        amplitude = 10 ** (gain_db / 40)
        omega = 2 * np.pi * center_hz / sample_rate
        alpha = np.sin(omega) / (2 * max(q, 0.1))
        cosine = np.cos(omega)
        b = np.array([1 + alpha * amplitude, -2 * cosine, 1 - alpha * amplitude])
        a = np.array([1 + alpha / amplitude, -2 * cosine, 1 - alpha / amplitude])
        b, a = b / a[0], a / a[0]
        try:
            result = filtfilt(b, a, audio)
        except ValueError:
            result = lfilter(b, a, audio)
        return np.asarray(result, dtype=np.float32)

    @staticmethod
    def _stft(audio: FloatAudio, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy.signal import stft

        nperseg = min(512, max(64, 2 ** int(np.log2(max(audio.size // 8, 64)))))
        frequencies, times, spectrum = stft(
            audio,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=nperseg * 3 // 4,
            boundary="zeros",
            padded=True,
        )
        return frequencies, times, spectrum

    @staticmethod
    def _istft(spectrum: np.ndarray, sample_rate: int, length: int) -> FloatAudio:
        from scipy.signal import istft

        nperseg = (spectrum.shape[0] - 1) * 2
        # This must mirror _stft's 75% overlap. SciPy's 50% default changes the
        # time grid and causes severe waveform corruption even when the spectrum
        # is passed through unchanged.
        _, result = istft(
            spectrum,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=nperseg * 3 // 4,
        )
        if result.size < length:
            result = np.pad(result, (0, length - result.size))
        return np.asarray(result[:length], dtype=np.float32)

    def _spectral_subtraction(
        self, audio: FloatAudio, sample_rate: int, action: DSPAction
    ) -> FloatAudio:
        frequencies, _times, spectrum = self._stft(audio, sample_rate)
        magnitude, phase = np.abs(spectrum), np.exp(1j * np.angle(spectrum))
        noise = np.quantile(magnitude, 0.20, axis=1, keepdims=True)
        alpha = min(2.5, max(0.2, (action.reduction_db or 6.0) / 8.0))
        cleaned = np.maximum(magnitude - alpha * noise, 0.05 * magnitude)
        low, high = action.low_hz or 0.0, action.high_hz or sample_rate / 2 - 1
        selected = (frequencies >= low) & (frequencies <= high)
        output_magnitude = magnitude.copy()
        output_magnitude[selected] = cleaned[selected]
        return self._istft(output_magnitude * phase, sample_rate, audio.size)

    def _dereverb(self, audio: FloatAudio, sample_rate: int, reduction_db: float) -> FloatAudio:
        _frequencies, _times, spectrum = self._stft(audio, sample_rate)
        magnitude, phase = np.abs(spectrum), np.exp(1j * np.angle(spectrum))
        late = np.zeros_like(magnitude)
        decay = 0.82
        for frame in range(1, magnitude.shape[1]):
            late[:, frame] = decay * late[:, frame - 1] + (1 - decay) * magnitude[:, frame - 1]
        strength = min(0.75, reduction_db / 24)
        cleaned = np.maximum(magnitude - strength * late, 0.10 * magnitude)
        return self._istft(cleaned * phase, sample_rate, audio.size)

    def _spectral_gate(self, audio: FloatAudio, sample_rate: int, quantile: float) -> FloatAudio:
        _frequencies, _times, spectrum = self._stft(audio, sample_rate)
        magnitude = np.abs(spectrum)
        threshold = np.quantile(magnitude, quantile, axis=1, keepdims=True)
        mask = np.clip((magnitude - threshold) / (threshold + 1e-8), 0.05, 1.0)
        return self._istft(spectrum * mask, sample_rate, audio.size)
