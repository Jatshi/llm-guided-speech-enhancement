"""Bounded-latency windowed DSP for stream-safe enhancement prescriptions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from .dsp import ProductionDSPExecutor, plan_from_prescription

FloatAudio = NDArray[np.float32]
STREAM_SAFE_ACTIONS = {
    "bandstop",
    "dc_remove",
    "equalizer",
    "eq",
    "gain",
    "highpass",
    "limiter",
    "lowpass",
    "notch",
}


class StreamingEnhancer:
    """Process overlapping windows and cross-fade boundaries.

    The class deliberately rejects global spectral subtraction and dereverberation:
    those algorithms need longer-term noise/reverb state and pretending they are
    sample-streaming would create audible discontinuities.
    """

    def __init__(
        self,
        prescription: str,
        *,
        sample_rate: int = 16_000,
        window_samples: int = 8_000,
        hop_samples: int = 4_000,
    ) -> None:
        if sample_rate < 8_000 or window_samples < 2 or not 0 < hop_samples < window_samples:
            raise ValueError("invalid streaming sample rate/window/hop")
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self.plan = plan_from_prescription(prescription)
        unsafe = [
            action.kind for action in self.plan.actions if action.kind not in STREAM_SAFE_ACTIONS
        ]
        if unsafe:
            raise ValueError(f"actions are not stream-safe: {unsafe}")
        self.executor = ProductionDSPExecutor()
        self._input = np.empty(0, dtype=np.float32)
        self._pending_overlap: FloatAudio | None = None

    def _process_window(self, window: FloatAudio) -> FloatAudio:
        processed = self.executor.execute(window, self.sample_rate, self.plan)
        head = processed[: self.hop_samples].copy()
        if self._pending_overlap is not None:
            overlap = min(head.size, self._pending_overlap.size)
            fade_in = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
            head[:overlap] = (
                self._pending_overlap[:overlap] * (1.0 - fade_in) + head[:overlap] * fade_in
            )
        self._pending_overlap = processed[self.hop_samples :].copy()
        return head

    def push(self, samples: Iterable[float] | FloatAudio) -> list[FloatAudio]:
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(chunk)):
            raise ValueError("stream chunk must contain finite samples")
        self._input = np.concatenate([self._input, chunk])
        output: list[FloatAudio] = []
        while self._input.size >= self.window_samples:
            output.append(self._process_window(self._input[: self.window_samples]))
            self._input = self._input[self.hop_samples :]
        return output

    def flush(self) -> FloatAudio:
        if self._input.size:
            valid = self._input.size
            padded = np.pad(self._input, (0, max(0, self.window_samples - valid)))
            final = self._process_window(
                np.asarray(padded[: self.window_samples], dtype=np.float32)
            )
            final = final[: min(valid, self.hop_samples)]
            remainder = max(0, valid - final.size)
            tail = (
                self._pending_overlap[:remainder]
                if self._pending_overlap is not None and remainder
                else np.empty(0, dtype=np.float32)
            )
            result = np.concatenate([final, tail]).astype(np.float32)
        elif self._pending_overlap is not None:
            result = self._pending_overlap.astype(np.float32)
        else:
            result = np.empty(0, dtype=np.float32)
        self._input = np.empty(0, dtype=np.float32)
        self._pending_overlap = None
        return result
