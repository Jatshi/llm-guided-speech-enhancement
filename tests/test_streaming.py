from __future__ import annotations

import json

import numpy as np

from lse_v2.streaming import StreamingEnhancer


def test_streaming_enhancer_emits_finite_chunks_and_flushes_tail() -> None:
    prescription = json.dumps(
        {
            "diagnosis": {},
            "actions": [{"type": "gain", "gain_db": -3.0}],
            "rationale": "prevent clipping",
            "confidence": 0.8,
        }
    )
    stream = StreamingEnhancer(
        prescription,
        sample_rate=16000,
        window_samples=1600,
        hop_samples=800,
    )
    signal = np.ones(3200, dtype=np.float32) * 0.5

    chunks = stream.push(signal[:1200]) + stream.push(signal[1200:])
    tail = stream.flush()
    output = np.concatenate([*chunks, tail])

    assert chunks
    assert np.all(np.isfinite(output))
    assert output.size > 0
    assert float(np.max(np.abs(output))) < 0.5


def test_streaming_enhancer_rejects_non_stream_safe_plan() -> None:
    import pytest

    prescription = json.dumps(
        {
            "diagnosis": {},
            "actions": [{"type": "dereverb", "reduction_db": 6}],
            "rationale": "x",
            "confidence": 0.8,
        }
    )

    with pytest.raises(ValueError, match="not stream-safe"):
        StreamingEnhancer(prescription)
