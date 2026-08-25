from __future__ import annotations

import json

import pytest

from lse_v2.native_predict import _chunks, _embedding_paths, _prediction, latency_summary


def test_latency_summary_reports_p50_and_p95_in_milliseconds() -> None:
    report = latency_summary([0.01, 0.02, 0.03, 0.04])

    assert report["samples"] == 4
    assert report["mean_ms"] == 25.0
    assert report["p50_ms"] == 25.0
    assert report["p95_ms"] > report["p50_ms"]


def test_chunks_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        list(_chunks([], 0))


def test_embedding_paths_rejects_duplicate_ids(tmp_path) -> None:
    index = tmp_path / "index.jsonl"
    rows = [
        {"sample_id": "same", "embedding_path": "/tmp/one.npz"},
        {"sample_id": "same", "embedding_path": "/tmp/two.npz"},
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _embedding_paths(index)


def test_prediction_keeps_invalid_raw_response_for_forensics() -> None:
    result = _prediction({"sample_id": "x"}, '{"bad": ?}', 0.1, "test")
    assert result["status"] == "failed"
    assert result["raw_response"] == '{"bad": ?}'
    assert result["prescription"] is None


def test_prediction_accepts_json_object() -> None:
    result = _prediction({"sample_id": "x"}, '{"confidence": 0.5}', 0.1, "test")
    assert result["status"] == "ok"
    assert json.loads(result["prescription"]) == {"confidence": 0.5}
