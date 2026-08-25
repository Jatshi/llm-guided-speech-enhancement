from __future__ import annotations

from lse_v2.generalization import _failed_prediction_result, aggregate_benchmark_rows


def test_benchmark_aggregation_reports_slices_and_no_fake_metrics() -> None:
    rows = [
        {
            "sample_id": "a",
            "dataset": "libritts-demand",
            "noise_type": "cafe",
            "snr_bin": "low",
            "decision": "accept",
            "metric_deltas": {"si_sdr": 1.5},
        },
        {
            "sample_id": "b",
            "dataset": "dns-test",
            "noise_type": "babble",
            "snr_bin": "mid",
            "decision": "rollback",
            "metric_deltas": {},
        },
    ]

    report = aggregate_benchmark_rows(rows)

    assert report["overall"]["samples"] == 2
    assert report["overall"]["accept_rate"] == 0.5
    assert report["overall"]["metrics"]["si_sdr"]["available_samples"] == 1
    assert "dns-test" in report["by_dataset"]
    assert "unknown" in report["by_language"]
    assert "unknown" in report["by_device"]


def test_invalid_policy_output_is_counted_as_explicit_rollback() -> None:
    row = {
        "sample_id": "bad-json",
        "provenance": {"dataset": "held-out", "language": "en"},
        "reward_context": {"noise_type": "cafe", "snr_db": 7.0},
    }

    result = _failed_prediction_result(
        row,
        {"status": "failed", "error": "JSONDecodeError: invalid JSON"},
    )
    report = aggregate_benchmark_rows([result])

    assert result["decision"] == "rollback"
    assert result["metric_deltas"] == {}
    assert result["prediction_error"].startswith("JSONDecodeError")
    assert report["overall"]["rollback_rate"] == 1.0
