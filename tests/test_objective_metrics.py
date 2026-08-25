from __future__ import annotations

import numpy as np

from lse_v2.metrics import MetricValue, ObjectiveMetricSuite, si_sdr


def test_si_sdr_ranks_cleaner_waveform_higher() -> None:
    rng = np.random.default_rng(11)
    clean = np.sin(np.linspace(0, 40 * np.pi, 8000)).astype(np.float32)
    noisy = clean + 0.2 * rng.standard_normal(clean.size)
    less_noisy = clean + 0.05 * rng.standard_normal(clean.size)

    assert si_sdr(less_noisy, clean) > si_sdr(noisy, clean)


def test_metric_suite_marks_optional_backend_unavailable_instead_of_zero() -> None:
    clean = np.sin(np.linspace(0, 20 * np.pi, 16000)).astype(np.float32)
    suite = ObjectiveMetricSuite(optional_backends={})

    report = suite.measure(clean, 16000, clean_reference=clean)

    assert report.values["si_sdr"].available
    assert report.values["si_sdr"].value is not None
    assert report.values["stoi"] == MetricValue.unavailable("backend_not_configured")
    assert report.values["pesq"] == MetricValue.unavailable("backend_not_configured")
    assert report.values["dnsmos"] == MetricValue.unavailable("backend_not_configured")


def test_metric_delta_uses_only_metrics_available_on_both_sides() -> None:
    clean = np.sin(np.linspace(0, 20 * np.pi, 16000)).astype(np.float32)
    noisy = clean + 0.15 * np.random.default_rng(13).standard_normal(clean.size)
    suite = ObjectiveMetricSuite(optional_backends={})

    comparison = suite.compare(noisy, clean, 16000, clean_reference=clean)

    assert set(comparison.deltas) == {"si_sdr"}
    assert comparison.deltas["si_sdr"] > 0


def test_frame_comparison_reports_distribution_not_only_global_mean() -> None:
    time = np.arange(16000, dtype=np.float32) / 16000
    clean = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    noisy = clean + 0.1 * np.random.default_rng(2).standard_normal(clean.size)
    improved = clean + 0.03 * np.random.default_rng(3).standard_normal(clean.size)
    suite = ObjectiveMetricSuite()

    report = suite.compare_frames(
        noisy.astype(np.float32),
        improved.astype(np.float32),
        16000,
        clean_reference=clean,
        frame_seconds=0.25,
    )

    assert report["frames"] == 4
    assert report["si_sdr_gain_db"]["available_frames"] == 4
    assert report["si_sdr_gain_db"]["mean"] > 0
    assert report["si_sdr_gain_db"]["p10"] <= report["si_sdr_gain_db"]["p90"]
