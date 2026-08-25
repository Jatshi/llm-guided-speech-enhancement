"""Out-of-domain waveform benchmark with dataset/noise/SNR slice reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .end_to_end import EnhancementRequest, run_enhancement
from .io import read_jsonl, utc_now, write_json_atomic, write_jsonl


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted({name for row in rows for name in row.get("metric_deltas", {})})
    metrics: dict[str, Any] = {}
    for name in metric_names:
        values = [
            row["metric_deltas"][name] for row in rows if name in row.get("metric_deltas", {})
        ]
        metrics[name] = {
            "available_samples": len(values),
            "unavailable_samples": len(rows) - len(values),
            "mean_gain": mean(values) if values else None,
        }
    return {
        "samples": len(rows),
        "accept_rate": sum(row["decision"] == "accept" for row in rows) / len(rows)
        if rows
        else None,
        "rollback_rate": sum(row["decision"] == "rollback" for row in rows) / len(rows)
        if rows
        else None,
        "metrics": metrics,
    }


def aggregate_benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"overall": _aggregate(rows)}
    for output_name, field in (
        ("by_dataset", "dataset"),
        ("by_noise_type", "noise_type"),
        ("by_snr_bin", "snr_bin"),
        ("by_language", "language"),
        ("by_device", "device"),
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field) or "unknown")].append(row)
        report[output_name] = {name: _aggregate(group) for name, group in sorted(groups.items())}
    return report


def _snr_bin(value: Any) -> str:
    if not isinstance(value, int | float):
        return "unknown"
    if value < 5:
        return "very_low_<5"
    if value < 10:
        return "low_5_10"
    if value < 20:
        return "mid_10_20"
    return "high_>=20"


def _failed_prediction_result(
    row: dict[str, Any], prediction: dict[str, Any]
) -> dict[str, Any]:
    """Represent an unusable policy output as an explicit safety rollback."""

    provenance = row.get("provenance", {})
    acoustics = row.get("acoustics", row.get("reward_context", {}))
    return {
        "sample_id": row["sample_id"],
        "dataset": provenance.get("dataset", "unknown"),
        "language": provenance.get("language", "unknown"),
        "device": provenance.get("device", "unknown"),
        "noise_type": acoustics.get("noise_type", "unknown"),
        "snr_bin": _snr_bin(acoustics.get("snr_db")),
        "decision": "rollback",
        "metric_deltas": {},
        "audit_report": None,
        "prediction_status": prediction.get("status", "failed"),
        "prediction_error": prediction.get("error") or "missing prescription text",
    }


def run_generalization_benchmark(
    manifest_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    minimum_si_sdr_gain_db: float = 0.0,
    split: str = "test",
) -> dict[str, Any]:
    manifest = [row for row in read_jsonl(manifest_path) if row.get("split") == split]
    if not manifest:
        raise ValueError(f"manifest has no {split} records")
    predictions = {row["sample_id"]: row for row in read_jsonl(predictions_path)}
    target = Path(output_dir).expanduser().resolve()
    results: list[dict[str, Any]] = []
    for row in manifest:
        sample_id = row["sample_id"]
        if sample_id not in predictions:
            raise ValueError(f"missing prediction for {sample_id}")
        audio = row.get("audio", {})
        acoustics = row.get("acoustics", row.get("reward_context", {}))
        audio_path = audio.get("noisy_path") or row.get("audio_path")
        clean_path = audio.get("clean_path") or row.get("clean_path")
        prediction = predictions[sample_id]
        prescription = prediction.get("prescription") or prediction.get("completion")
        if not isinstance(prescription, str):
            results.append(_failed_prediction_result(row, prediction))
            continue
        run = run_enhancement(
            EnhancementRequest(
                audio_path=Path(audio_path),
                clean_reference=Path(clean_path) if clean_path else None,
                prescription=prescription,
                output_dir=(
                    target / "samples" / hashlib.sha256(sample_id.encode()).hexdigest()[:24]
                ),
                minimum_si_sdr_gain_db=minimum_si_sdr_gain_db,
            )
        )
        provenance = row.get("provenance", {})
        results.append(
            {
                "sample_id": sample_id,
                "dataset": provenance.get("dataset", "unknown"),
                "language": provenance.get("language", "unknown"),
                "device": provenance.get("device", "unknown"),
                "noise_type": acoustics.get("noise_type", "unknown"),
                "snr_bin": _snr_bin(acoustics.get("snr_db")),
                "decision": run.decision,
                "metric_deltas": run.metrics.deltas,
                "audit_report": str(run.audit_report),
                "prediction_status": prediction.get("status", "ok"),
                "prediction_error": prediction.get("error"),
            }
        )
    write_jsonl(target / "sample_results.jsonl", results)
    report = {
        "schema_version": "lse.generalization_benchmark.v1",
        "created_at": utc_now(),
        "manifest": str(Path(manifest_path).resolve()),
        "predictions": str(Path(predictions_path).resolve()),
        "split": split,
        **aggregate_benchmark_rows(results),
    }
    write_json_atomic(target / "benchmark_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-si-sdr-gain-db", type=float, default=0.0)
    parser.add_argument("--split", choices=("train", "eval", "test"), default="test")
    args = parser.parse_args(argv)
    report = run_generalization_benchmark(
        args.manifest,
        args.predictions,
        args.output_dir,
        minimum_si_sdr_gain_db=args.minimum_si_sdr_gain_db,
        split=args.split,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
