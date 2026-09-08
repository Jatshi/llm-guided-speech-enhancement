"""Compare recovered GRPO predictions with the verified SFT baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .io import read_jsonl, utc_now, write_json_atomic
from .rewards import score_prescription


def _prediction_metrics(
    manifest_by_id: dict[str, dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(predictions) != len({str(row.get("sample_id")) for row in predictions}):
        raise ValueError("prediction sample IDs must be unique")
    if set(manifest_by_id) != {str(row.get("sample_id")) for row in predictions}:
        raise ValueError("predictions must cover the exact manifest sample set")
    scores = []
    valid = 0
    placeholders = 0
    for prediction in predictions:
        sample_id = str(prediction["sample_id"])
        text = str(prediction.get("raw_response") or prediction.get("prescription") or "")
        breakdown = score_prescription(text, manifest_by_id[sample_id]["reward_context"])
        scores.append(breakdown.total)
        valid += int(breakdown.valid_json)
        placeholders += int("?" in text)
    count = len(predictions)
    return {
        "records": count,
        "valid_json_rate": valid / count,
        "placeholder_rate": placeholders / count,
        "mean_reward": statistics.fmean(scores),
    }


def evaluate_recovery_predictions(
    manifest: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    candidate_predictions: list[dict[str, Any]],
    *,
    min_valid_json_rate: float = 0.99,
    max_reward_regression: float = 0.005,
) -> dict[str, Any]:
    if not manifest:
        raise ValueError("manifest must be non-empty")
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("manifest sample IDs must be unique")
    baseline = _prediction_metrics(manifest_by_id, baseline_predictions)
    candidate = _prediction_metrics(manifest_by_id, candidate_predictions)
    failed: list[str] = []
    if candidate["valid_json_rate"] < min_valid_json_rate:
        failed.append("valid_json_rate")
    if candidate["placeholder_rate"] > 0:
        failed.append("placeholder_rate")
    if candidate["mean_reward"] < baseline["mean_reward"] - max_reward_regression:
        failed.append("mean_reward_regression")
    return {
        "schema_version": "lse.grpo_recovery_acceptance.v1",
        "status": "failed" if failed else "passed",
        "created_at": utc_now(),
        "baseline": baseline,
        "candidate": candidate,
        "thresholds": {
            "min_valid_json_rate": min_valid_json_rate,
            "max_reward_regression": max_reward_regression,
            "max_placeholder_rate": 0.0,
        },
        "failed_checks": failed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval", "test"), default="test")
    parser.add_argument("--min-valid-json-rate", type=float, default=0.99)
    parser.add_argument("--max-reward-regression", type=float, default=0.005)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = [row for row in read_jsonl(args.manifest) if row.get("split") == args.split]
    report = evaluate_recovery_predictions(
        manifest,
        read_jsonl(args.baseline_predictions),
        read_jsonl(args.candidate_predictions),
        min_valid_json_rate=args.min_valid_json_rate,
        max_reward_regression=args.max_reward_regression,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
