"""Offline reward evaluation and reward-component ablation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from .contracts import GRPO_SCHEMA, validate_alignment_record
from .io import read_jsonl, utc_now, write_json_atomic
from .rewards import DEFAULT_WEIGHTS, score_prescription


def load_predictions(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    predictions: dict[str, str] = {}
    for record in read_jsonl(path):
        sample_id = str(record.get("sample_id", ""))
        response = record.get("response")
        if not sample_id or not isinstance(response, str):
            raise ValueError("Predictions require sample_id and string response")
        if sample_id in predictions:
            raise ValueError(f"Duplicate prediction sample_id: {sample_id}")
        predictions[sample_id] = response
    return predictions


def evaluate_rewards(
    dataset_path: str | Path,
    *,
    predictions_path: str | Path | None = None,
    include_ablations: bool = True,
) -> dict[str, Any]:
    records = read_jsonl(dataset_path)
    predictions = load_predictions(predictions_path)
    prediction_mode = predictions_path is not None
    aggregates: dict[str, list[float]] = defaultdict(list)
    examples: list[dict[str, Any]] = []
    dataset_ids: set[str] = set()
    evaluated: list[tuple[dict[str, Any], str, str]] = []
    for record in records:
        validate_alignment_record(record)
        if record["schema_version"] != GRPO_SCHEMA:
            raise ValueError("Offline reward evaluation requires lse.grpo.v2 records")
        sample_id = str(record["sample_id"])
        if sample_id in dataset_ids:
            raise ValueError(f"Duplicate dataset sample_id: {sample_id}")
        dataset_ids.add(sample_id)
        response = predictions.get(record["sample_id"])
        source = "prediction"
        if response is None:
            if prediction_mode:
                continue
            response = record["reward_context"].get("expected_response")
            source = "reference"
        if not isinstance(response, str):
            raise ValueError(f"No response available for {record['sample_id']}")
        evaluated.append((record, response, source))
    unknown_prediction_ids = sorted(set(predictions).difference(dataset_ids))
    if unknown_prediction_ids:
        raise ValueError(
            "Predictions contain sample IDs absent from the dataset: "
            + ", ".join(unknown_prediction_ids[:10])
        )
    if prediction_mode and not evaluated:
        raise ValueError("Prediction file did not match any dataset sample IDs")
    for record, response, source in evaluated:
        breakdown = score_prescription(response, record["reward_context"])
        for key, value in breakdown.to_dict().items():
            if isinstance(value, int | float) and key != "valid_json":
                aggregates[key].append(float(value))
        if len(examples) < 20:
            examples.append(
                {
                    "sample_id": record["sample_id"],
                    "source": source,
                    "score": breakdown.to_dict(),
                }
            )
    metrics = {
        key: round(fmean(values), 6) if values else 0.0 for key, values in aggregates.items()
    }
    ablations: dict[str, float] = {}
    if include_ablations:
        for omitted in DEFAULT_WEIGHTS:
            weights = {key: (0.0 if key == omitted else 1.0) for key in DEFAULT_WEIGHTS}
            scores: list[float] = []
            for record, response, _ in evaluated:
                scores.append(score_prescription(response, record["reward_context"], weights).total)
            ablations[f"without_{omitted}"] = round(fmean(scores), 6) if scores else 0.0
    evaluated_samples = len(evaluated)
    missing_predictions = len(records) - evaluated_samples if prediction_mode else 0
    return {
        "schema_version": "lse.reward_report.v2",
        "created_at": utc_now(),
        "dataset": str(Path(dataset_path).resolve()),
        "predictions": str(Path(predictions_path).resolve()) if predictions_path else None,
        "num_samples": evaluated_samples,
        "dataset_samples": len(records),
        "prediction_coverage": (
            round(evaluated_samples / len(records), 6) if prediction_mode and records else None
        ),
        "missing_predictions_skipped": missing_predictions,
        "missing_predictions_filled_with_reference": len(records) if not prediction_mode else 0,
        "metrics": metrics,
        "reward_ablations": ablations,
        "examples": examples,
        "note": (
            "Reference scoring validates reward behavior only; it is not a model benchmark."
            if predictions_path is None
            else (
                "Metrics score only supplied model predictions; dataset rows without a "
                "prediction are excluded, never filled with references."
            )
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-ablations", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_rewards(
        args.dataset,
        predictions_path=args.predictions,
        include_ablations=not args.no_ablations,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Report written to {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
