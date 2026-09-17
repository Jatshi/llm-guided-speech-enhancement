"""Attach held-out planner JSON predictions to paired MM-DiT records by sample id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lse_v2.io import read_jsonl, write_jsonl

from .contracts import validate_pair_record, validate_prescription


def _prediction(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("predicted_prescription") or row.get("prediction") or row.get("raw_response")
    if isinstance(value, str):
        value = json.loads(value)
    validate_prescription(value, f"{row.get('sample_id')}.prediction")
    return value


def attach_predictions(pairs: Path, predictions: Path, output: Path) -> dict[str, Any]:
    pair_rows = read_jsonl(pairs)
    prediction_rows = read_jsonl(predictions)
    index = {}
    invalid = 0
    for row in prediction_rows:
        try:
            index[row["sample_id"]] = _prediction(row)
        except (ValueError, TypeError, json.JSONDecodeError):
            invalid += 1
    attached = 0
    result = []
    for row in pair_rows:
        copy = json.loads(json.dumps(row))
        if copy["sample_id"] in index:
            copy["predicted_prescription"] = index[copy["sample_id"]]
            attached += 1
        validate_pair_record(copy)
        result.append(copy)
    write_jsonl(output, result)
    report = {
        "pairs": len(pair_rows),
        "prediction_rows": len(prediction_rows),
        "attached": attached,
        "missing": len(pair_rows) - attached,
        "unused_predictions": len(set(index) - {row["sample_id"] for row in pair_rows}),
        "invalid_predictions": invalid,
    }
    output.with_suffix(".prediction_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(attach_predictions(args.pairs, args.predictions, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
