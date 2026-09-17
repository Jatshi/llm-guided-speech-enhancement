"""Fuse a supervised acoustic router with auditable LLM planner outputs."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from lse_v2.contracts import extract_audio_features
from lse_v2.io import read_jsonl, write_jsonl

from .contracts import validate_prescription

FEATURE_KEYS = (
    "rms",
    "spectral_flatness",
    "spectral_centroid_hz",
    "zero_crossing_rate",
    "duration_seconds",
)


def _features(record: dict[str, Any]) -> list[float]:
    measured = extract_audio_features(
        record["audio"]["noisy_path"], int(record["audio"]["sample_rate"])
    )
    return [float(measured[key]) for key in FEATURE_KEYS]


def fuse_router_diagnosis(
    llm_prediction: dict[str, Any], router_label: str, router_confidence: float
) -> dict[str, Any]:
    """Use the discriminative head for diagnosis while retaining LLM actions."""

    result = copy.deepcopy(llm_prediction)
    result["diagnosis"] = {
        "noise_type": router_label,
        "reverb": router_label == "reverb",
        "band_limited": router_label == "telephone",
    }
    result["confidence"] = float(max(0.0, min(1.0, router_confidence)))
    validate_prescription(result, "hybrid_prediction")
    return result


def fit_router_and_fuse(
    pairs: Path,
    llm_predictions: Path,
    output: Path,
    model_output: Path,
    *,
    workers: int = 8,
    supported_diagnoses: set[str] | None = None,
) -> dict[str, Any]:
    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    if workers < 1:
        raise ValueError("workers must be positive")
    records = read_jsonl(pairs)
    train = [row for row in records if row.get("split") == "train"]
    validation = [row for row in records if row.get("split") == "validation"]
    test = [row for row in records if row.get("split") == "test"]
    if not train or not validation or not test:
        raise ValueError("router requires train, validation, and test splits")
    selected = train + validation + test
    with ThreadPoolExecutor(max_workers=workers) as executor:
        feature_rows = list(executor.map(_features, selected))
    train_end = len(train)
    validation_end = train_end + len(validation)
    x_train = np.asarray(feature_rows[:train_end], dtype=np.float64)
    x_validation = np.asarray(feature_rows[train_end:validation_end], dtype=np.float64)
    x_test = np.asarray(feature_rows[validation_end:], dtype=np.float64)

    def labels(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([row["oracle_prescription"]["diagnosis"]["noise_type"] for row in rows])

    y_train = labels(train)
    y_validation = labels(validation)
    y_test = labels(test)
    model = ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    validation_prediction = model.predict(x_validation)
    test_prediction = model.predict(x_test)
    test_probabilities = model.predict_proba(x_test)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_keys": FEATURE_KEYS, "schema_version": 1},
        model_output,
    )

    llm_index = {row["sample_id"]: row for row in read_jsonl(llm_predictions)}
    output_rows = []
    safety_counts: Counter[str] = Counter()
    for index, (record, router_label) in enumerate(zip(test, test_prediction, strict=True)):
        llm_row = llm_index.get(record["sample_id"])
        if llm_row is None or not isinstance(llm_row.get("prediction"), dict):
            raise ValueError(f"missing valid LLM prediction for {record['sample_id']}")
        probabilities = {
            str(label): float(value)
            for label, value in zip(model.classes_, test_probabilities[index], strict=True)
        }
        confidence = probabilities[str(router_label)]
        hybrid = fuse_router_diagnosis(llm_row["prediction"], str(router_label), confidence)
        supported = supported_diagnoses is None or str(router_label) in supported_diagnoses
        safety_decision = "execute_supported_action"
        if not supported:
            hybrid["confidence"] = 0.0
            safety_decision = "abstain_unsupported_action"
        safety_counts[safety_decision] += 1
        output_rows.append(
            {
                "sample_id": record["sample_id"],
                "prediction": hybrid,
                "llm_prediction": llm_row["prediction"],
                "raw_response": llm_row.get("raw_response"),
                "parse_status": llm_row.get("parse_status"),
                "router_prediction": str(router_label),
                "router_confidence": confidence,
                "router_probabilities": probabilities,
                "arbitration": "router_diagnosis+llm_actions",
                "safety_decision": safety_decision,
                "used_oracle_labels": False,
            }
        )
    write_jsonl(output, output_rows)
    ordered_labels = sorted(str(item) for item in model.classes_)
    matrix = confusion_matrix(y_test, test_prediction, labels=ordered_labels)
    report = {
        "schema_version": "lse.mmdit.hybrid_planner_manifest.v1",
        "train_records": len(train),
        "validation_records": len(validation),
        "test_records": len(test),
        "test_labels_used_for_fit": False,
        "feature_keys": list(FEATURE_KEYS),
        "validation_accuracy": float(accuracy_score(y_validation, validation_prediction)),
        "test_accuracy": float(accuracy_score(y_test, test_prediction)),
        "test_prediction_counts": dict(sorted(Counter(test_prediction).items())),
        "supported_diagnoses": (
            sorted(supported_diagnoses) if supported_diagnoses is not None else None
        ),
        "safety_decision_counts": dict(sorted(safety_counts.items())),
        "confusion_labels": ordered_labels,
        "test_confusion_matrix": matrix.tolist(),
        "llm_predictions": str(llm_predictions.resolve()),
        "router_model": str(model_output.resolve()),
        "output": str(output.resolve()),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--llm-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--supported-diagnosis",
        action="append",
        dest="supported_diagnoses",
        help="diagnosis supported by the current executable action family; repeatable",
    )
    args = parser.parse_args(argv)
    report = fit_router_and_fuse(
        args.pairs,
        args.llm_predictions,
        args.output,
        args.model_output,
        workers=args.workers,
        supported_diagnoses=(set(args.supported_diagnoses) if args.supported_diagnoses else None),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
