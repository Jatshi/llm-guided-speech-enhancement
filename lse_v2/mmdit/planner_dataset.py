"""Build leakage-free SFT records for the MM-DiT acoustic planner."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from lse_v2.contracts import SFT_SCHEMA, SYSTEM_PROMPT, validate_alignment_record
from lse_v2.io import read_jsonl, write_jsonl

from .contracts import validate_pair_record
from .planner_export import build_planner_prompt


def _sft_record(pair: dict[str, Any]) -> dict[str, Any]:
    validate_pair_record(pair, check_files=True)
    split = pair["split"]
    if split not in {"train", "validation"}:
        raise ValueError(f"planner SFT cannot consume split={split!r}")
    prompt = build_planner_prompt(pair["audio"]["noisy_path"], int(pair["audio"]["sample_rate"]))
    target = json.dumps(pair["oracle_prescription"], ensure_ascii=False, sort_keys=True)
    record = {
        "schema_version": SFT_SCHEMA,
        "sample_id": pair["sample_id"],
        "split": "train" if split == "train" else "eval",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "evidence": {
            "audio": pair["audio"],
            "oracle_labels_exposed_in_prompt": False,
        },
    }
    validate_alignment_record(record)
    return record


def build_planner_sft_dataset(pairs: Path, output_dir: Path, *, workers: int = 4) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    source = read_jsonl(pairs)
    selected = [row for row in source if row.get("split") in {"train", "validation"}]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_sft_record, selected))
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "eval"]
    if not train or not validation:
        raise ValueError("planner SFT requires non-empty train and validation splits")
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(eval_path, validation)

    def conditions(rows: list[dict[str, Any]]) -> dict[str, int]:
        counter = Counter(
            json.loads(row["messages"][-1]["content"])["diagnosis"]["noise_type"] for row in rows
        )
        return dict(sorted(counter.items()))

    report = {
        "schema_version": "lse.mmdit.planner_sft_manifest.v1",
        "source_pairs": len(source),
        "train_records": len(train),
        "validation_records": len(validation),
        "test_records_consumed": 0,
        "oracle_labels_exposed_in_prompt": False,
        "train_conditions": conditions(train),
        "validation_conditions": conditions(validation),
        "train_path": str(train_path.resolve()),
        "eval_path": str(eval_path.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    report = build_planner_sft_dataset(args.pairs, args.output_dir, workers=args.workers)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
