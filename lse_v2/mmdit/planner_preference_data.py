"""Build outcome-grounded Planner preference pairs from enhancement measurements."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from lse_v2.io import read_jsonl, write_jsonl

from .enhancement_reward import score_enhancement_outcome, select_preference_pair


def build_preference_rows(
    candidates: list[dict[str, Any]], *, min_gap: float = 0.05
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair best and worst prescriptions using measured downstream audio quality.

    Each candidate must contain ``sample_id``, ``prescription`` and ``metrics``.
    Optional prompt/context fields are copied to the pair so the output can be
    transformed into the exact DPO chat template used by a selected base model.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(candidates):
        sample_id = str(raw.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"candidate {index} requires sample_id")
        if not isinstance(raw.get("prescription"), dict):
            raise ValueError(f"candidate {index} requires prescription object")
        if not isinstance(raw.get("metrics"), dict):
            raise ValueError(f"candidate {index} requires metrics object")
        item = copy.deepcopy(raw)
        item["reward"] = score_enhancement_outcome(item["metrics"])
        grouped[sample_id].append(item)

    pairs: list[dict[str, Any]] = []
    skipped = 0
    for sample_id, rows in sorted(grouped.items()):
        selected = select_preference_pair(rows, min_gap=min_gap)
        if selected is None:
            skipped += 1
            continue
        chosen = selected["chosen"]
        rejected = selected["rejected"]
        pairs.append(
            {
                "schema_version": "lse.mmdit.planner_preference.v1",
                "sample_id": sample_id,
                "prompt": chosen.get("prompt", rejected.get("prompt")),
                "chosen": chosen["prescription"],
                "rejected": rejected["prescription"],
                "chosen_reward": chosen["reward"],
                "rejected_reward": rejected["reward"],
                "reward_gap": selected["reward_gap"],
                "chosen_candidate_id": chosen.get("candidate_id"),
                "rejected_candidate_id": rejected.get("candidate_id"),
                "reward_source": "executed_enhancement_metrics",
            }
        )
    report = {
        "schema_version": "lse.mmdit.planner_preference_report.v1",
        "candidate_records": len(candidates),
        "sample_groups": len(grouped),
        "preference_pairs": len(pairs),
        "skipped_small_gap_or_singleton": skipped,
        "minimum_reward_gap": float(min_gap),
        "reward_source": "executed_enhancement_metrics",
    }
    return pairs, report


def build_preference_file(source: Path, output: Path, *, min_gap: float = 0.05) -> dict[str, Any]:
    pairs, report = build_preference_rows(read_jsonl(source), min_gap=min_gap)
    write_jsonl(output, pairs)
    report.update({"source": str(source.resolve()), "output": str(output.resolve())})
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-gap", type=float, default=0.05)
    args = parser.parse_args(argv)
    report = build_preference_file(args.input, args.output, min_gap=args.min_gap)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
