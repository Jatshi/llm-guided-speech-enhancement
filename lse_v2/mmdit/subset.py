"""Create deterministic, speaker-preserving canary subsets from a pair manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lse_v2.io import read_jsonl, write_jsonl


def make_subset(
    source: Path,
    output: Path,
    *,
    train: int,
    validation: int,
    test: int,
    seed: int,
) -> dict:
    rows = read_jsonl(source)
    result = []
    requested = {"train": train, "validation": validation, "test": test}
    for split, count in requested.items():
        candidates = [row for row in rows if row.get("split") == split]
        candidates.sort(
            key=lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).digest()
        )
        if len(candidates) < count:
            raise ValueError(f"split {split} has {len(candidates)} records; need {count}")
        result.extend(candidates[:count])
    write_jsonl(output, result)
    report = {
        "schema_version": "lse.mmdit.subset.v1",
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "seed": seed,
        "counts": requested,
    }
    output.with_suffix(".subset.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=32)
    parser.add_argument("--validation", type=int, default=8)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            make_subset(
                args.source,
                args.output,
                train=args.train,
                validation=args.validation,
                test=args.test,
                seed=args.seed,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
