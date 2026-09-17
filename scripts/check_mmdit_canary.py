#!/usr/bin/env python3
"""Machine-checkable finite-loss and learning-direction gate for the paid run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-last-to-first-ratio", type=float, default=0.98)
    parser.add_argument("--min-steps", type=int, default=50)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    losses = [float(row["loss"]) for row in rows]
    if not losses:
        raise ValueError(f"no loss rows found in {args.metrics}")
    if args.min_steps < 1 or args.max_last_to_first_ratio <= 0:
        raise ValueError("min-steps and max-last-to-first-ratio must be positive")
    window = min(10, max(1, len(losses) // 4))
    first = statistics.fmean(losses[:window])
    last = statistics.fmean(losses[-window:])
    finite = all(math.isfinite(value) for value in losses)
    enough_steps = len(losses) >= args.min_steps
    learned = enough_steps and last <= first * args.max_last_to_first_ratio
    report = {
        "schema_version": "lse.mmdit.canary_gate.v1",
        "status": "PASSED" if finite and learned else "BLOCKED",
        "steps": len(losses),
        "min_steps": args.min_steps,
        "window": window,
        "first_loss_mean": first,
        "last_loss_mean": last,
        "last_to_first_ratio": last / first,
        "all_finite": finite,
        "enough_steps": enough_steps,
        "learning_direction_gate": learned,
        "quality_claim": "NOT_MEASURED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
