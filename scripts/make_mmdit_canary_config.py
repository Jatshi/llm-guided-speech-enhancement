#!/usr/bin/env python3
"""Derive a short canary config without hand-editing the locked formal config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", default="data/mmdit/canary.jsonl")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    config = json.loads(args.base.read_text(encoding="utf-8"))
    config["project"]["name"] += "-canary"
    config["data"]["manifest"] = args.manifest
    config["training"].update(
        {
            "output_dir": "outputs/mmdit-canary",
            "max_steps": args.max_steps,
            "warmup_steps": min(20, max(1, args.max_steps // 10)),
            "log_every": 1,
            "eval_every": 25,
            "save_every": 25,
            "eval_batches": 8,
        }
    )
    config["evaluation"].update(
        {
            "output_dir": "results/mmdit-canary",
            "max_samples": 20,
            "deepfilternet": {"enabled": False, "model": "DeepFilterNet3"},
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
