"""Backward-compatible GRPO entry point."""

from __future__ import annotations

import argparse
import json

from lse_v2.training import train_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/autodl_4090.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--deepspeed")
    args = parser.parse_args()
    print(
        json.dumps(
            train_stage(
                args.config,
                "grpo",
                dry_run=args.dry_run,
                resume_mode=args.resume,
                deepspeed_override=args.deepspeed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
