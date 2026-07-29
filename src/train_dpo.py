"""Backward-compatible DPO entry point."""

from __future__ import annotations

import argparse
import json

from lse_v2.contracts import corrupt_target
from lse_v2.training import train_stage


def corrupt(good: str) -> str:
    """Compatibility helper for callers of the original corruption function."""
    import json as json_module

    try:
        target = json_module.loads(good)
    except json_module.JSONDecodeError:
        target = {
            "diagnosis": {"noise_type": "unknown"},
            "actions": [{"type": "spectral_subtraction", "reduction_db": 8.0}],
            "rationale": good,
            "confidence": 0.5,
        }
    return json_module.dumps(corrupt_target(target), ensure_ascii=False, sort_keys=True)


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
                "dpo",
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
