"""Backward-compatible data builder for the v2 alignment contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lse_v2.contracts import build_alignment_datasets, manifest_from_legacy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/training/audio_manifest.v2.jsonl")
    parser.add_argument("--legacy-metadata", default="data/training/metadata.json")
    parser.add_argument("--output-dir", default="data/v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_file():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_from_legacy(args.legacy_metadata, manifest)
    result = build_alignment_datasets(
        manifest, args.output_dir, seed=args.seed, eval_ratio=args.eval_ratio
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
