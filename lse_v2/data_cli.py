"""CLI for manifest migration, validation, and alignment dataset construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import (
    build_alignment_datasets,
    manifest_from_legacy,
    validate_audio_record,
)
from .io import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate an audio manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--check-audio-files", action="store_true")

    migrate = sub.add_parser("migrate", help="Convert legacy metadata.json to v2 JSONL")
    migrate.add_argument("--legacy-metadata", required=True)
    migrate.add_argument("--output", required=True)
    migrate.add_argument("--sample-rate", type=int, default=16000)
    migrate.add_argument("--extract-features", action="store_true")

    build = sub.add_parser("build", help="Build SFT/DPO/GRPO datasets")
    build.add_argument("--manifest", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--eval-ratio", type=float, default=0.05)
    build.add_argument("--check-audio-files", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        records = read_jsonl(args.manifest)
        for record in records:
            validate_audio_record(record, check_files=args.check_audio_files)
        print(json.dumps({"valid": True, "records": len(records)}, indent=2))
        return 0
    if args.command == "migrate":
        count = manifest_from_legacy(
            args.legacy_metadata,
            args.output,
            sample_rate=args.sample_rate,
            extract_features=args.extract_features,
        )
        print(json.dumps({"output": str(Path(args.output).resolve()), "records": count}, indent=2))
        return 0
    result = build_alignment_datasets(
        args.manifest,
        args.output_dir,
        seed=args.seed,
        eval_ratio=args.eval_ratio,
        check_audio_files=args.check_audio_files,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
