"""Unified command line for LSE 2.0."""

from __future__ import annotations

import argparse

from . import data_cli, deepspeed, evaluation, inference, pipeline, training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lse-v2")
    parser.add_argument(
        "command",
        choices=("data", "deepspeed", "train", "pipeline", "predict", "evaluate"),
    )
    args, rest = parser.parse_known_args(argv)
    return {
        "data": data_cli.main,
        "deepspeed": deepspeed.main,
        "train": training.main,
        "pipeline": pipeline.main,
        "predict": inference.main,
        "evaluate": evaluation.main,
    }[args.command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
