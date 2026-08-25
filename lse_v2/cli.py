"""Unified command line for LSE 2.0."""

from __future__ import annotations

import argparse

from . import (
    catalog,
    data_cli,
    deepspeed,
    end_to_end,
    evaluation,
    generalization,
    inference,
    listening_test,
    materialization,
    native_data,
    native_pipeline,
    pipeline,
    service,
    training,
)


def _native_predict_main(argv: list[str] | None = None) -> int:
    from . import native_predict

    return native_predict.main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lse-v2")
    parser.add_argument(
        "command",
        choices=(
            "catalog",
            "data",
            "deepspeed",
            "enhance",
            "evaluate",
            "generalization",
            "materialize",
            "listening-test",
            "native-data",
            "native-train",
            "native-predict",
            "pipeline",
            "predict",
            "serve",
            "train",
        ),
    )
    args, rest = parser.parse_known_args(argv)
    return {
        "catalog": catalog.main,
        "data": data_cli.main,
        "deepspeed": deepspeed.main,
        "enhance": end_to_end.main,
        "generalization": generalization.main,
        "materialize": materialization.main,
        "listening-test": listening_test.main,
        "native-data": native_data.main,
        "native-train": native_pipeline.main,
        "native-predict": _native_predict_main,
        "serve": service.main,
        "train": training.main,
        "pipeline": pipeline.main,
        "predict": inference.main,
        "evaluate": evaluation.main,
    }[args.command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
