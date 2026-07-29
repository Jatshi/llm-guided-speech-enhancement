"""Resumable SFT -> DPO -> GRPO -> offline-evaluation orchestrator."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path
from .evaluation import evaluate_rewards
from .inference import generate_predictions
from .io import utc_now, write_json_atomic
from .training import STAGES, train_stage


def release_training_memory() -> None:
    """Collect Trainer/DeepSpeed cycles before loading the evaluation model."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pipeline(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    resume_mode: str = "auto",
    stages: tuple[str, ...] = STAGES,
    deepspeed_override: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    status_path = resolve_path(
        config, config["training"].get("pipeline_status", "outputs/v2/pipeline.json")
    )
    state: dict[str, Any] = {
        "schema_version": "lse.pipeline_run.v2",
        "config": str(Path(config_path).resolve()),
        "started_at": utc_now(),
        "dry_run": dry_run,
        "status": "running",
        "stages": {},
    }
    write_json_atomic(status_path, state)
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        try:
            result = train_stage(
                config_path,
                stage,
                dry_run=dry_run,
                resume_mode=resume_mode,
                deepspeed_override=deepspeed_override,
            )
            state["stages"][stage] = {
                "status": result["status"],
                "manifest": str(Path(result["output_dir"]) / "stage_manifest.json"),
            }
            write_json_atomic(status_path, state)
        except Exception as exc:
            state["status"] = "failed"
            state["failed_stage"] = stage
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["finished_at"] = utc_now()
            write_json_atomic(status_path, state)
            raise
    if not dry_run:
        release_training_memory()
    eval_dataset = resolve_path(config, config["data"]["grpo_eval"])
    eval_output = resolve_path(
        config, config["evaluation"].get("report", "outputs/v2/evaluation/reward_report.json")
    )
    predictions_path = resolve_path(
        config,
        config["evaluation"].get("predictions", "outputs/v2/evaluation/predictions.jsonl"),
    )
    if dry_run:
        report = evaluate_rewards(eval_dataset)
    else:
        adapter_path = (
            resolve_path(
                config,
                config["training"]["stages"]["grpo"].get("output_dir", "outputs/v2/grpo"),
            )
            / "final"
        )
        generate_predictions(
            config_path,
            eval_dataset,
            adapter_path,
            predictions_path,
            max_samples=config["evaluation"].get("max_samples"),
        )
        report = evaluate_rewards(
            eval_dataset,
            predictions_path=predictions_path,
        )
    write_json_atomic(eval_output, report)
    state["evaluation"] = {
        "status": ("reference_reward_validation" if dry_run else "model_predictions_scored"),
        "report": str(eval_output),
        "predictions": None if dry_run else str(predictions_path),
        "note": (
            "Reference-only reward validation; not a model benchmark."
            if dry_run
            else "Scores are computed from saved model predictions."
        ),
    }
    state["status"] = "validated" if dry_run else "complete"
    state["finished_at"] = utc_now()
    write_json_atomic(status_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/autodl_4090.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", default="auto")
    parser.add_argument(
        "--deepspeed",
        help="Override all stage DeepSpeed profiles; use 'none' to disable",
    )
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help="Comma-separated stage subset in execution order",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    state = run_pipeline(
        args.config,
        dry_run=args.dry_run,
        resume_mode=args.resume,
        stages=stages,
        deepspeed_override=args.deepspeed,
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
