"""Recover GRPO from a verified SFT stage without rerunning SFT or DPO."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import set_global_seed
from .io import git_commit, utc_now, write_json_atomic
from .native_pipeline import NativePipelineConfig, load_native_pipeline_config, load_native_records


@dataclass(frozen=True, slots=True)
class GrpoRecoveryConfig:
    native: NativePipelineConfig
    source_stage_dir: Path
    audio_cache_dir: Path


def _resolve_path(config_path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"recovery.{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def load_grpo_recovery_config(path: str | Path) -> GrpoRecoveryConfig:
    native = load_native_pipeline_config(path)
    if native.stages["grpo"].get("input_stage") != "sft":
        raise ValueError("GRPO recovery must bootstrap from the verified SFT stage")
    payload = json.loads(native.config_path.read_text(encoding="utf-8"))
    recovery = payload.get("recovery")
    if not isinstance(recovery, dict):
        raise ValueError("recovery config must be an object")
    return GrpoRecoveryConfig(
        native=native,
        source_stage_dir=_resolve_path(
            native.config_path, recovery.get("source_stage_dir"), "source_stage_dir"
        ),
        audio_cache_dir=_resolve_path(
            native.config_path, recovery.get("audio_cache_dir"), "audio_cache_dir"
        ),
    )


def validate_source_stage(source_stage_dir: Path) -> dict[str, Any]:
    from .native_training_pipeline import _saved_adapter_path

    manifest_path = source_stage_dir / "artifact_manifest.json"
    projector_path = source_stage_dir / "audio_projector.pt"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not projector_path.is_file() or projector_path.stat().st_size == 0:
        raise FileNotFoundError(projector_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "lse.native_stage_artifacts.v1"
        or manifest.get("stage") != "sft"
    ):
        raise ValueError("source artifact manifest must prove a native SFT stage")
    adapter = _saved_adapter_path(source_stage_dir).resolve()
    return {
        "schema_version": "lse.grpo_recovery_source.v1",
        "stage": "sft",
        "source_stage_dir": str(source_stage_dir.resolve()),
        "adapter": str(adapter),
        "audio_projector": str(projector_path.resolve()),
    }


def recovery_plan(config: GrpoRecoveryConfig, *, check_artifacts: bool = False) -> dict[str, Any]:
    source = (
        validate_source_stage(config.source_stage_dir)
        if check_artifacts
        else {
            "stage": "sft",
            "source_stage_dir": str(config.source_stage_dir),
            "exists": config.source_stage_dir.is_dir(),
        }
    )
    return {
        "schema_version": "lse.grpo_recovery_plan.v1",
        "status": "validated",
        "config": str(config.native.config_path),
        "manifest": str(config.native.manifest),
        "output_dir": str(config.native.output_dir),
        "audio_cache_dir": str(config.audio_cache_dir),
        "source": source,
        "grpo": config.native.stages["grpo"],
        "gpu_claim": False,
    }


def run_grpo_recovery(config: GrpoRecoveryConfig) -> dict[str, Any]:
    import torch

    from .native_training_pipeline import (
        _completed_stage_report,
        _gpu_description,
        _run_stage,
        _versions,
        precompute_audio_embeddings,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("GRPO recovery requires a CUDA GPU")
    source = validate_source_stage(config.source_stage_dir)
    set_global_seed(config.native.seed)
    config.native.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_native_records(config.native.manifest)
    all_record_count = len(records)
    sample_limit = int(config.native.stages["grpo"].get("max_train_records", 0))
    if sample_limit:
        if sample_limit < 1:
            raise ValueError("max_train_records must be positive when set")
        records = [row for row in records if row["split"] == "train"][:sample_limit]
    records, cache_report = precompute_audio_embeddings(
        records,
        whisper_model=config.native.whisper_model,
        cache_dir=config.audio_cache_dir,
        batch_size=int(config.native.stages["grpo"].get("embedding_batch_size", 16)),
    )
    first = np.load(records[0]["embedding_path"], allow_pickle=False)
    encoder_dim = int(first["hidden"].shape[1])
    first.close()
    expected_steps = int(config.native.stages["grpo"]["max_steps"])
    stage_report = _completed_stage_report(config.native.output_dir, "grpo", expected_steps)
    if stage_report is None:
        stage_report = _run_stage(
            config.native,
            "grpo",
            records,
            encoder_dim,
            config.source_stage_dir,
        )
    report = {
        "schema_version": "lse.grpo_recovery_run.v1",
        "status": "completed",
        "completed_at": utc_now(),
        "config": str(config.native.config_path),
        "git_commit": git_commit(config.native.config_path.parent),
        "manifest": str(config.native.manifest),
        "records": len(records),
        "available_manifest_records": all_record_count,
        "subset_run": bool(sample_limit),
        "source": source,
        "audio_cache": cache_report,
        "stage": stage_report,
        "versions": _versions(),
        "gpu": _gpu_description(),
        "gpu_claim": True,
        "final_adapter": str(config.native.output_dir / "grpo" / "final"),
        "command": " ".join(sys.argv),
    }
    write_json_atomic(config.native.output_dir / "run_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-artifacts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_grpo_recovery_config(args.config)
    report = (
        recovery_plan(config, check_artifacts=args.check_artifacts)
        if args.dry_run
        else run_grpo_recovery(config)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
