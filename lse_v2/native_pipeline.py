"""Configuration, validation, and execution entrypoint for native-audio alignment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_jsonl, utc_now, write_json_atomic

NATIVE_AUDIO_SCHEMA = "lse.native_audio.v1"
NATIVE_STAGES = ("sft", "dpo", "grpo")


@dataclass(frozen=True, slots=True)
class NativePipelineConfig:
    config_path: Path
    manifest: Path
    output_dir: Path
    whisper_model: str
    language_model: str
    prefix_tokens: int
    stages: dict[str, dict[str, Any]]
    seed: int = 42


def load_native_pipeline_config(path: str | Path) -> NativePipelineConfig:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("manifest"), str) or not payload["manifest"].strip():
        raise ValueError("manifest must be a non-empty path")
    if not isinstance(payload.get("output_dir"), str) or not payload["output_dir"].strip():
        raise ValueError("output_dir must be a non-empty path")
    manifest = Path(payload["manifest"]).expanduser()
    output_dir = Path(payload["output_dir"]).expanduser()
    if not manifest.is_absolute():
        manifest = (config_path.parent / manifest).resolve()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    stages = payload.get("stages")
    if not isinstance(stages, dict) or list(stages) != list(NATIVE_STAGES):
        raise ValueError("stages must be ordered as sft, dpo, grpo")
    for stage in NATIVE_STAGES:
        if not isinstance(stages[stage], dict):
            raise ValueError(f"{stage} config must be an object")
        if int(stages[stage].get("max_steps", 0)) <= 0:
            raise ValueError(f"{stage}.max_steps must be positive")
        if int(stages[stage].get("gradient_accumulation_steps", 1)) <= 0:
            raise ValueError(f"{stage}.gradient_accumulation_steps must be positive")
        if float(stages[stage].get("learning_rate", 2e-4)) <= 0:
            raise ValueError(f"{stage}.learning_rate must be positive")
        dropout = float(stages[stage].get("audio_dropout", 0.1))
        if not 0 <= dropout < 1:
            raise ValueError(f"{stage}.audio_dropout must be in [0, 1)")
        deepspeed = stages[stage].get("deepspeed")
        if deepspeed:
            deepspeed_path = Path(str(deepspeed)).expanduser()
            if not deepspeed_path.is_absolute():
                deepspeed_path = (config_path.parent / deepspeed_path).resolve()
            if not deepspeed_path.is_file():
                raise FileNotFoundError(deepspeed_path)
    if stages["dpo"].get("input_stage") != "sft":
        raise ValueError("dpo.input_stage must be sft")
    if stages["grpo"].get("input_stage") not in {"sft", "dpo"}:
        raise ValueError("grpo.input_stage must be sft or dpo")
    if int(stages["grpo"].get("num_generations", 0)) < 2:
        raise ValueError("GRPO requires at least two generations")
    if float(stages["dpo"].get("beta", 0.1)) <= 0:
        raise ValueError("dpo.beta must be positive")
    if not 0 <= float(stages["dpo"].get("label_smoothing", 0.1)) < 0.5:
        raise ValueError("dpo.label_smoothing must be in [0, 0.5)")
    if float(stages["grpo"].get("beta", 0.04)) < 0:
        raise ValueError("grpo.beta must be non-negative")
    if not 0 < float(stages["grpo"].get("clip_epsilon", 0.2)) < 1:
        raise ValueError("grpo.clip_epsilon must be in (0, 1)")
    if float(stages["grpo"].get("sft_anchor_weight", 0.05)) < 0:
        raise ValueError("grpo.sft_anchor_weight must be non-negative")
    if int(stages["grpo"].get("canary_groups", 8)) <= 0:
        raise ValueError("grpo.canary_groups must be positive")
    if int(stages["grpo"].get("max_consecutive_saturated_groups", 16)) <= 0:
        raise ValueError("grpo.max_consecutive_saturated_groups must be positive")
    for name, default in (
        ("min_canary_valid_json_rate", 0.8),
        ("min_canary_non_saturated_group_rate", 0.2),
    ):
        if not 0 <= float(stages["grpo"].get(name, default)) <= 1:
            raise ValueError(f"grpo.{name} must be in [0, 1]")
    prefix_tokens = int(payload.get("prefix_tokens", 0))
    if prefix_tokens <= 0:
        raise ValueError("prefix_tokens must be positive")
    whisper_model = payload.get("whisper_model")
    language_model = payload.get("language_model")
    if not isinstance(whisper_model, str) or not whisper_model.strip():
        raise ValueError("whisper_model must be non-empty")
    if not isinstance(language_model, str) or not language_model.strip():
        raise ValueError("language_model must be non-empty")
    seed = int(payload.get("seed", 42))
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return NativePipelineConfig(
        config_path=config_path,
        manifest=manifest,
        output_dir=output_dir,
        whisper_model=whisper_model,
        language_model=language_model,
        prefix_tokens=prefix_tokens,
        stages=stages,
        seed=seed,
    )


def load_native_records(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("native manifest is empty")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("schema_version") != NATIVE_AUDIO_SCHEMA:
            raise ValueError(f"row {index} must use {NATIVE_AUDIO_SCHEMA}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"invalid or duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        for key in ("audio_path", "prompt_text", "chosen_text", "rejected_text", "source_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{sample_id} requires {key}")
        audio_path = Path(row["audio_path"]).expanduser()
        if not audio_path.is_absolute():
            audio_path = (path.parent / audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        row["audio_path"] = str(audio_path)
        if row.get("split") not in {"train", "eval", "test"}:
            raise ValueError(f"{sample_id} requires train/eval/test split")
        if not isinstance(row.get("reward_context"), dict):
            raise ValueError(f"{sample_id} requires reward_context")
    return rows


def dry_run_native_pipeline(config: NativePipelineConfig) -> dict[str, Any]:
    rows = load_native_records(config.manifest)
    source_splits: dict[str, set[str]] = {}
    for row in rows:
        source_splits.setdefault(row["source_id"], set()).add(row["split"])
    leaked = sorted(source for source, splits in source_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"source-level split leakage: {leaked[:10]}")
    return {
        "schema_version": "lse.native_pipeline_plan.v1",
        "status": "validated",
        "validated_at": utc_now(),
        "manifest": str(config.manifest),
        "records": len(rows),
        "materialized_audio_records": len(rows),
        "sources": len(source_splits),
        "models": {
            "audio_encoder": config.whisper_model,
            "language_model": config.language_model,
        },
        "prefix_tokens": config.prefix_tokens,
        "stages": {
            stage: {
                "status": "validated",
                "input_stage": config.stages[stage].get("input_stage"),
                "max_steps": int(config.stages[stage]["max_steps"]),
                "output_dir": str(config.output_dir / stage),
            }
            for stage in NATIVE_STAGES
        },
        "gpu_claim": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_native_pipeline_config(args.config)
    if not args.dry_run:
        from .native_training_pipeline import run_native_training_pipeline

        report = run_native_training_pipeline(config)
    else:
        report = dry_run_native_pipeline(config)
    output = config.output_dir / "pipeline_status.json"
    write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
