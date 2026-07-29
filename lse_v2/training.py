"""Config-driven SFT, DPO, and GRPO training with resumable checkpoints."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
from pathlib import Path
from typing import Any

from .config import (
    config_digest,
    find_latest_checkpoint,
    load_config,
    resolve_path,
    set_global_seed,
)
from .contracts import validate_alignment_record
from .deepspeed import resolve_deepspeed, write_runtime_config
from .io import git_commit, read_jsonl, utc_now, write_json_atomic
from .rewards import grpo_reward

STAGES = ("sft", "dpo", "grpo")
DEEPSPEED_DISTRIBUTED_ENV = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
)


def _ensure_single_process_deepspeed_env(world_size: int) -> dict[str, str]:
    """Prevent DeepSpeed from falling back to MPI discovery for world_size=1."""
    if world_size == 1:
        defaults = {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in DEEPSPEED_DISTRIBUTED_ENV if key in os.environ}


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "trl", "peft", "datasets"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _load_dataset(path: Path, expected_schema: str) -> Any:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    for row in rows:
        validate_alignment_record(row)
        if row["schema_version"] != expected_schema:
            raise ValueError(f"{path}: expected {expected_schema}, got {row['schema_version']}")
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc
    return Dataset.from_list(rows)


def _filter_kwargs(factory: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep one codebase compatible across adjacent TRL patch releases."""
    signature = inspect.signature(factory)
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _precision_flags(config: dict[str, Any]) -> tuple[bool, bool]:
    dtype = str(config["model"].get("dtype", "bfloat16")).lower()
    return dtype in {"bf16", "bfloat16"}, dtype in {"fp16", "float16"}


def _common_args(
    config: dict[str, Any],
    stage: str,
    output_dir: Path,
    deepspeed_config: dict[str, Any] | None,
) -> dict[str, Any]:
    params = config["training"]["stages"][stage]
    bf16, fp16 = _precision_flags(config)
    result = {
        "output_dir": str(output_dir),
        "run_name": f"{config['project'].get('name', 'lse-v2')}-{stage}",
        "seed": int(config["project"]["seed"]),
        "data_seed": int(config["project"]["seed"]),
        "num_train_epochs": float(params.get("num_train_epochs", 1)),
        "max_steps": int(params.get("max_steps", -1)),
        "per_device_train_batch_size": int(params["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(params.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(params.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(params.get("learning_rate", 2e-4)),
        "warmup_ratio": float(params.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": params.get("lr_scheduler_type", "cosine"),
        "logging_steps": int(params.get("logging_steps", 5)),
        "logging_dir": str(output_dir / "tensorboard"),
        "save_strategy": params.get("save_strategy", "steps"),
        "save_steps": int(params.get("save_steps", 100)),
        "save_total_limit": int(params.get("save_total_limit", 2)),
        "eval_strategy": params.get("eval_strategy", "steps"),
        "evaluation_strategy": params.get("eval_strategy", "steps"),
        "eval_steps": int(params.get("eval_steps", 100)),
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": bool(params.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": params.get("report_to", ["tensorboard"]),
        "remove_unused_columns": bool(params.get("remove_unused_columns", stage != "grpo")),
        "dataloader_num_workers": int(params.get("dataloader_num_workers", 2)),
        "max_grad_norm": float(params.get("max_grad_norm", 1.0)),
    }
    if deepspeed_config:
        result["deepspeed"] = str(write_runtime_config(deepspeed_config, output_dir))
    return result


def _lora_config(config: dict[str, Any]) -> Any:
    from peft import LoraConfig, TaskType

    params = config["model"].get("lora", {})
    return LoraConfig(
        r=int(params.get("r", 16)),
        lora_alpha=int(params.get("alpha", 32)),
        lora_dropout=float(params.get("dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=params.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )


def _load_base_and_adapter(
    config: dict[str, Any],
    adapter_path: Path,
    *,
    deepspeed_enabled: bool,
) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_name = str(config["model"].get("dtype", "bfloat16")).lower()
    dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float16
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config["model"].get("trust_remote_code", True)),
        "torch_dtype": dtype,
    }
    if not deepspeed_enabled:
        model_kwargs["device_map"] = config["model"].get("device_map", "auto")
    base = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    model.config.use_cache = False
    return tokenizer, model


def _train_sft(
    config: dict[str, Any],
    output_dir: Path,
    resume: Path | None,
    deepspeed_config: dict[str, Any] | None,
) -> None:
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    paths = config["data"]
    train_ds = _load_dataset(resolve_path(config, paths["sft_train"]), "lse.sft.v2")
    eval_ds = _load_dataset(resolve_path(config, paths["sft_eval"]), "lse.sft.v2")
    model_id = config["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    params = config["training"]["stages"]["sft"]
    kwargs = _common_args(config, "sft", output_dir, deepspeed_config)
    kwargs.update(
        {
            "max_length": int(params.get("max_length", 1024)),
            "packing": bool(params.get("packing", False)),
        }
    )
    args = SFTConfig(**_filter_kwargs(SFTConfig, kwargs))
    trainer_kwargs = {
        "model": model_id,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "peft_config": _lora_config(config),
    }
    trainer = SFTTrainer(**_filter_kwargs(SFTTrainer, trainer_kwargs))
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_state()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


def _train_dpo(
    config: dict[str, Any],
    output_dir: Path,
    resume: Path | None,
    deepspeed_config: dict[str, Any] | None,
) -> None:
    from trl import DPOConfig, DPOTrainer

    paths = config["data"]
    train_ds = _load_dataset(resolve_path(config, paths["dpo_train"]), "lse.dpo.v2")
    eval_ds = _load_dataset(resolve_path(config, paths["dpo_eval"]), "lse.dpo.v2")
    sft_adapter = resolve_path(config, config["training"]["stage_inputs"]["dpo"])
    if not (sft_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"SFT adapter not found: {sft_adapter}")
    tokenizer, model = _load_base_and_adapter(
        config, sft_adapter, deepspeed_enabled=deepspeed_config is not None
    )
    params = config["training"]["stages"]["dpo"]
    kwargs = _common_args(config, "dpo", output_dir, deepspeed_config)
    kwargs.update(
        {
            "beta": float(params.get("beta", 0.1)),
            "label_smoothing": float(params.get("label_smoothing", 0.0)),
            "max_length": int(params.get("max_length", 1024)),
            "max_prompt_length": int(params.get("max_prompt_length", 768)),
        }
    )
    args = DPOConfig(**_filter_kwargs(DPOConfig, kwargs))
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = DPOTrainer(**_filter_kwargs(DPOTrainer, trainer_kwargs))
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_state()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


def _train_grpo(
    config: dict[str, Any],
    output_dir: Path,
    resume: Path | None,
    deepspeed_config: dict[str, Any] | None,
) -> None:
    from trl import GRPOConfig, GRPOTrainer

    paths = config["data"]
    train_ds = _load_dataset(resolve_path(config, paths["grpo_train"]), "lse.grpo.v2")
    eval_ds = _load_dataset(resolve_path(config, paths["grpo_eval"]), "lse.grpo.v2")
    dpo_adapter = resolve_path(config, config["training"]["stage_inputs"]["grpo"])
    if not (dpo_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"DPO adapter not found: {dpo_adapter}")
    tokenizer, model = _load_base_and_adapter(
        config, dpo_adapter, deepspeed_enabled=deepspeed_config is not None
    )
    params = config["training"]["stages"]["grpo"]
    kwargs = _common_args(config, "grpo", output_dir, deepspeed_config)
    kwargs.update(
        {
            "max_prompt_length": int(params.get("max_prompt_length", 768)),
            "max_completion_length": int(params.get("max_completion_length", 256)),
            "num_generations": int(params.get("num_generations", 2)),
            "temperature": float(params.get("temperature", 0.7)),
            "beta": float(params.get("beta", 0.04)),
        }
    )
    args = GRPOConfig(**_filter_kwargs(GRPOConfig, kwargs))
    trainer_kwargs = {
        "model": model,
        "reward_funcs": [grpo_reward],
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = GRPOTrainer(**_filter_kwargs(GRPOTrainer, trainer_kwargs))
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_state()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


def _validate_stage(config: dict[str, Any], stage: str) -> dict[str, Any]:
    stage_schema = {"sft": "lse.sft.v2", "dpo": "lse.dpo.v2", "grpo": "lse.grpo.v2"}[stage]
    paths = config["data"]
    report: dict[str, Any] = {"stage": stage, "datasets": {}}
    for split in ("train", "eval"):
        path = resolve_path(config, paths[f"{stage}_{split}"])
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"{path} is empty")
        for row in rows:
            validate_alignment_record(row)
            if row["schema_version"] != stage_schema:
                raise ValueError(f"{path}: expected schema {stage_schema}")
        report["datasets"][split] = {"path": str(path), "records": len(rows)}
    params = config["training"]["stages"][stage]
    report["effective_batch_size"] = int(params["per_device_train_batch_size"]) * int(
        params.get("gradient_accumulation_steps", 1)
    )
    if stage == "grpo":
        generations = int(params.get("num_generations", 2))
        if int(params["per_device_train_batch_size"]) % generations != 0:
            raise ValueError(
                "GRPO per_device_train_batch_size must be divisible by num_generations"
            )
    return report


def train_stage(
    config_path: str | Path,
    stage: str,
    *,
    dry_run: bool = False,
    resume_mode: str = "auto",
    deepspeed_override: str | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    config = load_config(config_path)
    seed = int(config["project"]["seed"])
    set_global_seed(seed, include_torch=not dry_run)
    output_dir = resolve_path(
        config, config["training"]["stages"][stage].get("output_dir", f"outputs/v2/{stage}")
    )
    validation = _validate_stage(config, stage)
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    deepspeed_config = resolve_deepspeed(
        config,
        stage,
        cli_value=deepspeed_override,
        actual_world_size=actual_world_size,
    )
    launch_environment = (
        _ensure_single_process_deepspeed_env(actual_world_size)
        if deepspeed_config is not None and not dry_run
        else None
    )
    resume = find_latest_checkpoint(output_dir) if resume_mode == "auto" else None
    if resume_mode not in {"auto", "never"}:
        explicit = Path(resume_mode).expanduser().resolve()
        if not explicit.is_dir():
            raise FileNotFoundError(f"Resume checkpoint not found: {explicit}")
        resume = explicit
    result = {
        "schema_version": "lse.stage_run.v2",
        "stage": stage,
        "started_at": utc_now(),
        "dry_run": dry_run,
        "config": str(Path(config_path).resolve()),
        "config_sha256": config_digest(config),
        "output_dir": str(output_dir),
        "resume_from_checkpoint": str(resume) if resume else None,
        "seed": seed,
        "model": config["model"]["name_or_path"],
        "distributed": {
            "world_size": actual_world_size,
            "launch_environment": launch_environment,
            "deepspeed": (
                None
                if deepspeed_config is None
                else {
                    key: value for key, value in deepspeed_config.items() if key != "runtime_config"
                }
            ),
        },
        "validation": validation,
        "environment": _dependency_versions(),
        "git_commit": git_commit(resolve_path(config, ".")),
        "status": "validated" if dry_run else "running",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "stage_manifest.json", result)
    if dry_run:
        return result
    if not result["environment"].get("torch") or not result["environment"].get("trl"):
        raise RuntimeError("Training dependencies are not installed")
    dispatch = {"sft": _train_sft, "dpo": _train_dpo, "grpo": _train_grpo}
    try:
        dispatch[stage](config, output_dir, resume, deepspeed_config)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["finished_at"] = utc_now()
        write_json_atomic(output_dir / "stage_manifest.json", result)
        raise
    final_dir = output_dir / "final"
    if not (final_dir / "adapter_config.json").is_file():
        raise RuntimeError(f"{stage} finished without adapter_config.json in {final_dir}")
    result["status"] = "complete"
    result["finished_at"] = utc_now()
    result["artifacts"] = {
        "adapter": str(final_dir),
        "trainer_state": str(output_dir / "trainer_state.json"),
        "tensorboard": str(output_dir / "tensorboard"),
    }
    if deepspeed_config:
        result["artifacts"]["deepspeed_runtime"] = str(output_dir / "deepspeed_runtime.json")
    write_json_atomic(output_dir / "stage_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--config", default="configs/autodl_4090.json")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'never', or an explicit checkpoint directory",
    )
    parser.add_argument(
        "--deepspeed",
        help="DeepSpeed JSON path; use 'none' to disable a configured profile",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_stage(
        args.config,
        args.stage,
        dry_run=args.dry_run,
        resume_mode=args.resume,
        deepspeed_override=args.deepspeed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
