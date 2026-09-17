"""Single-GPU friendly, resumable MM-DiT training loop."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lse_v2.config import set_global_seed

from .checkpoint import atomic_torch_save
from .codec import ComplexSTFTCodec, STFTCodecConfig
from .config import config_digest, load_mmdit_config, resolve_path
from .data import PairedEnhancementDataset, batch_to_device, collate_pairs
from .flow import RectifiedFlow, RectifiedFlowConfig
from .model import MMDiTConfig, PrescriptionConditionedMMDiT


def _autocast(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bfloat16", "float16"}
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, enabled=enabled, dtype=dtype)


def _grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "float16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch 2.1 AutoDL images
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _make_dataset(config: dict[str, Any], split: str) -> PairedEnhancementDataset:
    return PairedEnhancementDataset(
        resolve_path(config, config["data"]["manifest"]),
        split=split,
        sample_rate=int(config["data"]["sample_rate"]),
        segment_seconds=float(config["data"]["segment_seconds"]),
        prescription_token_count=int(config["model"]["prescription_tokens"]),
        prescription_source="oracle",
        seed=int(config["project"]["seed"]),
    )


@torch.no_grad()
def _validate(
    model: PrescriptionConditionedMMDiT,
    flow: RectifiedFlow,
    codec: ComplexSTFTCodec,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    rows: list[tuple[float, float, float, float]] = []
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    torch.manual_seed(1729)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1729)
    try:
        for index, raw in enumerate(loader):
            if index >= max_batches:
                break
            batch = batch_to_device(raw, device)
            clean = codec.encode(batch["clean"].float())
            observed = codec.encode(batch["noisy"].float())
            with _autocast(device, precision):
                loss = flow.training_loss(
                    model,
                    clean,
                    observed,
                    batch["fields"],
                    batch["categories"],
                    batch["values"],
                )
            rows.append(
                (
                    float(loss.total),
                    float(loss.velocity_mse),
                    float(loss.x0_l1),
                    float(loss.mismatch_noop_mse),
                )
            )
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    model.train()
    if not rows:
        return {
            "loss": math.nan,
            "velocity_mse": math.nan,
            "x0_l1": math.nan,
            "mismatch_noop_mse": math.nan,
        }
    return {
        "loss": sum(row[0] for row in rows) / len(rows),
        "velocity_mse": sum(row[1] for row in rows) / len(rows),
        "x0_l1": sum(row[2] for row in rows) / len(rows),
        "mismatch_noop_mse": sum(row[3] for row in rows) / len(rows),
    }


def train(config_path: str | Path, *, resume: str | Path | None = None) -> dict[str, Any]:
    config = load_mmdit_config(config_path)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = config["training"]
    precision = str(training.get("precision", "bfloat16"))
    output = resolve_path(config, training["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = _make_dataset(config, "train")
    validation_dataset = _make_dataset(config, "validation")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_pairs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training.get("eval_batch_size", training["batch_size"])),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_pairs,
    )
    model_config = MMDiTConfig(**config["model"])
    flow_config = RectifiedFlowConfig(**config["flow"])
    codec_config = STFTCodecConfig(**config["codec"])
    model = PrescriptionConditionedMMDiT(model_config).to(device)
    flow = RectifiedFlow(flow_config)
    codec = ComplexSTFTCodec(codec_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    max_steps = int(training["max_steps"])
    warmup = int(training.get("warmup_steps", max(1, max_steps // 20)))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return max(1e-4, (step + 1) / warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = _grad_scaler(device, precision)
    step = 0
    micro_step = 0
    epoch = 0
    best_validation = float("inf")
    resume_path = Path(resume) if resume else None
    if resume_path:
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        step = int(payload["step"])
        micro_step = int(payload.get("micro_step", step))
        epoch = int(payload.get("epoch", 0))
        best_validation = float(payload.get("best_validation", best_validation))
    accumulation = int(training["gradient_accumulation_steps"])
    log_every = int(training.get("log_every", 10))
    eval_every = int(training.get("eval_every", 250))
    save_every = int(training.get("save_every", 250))
    clip = float(training.get("max_grad_norm", 1.0))
    log_path = output / "train_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    accumulated_total = 0.0
    accumulated_velocity = 0.0
    accumulated_x0 = 0.0
    accumulated_mismatch = 0.0
    started = time.perf_counter()
    while step < max_steps:
        train_dataset.set_epoch(epoch)
        for raw in train_loader:
            batch = batch_to_device(raw, device)
            clean = codec.encode(batch["clean"].float())
            observed = codec.encode(batch["noisy"].float())
            with _autocast(device, precision):
                losses = flow.training_loss(
                    model,
                    clean,
                    observed,
                    batch["fields"],
                    batch["categories"],
                    batch["values"],
                )
                scaled_loss = losses.total / accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_total += float(losses.total.detach())
            accumulated_velocity += float(losses.velocity_mse.detach())
            accumulated_x0 += float(losses.x0_l1.detach())
            accumulated_mismatch += float(losses.mismatch_noop_mse.detach())
            micro_step += 1
            if micro_step % accumulation:
                continue
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            row = {
                "step": step,
                "epoch": epoch,
                "loss": accumulated_total / accumulation,
                "velocity_mse": accumulated_velocity / accumulation,
                "x0_l1": accumulated_x0 / accumulation,
                "mismatch_noop_mse": accumulated_mismatch / accumulation,
                "grad_norm": grad_norm,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.perf_counter() - started,
            }
            accumulated_total = 0.0
            accumulated_velocity = 0.0
            accumulated_x0 = 0.0
            accumulated_mismatch = 0.0
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if step == 1 or step % log_every == 0:
                print(json.dumps(row), flush=True)
            should_evaluate = step % eval_every == 0 or step == max_steps
            validation = None
            if should_evaluate:
                validation = _validate(
                    model,
                    flow,
                    codec,
                    validation_loader,
                    device,
                    precision,
                    int(training.get("eval_batches", 20)),
                )
                print(json.dumps({"step": step, "validation": validation}), flush=True)
            improved = bool(validation and validation["loss"] < best_validation)
            if improved:
                best_validation = validation["loss"]
            payload = {
                "schema_version": "lse.mmdit.checkpoint.v1",
                "model_config": asdict(model_config),
                "flow_config": asdict(flow_config),
                "codec_config": asdict(codec_config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "micro_step": micro_step,
                "epoch": epoch,
                "best_validation": best_validation,
                "config_digest": config_digest(config),
            }
            if step % save_every == 0 or step == max_steps:
                atomic_torch_save(payload, output / "checkpoint-last.pt")
            if improved:
                atomic_torch_save(payload, output / "checkpoint-best.pt")
            if step >= max_steps:
                break
        epoch += 1
    report = {
        "schema_version": "lse.mmdit.training_report.v1",
        "status": "COMPLETED",
        "steps": step,
        "best_validation": best_validation,
        "elapsed_seconds": time.perf_counter() - started,
        "train_records": len(train_dataset),
        "validation_records": len(validation_dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config_digest": config_digest(config),
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(train(args.config, resume=args.resume), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
