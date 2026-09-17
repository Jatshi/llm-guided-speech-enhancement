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
from .curriculum import CurriculumPhase, CurriculumSchedule
from .data import PairedEnhancementDataset, batch_to_device, collate_pairs
from .flow import FlowLoss, RectifiedFlow, RectifiedFlowConfig
from .losses import (
    MultiResolutionSTFTLoss,
    load_frozen_semantic_teacher,
    semantic_consistency_loss,
)
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


def _phase_scoped_best(
    phase_name: str,
    validation_loss: float,
    *,
    best_phase: str | None,
    best_validation: float,
) -> tuple[float, str | None, bool]:
    """Select checkpoints within a curriculum phase, not across unlike objectives."""

    if not math.isfinite(validation_loss):
        return best_validation, best_phase, False
    if phase_name != best_phase or validation_loss < best_validation:
        return validation_loss, phase_name, True
    return best_validation, best_phase, False


def _prepare_metrics_log(path: Path, *, resume_step: int | None) -> None:
    """Make JSONL metrics consistent with the checkpoint used for this run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if resume_step is None or not path.exists():
        path.write_text("", encoding="utf-8")
        return
    rows_by_step: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row_step = int(row["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if row_step <= resume_step:
            rows_by_step[row_step] = json.dumps(row)
    content = "".join(f"{rows_by_step[key]}\n" for key in sorted(rows_by_step))
    path.write_text(content, encoding="utf-8")


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


def _auxiliary_losses(
    loss: FlowLoss,
    codec: ComplexSTFTCodec,
    clean_waveform: torch.Tensor,
    phase: CurriculumPhase,
    mrstft: MultiResolutionSTFTLoss,
    semantic_teacher: torch.nn.Module | None,
    *,
    semantic_layer: int | None,
    compute_semantic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = loss.total.new_zeros(())
    if phase.mrstft_weight == 0 and (phase.semantic_weight == 0 or not compute_semantic):
        return loss.total, zero, zero
    estimate = codec.decode(loss.estimated_x0.float(), length=clean_waveform.shape[-1])
    reference = clean_waveform.float()
    mrstft_loss = mrstft(estimate, reference) if phase.mrstft_weight else zero
    semantic_loss = zero
    if phase.semantic_weight and compute_semantic:
        if semantic_teacher is None:
            raise RuntimeError("semantic curriculum weight requires a loaded teacher")
        semantic_loss = semantic_consistency_loss(
            semantic_teacher,
            estimate,
            reference,
            layer=semantic_layer,
        )
    total = loss.total + phase.mrstft_weight * mrstft_loss + phase.semantic_weight * semantic_loss
    return total, mrstft_loss, semantic_loss


@torch.no_grad()
def _validate(
    model: PrescriptionConditionedMMDiT,
    flow: RectifiedFlow,
    codec: ComplexSTFTCodec,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    max_batches: int,
    phase: CurriculumPhase,
    mrstft: MultiResolutionSTFTLoss,
    semantic_teacher: torch.nn.Module | None,
    semantic_layer: int | None,
) -> dict[str, float]:
    model.eval()
    rows: list[tuple[float, float, float, float, float, float, float]] = []
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
                total, mrstft_loss, semantic_loss = _auxiliary_losses(
                    loss,
                    codec,
                    batch["clean"],
                    phase,
                    mrstft,
                    semantic_teacher,
                    semantic_layer=semantic_layer,
                    compute_semantic=True,
                )
            rows.append(
                (
                    float(total),
                    float(loss.total),
                    float(loss.velocity_mse),
                    float(loss.x0_l1),
                    float(loss.mismatch_noop_mse),
                    float(mrstft_loss),
                    float(semantic_loss),
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
            "base_loss": math.nan,
            "velocity_mse": math.nan,
            "x0_l1": math.nan,
            "mismatch_noop_mse": math.nan,
            "mrstft_loss": math.nan,
            "semantic_loss": math.nan,
        }
    return {
        "loss": sum(row[0] for row in rows) / len(rows),
        "base_loss": sum(row[1] for row in rows) / len(rows),
        "velocity_mse": sum(row[2] for row in rows) / len(rows),
        "x0_l1": sum(row[3] for row in rows) / len(rows),
        "mismatch_noop_mse": sum(row[4] for row in rows) / len(rows),
        "mrstft_loss": sum(row[5] for row in rows) / len(rows),
        "semantic_loss": sum(row[6] for row in rows) / len(rows),
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
    base_flow_config = RectifiedFlowConfig(**config["flow"])
    codec_config = STFTCodecConfig(**config["codec"])
    model = PrescriptionConditionedMMDiT(model_config).to(device)
    codec = ComplexSTFTCodec(codec_config)
    max_steps = int(training["max_steps"])
    curriculum = CurriculumSchedule.from_config(training.get("curriculum"), max_steps=max_steps)
    flows = {
        phase.name: RectifiedFlow(phase.flow_config(base_flow_config))
        for phase in curriculum.phases
    }
    auxiliary = training.get("auxiliary_losses", {})
    mrstft_config = auxiliary.get("multi_resolution_stft", {})
    mrstft = MultiResolutionSTFTLoss(
        fft_sizes=tuple(mrstft_config.get("fft_sizes", [256, 512, 1024])),
        hop_sizes=tuple(mrstft_config.get("hop_sizes", [64, 128, 256])),
    ).to(device)
    semantic_config = auxiliary.get("semantic", {})
    semantic_layer_raw = semantic_config.get("layer")
    semantic_layer = int(semantic_layer_raw) if semantic_layer_raw is not None else None
    semantic_every = int(semantic_config.get("every_steps", 4))
    if semantic_every < 1:
        raise ValueError("training.auxiliary_losses.semantic.every_steps must be positive")
    semantic_teacher = None
    if curriculum.needs_semantic_teacher:
        model_name = str(semantic_config.get("model_name", "")).strip()
        if not model_name:
            raise ValueError("semantic curriculum requires auxiliary_losses.semantic.model_name")
        semantic_teacher = load_frozen_semantic_teacher(model_name, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    warmup = int(training.get("warmup_steps", max(1, max_steps // 20)))

    def lr_factor(step: int) -> float:
        phase_step = min(max_steps, max(1, step + 1))
        phase_scale = curriculum.phase_for_step(phase_step).lr_scale
        if step < warmup:
            return phase_scale * max(1e-4, (step + 1) / warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return phase_scale * max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = _grad_scaler(device, precision)
    step = 0
    micro_step = 0
    epoch = 0
    best_validation = float("inf")
    best_phase: str | None = None
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
        # Old checkpoints did not identify which curriculum objective produced
        # their best score.  Treat that as unknown and establish a fresh best at
        # the next validation rather than comparing unlike phase objectives.
        best_phase = payload.get("best_phase")
    accumulation = int(training["gradient_accumulation_steps"])
    log_every = int(training.get("log_every", 10))
    eval_every = int(training.get("eval_every", 250))
    save_every = int(training.get("save_every", 250))
    clip = float(training.get("max_grad_norm", 1.0))
    log_path = output / "train_metrics.jsonl"
    _prepare_metrics_log(log_path, resume_step=step if resume_path else None)
    optimizer.zero_grad(set_to_none=True)
    accumulated_total = 0.0
    accumulated_velocity = 0.0
    accumulated_x0 = 0.0
    accumulated_mismatch = 0.0
    accumulated_mrstft = 0.0
    accumulated_semantic = 0.0
    accumulated_base = 0.0
    started = time.perf_counter()
    while step < max_steps:
        train_dataset.set_epoch(epoch)
        for raw in train_loader:
            phase = curriculum.phase_for_step(step + 1)
            flow = flows[phase.name]
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
                total_loss, mrstft_loss, semantic_loss = _auxiliary_losses(
                    losses,
                    codec,
                    batch["clean"],
                    phase,
                    mrstft,
                    semantic_teacher,
                    semantic_layer=semantic_layer,
                    compute_semantic=(micro_step + 1) % semantic_every == 0,
                )
                scaled_loss = total_loss / accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_total += float(total_loss.detach())
            accumulated_base += float(losses.total.detach())
            accumulated_velocity += float(losses.velocity_mse.detach())
            accumulated_x0 += float(losses.x0_l1.detach())
            accumulated_mismatch += float(losses.mismatch_noop_mse.detach())
            accumulated_mrstft += float(mrstft_loss.detach())
            accumulated_semantic += float(semantic_loss.detach())
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
                "phase": phase.name,
                "loss": accumulated_total / accumulation,
                "base_loss": accumulated_base / accumulation,
                "velocity_mse": accumulated_velocity / accumulation,
                "x0_l1": accumulated_x0 / accumulation,
                "mismatch_noop_mse": accumulated_mismatch / accumulation,
                "mrstft_loss": accumulated_mrstft / accumulation,
                "semantic_loss": accumulated_semantic / accumulation,
                "grad_norm": grad_norm,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.perf_counter() - started,
            }
            accumulated_total = 0.0
            accumulated_base = 0.0
            accumulated_velocity = 0.0
            accumulated_x0 = 0.0
            accumulated_mismatch = 0.0
            accumulated_mrstft = 0.0
            accumulated_semantic = 0.0
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
                    phase,
                    mrstft,
                    semantic_teacher,
                    semantic_layer,
                )
                print(json.dumps({"step": step, "validation": validation}), flush=True)
            improved = False
            if validation is not None:
                best_validation, best_phase, improved = _phase_scoped_best(
                    phase.name,
                    float(validation["loss"]),
                    best_phase=best_phase,
                    best_validation=best_validation,
                )
            payload = {
                "schema_version": "lse.mmdit.checkpoint.v1",
                "model_config": asdict(model_config),
                "flow_config": asdict(base_flow_config),
                "codec_config": asdict(codec_config),
                "curriculum_phase": phase.name,
                "curriculum": [asdict(item) for item in curriculum.phases],
                "auxiliary_losses": auxiliary,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "micro_step": micro_step,
                "epoch": epoch,
                "best_validation": best_validation,
                "best_phase": best_phase,
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
        "best_phase": best_phase,
        "elapsed_seconds": time.perf_counter() - started,
        "train_records": len(train_dataset),
        "validation_records": len(validation_dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "curriculum": [asdict(item) for item in curriculum.phases],
        "semantic_teacher": semantic_config.get("model_name") if semantic_teacher else None,
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
