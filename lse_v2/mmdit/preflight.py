"""Fail-fast data, architecture, and runtime checks before paid GPU training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lse_v2.io import read_jsonl

from .codec import ComplexSTFTCodec, STFTCodecConfig
from .config import config_digest, load_mmdit_config, resolve_path
from .contracts import validate_pair_record
from .curriculum import CurriculumSchedule
from .data import PairedEnhancementDataset, batch_to_device, collate_pairs
from .model import MMDiTConfig, PrescriptionConditionedMMDiT


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    value = result.stdout.strip()
    return value or None


def preflight(config_path: Path, output: Path, *, runtime_smoke: bool = False) -> dict[str, Any]:
    config = load_mmdit_config(config_path)
    root = resolve_path(config, ".")
    manifest = resolve_path(config, config["data"]["manifest"])
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError("MM-DiT manifest is empty")
    check_files = bool(config["data"].get("check_files", True))
    speakers: dict[str, set[str]] = defaultdict(set)
    errors = []
    for row in rows:
        try:
            validate_pair_record(row, check_files=check_files)
        except Exception as exc:
            errors.append(f"{row.get('sample_id')}: {exc}")
        speakers[str(row.get("speaker_id"))].add(str(row.get("split")))
    leakage = {key: sorted(value) for key, value in speakers.items() if len(value) > 1}
    split_counts = Counter(str(row.get("split")) for row in rows)
    model_config = MMDiTConfig(**config["model"])
    model = PrescriptionConditionedMMDiT(model_config)
    curriculum = CurriculumSchedule.from_config(
        config["training"].get("curriculum"),
        max_steps=int(config["training"]["max_steps"]),
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    sample_rate = int(config["data"]["sample_rate"])
    samples = round(sample_rate * float(config["data"]["segment_seconds"]))
    n_fft = int(config["codec"]["n_fft"])
    hop = int(config["codec"]["hop_length"])
    frequency_bins = n_fft // 2 + 1
    frames = samples // hop + 1
    audio_tokens = (
        (frequency_bins + model_config.patch_frequency - 1) // model_config.patch_frequency
    ) * ((frames + model_config.patch_time - 1) // model_config.patch_time)
    joint_tokens = 2 * audio_tokens + model_config.prescription_tokens
    gates = {
        "manifest_records_present": bool(rows),
        "pair_contracts_valid": not errors,
        "train_split_present": split_counts["train"] > 0,
        "validation_split_present": split_counts["validation"] > 0,
        "test_split_present": split_counts["test"] > 0,
        "speaker_disjoint": not leakage,
        "parameter_budget": parameters <= int(config["training"].get("max_parameters", 80_000_000)),
        "joint_token_budget": joint_tokens <= int(config["training"].get("max_joint_tokens", 1536)),
        "identity_initialization": (
            not model_config.identity_init
            or (
                bool(torch.count_nonzero(model.output.weight) == 0)
                and all(
                    bool(torch.count_nonzero(block.modulation[-1].weight) == 0)
                    for block in model.blocks
                )
            )
        ),
        "curriculum_covers_training": (
            curriculum.phases[-1].until_step >= int(config["training"]["max_steps"])
        ),
    }
    if config["evaluation"].get("deepfilternet", {}).get("enabled", False):
        gates["strong_baseline_package"] = importlib.util.find_spec("df") is not None
    runtime = {"requested": runtime_smoke, "status": "NOT_RUN"}
    if runtime_smoke and all(gates.values()):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = PairedEnhancementDataset(
            manifest,
            split="train",
            sample_rate=sample_rate,
            segment_seconds=float(config["data"]["segment_seconds"]),
            prescription_token_count=model_config.prescription_tokens,
            seed=int(config["project"]["seed"]),
        )
        codec = ComplexSTFTCodec(STFTCodecConfig(**config["codec"]))
        probe_batches = min(len(dataset), int(config["training"].get("latent_probe_batches", 16)))
        latent_stds = []
        batch = None
        clean = None
        observed = None
        for index, raw_batch in enumerate(
            DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pairs)
        ):
            device_batch = batch_to_device(raw_batch, device)
            encoded_clean = codec.encode(device_batch["clean"].float())
            latent_stds.append(float(encoded_clean.std().item()))
            if batch is None:
                batch = device_batch
                clean = encoded_clean
                observed = codec.encode(device_batch["noisy"].float())
            if index + 1 >= probe_batches:
                break
        assert batch is not None and clean is not None and observed is not None
        model = model.to(device).eval()
        clean_latent_std = statistics.median(latent_stds)
        latent_scale_sane = 0.2 <= clean_latent_std <= 4.0
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            predicted = model(
                clean,
                observed,
                batch["fields"],
                batch["categories"],
                batch["values"],
                torch.full((1,), 0.5, device=device),
            )
        runtime = {
            "requested": True,
            "status": "PASSED",
            "device": str(device),
            "input_shape": list(clean.shape),
            "output_shape": list(predicted.shape),
            "finite": bool(torch.isfinite(predicted).all()),
            "clean_latent_mean": float(clean.mean().item()),
            "clean_latent_std": clean_latent_std,
            "clean_latent_std_min": min(latent_stds),
            "clean_latent_std_max": max(latent_stds),
            "latent_probe_batches": len(latent_stds),
            "clean_latent_abs_max": float(clean.abs().max().item()),
            "gpu_peak_memory_bytes": (
                torch.cuda.max_memory_allocated() if device.type == "cuda" else None
            ),
        }
        gates["runtime_forward"] = runtime["finite"] and clean.shape == predicted.shape
        gates["latent_scale_sane"] = latent_scale_sane
    report = {
        "schema_version": "lse.mmdit.preflight.v1",
        "status": "READY" if all(gates.values()) else "BLOCKED",
        "measurement_status": "NOT_MEASURED",
        "config": str(config_path.resolve()),
        "config_digest": config_digest(config),
        "manifest": str(manifest),
        "records": len(rows),
        "split_counts": dict(split_counts),
        "speakers": len(speakers),
        "speaker_leakage": dict(list(leakage.items())[:20]),
        "predicted_prescriptions": sum(
            row.get("predicted_prescription") is not None for row in rows
        ),
        "contract_errors": errors[:20],
        "parameters": parameters,
        "audio_tokens_per_stream": audio_tokens,
        "joint_attention_tokens": joint_tokens,
        "curriculum": [
            {
                "name": phase.name,
                "start_step": phase.start_step,
                "until_step": phase.until_step,
                "mrstft_weight": phase.mrstft_weight,
                "semantic_weight": phase.semantic_weight,
                "lr_scale": phase.lr_scale,
            }
            for phase in curriculum.phases
        ],
        "gates": gates,
        "runtime_smoke": runtime,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": _git(root, "rev-parse", "HEAD"),
            "git_dirty": bool(_git(root, "status", "--porcelain")),
        },
        "claim_boundary": (
            "Preflight verifies wiring only. Enhancement quality remains NOT_MEASURED until "
            "the held-out five-arm evaluation completes."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-smoke", action="store_true")
    args = parser.parse_args(argv)
    report = preflight(args.config, args.output, runtime_smoke=args.runtime_smoke)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
