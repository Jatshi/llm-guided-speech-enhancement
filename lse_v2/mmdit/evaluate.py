"""Five-arm causal evaluation for the prescription-conditioned MM-DiT."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_model_checkpoint
from .codec import ComplexSTFTCodec, STFTCodecConfig
from .config import load_mmdit_config, resolve_path
from .contracts import prescription_tokens
from .data import PairedEnhancementDataset, batch_to_device, collate_pairs
from .metrics import finite_or_none, log_spectral_distance, optional_pesq_stoi, si_sdr, snr
from .safety import EnhancementGate, EnhancementGateConfig


def _token_tensors(prescription: dict[str, Any] | None, count: int, device: torch.device):
    tokens = prescription_tokens(prescription, max_tokens=count)
    fields, categories, values = zip(*tokens, strict=True)
    return (
        torch.tensor(fields, device=device, dtype=torch.long).unsqueeze(0),
        torch.tensor(categories, device=device, dtype=torch.long).unsqueeze(0),
        torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0),
    )


def _spectral_subtraction(
    noisy: torch.Tensor,
    codec: ComplexSTFTCodec,
    prescription: dict[str, Any],
    sample_rate: int,
) -> torch.Tensor:
    reduction = max(
        (
            float(action.get("reduction_db", 0.0))
            for action in prescription.get("actions", [])
            if isinstance(action, dict)
        ),
        default=0.0,
    )
    strength = max(0.0, min(2.0, reduction / 12.0))
    latent = codec.encode(noisy)
    spectrum = torch.complex(latent[:, 0], latent[:, 1])
    magnitude, phase = spectrum.abs(), spectrum.angle()
    noise_frames = max(
        1, min(magnitude.shape[-1], round(0.3 * sample_rate / codec.config.hop_length))
    )
    noise = magnitude[..., :noise_frames].mean(dim=-1, keepdim=True)
    enhanced = (magnitude - strength * noise).clamp_min(0) * torch.exp(1j * phase)
    packed = torch.stack((enhanced.real, enhanced.imag), dim=1)
    return codec.decode(packed, length=noisy.shape[-1])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    result = []
    for variant, items in sorted(grouped.items()):
        summary: dict[str, Any] = {"variant": variant, "records": len(items)}
        for metric in (
            "si_sdr",
            "si_sdr_improvement",
            "snr",
            "snr_improvement",
            "lsd",
            "raw_si_sdr",
            "raw_si_sdr_improvement",
            "raw_snr",
            "raw_snr_improvement",
            "raw_lsd",
            "pesq",
            "stoi",
            "latency_ms",
        ):
            values = [float(item[metric]) for item in items if item.get(metric) not in {None, ""}]
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            summary[f"{metric}_median"] = statistics.median(values) if values else ""
        summary["fallback_rate"] = statistics.fmean(
            float(bool(item.get("used_fallback"))) for item in items
        )
        result.append(summary)
    return result


def _degradation_summaries(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Keep the clean safety slice separate from the enhancement task slices."""

    degradation_types = sorted({str(row.get("degradation_type", "unknown")) for row in rows})
    grouped = {
        degradation: _summaries(
            [row for row in rows if str(row.get("degradation_type", "unknown")) == degradation]
        )
        for degradation in degradation_types
    }
    grouped["corrupted_only"] = _summaries(
        [row for row in rows if str(row.get("degradation_type", "unknown")) != "clean"]
    )
    return grouped


@torch.inference_mode()
def evaluate(config_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    config = load_mmdit_config(config_path)
    evaluation = config["evaluation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, flow, checkpoint = load_model_checkpoint(checkpoint_path, device=device)
    model.eval()
    codec = ComplexSTFTCodec(STFTCodecConfig(**checkpoint["codec_config"]))
    gate = EnhancementGate(EnhancementGateConfig(**evaluation.get("gate", {})))
    deepfilter = None
    if evaluation.get("deepfilternet", {}).get("enabled", False):
        from .baselines import DeepFilterNetBaseline

        deepfilter = DeepFilterNetBaseline(
            str(evaluation.get("deepfilternet", {}).get("model", "DeepFilterNet3"))
        )
    dataset = PairedEnhancementDataset(
        resolve_path(config, config["data"]["manifest"]),
        split="test",
        sample_rate=int(config["data"]["sample_rate"]),
        segment_seconds=float(config["data"]["segment_seconds"]),
        prescription_token_count=int(config["model"]["prescription_tokens"]),
        seed=int(config["project"]["seed"]),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pairs)
    output = resolve_path(config, evaluation["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    max_samples = min(len(dataset), int(evaluation.get("max_samples", len(dataset))))
    token_count = int(config["model"]["prescription_tokens"])
    steps = int(evaluation.get("sampling_steps", config["flow"]["steps"]))
    guidance = float(evaluation.get("guidance_scale", config["flow"].get("guidance_scale", 1)))
    for index, raw in enumerate(loader):
        if index >= max_samples:
            break
        batch = batch_to_device(raw, device)
        noisy, clean = batch["noisy"], batch["clean"]
        record = raw["record"][0]
        degradation = record.get("provenance", {}).get("degradation", {})
        observed = codec.encode(noisy.float())
        input_si_sdr = si_sdr(noisy, clean)
        input_snr = snr(noisy, clean)
        variants: list[tuple[str, dict[str, Any] | None]] = [
            ("spectral_subtraction", record["oracle_prescription"]),
            ("mmdit_no_prescription", None),
            ("mmdit_oracle_prescription", record["oracle_prescription"]),
            (
                "mmdit_shuffled_prescription",
                dataset.records[(index + 1) % len(dataset.records)]["oracle_prescription"],
            ),
        ]
        if deepfilter is not None:
            variants.insert(1, ("deepfilternet3", None))
        if record.get("predicted_prescription") is not None:
            variants.append(("mmdit_predicted_prescription", record["predicted_prescription"]))
        for variant, prescription in variants:
            started = time.perf_counter()
            if variant == "spectral_subtraction":
                raw_candidate = _spectral_subtraction(
                    noisy, codec, prescription or {}, int(config["data"]["sample_rate"])
                )
            elif variant == "deepfilternet3":
                raw_candidate = deepfilter(noisy, int(config["data"]["sample_rate"]))
            else:
                fields, categories, values = _token_tensors(prescription, token_count, device)
                generated = flow.sample(
                    model,
                    observed,
                    fields,
                    categories,
                    values,
                    seed=int(config["project"]["seed"]) + index,
                    steps=steps,
                    guidance_scale=guidance,
                )
                raw_candidate = codec.decode(generated.float(), length=noisy.shape[-1])
            latency = (time.perf_counter() - started) * 1000
            raw_si_sdr = si_sdr(raw_candidate, clean)
            raw_snr = snr(raw_candidate, clean)
            raw_lsd = log_spectral_distance(raw_candidate, clean, codec)
            confidence = float((prescription or {}).get("confidence", 0.0))
            if variant in {"spectral_subtraction", "deepfilternet3"}:
                candidate, gate_report = raw_candidate, {"used_fallback": False, "reasons": []}
            else:
                gate_confidence = 1.0 if variant == "mmdit_no_prescription" else confidence
                candidate, gate_report = gate.select(
                    noisy, raw_candidate, planner_confidence=gate_confidence
                )
            objective = optional_pesq_stoi(candidate, clean, int(config["data"]["sample_rate"]))
            candidate_si_sdr = si_sdr(candidate, clean)
            candidate_snr = snr(candidate, clean)
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "variant": variant,
                    "degradation_type": degradation.get("noise_type", "unknown"),
                    "degradation_snr_db": degradation.get("snr_db", ""),
                    "degradation_reverb_rt60": degradation.get("reverb_rt60", ""),
                    "degradation_bandlimit_hz": degradation.get("bandlimit_hz", ""),
                    "input_si_sdr": finite_or_none(input_si_sdr),
                    "input_snr": finite_or_none(input_snr),
                    "si_sdr": finite_or_none(candidate_si_sdr),
                    "si_sdr_improvement": finite_or_none(candidate_si_sdr - input_si_sdr),
                    "snr": finite_or_none(candidate_snr),
                    "snr_improvement": finite_or_none(candidate_snr - input_snr),
                    "lsd": finite_or_none(log_spectral_distance(candidate, clean, codec)),
                    "raw_si_sdr": finite_or_none(raw_si_sdr),
                    "raw_si_sdr_improvement": finite_or_none(raw_si_sdr - input_si_sdr),
                    "raw_snr": finite_or_none(raw_snr),
                    "raw_snr_improvement": finite_or_none(raw_snr - input_snr),
                    "raw_lsd": finite_or_none(raw_lsd),
                    **objective,
                    "latency_ms": latency,
                    "used_fallback": gate_report["used_fallback"],
                    "fallback_reasons": json.dumps(gate_report["reasons"]),
                    "gate_peak": gate_report.get("peak", ""),
                    "gate_energy_ratio": gate_report.get("energy_ratio", ""),
                    "gate_change_rms_ratio": gate_report.get("change_rms_ratio", ""),
                    "planner_confidence": confidence,
                    "sampling_steps": (
                        0 if variant in {"spectral_subtraction", "deepfilternet3"} else steps
                    ),
                }
            )
        print(f"evaluated {index + 1}/{max_samples}", flush=True)
    summaries = _summaries(rows)
    _write_csv(output / "per_sample.csv", rows)
    _write_csv(output / "summary.csv", summaries)
    predicted_count = sum(row["variant"] == "mmdit_predicted_prescription" for row in rows)
    report = {
        "schema_version": "lse.mmdit.evaluation.v1",
        "status": "MEASURED" if rows else "BLOCKED",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": checkpoint["step"],
        "test_records": max_samples,
        "rows": len(rows),
        "predicted_prescription_records": predicted_count,
        "predicted_arm_status": "MEASURED"
        if predicted_count == max_samples
        else "PARTIAL_OR_BLOCKED",
        "metrics": {
            "pesq": "optional; blank when package unavailable",
            "stoi": "optional; blank when package unavailable",
            "si_sdr": "dependency-light paired objective metric",
            "lsd": "complex-STFT log-spectral distance",
        },
        "summary": summaries,
        "summary_by_degradation": _degradation_summaries(rows),
        "interpretation_note": (
            "Use corrupted_only and per-degradation summaries for enhancement quality; "
            "clean identity pairs are a safety slice and can dominate mean SI-SDR improvement."
        ),
    }
    (output / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.config, args.checkpoint), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
