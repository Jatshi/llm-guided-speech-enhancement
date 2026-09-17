"""Enhance one waveform with a trained prescription-conditioned MM-DiT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint
from .codec import ComplexSTFTCodec, STFTCodecConfig
from .contracts import prescription_tokens, validate_prescription
from .data import _load_audio, _resample
from .safety import EnhancementGate, EnhancementGateConfig


def enhance_file(
    checkpoint_path: Path,
    input_path: Path,
    prescription_path: Path,
    output_path: Path,
    *,
    sample_rate: int = 16000,
    steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int = 42,
    chunk_seconds: float = 2.0,
    overlap_seconds: float = 0.25,
) -> dict:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for waveform output") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, flow, payload = load_model_checkpoint(checkpoint_path, device=device)
    model.eval()
    prescription = json.loads(prescription_path.read_text(encoding="utf-8"))
    validate_prescription(prescription)
    waveform, source_rate = _load_audio(input_path)
    waveform = _resample(waveform, source_rate, sample_rate).unsqueeze(0).to(device)
    codec = ComplexSTFTCodec(STFTCodecConfig(**payload["codec_config"]))
    tokens = prescription_tokens(
        prescription, max_tokens=int(payload["model_config"]["prescription_tokens"])
    )
    fields, categories, values = zip(*tokens, strict=True)
    fields_t = torch.tensor(fields, device=device, dtype=torch.long).unsqueeze(0)
    categories_t = torch.tensor(categories, device=device, dtype=torch.long).unsqueeze(0)
    values_t = torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0)
    chunk_samples = round(chunk_seconds * sample_rate)
    overlap_samples = round(overlap_seconds * sample_rate)
    if chunk_samples <= overlap_samples or overlap_samples < 0:
        raise ValueError("require chunk_seconds > overlap_seconds >= 0")
    hop = chunk_samples - overlap_samples
    total = waveform.shape[-1]
    starts = list(range(0, max(1, total), hop))
    accumulation = torch.zeros_like(waveform)
    weights = torch.zeros_like(waveform)
    gate_reports = []
    gate = EnhancementGate(EnhancementGateConfig())
    for chunk_index, start in enumerate(starts):
        end = min(total, start + chunk_samples)
        valid = end - start
        chunk = waveform[:, start:end]
        if valid < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - valid))
        observed = codec.encode(chunk.float())
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            generated = flow.sample(
                model,
                observed,
                fields_t,
                categories_t,
                values_t,
                seed=seed + chunk_index,
                steps=steps,
                guidance_scale=guidance_scale,
            )
        raw_candidate = codec.decode(generated.float(), length=chunk_samples)
        selected, gate_report = gate.select(
            chunk,
            raw_candidate,
            planner_confidence=float(prescription.get("confidence", 0.0)),
        )
        gate_reports.append(gate_report)
        window = torch.ones(valid, device=device)
        fade = min(overlap_samples, valid)
        if start > 0 and fade:
            window[:fade] = torch.linspace(0, 1, fade, device=device)
        if end < total and fade:
            window[-fade:] = torch.minimum(
                window[-fade:], torch.linspace(1, 0, fade, device=device)
            )
        accumulation[:, start:end] += selected[:, :valid] * window
        weights[:, start:end] += window
        if end == total:
            break
    selected = accumulation / weights.clamp_min(1e-6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, selected.squeeze(0).cpu().numpy(), sample_rate)
    report = {
        "schema_version": "lse.mmdit.inference.v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": payload["step"],
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "sample_rate": sample_rate,
        "samples": selected.shape[-1],
        "chunks": len(gate_reports),
        "gate": {
            "fallback_chunks": sum(report["used_fallback"] for report in gate_reports),
            "reports": gate_reports,
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prescription", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.25)
    args = parser.parse_args(argv)
    report = enhance_file(
        args.checkpoint,
        args.input,
        args.prescription,
        args.output,
        sample_rate=args.sample_rate,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
