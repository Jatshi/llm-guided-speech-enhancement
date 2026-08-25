"""Materialize real noisy waveforms with deterministic source-level splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import AUDIO_SCHEMA, validate_audio_record
from .io import read_jsonl, utc_now, write_json_atomic, write_jsonl


@dataclass(frozen=True, slots=True)
class MaterializationConfig:
    clean_manifest: Path
    output_dir: Path
    noise_manifest: Path | None = None
    rir_manifest: Path | None = None
    variants_per_source: int = 1
    max_sources: int | None = None
    sample_rate: int = 16_000
    max_duration_seconds: float = 8.0
    eval_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.variants_per_source <= 0:
            raise ValueError("variants_per_source must be positive")
        if self.max_sources is not None and self.max_sources <= 0:
            raise ValueError("max_sources must be positive")
        if self.sample_rate < 8_000 or self.sample_rate > 192_000:
            raise ValueError("sample_rate must be in [8000, 192000]")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        if self.eval_ratio < 0 or self.test_ratio < 0 or self.eval_ratio + self.test_ratio >= 1:
            raise ValueError("eval_ratio and test_ratio must be non-negative and sum below one")


@dataclass(frozen=True, slots=True)
class MaterializationReport:
    manifest: str
    materialized_audio_records: int
    clean_sources: int
    split_counts: dict[str, int]
    created_at: str


def _resolve_manifest_path(manifest: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def _load_catalog(
    path: Path | None, *, required_audio_key: str = "audio_path"
) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"catalog is empty: {path}")
    for index, row in enumerate(rows):
        raw = row.get(required_audio_key)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"catalog row {index} requires {required_audio_key}")
        resolved = _resolve_manifest_path(path, raw)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        row[required_audio_key] = str(resolved)
    return rows


def _source_split(identity: str, seed: int, eval_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < test_ratio:
        return "test"
    if value < test_ratio + eval_ratio:
        return "eval"
    return "train"


def _read_audio(path: Path, target_rate: int, max_samples: int) -> np.ndarray:
    import soundfile as sf  # type: ignore[import-untyped]
    from scipy.signal import resample_poly

    waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError(f"invalid clean/noise audio: {path}")
    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        waveform = resample_poly(waveform, target_rate // divisor, source_rate // divisor)
    waveform = np.asarray(waveform[:max_samples], dtype=np.float32)
    peak = float(np.max(np.abs(waveform)))
    return waveform / max(peak / 0.9, 1.0)


def _synthetic_noise(
    kind: str, length: int, sample_rate: int, rng: np.random.Generator
) -> np.ndarray:
    if kind == "pink":
        spectrum = np.fft.rfft(rng.standard_normal(length))
        spectrum /= np.sqrt(np.arange(spectrum.size) + 1)
        noise = np.fft.irfft(spectrum, n=length)
    elif kind == "hvac":
        time = np.arange(length) / sample_rate
        noise = sum(np.sin(2 * np.pi * frequency * time) for frequency in (120, 240, 360))
        noise += 0.15 * rng.standard_normal(length)
    elif kind == "cafe":
        noise = rng.standard_normal(length)
        noise = np.convolve(noise, np.ones(9) / 9, mode="same")
    else:
        noise = rng.standard_normal(length)
    return np.asarray(noise, dtype=np.float32)


def _fit_noise(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if noise.size < length:
        noise = np.tile(noise, math.ceil(length / noise.size))
    start = int(rng.integers(0, max(1, noise.size - length + 1)))
    return np.asarray(noise[start : start + length], dtype=np.float32)


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_power = float(np.mean(clean**2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    scale = math.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
    return np.asarray(clean + scale * noise, dtype=np.float32)


def _target_for(noise_type: str, snr_db: float, rt60: float | None, bandlimit: bool) -> dict:
    actions: list[dict[str, float | str]] = []
    if noise_type == "hvac":
        actions.append({"type": "notch", "low_hz": 100, "high_hz": 400, "q": 4, "reduction_db": 8})
    else:
        actions.append(
            {
                "type": "spectral_subtraction",
                "reduction_db": float(np.clip(14 - 0.35 * snr_db, 4, 14)),
                "low_hz": 80,
                "high_hz": 7_600,
            }
        )
    if rt60 is not None:
        actions.append({"type": "dereverb", "reduction_db": float(np.clip(rt60 * 12, 2, 8))})
    if bandlimit:
        actions.append(
            {"type": "bandwidth_extension", "low_hz": 3_400, "high_hz": 7_000, "gain_db": 2}
        )
    return {
        "diagnosis": {
            "noise_type": noise_type,
            "reverb": rt60 is not None,
            "band_limited": bandlimit,
        },
        "actions": actions,
        "rationale": (
            "Conservative prescription derived from a physically materialized degradation."
        ),
        "confidence": 0.75,
    }


def _measured_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Compute observable waveform evidence without exposing degradation labels."""

    frame_size = min(512, max(64, audio.size // 16))
    frames = [audio[index : index + frame_size] for index in range(0, audio.size, frame_size)]
    matrix = np.stack(
        [np.pad(frame, (0, frame_size - frame.size)) for frame in frames if frame.size]
    )
    magnitude = np.abs(np.fft.rfft(matrix * np.hanning(frame_size), axis=1)) + 1e-8
    frequencies = np.fft.rfftfreq(frame_size, 1 / sample_rate)
    mean_spectrum = magnitude.mean(axis=0)
    centroid = float(np.sum(frequencies * mean_spectrum) / np.sum(mean_spectrum))
    flatness = float(np.mean(np.exp(np.mean(np.log(magnitude), axis=1)) / magnitude.mean(axis=1)))
    return {
        "rms": round(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))), 6),
        "zero_crossing_rate": round(float(np.mean(np.diff(np.signbit(audio)) != 0)), 6),
        "spectral_flatness": round(flatness, 6),
        "spectral_centroid_hz": round(centroid, 3),
        "clipped_fraction": round(float(np.mean(np.abs(audio) >= 0.99)), 6),
    }


def materialize_dataset(config: MaterializationConfig) -> MaterializationReport:
    import soundfile as sf  # type: ignore[import-untyped]
    from scipy.signal import butter, fftconvolve, sosfiltfilt

    clean_manifest = config.clean_manifest.expanduser().resolve()
    clean_rows = _load_catalog(clean_manifest)
    if config.max_sources is not None:
        clean_rows = clean_rows[: config.max_sources]
    noises = _load_catalog(config.noise_manifest)
    rirs = _load_catalog(config.rir_manifest)
    output_dir = config.output_dir.expanduser().resolve()
    audio_dir = output_dir / "audio"
    clean_dir = output_dir / "clean"
    audio_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    max_samples = round(config.sample_rate * config.max_duration_seconds)
    for clean_index, row in enumerate(clean_rows):
        source_id = str(row.get("source_id", f"source-{clean_index:06d}"))
        speaker_id = str(row.get("speaker_id") or source_id)
        clean_source = Path(row["audio_path"])
        clean = _read_audio(clean_source, config.sample_rate, max_samples)
        clean_copy = clean_dir / f"{hashlib.sha256(source_id.encode()).hexdigest()[:16]}.wav"
        if not clean_copy.exists():
            sf.write(clean_copy, clean, config.sample_rate, subtype="PCM_16")
        split = _source_split(speaker_id, config.seed, config.eval_ratio, config.test_ratio)
        for variant in range(config.variants_per_source):
            seed_payload = f"{config.seed}:{source_id}:{variant}"
            sample_hash = hashlib.sha256(seed_payload.encode()).hexdigest()
            rng = np.random.default_rng(int(sample_hash[:16], 16))
            noise_type = ["white", "pink", "hvac", "cafe"][int(rng.integers(0, 4))]
            snr_db = round(float(rng.uniform(3, 25)), 2)
            rt60 = round(float(rng.uniform(0.15, 0.75)), 3) if rng.random() < 0.45 else None
            bandlimit = bool(rng.random() < 0.20)
            degraded = clean.copy()
            noise_provenance: dict[str, Any]
            if noises:
                noise_row = noises[int(rng.integers(0, len(noises)))]
                noise = _read_audio(
                    Path(noise_row["audio_path"]), config.sample_rate, max_samples * 2
                )
                noise_type = str(noise_row.get("noise_type") or noise_row.get("dataset") or "real")
                noise_provenance = {
                    "dataset": noise_row.get("dataset"),
                    "license": noise_row.get("license"),
                    "audio_path": noise_row["audio_path"],
                }
            else:
                noise = _synthetic_noise(noise_type, degraded.size, config.sample_rate, rng)
                noise_provenance = {"dataset": "procedural", "license": "generated"}
            if rt60 is not None:
                if rirs:
                    rir_row = rirs[int(rng.integers(0, len(rirs)))]
                    rir = _read_audio(Path(rir_row["audio_path"]), config.sample_rate, max_samples)
                else:
                    rir_length = max(32, round(rt60 * config.sample_rate))
                    rir = np.exp(-np.arange(rir_length) / max(rt60 * config.sample_rate / 6.91, 1))
                # Avoid a platform-specific MKL abort seen in ``np.linalg.norm`` on
                # long one-dimensional arrays.  This is the same L2 norm and keeps
                # materialisation deterministic across CPU environments.
                rir_energy = float(np.sqrt(np.sum(np.square(rir, dtype=np.float64))))
                rir = rir / (rir_energy + 1e-8)
                degraded = fftconvolve(degraded, rir, mode="full")[: clean.size].astype(np.float32)
            degraded = _mix_at_snr(degraded, _fit_noise(noise, degraded.size, rng), snr_db)
            if bandlimit:
                sos = butter(6, [300, 3400], btype="bandpass", fs=config.sample_rate, output="sos")
                degraded = sosfiltfilt(sos, degraded).astype(np.float32)
            peak = float(np.max(np.abs(degraded)))
            degraded = degraded / max(peak / 0.95, 1.0)
            sample_id = f"mat-{sample_hash[:20]}"
            noisy_path = audio_dir / f"{sample_id}.wav"
            sf.write(noisy_path, degraded, config.sample_rate, subtype="PCM_16")
            record = {
                "schema_version": AUDIO_SCHEMA,
                "sample_id": sample_id,
                "split": split,
                "audio": {
                    "noisy_path": str(noisy_path),
                    "clean_path": str(clean_copy),
                    "source_role": "materialized_noisy_audio",
                    "sample_rate": config.sample_rate,
                    "duration_seconds": round(degraded.size / config.sample_rate, 6),
                },
                "acoustics": {
                    "noise_type": noise_type,
                    "snr_db": snr_db,
                    "reverb_rt60": rt60,
                    "bandlimit_hz": [300, 3400] if bandlimit else None,
                    "features": _measured_features(degraded, config.sample_rate),
                },
                "target": _target_for(noise_type, snr_db, rt60, bandlimit),
                "provenance": {
                    "dataset": row.get("dataset"),
                    "license": row.get("license"),
                    "source_id": source_id,
                    "speaker_id": speaker_id,
                    "language": row.get("language"),
                    "device": row.get("device"),
                    "source_audio": str(clean_source),
                    "noise": noise_provenance,
                    "materialization_seed": int(sample_hash[:16], 16),
                },
            }
            validate_audio_record(record, check_files=True)
            records.append(record)
    manifest_path = output_dir / "audio_manifest.v2.jsonl"
    write_jsonl(manifest_path, records)
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "eval", "test")
        if any(record["split"] == split for record in records)
    }
    report = MaterializationReport(
        manifest=str(manifest_path),
        materialized_audio_records=len(records),
        clean_sources=len(clean_rows),
        split_counts=split_counts,
        created_at=utc_now(),
    )
    write_json_atomic(output_dir / "materialization_report.json", asdict(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-manifest", type=Path)
    parser.add_argument("--rir-manifest", type=Path)
    parser.add_argument("--variants-per-source", type=int, default=1)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--max-duration-seconds", type=float, default=8.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = materialize_dataset(
        MaterializationConfig(
            clean_manifest=args.clean_manifest,
            output_dir=args.output_dir,
            noise_manifest=args.noise_manifest,
            rir_manifest=args.rir_manifest,
            variants_per_source=args.variants_per_source,
            max_sources=args.max_sources,
            sample_rate=args.sample_rate,
            max_duration_seconds=args.max_duration_seconds,
            eval_ratio=args.eval_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
