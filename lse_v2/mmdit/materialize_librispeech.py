"""Build a deterministic speaker-disjoint paired corpus from LibriSpeech audio."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from lse_v2.io import write_jsonl

from .contracts import MMDIT_PAIR_SCHEMA, validate_pair_record

DEGRADATIONS = ("clean", "white", "pink", "reverb", "telephone")


def _rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def _seed(seed: int, value: str) -> int:
    return int.from_bytes(_rank(seed, value)[:8], "big")


def _speaker(path: Path) -> str:
    return path.stem.split("-", 1)[0]


def select_speaker_disjoint(
    paths: list[Path], counts: dict[str, int], *, seed: int
) -> dict[str, list[Path]]:
    """Fill split quotas without allowing a speaker in two splits."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[_speaker(path)].append(path)
    speakers = sorted(grouped, key=lambda value: _rank(seed, value))
    result = {split: [] for split in counts}
    cursor = 0
    for split, count in counts.items():
        while len(result[split]) < count and cursor < len(speakers):
            speaker = speakers[cursor]
            cursor += 1
            utterances = sorted(grouped[speaker], key=lambda path: _rank(seed + 1, str(path)))
            result[split].extend(utterances[: count - len(result[split])])
        if len(result[split]) != count:
            raise ValueError(
                f"not enough speaker-disjoint utterances for {split}: "
                f"got {len(result[split])}, need {count}"
            )
    return result


def _load_segment(path: Path, sample_rate: int, samples: int, seed: int) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    waveform, source_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = waveform.mean(axis=1)
    if source_rate != sample_rate:
        divisor = np.gcd(source_rate, sample_rate)
        waveform = resample_poly(waveform, sample_rate // divisor, source_rate // divisor)
    if waveform.size >= samples:
        start = seed % (waveform.size - samples + 1)
        waveform = waveform[start : start + samples]
    else:
        waveform = np.pad(waveform, (0, samples - waveform.size))
    waveform = waveform.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(waveform)))
    return waveform * (0.8 / peak) if peak > 1e-7 else waveform


def _colored_noise(rng: np.random.Generator, samples: int, pink: bool) -> np.ndarray:
    noise = rng.standard_normal(samples)
    if pink:
        spectrum = np.fft.rfft(noise)
        spectrum /= np.sqrt(np.arange(1, spectrum.size + 1))
        noise = np.fft.irfft(spectrum, n=samples)
    rms = float(np.sqrt(np.mean(noise**2)))
    return (noise / max(rms, 1e-8)).astype(np.float32)


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = float(np.sqrt(np.mean(clean**2)))
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    scale = clean_rms / max(noise_rms * 10 ** (snr_db / 20), 1e-8)
    return clean + noise * scale


def _degrade(
    clean: np.ndarray, kind: str, sample_rate: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    from scipy.signal import butter, fftconvolve, sosfiltfilt

    snr_db: float | None = None
    rt60: float | None = None
    band_limited = False
    if kind == "clean":
        noisy = clean.copy()
        actions = [
            {
                "type": "spectral_subtraction",
                "reduction_db": 0.0,
                "low_hz": 80,
                "high_hz": 7600,
            }
        ]
    elif kind in {"white", "pink"}:
        snr_db = float(rng.uniform(5.0, 20.0))
        noise = _colored_noise(rng, clean.size, pink=kind == "pink")
        noisy = _mix_at_snr(clean, noise, snr_db)
        actions = [
            {
                "type": "spectral_subtraction",
                "reduction_db": float(np.clip(18.0 - 0.5 * snr_db, 6.0, 14.0)),
                "low_hz": 80,
                "high_hz": 7600,
            }
        ]
    elif kind == "reverb":
        rt60 = float(rng.uniform(0.25, 0.8))
        ir_length = max(64, round(rt60 * sample_rate))
        time = np.arange(ir_length) / sample_rate
        impulse = rng.standard_normal(ir_length) * np.exp(-6.91 * time / rt60)
        impulse[0] += 8.0
        impulse /= np.sqrt(np.sum(impulse**2)) + 1e-8
        wet = fftconvolve(clean, impulse, mode="full")[: clean.size]
        noisy = 0.35 * clean + 0.65 * wet
        actions = [{"type": "dereverb", "reduction_db": float(3.0 + 5.0 * rt60)}]
    elif kind == "telephone":
        band_limited = True
        sos = butter(6, [300, 3400], btype="bandpass", fs=sample_rate, output="sos")
        noisy = sosfiltfilt(sos, clean)
        snr_db = float(rng.uniform(15.0, 25.0))
        noisy = _mix_at_snr(noisy, _colored_noise(rng, clean.size, pink=False), snr_db)
        actions = [
            {
                "type": "bandwidth_extension",
                "gain_db": 2.0,
                "low_hz": 300,
                "high_hz": 3400,
            }
        ]
    else:
        raise ValueError(f"unsupported degradation: {kind}")
    joint_peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(noisy))), 1.0)
    noisy = np.clip(noisy / joint_peak, -0.999, 0.999).astype(np.float32)
    clean = (clean / joint_peak).astype(np.float32)
    prescription = {
        "diagnosis": {
            "noise_type": kind,
            "reverb": kind == "reverb",
            "band_limited": band_limited,
        },
        "actions": actions,
        "rationale": "Deterministic prescription from the public paired-corpus protocol.",
        "confidence": 0.9,
    }
    acoustics = {
        "noise_type": kind,
        "snr_db": None if snr_db is None else round(snr_db, 4),
        "reverb_rt60": None if rt60 is None else round(rt60, 4),
        "bandlimit_hz": [300, 3400] if band_limited else None,
    }
    return noisy, prescription, acoustics


def materialize(
    librispeech_root: Path,
    output_root: Path,
    *,
    train: int,
    validation: int,
    test: int,
    seed: int = 42,
    sample_rate: int = 16000,
    seconds: float = 4.0,
) -> dict[str, Any]:
    import soundfile as sf

    paths = sorted(librispeech_root.rglob("*.flac"))
    if not paths:
        raise FileNotFoundError(f"no FLAC files below {librispeech_root}")
    counts = {"train": train, "validation": validation, "test": test}
    if any(value < len(DEGRADATIONS) or value % len(DEGRADATIONS) for value in counts.values()):
        raise ValueError(f"split counts must be positive multiples of {len(DEGRADATIONS)}")
    selected = select_speaker_disjoint(paths, counts, seed=seed)
    clean_dir = output_root / "clean"
    noisy_dir = output_root / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    segment_samples = round(sample_rate * seconds)
    rows = []
    for split, utterances in selected.items():
        ordered = sorted(utterances, key=lambda path: _rank(seed + 2, str(path)))
        for index, source in enumerate(ordered):
            kind = DEGRADATIONS[index % len(DEGRADATIONS)]
            sample_id = f"lsdev-{split}-{source.stem}-{kind}"
            item_seed = _seed(seed, sample_id)
            rng = np.random.default_rng(item_seed)
            clean = _load_segment(source, sample_rate, segment_samples, item_seed)
            noisy, prescription, acoustics = _degrade(clean, kind, sample_rate, rng)
            clean_path = (clean_dir / f"{sample_id}.wav").resolve()
            noisy_path = (noisy_dir / f"{sample_id}.wav").resolve()
            sf.write(clean_path, clean, sample_rate, subtype="PCM_16")
            sf.write(noisy_path, noisy, sample_rate, subtype="PCM_16")
            row = {
                "schema_version": MMDIT_PAIR_SCHEMA,
                "sample_id": sample_id,
                "speaker_id": _speaker(source),
                "split": split,
                "audio": {
                    "noisy_path": str(noisy_path),
                    "clean_path": str(clean_path),
                    "sample_rate": sample_rate,
                },
                "oracle_prescription": prescription,
                "predicted_prescription": None,
                "provenance": {
                    "dataset": "LibriSpeech dev-clean",
                    "license": "CC BY 4.0",
                    "source_audio": str(source.resolve()),
                    "materialization_seed": item_seed,
                    "degradation": acoustics,
                },
            }
            validate_pair_record(row, check_files=True)
            rows.append(row)
    manifest = output_root / "pairs.jsonl"
    write_jsonl(manifest, rows)
    split_speakers = {
        split: {row["speaker_id"] for row in rows if row["split"] == split} for split in counts
    }
    leakage = {
        speaker
        for first_index, first in enumerate(counts)
        for second in list(counts)[first_index + 1 :]
        for speaker in split_speakers[first] & split_speakers[second]
    }
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report = {
        "schema_version": "lse.mmdit.public_corpus.v1",
        "manifest": str(manifest.resolve()),
        "sha256": digest,
        "records": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "degradation_counts": dict(
            Counter(row["provenance"]["degradation"]["noise_type"] for row in rows)
        ),
        "speakers_by_split": {key: len(value) for key, value in split_speakers.items()},
        "speaker_leakage": sorted(leakage),
        "sample_rate": sample_rate,
        "seconds": seconds,
        "seed": seed,
    }
    (output_root / "corpus_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if leakage:
        raise RuntimeError(f"speaker leakage detected: {sorted(leakage)[:5]}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--validation", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize(
                args.librispeech_root,
                args.output_root,
                train=args.train,
                validation=args.validation,
                test=args.test,
                seed=args.seed,
                sample_rate=args.sample_rate,
                seconds=args.seconds,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
