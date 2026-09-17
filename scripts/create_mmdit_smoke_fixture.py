#!/usr/bin/env python3
"""Create a tiny, fully local paired-audio fixture for end-to-end smoke tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def create(output: Path, sample_rate: int = 16000) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    splits = ["train"] * 6 + ["validation"] * 2 + ["test"] * 2
    duration = 0.5
    timeline = np.arange(round(sample_rate * duration), dtype=np.float32) / sample_rate
    for index, split in enumerate(splits):
        base_frequency = 150 + 13 * index
        clean = 0.22 * np.sin(2 * math.pi * base_frequency * timeline) + 0.08 * np.sin(
            2 * math.pi * 2 * base_frequency * timeline
        )
        clean *= 0.7 + 0.3 * np.sin(2 * math.pi * 3 * timeline) ** 2
        rng = np.random.default_rng(1000 + index)
        if index % 2:
            noise = 0.05 * rng.standard_normal(clean.shape)
            noise_type = "white"
        else:
            noise = 0.07 * np.sin(2 * math.pi * 60 * timeline)
            noise_type = "hum"
        noisy = np.clip(clean + noise, -0.95, 0.95).astype(np.float32)
        clean = clean.astype(np.float32)
        noisy_path = output / f"sample-{index:02d}-noisy.wav"
        clean_path = output / f"sample-{index:02d}-clean.wav"
        sf.write(noisy_path, noisy, sample_rate)
        sf.write(clean_path, clean, sample_rate)
        prescription = {
            "diagnosis": {
                "noise_type": noise_type,
                "reverb": False,
                "band_limited": False,
            },
            "actions": [
                {
                    "type": "spectral_subtraction" if noise_type == "white" else "notch",
                    "reduction_db": 6.0 if noise_type == "white" else 4.0,
                    "low_hz": 50,
                    "high_hz": 7600,
                }
            ],
            "rationale": "Synthetic smoke prescription; not a quality claim.",
            "confidence": 0.85,
        }
        rows.append(
            {
                "schema_version": "lse.mmdit_pair.v1",
                "sample_id": f"smoke-{index:02d}",
                "speaker_id": f"speaker-{index:02d}",
                "split": split,
                "audio": {
                    "noisy_path": str(noisy_path.resolve()),
                    "clean_path": str(clean_path.resolve()),
                    "sample_rate": sample_rate,
                },
                "oracle_prescription": prescription,
                "predicted_prescription": prescription,
                "provenance": {"dataset": "generated-smoke", "quality_claim": False},
            }
        )
    manifest = output / "pairs.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/mmdit_smoke"))
    args = parser.parse_args()
    print(create(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
