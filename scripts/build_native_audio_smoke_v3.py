"""Generate a redistributable two-example native-audio smoke manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    rows = []
    for index, frequency in enumerate((220.0, 440.0), start=1):
        rng = np.random.default_rng(index)
        clean = 0.08 * np.sin(2 * np.pi * frequency * time)
        noisy = clean + 0.015 * rng.standard_normal(time.shape)
        audio = args.output_dir / f"smoke_{index}.wav"
        sf.write(audio, noisy.astype(np.float32), sample_rate)
        target = {
            "diagnosis": {"noise_type": "broadband", "reverb": False},
            "actions": [{"type": "spectral_gate", "strength": 0.2}],
            "confidence": 0.7,
        }
        rows.append(
            {
                "sample_id": f"native-smoke-{index}",
                "audio_path": audio.name,
                "target_text": json.dumps(target, ensure_ascii=False, sort_keys=True),
            }
        )
    manifest = args.output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
