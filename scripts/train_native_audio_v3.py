"""CLI for native audio-prefix projector training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lse_v2.native_audio_training import NativeAudioTrainConfig, train_native_audio_projector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--whisper-model", default="openai/whisper-small")
    parser.add_argument("--language-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--pooling-stride", type=int, default=16)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    args = parser.parse_args()
    report = train_native_audio_projector(
        NativeAudioTrainConfig(
            manifest=args.manifest,
            output_dir=args.output_dir,
            whisper_model=args.whisper_model,
            language_model=args.language_model,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            pooling_stride=args.pooling_stride,
            max_target_tokens=args.max_target_tokens,
        )
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
