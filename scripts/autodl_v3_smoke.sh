#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${LSE_V3_VENV:-/root/autodl-tmp/portfolio-v3/envs/audio-v3}"
SMOKE_DIR="${LSE_V3_SMOKE_DIR:-$REPO_ROOT/artifacts/v3/smoke}"
DATA_DIR="$SMOKE_DIR/data"
source "$VENV_DIR/bin/activate"
cd "$REPO_ROOT"
mkdir -p "$SMOKE_DIR"
python -m pytest -q tests/test_audio_conditioning_v3.py tests/test_closed_loop_v3.py
python scripts/build_native_audio_smoke_v3.py --output-dir "$DATA_DIR"
python scripts/train_native_audio_v3.py \
  --manifest "$DATA_DIR/manifest.jsonl" \
  --output-dir "$SMOKE_DIR/native_audio" \
  --epochs 1 \
  --pooling-stride 32 \
  --max-target-tokens 96
test -s "$SMOKE_DIR/native_audio/audio_projector.pt"
test -s "$SMOKE_DIR/native_audio/run_manifest.json"
