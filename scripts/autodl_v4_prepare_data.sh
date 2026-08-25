#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${LSE_V4_VENV:-/root/autodl-tmp/lse-v4-env}"
DATA_ROOT="${LSE_DATA_ROOT:-/root/autodl-tmp/lse-v4-data}"
CLEAN_MANIFEST="${LSE_CLEAN_MANIFEST:?Set LSE_CLEAN_MANIFEST to a licensed clean-audio JSONL catalog}"
NOISE_MANIFEST="${LSE_NOISE_MANIFEST:-}"
RIR_MANIFEST="${LSE_RIR_MANIFEST:-}"
MAX_SOURCES="${LSE_MAX_SOURCES:-10000}"
VARIANTS="${LSE_VARIANTS_PER_SOURCE:-2}"
PIPELINE_CONFIG="${LSE_CONFIG:-$REPO_ROOT/configs/native_audio_4090.json}"

source "$VENV_DIR/bin/activate"
mkdir -p "$DATA_ROOT/materialized" "$REPO_ROOT/data/native"

ARGS=(
  --clean-manifest "$CLEAN_MANIFEST"
  --output-dir "$DATA_ROOT/materialized"
  --max-sources "$MAX_SOURCES"
  --variants-per-source "$VARIANTS"
  --seed 42
  --eval-ratio 0.1
  --test-ratio 0.1
)
if [[ -n "$NOISE_MANIFEST" ]]; then ARGS+=(--noise-manifest "$NOISE_MANIFEST"); fi
if [[ -n "$RIR_MANIFEST" ]]; then ARGS+=(--rir-manifest "$RIR_MANIFEST"); fi

python -m lse_v2.materialization "${ARGS[@]}"
python -m lse_v2.native_data \
  --audio-manifest "$DATA_ROOT/materialized/audio_manifest.v2.jsonl" \
  --output "$REPO_ROOT/data/native/native_audio.v1.jsonl"
python -m lse_v2.native_pipeline \
  --config "$PIPELINE_CONFIG" \
  --dry-run
