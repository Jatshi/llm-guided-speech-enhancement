#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${PORTFOLIO_V2_MODE:-full}"
VENV_DIR="${LSE_V2_VENV:-/root/autodl-tmp/lse-v2-venv}"
CONFIG="${LSE_V2_CONFIG:-$REPO_ROOT/configs/autodl_4090.json}"
DATA_DIR="$REPO_ROOT/data/v2"
MANIFEST_DEFAULT="$REPO_ROOT/data/training/audio_manifest.v2.jsonl"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
fi
cd "$REPO_ROOT"

if [[ "$MODE" == "smoke" ]]; then
  CONFIG="$REPO_ROOT/configs/smoke.json"
  export LSE_V2_CONFIG="$CONFIG"
  SMOKE_DATA="$REPO_ROOT/outputs/smoke/data"
  python -m lse_v2.data_cli build \
    --manifest "$REPO_ROOT/examples/audio_manifest.smoke.jsonl" \
    --output-dir "$SMOKE_DATA" \
    --seed 42
  bash "$REPO_ROOT/scripts/autodl_v2_preflight.sh"
  python -m lse_v2.pipeline --config "$CONFIG" --dry-run
  python -m pytest
  echo "RUN_OK mode=smoke (no GPU training was performed)"
  exit 0
fi

if [[ -n "${AUDIO_MANIFEST:-}" ]]; then
  test -f "$AUDIO_MANIFEST" || { echo "AUDIO_MANIFEST not found: $AUDIO_MANIFEST" >&2; exit 2; }
  mkdir -p "$(dirname "$MANIFEST_DEFAULT")"
  cp "$AUDIO_MANIFEST" "$MANIFEST_DEFAULT"
fi

if [[ ! -f "$MANIFEST_DEFAULT" ]]; then
  LEGACY="${LSE_V2_LEGACY_METADATA:-$REPO_ROOT/data/training/metadata.json}"
  if [[ ! -f "$LEGACY" ]]; then
    echo "No v2 audio manifest or legacy metadata found." >&2
    echo "Set AUDIO_MANIFEST=/absolute/path/audio_manifest.v2.jsonl" >&2
    exit 2
  fi
  python -m lse_v2.data_cli migrate \
    --legacy-metadata "$LEGACY" \
    --output "$MANIFEST_DEFAULT"
fi

python -m lse_v2.data_cli build \
  --manifest "$MANIFEST_DEFAULT" \
  --output-dir "$DATA_DIR" \
  --seed 42 \
  --eval-ratio 0.05

bash "$REPO_ROOT/scripts/autodl_v2_preflight.sh"
python -m lse_v2.pipeline --config "$CONFIG" --resume auto
echo "RUN_OK mode=full"
