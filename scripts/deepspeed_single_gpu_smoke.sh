#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${LSE_V2_SMOKE_CONFIG:-$REPO_ROOT/configs/smoke.json}"
DS_CONFIG="${LSE_V2_DEEPSPEED_CONFIG:-$REPO_ROOT/configs/deepspeed/ds_zero2.json}"
VENV_DIR="${LSE_V2_VENV:-/root/autodl-tmp/lse-v2-venv}"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
fi
cd "$REPO_ROOT"
command -v deepspeed >/dev/null || {
  echo "DeepSpeed is not installed; run autodl_v2_bootstrap.sh in full mode." >&2
  exit 2
}
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WORLD_SIZE=1
python -m lse_v2.data_cli build \
  --manifest examples/audio_manifest.smoke.jsonl \
  --output-dir outputs/smoke/data \
  --seed 42

deepspeed --num_gpus=1 scripts/deepspeed_stage_entry.py \
  --config "$CONFIG" \
  --stage sft \
  --resume never \
  --deepspeed "$DS_CONFIG"

test -f outputs/smoke/sft/final/adapter_config.json
echo "DEEPSPEED_SINGLE_GPU_SMOKE_OK world_size=1 stage=sft config=$DS_CONFIG"

