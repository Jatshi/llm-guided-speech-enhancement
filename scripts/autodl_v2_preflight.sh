#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${PORTFOLIO_V2_MODE:-full}"
CONFIG="${LSE_V2_CONFIG:-$REPO_ROOT/configs/autodl_4090.json}"

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "PORTFOLIO_V2_MODE must be smoke or full" >&2
  exit 2
fi

python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required")
print("python", sys.version.split()[0])
PY

test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 2; }
python -m lse_v2.deepspeed \
  --config "$CONFIG" \
  --stage sft \
  --stage dpo \
  --stage grpo \
  --world-size 1 >/dev/null
DATA_STATUS="validated"
python -m lse_v2.training --config "$CONFIG" --stage sft --dry-run >/dev/null 2>&1 || {
  DATA_STATUS="deferred"
  if [[ "$MODE" == "full" ]]; then
    echo "Full-mode datasets are not ready yet; autodl_v2_run.sh can build them from AUDIO_MANIFEST or legacy metadata." >&2
  fi
}

if [[ "$MODE" == "full" ]]; then
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
  VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ -z "$VRAM_MB" || "$VRAM_MB" -lt 20000 ]]; then
    echo "At least 20 GB GPU memory is required by the conservative full config; found ${VRAM_MB:-unknown} MB" >&2
    exit 2
  fi
  FREE_GB="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
  if [[ "$FREE_GB" -lt 35 ]]; then
    echo "At least 35 GB free disk is required; found ${FREE_GB} GB" >&2
    exit 2
  fi
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA")
print("gpu", torch.cuda.get_device_name(0))
print("cuda", torch.version.cuda)
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY
fi

echo "PREFLIGHT_OK mode=$MODE data=$DATA_STATUS config=$CONFIG"
