#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${LSE_DATA_ROOT:-/root/autodl-tmp/lse-v4-data}"
MIN_FREE_GIB="${LSE_MIN_FREE_GIB:-80}"
CONNECTIVITY_URL="${LSE_CONNECTIVITY_URL:-https://huggingface.co/}"

python - <<'PY'
import shutil
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible")
props = torch.cuda.get_device_properties(0)
print(f"GPU={props.name} VRAM_GiB={props.total_memory / 1024**3:.2f}")
if props.total_memory < 20 * 1024**3:
    raise SystemExit("at least 20 GiB VRAM is required by the checked-in profile")
PY

mkdir -p "$DATA_ROOT"
FREE_GIB="$(df -Pk "$DATA_ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
echo "DATA_ROOT=$DATA_ROOT FREE_GiB=$FREE_GIB REQUIRED_GiB=$MIN_FREE_GIB"
if (( FREE_GIB < MIN_FREE_GIB )); then
  echo "Insufficient free disk. No downloads or training were started." >&2
  exit 2
fi

python -m pip check
python -m lse_v2.native_pipeline --help >/dev/null
echo "CONNECTIVITY_URL=$CONNECTIVITY_URL"
curl --fail --location --max-time 20 --output /dev/null "$CONNECTIVITY_URL"
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$REPO_ROOT" status --short
else
  echo "SOURCE_TREE=archive (no .git metadata)"
fi
nvidia-smi
