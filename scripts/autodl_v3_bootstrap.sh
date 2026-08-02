#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LSE_V2_VENV="${LSE_V3_VENV:-/root/autodl-tmp/portfolio-v3/envs/audio-v3}"
export PORTFOLIO_V2_MODE=full
mkdir -p "$REPO_ROOT/artifacts/v3"
bash "$REPO_ROOT/scripts/autodl_v2_bootstrap.sh"
source "$LSE_V2_VENV/bin/activate"
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -m pip freeze > "$REPO_ROOT/artifacts/v3/environment.freeze.txt"
