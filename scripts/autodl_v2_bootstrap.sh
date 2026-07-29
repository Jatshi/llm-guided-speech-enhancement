#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${PORTFOLIO_V2_MODE:-full}"
VENV_DIR="${LSE_V2_VENV:-/root/autodl-tmp/lse-v2-venv}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/.cache/pip}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/.cache/torch}"
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}"

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip==25.0.1" "setuptools>=69" wheel

if [[ "$MODE" == "smoke" ]]; then
  python -m pip install -e "$REPO_ROOT[test]"
else
  python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1"
  python -m pip install -e "$REPO_ROOT[train,audio,test]"
fi

python -m pip check
python - <<'PY'
import lse_v2
print("lse_v2", lse_v2.__version__)
PY

echo "BOOTSTRAP_OK mode=$MODE venv=$VENV_DIR"
