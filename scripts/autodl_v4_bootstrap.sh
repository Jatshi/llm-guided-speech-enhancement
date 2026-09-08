#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${LSE_V4_VENV:-/root/autodl-tmp/lse-v4-env}"

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install -e "$REPO_ROOT[train,audio,metrics,serve,test]"
python -m pip check
cd "$REPO_ROOT"
python -m pytest -q "$REPO_ROOT/tests"
echo "Environment ready at $VENV_DIR"
