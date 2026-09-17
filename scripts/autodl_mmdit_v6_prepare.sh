#!/usr/bin/env bash
set -euo pipefail

PROJECT=${LSE_PROJECT_DIR:-/root/autodl-tmp/llm-guided-speech-enhancement}
VENV=${LSE_MMDIT_VENV:-/root/autodl-tmp/lse-mmdit-env}
SOURCE=${MMDIT_V6_SOURCE:-data/mmdit/pairs.hybrid_safe_predictions.jsonl}
TARGET=data/mmdit/pairs.hybrid_safe_predictions.enhance_script.jsonl

cd "$PROJECT"
if [[ ! -x "$VENV/bin/python" ]]; then
  bash scripts/autodl_mmdit_bootstrap.sh
fi
source "$VENV/bin/activate"
if ! python -c 'import df, pytest, ruff, soundfile' >/dev/null 2>&1; then
  deactivate
  bash scripts/autodl_mmdit_bootstrap.sh
  source "$VENV/bin/activate"
fi
python -m pip install -e . --no-deps
if ! python -c 'import transformers; assert transformers.__version__ == "4.48.3"' \
  >/dev/null 2>&1; then
  python -m pip install 'transformers==4.48.3'
fi
python -m pytest tests/test_mmdit_v6_stepaudio3.py -q
python -m ruff check lse_v2/mmdit tests/test_mmdit_v6_stepaudio3.py

if [[ ! -s "$SOURCE" ]]; then
  echo "Missing v6 source manifest: $SOURCE" >&2
  echo 'Restore the measured MM-DiT data before starting paid training.' >&2
  exit 2
fi
python -m lse_v2.mmdit.enhance_script \
  --input "$SOURCE" \
  --output "$TARGET" \
  --duration-ms 2000
python scripts/probe_mmdit_v6_teacher.py \
  --output results/mmdit-v6-stepaudio3/semantic_teacher_probe.json
python -m lse_v2.mmdit.preflight \
  --config configs/mmdit_v6_stepaudio3_32gb.json \
  --output results/mmdit-v6-stepaudio3/preflight.json \
  --runtime-smoke
echo 'MM-DiT v6 formal data, teacher, and runtime preflight are ready.'
