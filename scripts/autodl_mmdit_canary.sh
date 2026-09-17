#!/usr/bin/env bash
set -euo pipefail

PROJECT=${LSE_PROJECT_DIR:-/root/autodl-tmp/llm-guided-speech-enhancement}
VENV=${LSE_MMDIT_VENV:-/root/autodl-tmp/lse-mmdit-env}
BASE_CONFIG=${MMDIT_CONFIG:-configs/mmdit_4090.json}
cd "$PROJECT"
source "$VENV/bin/activate"
if [[ -e outputs/mmdit-canary/train_metrics.jsonl ]]; then
  echo 'Canary outputs already exist. Archive or remove the dedicated canary directory explicitly.' >&2
  exit 2
fi
mkdir -p outputs/mmdit-canary results/mmdit-canary data/mmdit

python -m lse_v2.mmdit.subset \
  --source data/mmdit/pairs.jsonl \
  --output data/mmdit/canary.jsonl \
  --train 32 --validation 8 --test 20 --seed 42
python scripts/make_mmdit_canary_config.py \
  --base "$BASE_CONFIG" \
  --output configs/mmdit_canary.runtime.json \
  --max-steps 200
python -m lse_v2.mmdit.preflight \
  --config configs/mmdit_canary.runtime.json \
  --output results/mmdit-canary/preflight.json \
  --runtime-smoke
python -u -m lse_v2.mmdit.train --config configs/mmdit_canary.runtime.json \
  2>&1 | tee outputs/mmdit-canary/canary_train.log
python scripts/check_mmdit_canary.py \
  --metrics outputs/mmdit-canary/train_metrics.jsonl \
  --output results/mmdit-canary/canary_gate.json
python -m lse_v2.mmdit.evaluate \
  --config configs/mmdit_canary.runtime.json \
  --checkpoint outputs/mmdit-canary/checkpoint-best.pt
