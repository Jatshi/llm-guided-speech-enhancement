#!/usr/bin/env bash
set -euo pipefail

PROJECT=${LSE_PROJECT_DIR:-/root/autodl-tmp/llm-guided-speech-enhancement}
VENV=${LSE_MMDIT_VENV:-/root/autodl-tmp/lse-mmdit-env}
: "${MMDIT_SOURCE_MANIFEST:?set MMDIT_SOURCE_MANIFEST to the paired source manifest}"

cd "$PROJECT"
source "$VENV/bin/activate"
mkdir -p data/mmdit results/mmdit-preflight
python -m lse_v2.mmdit.prepare \
  --input "$MMDIT_SOURCE_MANIFEST" \
  --output data/mmdit/pairs.base.jsonl \
  --check-files \
  --require-speaker-disjoint

if [[ -n "${MMDIT_PREDICTIONS:-}" ]]; then
  python -m lse_v2.mmdit.attach_predictions \
    --pairs data/mmdit/pairs.base.jsonl \
    --predictions "$MMDIT_PREDICTIONS" \
    --output data/mmdit/pairs.jsonl
elif [[ -n "${MMDIT_BASE_MODEL:-}" && -n "${MMDIT_PLANNER_ADAPTER:-}" ]]; then
  python -m lse_v2.mmdit.planner_export \
    --pairs data/mmdit/pairs.base.jsonl \
    --output data/mmdit/planner_predictions.jsonl \
    --base-model "$MMDIT_BASE_MODEL" \
    --adapter "$MMDIT_PLANNER_ADAPTER" \
    --split test \
    --batch-size "${MMDIT_PLANNER_BATCH_SIZE:-8}"
  python -m lse_v2.mmdit.attach_predictions \
    --pairs data/mmdit/pairs.base.jsonl \
    --predictions data/mmdit/planner_predictions.jsonl \
    --output data/mmdit/pairs.jsonl
else
  cp data/mmdit/pairs.base.jsonl data/mmdit/pairs.jsonl
  echo 'No MMDIT_PREDICTIONS supplied: predicted-prescription arm will be BLOCKED, not fabricated.'
fi
