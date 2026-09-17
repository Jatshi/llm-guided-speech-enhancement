#!/usr/bin/env bash
set -euo pipefail

PROJECT=${LSE_PROJECT_DIR:-/root/autodl-tmp/llm-guided-speech-enhancement}
VENV=${LSE_MMDIT_VENV:-/root/autodl-tmp/lse-mmdit-env}
CONFIG=${MMDIT_CONFIG:-configs/mmdit_4090.json}
cd "$PROJECT"
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/autodl_mmdit_smoke.sh
if [[ -s results/mmdit-canary/canary_gate.json ]] && \
   grep -q '"status": "PASSED"' results/mmdit-canary/canary_gate.json; then
  echo 'Existing passed canary gate accepted.'
else
  bash scripts/autodl_mmdit_canary.sh
fi
python -m lse_v2.mmdit.preflight \
  --config "$CONFIG" \
  --output results/mmdit-preflight/formal.json \
  --runtime-smoke

OUTPUT_DIR=$(python - "$CONFIG" <<'PY'
import sys
from lse_v2.mmdit.config import load_mmdit_config, resolve_path
c=load_mmdit_config(sys.argv[1])
print(resolve_path(c, c['training']['output_dir']))
PY
)
mkdir -p "$OUTPUT_DIR"
RESUME_ARGS=()
if [[ -s "$OUTPUT_DIR/checkpoint-last.pt" ]]; then
  RESUME_ARGS=(--resume "$OUTPUT_DIR/checkpoint-last.pt")
fi
python -u -m lse_v2.mmdit.train --config "$CONFIG" "${RESUME_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/formal_train.log"
python -m lse_v2.mmdit.evaluate \
  --config "$CONFIG" \
  --checkpoint "$OUTPUT_DIR/checkpoint-best.pt"
