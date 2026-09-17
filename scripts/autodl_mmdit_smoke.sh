#!/usr/bin/env bash
set -euo pipefail

PROJECT=${LSE_PROJECT_DIR:-/root/autodl-tmp/llm-guided-speech-enhancement}
VENV=${LSE_MMDIT_VENV:-/root/autodl-tmp/lse-mmdit-env}
cd "$PROJECT"
source "$VENV/bin/activate"

if [[ "$(pwd -P)" != "$(cd "$PROJECT" && pwd -P)" ]]; then
  echo "Refusing smoke cleanup outside the configured project directory." >&2
  exit 2
fi
rm -rf -- outputs/mmdit-smoke-v2 results/mmdit-smoke-v2 data/mmdit_smoke_v2
python scripts/create_mmdit_smoke_fixture.py --output data/mmdit_smoke_v2
python -m lse_v2.mmdit.preflight \
  --config configs/mmdit_smoke.json \
  --output results/mmdit-smoke-v2/preflight.json \
  --runtime-smoke
python -m lse_v2.mmdit.train --config configs/mmdit_smoke.json
python -m lse_v2.mmdit.evaluate \
  --config configs/mmdit_smoke.json \
  --checkpoint outputs/mmdit-smoke-v2/checkpoint-best.pt
test -s results/mmdit-smoke-v2/evaluation_report.json
echo 'MM-DiT smoke chain completed; this validates wiring, not enhancement quality.'
