#!/usr/bin/env bash
set -euo pipefail

PROJECT=${LSE_PROJECT_DIR:-/root/autodl-tmp/llm-guided-speech-enhancement}
VENV=${LSE_MMDIT_VENV:-/root/autodl-tmp/lse-mmdit-env}

cd "$PROJECT"
if [[ ! -x "$VENV/bin/python" ]]; then
  bash scripts/autodl_mmdit_bootstrap.sh
fi
source "$VENV/bin/activate"
if [[ "$(pwd -P)" != "$(cd "$PROJECT" && pwd -P)" ]]; then
  echo 'Refusing smoke cleanup outside the configured project directory.' >&2
  exit 2
fi
rm -rf -- outputs/mmdit-v6-smoke results/mmdit-v6-smoke data/mmdit_smoke_v6
python scripts/create_mmdit_smoke_fixture.py --output data/mmdit_smoke_v6
python -m lse_v2.mmdit.enhance_script \
  --input data/mmdit_smoke_v6/pairs.jsonl \
  --output data/mmdit_smoke_v6/pairs.enhance_script.jsonl \
  --duration-ms 500
python -m lse_v2.mmdit.preflight \
  --config configs/mmdit_v6_stepaudio3_smoke.json \
  --output results/mmdit-v6-smoke/preflight.json \
  --runtime-smoke
python -u -m lse_v2.mmdit.train --config configs/mmdit_v6_stepaudio3_smoke.json
python -m lse_v2.mmdit.evaluate \
  --config configs/mmdit_v6_stepaudio3_smoke.json \
  --checkpoint outputs/mmdit-v6-smoke/checkpoint-best.pt
test -s results/mmdit-v6-smoke/evaluation_report.json
echo 'MM-DiT v6 smoke completed; wiring passed, enhancement quality remains unmeasured.'
