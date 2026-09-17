#!/usr/bin/env bash
set -euo pipefail

PROJECT=${LSE_PROJECT_DIR:-/root/autodl-tmp/llm-guided-speech-enhancement}
VENV=${LSE_MMDIT_VENV:-/root/autodl-tmp/lse-mmdit-env}
CONFIG=configs/mmdit_v6_stepaudio3_canary.json

cd "$PROJECT"
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs/mmdit-v6-canary results/mmdit-v6-canary
python -m lse_v2.mmdit.preflight \
  --config "$CONFIG" \
  --output results/mmdit-v6-canary/preflight.json \
  --runtime-smoke
python -u -m lse_v2.mmdit.train --config "$CONFIG" \
  2>&1 | tee outputs/mmdit-v6-canary/canary_train.log
python -m lse_v2.mmdit.evaluate \
  --config "$CONFIG" \
  --checkpoint outputs/mmdit-v6-canary/checkpoint-best.pt
test -s outputs/mmdit-v6-canary/training_report.json
test -s results/mmdit-v6-canary/evaluation_report.json
python - <<'PY'
import json
import math
from pathlib import Path

training = json.loads(Path("outputs/mmdit-v6-canary/training_report.json").read_text())
evaluation = json.loads(Path("results/mmdit-v6-canary/evaluation_report.json").read_text())
if training.get("status") != "COMPLETED" or evaluation.get("status") != "MEASURED":
    raise SystemExit("v6 canary did not complete its measured chain")
if not math.isfinite(float(training["best_validation"])):
    raise SystemExit("v6 canary validation produced a non-finite value")
print({"canary": "PASSED", "best_validation": training["best_validation"]})
PY
echo 'MM-DiT v6 full-objective canary passed.'
