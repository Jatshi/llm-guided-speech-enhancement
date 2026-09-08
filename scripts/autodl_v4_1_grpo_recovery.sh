#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${LSE_V4_VENV:-/root/autodl-tmp/lse-v4-env}"
RUN_MODE="${LSE_GRPO_RECOVERY_MODE:-all}"
CANARY_CONFIG="${LSE_GRPO_CANARY_CONFIG:-$REPO_ROOT/configs/native_audio_autodl_grpo_recovery_canary.json}"
FULL_CONFIG="${LSE_GRPO_FULL_CONFIG:-$REPO_ROOT/configs/native_audio_autodl_grpo_recovery_32gb.json}"
CANARY_ROOT="$(python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]).resolve(); print((p.parent / json.loads(p.read_text())["output_dir"]).resolve())' "$CANARY_CONFIG")"
OUTPUT_ROOT="$(python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]).resolve(); print((p.parent / json.loads(p.read_text())["output_dir"]).resolve())' "$FULL_CONFIG")"
BASELINE_ROOT="$REPO_ROOT/outputs/native_v4"
MANIFEST="$REPO_ROOT/data/native/native_audio.v1.jsonl"
MODEL_ROOT="${LSE_V4_MODEL_ROOT:-/root/autodl-tmp/lse-v4-models}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  bash "$REPO_ROOT/scripts/autodl_v4_bootstrap.sh"
fi
source "$VENV_DIR/bin/activate"
cd "$REPO_ROOT"

case "$RUN_MODE" in
  preflight|canary|full|all) ;;
  *)
    echo "LSE_GRPO_RECOVERY_MODE must be preflight, canary, full, or all" >&2
    exit 2
    ;;
esac

python scripts/ensure_v4_models.py \
  --root "$MODEL_ROOT" \
  --endpoint "$HF_ENDPOINT_VALUE" \
  --max-workers "${LSE_MODEL_DOWNLOAD_WORKERS:-8}"
test -s "$MANIFEST"
test -s "$BASELINE_ROOT/sft/final/audio_projector.pt"
# Recovery regenerates a missing embedding cache from retained physical waveforms.

export LSE_CONNECTIVITY_URL="$HF_ENDPOINT_VALUE"
LSE_MIN_FREE_GIB="${LSE_MIN_FREE_GIB:-20}" \
LSE_MIN_VRAM_GIB="${LSE_MIN_VRAM_GIB:-20}" \
  bash scripts/autodl_v4_preflight.sh
python -m pytest -q tests/test_rewards.py tests/test_native_alignment_losses.py \
  tests/test_native_training_objectives.py tests/test_grpo_recovery.py \
  tests/test_grpo_acceptance.py
python -m lse_v2.grpo_recovery --config "$CANARY_CONFIG" --dry-run --check-artifacts

if [[ "$RUN_MODE" == "preflight" ]]; then
  exit 0
fi

if [[ "$RUN_MODE" == "canary" || "$RUN_MODE" == "all" ]]; then
  python -m lse_v2.grpo_recovery --config "$CANARY_CONFIG"
fi

python - "$CANARY_ROOT/grpo/stage_manifest.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
checks = {
    "status": report.get("status") == "completed",
    "canary": report.get("canary", {}).get("status") == "passed",
    "valid_json_rate": float(report.get("valid_json_rate", 0.0)) >= 0.8,
    "non_saturated_group_rate": float(report.get("non_saturated_group_rate", 0.0)) >= 0.2,
    "anchor_loss": report.get("mean_anchor_loss") is not None,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"short GRPO canary did not pass: {failed}; inspect {path}")
print(json.dumps({"short_canary": "passed", "checks": checks}, indent=2))
PY

if [[ "$RUN_MODE" == "canary" ]]; then
  exit 0
fi

test -s "$BASELINE_ROOT/sft_test_predictions/predictions.jsonl"
python -m lse_v2.grpo_recovery --config "$FULL_CONFIG" --dry-run --check-artifacts
python -m lse_v2.grpo_recovery --config "$FULL_CONFIG"

python -m lse_v2.native_predict \
  --manifest "$MANIFEST" \
  --stage-dir "$OUTPUT_ROOT/grpo/final" \
  --output-dir "$OUTPUT_ROOT/test_predictions" \
  --whisper-model "/root/autodl-tmp/lse-v4-models/whisper-small" \
  --language-model "/root/autodl-tmp/lse-v4-models/Qwen2.5-1.5B-Instruct" \
  --prefix-tokens 16 \
  --split test \
  --embedding-index "$BASELINE_ROOT/audio_embedding_cache/index.jsonl" \
  --batch-size 16 \
  --max-new-tokens 256 \
  --temperature 0 \
  --resume

python -m lse_v2.generalization \
  --manifest "$MANIFEST" \
  --predictions "$OUTPUT_ROOT/test_predictions/predictions.jsonl" \
  --output-dir "$OUTPUT_ROOT/generalization" \
  --split test

python -m lse_v2.grpo_acceptance \
  --manifest "$MANIFEST" \
  --baseline-predictions "$BASELINE_ROOT/sft_test_predictions/predictions.jsonl" \
  --candidate-predictions "$OUTPUT_ROOT/test_predictions/predictions.jsonl" \
  --output "$OUTPUT_ROOT/grpo_acceptance.json" \
  --split test \
  --min-valid-json-rate 0.99 \
  --max-reward-regression 0.005

python -m pip freeze > "$OUTPUT_ROOT/environment.freeze.txt"
echo "GRPO v4.1 recovery and acceptance checks completed: $OUTPUT_ROOT"
