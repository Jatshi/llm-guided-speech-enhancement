#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${LSE_V4_VENV:-/root/autodl-tmp/lse-v4-env}"
RUN_MODE="${LSE_RUN_MODE:-full}"
PIPELINE_CONFIG="${LSE_CONFIG:-$REPO_ROOT/configs/native_audio_4090.json}"

source "$VENV_DIR/bin/activate"
cd "$REPO_ROOT"
bash scripts/autodl_v4_preflight.sh

if [[ "$RUN_MODE" == "dry-run" ]]; then
  python -m lse_v2.native_pipeline --config "$PIPELINE_CONFIG" --dry-run
  exit 0
fi
if [[ "$RUN_MODE" != "full" ]]; then
  echo "LSE_RUN_MODE must be dry-run or full" >&2
  exit 2
fi

python -m lse_v2.native_pipeline --config "$PIPELINE_CONFIG"
test -s outputs/native_v4/run_manifest.json
test -s outputs/native_v4/grpo/final/audio_projector.pt
WHISPER_MODEL="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["whisper_model"])' "$PIPELINE_CONFIG")"
LANGUAGE_MODEL="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["language_model"])' "$PIPELINE_CONFIG")"
PREFIX_TOKENS="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["prefix_tokens"])' "$PIPELINE_CONFIG")"
python -m lse_v2.native_predict \
  --manifest data/native/native_audio.v1.jsonl \
  --stage-dir outputs/native_v4/grpo/final \
  --output-dir outputs/native_v4/test_predictions \
  --whisper-model "$WHISPER_MODEL" \
  --language-model "$LANGUAGE_MODEL" \
  --prefix-tokens "$PREFIX_TOKENS" \
  --split test
python -m lse_v2.generalization \
  --manifest data/native/native_audio.v1.jsonl \
  --predictions outputs/native_v4/test_predictions/predictions.jsonl \
  --output-dir outputs/native_v4/generalization \
  --split test
python -m pip freeze > outputs/native_v4/environment.freeze.txt
echo "Native audio SFT -> cDPO -> GRPO completed. Read outputs/native_v4/run_manifest.json."
