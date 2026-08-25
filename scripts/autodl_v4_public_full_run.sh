#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${LSE_V4_VENV:-/root/autodl-tmp/lse-v4-env}"
DATA_ROOT="${LSE_DATA_ROOT:-/root/autodl-tmp/lse-v4-data}"
CONFIG="${LSE_CONFIG:-$REPO_ROOT/configs/native_audio_autodl_32gb.json}"
LOG_ROOT="${LSE_LOG_ROOT:-/root/autodl-tmp/lse-v4-logs}"

mkdir -p "$LOG_ROOT"
source "$VENV_DIR/bin/activate"

markers=(
  "$DATA_ROOT/.libri_download_complete"
  "$DATA_ROOT/.noise_download_complete"
  "$DATA_ROOT/.rir_download_complete"
  "/root/autodl-tmp/lse-v4-models/.models_download_complete"
)
while true; do
  missing=0
  for marker in "${markers[@]}"; do
    if [[ ! -f "$marker" ]]; then
      echo "WAITING_FOR=$marker"
      missing=1
    fi
  done
  (( missing == 0 )) && break
  sleep 30
done

python "$REPO_ROOT/scripts/prepare_public_audio.py" \
  --parquet-root "$DATA_ROOT/sources/librispeech_asr/all/train.clean.100" \
  --esc50-zip "$DATA_ROOT/sources/esc50/raw/master.zip" \
  --rir-zip "$DATA_ROOT/sources/rirs_noises/rirs_noises.zip" \
  --output-root "$DATA_ROOT/prepared" \
  --max-clean 10000 \
  | tee "$LOG_ROOT/public_audio_prepare.log"

export LSE_CLEAN_MANIFEST="$DATA_ROOT/prepared/catalogs/clean.jsonl"
export LSE_NOISE_MANIFEST="$DATA_ROOT/prepared/catalogs/noise.jsonl"
export LSE_RIR_MANIFEST="$DATA_ROOT/prepared/catalogs/rir.jsonl"
export LSE_MAX_SOURCES=10000
export LSE_VARIANTS_PER_SOURCE=2
export LSE_CONFIG="$CONFIG"
bash "$REPO_ROOT/scripts/autodl_v4_prepare_data.sh" \
  2>&1 | tee "$LOG_ROOT/materialize.log"

python - <<'PY'
from pathlib import Path
from lse_v2.io import read_jsonl, write_jsonl

root = Path("/root/autodl-tmp/lse-v4")
rows = [row for row in read_jsonl(root / "data/native/native_audio.v1.jsonl") if row["split"] == "train"]
if len(rows) < 8:
    raise SystemExit("at least eight train rows are required for the native-audio canary")
write_jsonl(root / "data/native/native_audio.canary.jsonl", rows[:8])
PY
python -m lse_v2.native_pipeline \
  --config "$REPO_ROOT/configs/native_audio_autodl_canary.json" \
  2>&1 | tee "$LOG_ROOT/canary.log"

export LSE_CONNECTIVITY_URL="https://www.modelscope.cn/"
export LSE_MIN_FREE_GIB=35
export LSE_RUN_MODE=full
bash "$REPO_ROOT/scripts/autodl_v4_run.sh" \
  2>&1 | tee "$LOG_ROOT/full_run.log"
