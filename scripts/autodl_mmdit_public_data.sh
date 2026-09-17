#!/usr/bin/env bash
set -euo pipefail

PROJECT=${LSE_PROJECT_DIR:-/root/autodl-tmp/llm-guided-speech-enhancement}
VENV=${LSE_MMDIT_VENV:-/root/autodl-tmp/lse-mmdit-env}
CORPUS_ROOT=${LSE_MMDIT_CORPUS_ROOT:-/root/autodl-tmp/mmdit-corpus}
ARCHIVE="$CORPUS_ROOT/downloads/dev-clean.tar.gz"
SOURCE="$CORPUS_ROOT/source/LibriSpeech/dev-clean"

mkdir -p "$CORPUS_ROOT/downloads" "$CORPUS_ROOT/source"
if [[ ! -s "$ARCHIVE" ]]; then
  wget -c https://www.openslr.org/resources/12/dev-clean.tar.gz -O "$ARCHIVE"
fi
echo '42e2234ba48799c1f50f24a7926300a1  '"$ARCHIVE" | md5sum -c -
if [[ ! -d "$SOURCE" ]]; then
  tar -xzf "$ARCHIVE" -C "$CORPUS_ROOT/source"
fi

cd "$PROJECT"
source "$VENV/bin/activate"
if [[ -e data/mmdit/pairs.jsonl ]]; then
  echo 'data/mmdit/pairs.jsonl already exists; refusing to overwrite a measured corpus.' >&2
  exit 2
fi
python -m lse_v2.mmdit.materialize_librispeech \
  --librispeech-root "$SOURCE" \
  --output-root data/mmdit \
  --train 1600 --validation 200 --test 200 \
  --seed 42 --sample-rate 16000 --seconds 4.0
test -s data/mmdit/corpus_report.json
