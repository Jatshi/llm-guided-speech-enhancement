#!/usr/bin/env bash
set -euo pipefail

PROJECT=${LSE_PROJECT_DIR:-/root/autodl-tmp/llm-guided-speech-enhancement}
VENV=${LSE_MMDIT_VENV:-/root/autodl-tmp/lse-mmdit-env}

cd "$PROJECT"
if [[ ! -x "$VENV/bin/python" ]]; then
  # AutoDL images already ship a CUDA-matched PyTorch build. Reuse it instead of
  # downloading another ~GB wheel into the paid instance.
  python3 -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
if [[ "${LSE_MMDIT_INSTALL_FULL_TRAIN:-0}" == "1" ]]; then
  python -m pip install -e '.[train,audio,test,mmdit-eval]'
else
  # MM-DiT itself does not import transformers/librosa. Keep the paid-machine
  # bootstrap small and install only the runtime/evaluation packages it uses.
  python -m pip install -e . --no-deps
  python -m pip install \
    'numpy==1.26.4' 'scipy==1.15.3' 'soundfile==0.13.1' \
    'pytest==8.3.4' 'ruff==0.9.6' \
    'deepfilternet==0.5.6' 'pesq==0.0.4' 'pystoi==0.4.1'
fi

# DeepFilterNet imports torchaudio. Match the wheel to the CUDA-enabled torch
# already shipped by AutoDL, and never let pip replace that torch build.
if ! python -c 'import torchaudio' >/dev/null 2>&1; then
  TORCH_VERSION=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
  CUDA_WHEEL=$(python -c 'import torch; print("cu" + torch.version.cuda.replace(".", ""))')
  python -m pip install --force-reinstall --no-deps \
    "torchaudio==$TORCH_VERSION" \
    --index-url "https://download.pytorch.org/whl/$CUDA_WHEEL"
fi

python - <<'PY'
import torch
print({"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the paid AutoDL run")
PY

# The package downloads this archive from GitHub on first use. GitHub raw can
# stall on AutoDL, so make the download bounded and retain an explicit proxy
# fallback. Override DEEPFILTER_MODEL_URL to use an institutional mirror.
DF_CACHE=${DEEPFILTER_CACHE_DIR:-/root/.cache/DeepFilterNet}
DF_ARCHIVE="$DF_CACHE/DeepFilterNet3.zip"
DF_MODEL_DIR="$DF_CACHE/DeepFilterNet3"
DF_PRIMARY_URL=${DEEPFILTER_MODEL_URL:-https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip}
DF_PROXY_URL=https://ghproxy.net/https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip
if [[ ! -f "$DF_MODEL_DIR/config.ini" ]]; then
  mkdir -p "$DF_CACHE"
  DF_PART="$DF_ARCHIVE.part"
  rm -f "$DF_PART"
  if ! curl --fail --location --connect-timeout 15 --max-time 120 \
    --speed-time 20 --speed-limit 1024 --output "$DF_PART" "$DF_PRIMARY_URL"; then
    rm -f "$DF_PART"
    curl --fail --location --connect-timeout 15 --max-time 600 \
      --speed-time 30 --speed-limit 1024 --output "$DF_PART" "$DF_PROXY_URL"
  fi
  mv "$DF_PART" "$DF_ARCHIVE"
  python -m zipfile -t "$DF_ARCHIVE"
  python -m zipfile -e "$DF_ARCHIVE" "$DF_CACHE"
fi

python - <<'PY'
try:
    from df.enhance import init_df
except ImportError:
    print("DeepFilterNet optional baseline unavailable; formal preflight will remain BLOCKED")
else:
    loaded = init_df(model_base_dir="DeepFilterNet3", log_file=None)
    print({"deepfilternet3": "ready", "sample_rate": int(loaded[1].sr())})
PY
