#!/usr/bin/env bash
# Run inside screen; preserve the exit code even after the screen disappears.
set -uo pipefail
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TMPDIR=/root/autodl-tmp/lse-v4-1-tmp
export LSE_V4_VENV=/root/autodl-tmp/lse-v4-1-env
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LSE_GRPO_RECOVERY_MODE="${1:-canary}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT=/root/autodl-tmp/lse-v4-1-logs
mkdir -p "$LOG_ROOT" "$TMPDIR"
RUN_ID="${LSE_GRPO_RECOVERY_MODE}-$(date -u +%Y%m%dT%H%M%SZ)"
bash "$REPO_ROOT/scripts/autodl_v4_1_grpo_recovery.sh" > "$LOG_ROOT/$RUN_ID.log" 2>&1
status=$?
printf '%s\n' "$status" > "$LOG_ROOT/$RUN_ID.exitcode"
exit "$status"
