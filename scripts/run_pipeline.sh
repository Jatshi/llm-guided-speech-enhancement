#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PORTFOLIO_V2_MODE="${PORTFOLIO_V2_MODE:-full}"
exec bash "$SCRIPT_DIR/autodl_v2_run.sh" "$@"
