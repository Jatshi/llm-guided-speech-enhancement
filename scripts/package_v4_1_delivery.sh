#!/usr/bin/env bash
set -euo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
ARCHIVE="${LSE_V4_1_ARCHIVE:-$AUTODL_ROOT/lse-v4-1-delivery-20260909.tar}"
REPO_NAME="${LSE_V4_1_REPO_NAME:-lse-v4-1}"

rm -f "$ARCHIVE" "$ARCHIVE.sha256"

tar -C "$AUTODL_ROOT" -cf "$ARCHIVE" \
  --exclude="$REPO_NAME/outputs/*" \
  --exclude="$REPO_NAME/data/*" \
  --exclude="$REPO_NAME/.pytest_cache" \
  --exclude="$REPO_NAME/.ruff_cache" \
  "$REPO_NAME"

tar -C "$AUTODL_ROOT" -rf "$ARCHIVE" \
  "$REPO_NAME/outputs/native_v4_1_grpo_12gb" \
  "$REPO_NAME/data/native" \
  "$REPO_NAME-logs"

sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
stat -c "%s %n" "$ARCHIVE"
