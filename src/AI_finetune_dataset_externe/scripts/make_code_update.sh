#!/usr/bin/env bash
# Small tar with only the fte code/scripts (for re-syncing the laptop after code changes).
set -euo pipefail
FTE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$FTE/../.." && pwd)"
OUT="${1:-$FTE/work/bundle/fte_code_update.tar}"
mkdir -p "$(dirname "$OUT")"
cd "$(dirname "$REPO")"
R="$(basename "$REPO")"
tar -cf "$OUT" --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  "$R/src/AI_finetune_dataset_externe/fte" "$R/src/AI_finetune_dataset_externe/scripts" \
  "$R/src/AI_finetune_dataset_externe/configs"
ls -lh "$OUT"
