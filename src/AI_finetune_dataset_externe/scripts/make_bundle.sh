#!/usr/bin/env bash
# Pack everything the CUDA laptop needs to train/infer fte models into ONE tar (keeps the repo layout,
# symlinks such as work/runs/LATEST_REFERENCE, and spaces in paths). Run on the Mac:
#   bash src/AI_finetune_dataset_externe/scripts/make_bundle.sh [out.tar]
# Copy the tar to the USB stick; on the laptop: tar -xf fte_bundle.tar -C ~/  (creates ~/Vin Gigahack/...)
set -euo pipefail
FTE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$FTE/../.." && pwd)"
REPO_NAME="$(basename "$REPO")"
OUT="${1:-$FTE/work/bundle/fte_bundle.tar}"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
cd "$(dirname "$REPO")"
R="$REPO_NAME"
REF_RUN="$(readlink "$R/src/AI/work/runs/LATEST_REFERENCE")"
EXCL=(--exclude '__pycache__' --exclude '.DS_Store' --exclude '*.pyc' --exclude '.pytest_cache' --exclude '.ruff_cache')
paths=(
  "$R/CLAUDE.md"
  "$R/data & info/02_route"
  "$R/data & info/05_examples/siret3_examples_cvat"
  "$R/src/AI/vineyard" "$R/src/AI/configs" "$R/src/AI/pyproject.toml" "$R/src/AI/models/waste-probe"
  "$R/src/AI/work/tiles" "$R/src/AI/work/cache" "$R/src/AI/work/tile_index.parquet" "$R/src/AI/work/layers"
  "$R/src/AI/work/runs/complete-v4/annset" "$R/src/AI/work/runs/complete-v4/layers"
  "$R/src/AI/work/runs/complete-v4/metrics" "$R/src/AI/work/runs/complete-v4/run.json"
  "$R/src/AI/work/runs/complete-v4/config.json"
  "$R/src/AI/work/runs/$REF_RUN" "$R/src/AI/work/runs/LATEST_REFERENCE"
  "$R/src/AI_finetune_dataset_externe/fte" "$R/src/AI_finetune_dataset_externe/scripts"
  "$R/src/AI_finetune_dataset_externe/configs" "$R/src/AI_finetune_dataset_externe/pyproject.toml"
  "$R/src/AI_finetune_dataset_externe/README.md" "$R/src/AI_finetune_dataset_externe/DATASETS.md"
  "$R/src/AI_finetune_dataset_externe/RUNBOOK_LINUX.md" "$R/src/AI_finetune_dataset_externe/tests"
  "$R/src/AI_finetune_dataset_externe/models/fte-canopy/v1b"
  "$R/src/AI_finetune_dataset_externe/work/stores/canopy_v1"
  "$R/src/AI_finetune_dataset_externe/work/data"
  "$R/src/AI_finetune_dataset_externe/work/raw/icaerus"
)
existing=()
for p in "${paths[@]}"; do [ -e "$p" ] && existing+=("$p") || echo "skip (missing): $p"; done
tar -cf "$OUT" "${EXCL[@]}" "${existing[@]}"
# smp ImageNet encoder weights (dereferenced HF cache entry) so the laptop needs no download
HF="$HOME/.cache/huggingface/hub"
if [ -d "$HF/models--smp-hub--resnet34.imagenet" ]; then
  tar -rhf "$OUT" -C "$HOME/.cache/huggingface" -s ",^hub,$R/hf_cache/hub," "hub/models--smp-hub--resnet34.imagenet"
fi
ls -lh "$OUT"
shasum -a 256 "$OUT" | tee "$OUT.sha256"
