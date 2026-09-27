#!/usr/bin/env bash
# Train + infer on the CUDA laptop (see RUNBOOK_LINUX.md), then pack results for the Mac.
#   bash scripts/run_laptop.sh waste    # P1: waste train (time box) + infer 311 tiles + select + export
#   bash scripts/run_laptop.sh canopy   # P3: canopy v2, warm start from v1b, AMP + inference (c0/c1/c2)
#   bash scripts/run_laptop.sh pack     # -> work/bundle/fte_results.tar (models, preds, reports, logs)
set -euo pipefail
FTE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$FTE/../.." && pwd)"
PY="$FTE/.venv-cuda/bin/python"
export PYTHONPATH="$REPO/src/AI:$FTE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GDAL_NUM_THREADS=1 GDAL_CACHEMAX=256
unset PROJ_DATA PROJ_LIB GDAL_DATA CONDA_PREFIX || true
NW="${WORKERS:-$(( $(nproc) > 10 ? 8 : $(nproc) - 2 ))}"
cd "$FTE"
mkdir -p work/logs
case "${1:-}" in
  waste)
    $PY -m fte.waste.train --device cuda --epochs "${EPOCHS:-40}" --patches-per-epoch 4000 --batch "${BATCH:-12}" \
      --lr 3e-4 --time-box-min "${TBOX:-40}" --patience "${PATIENCE:-6}" --workers "$NW" \
      --out models/fte-waste/v1 2>&1 | tee work/logs/waste_train_cuda.log
    $PY -m fte.waste.infer --device cuda --weights models/fte-waste/v1/best.pt --tiles all \
      --out work/preds/waste/v1 2>&1 | tee work/logs/waste_infer_cuda.log
    $PY -m fte.waste.select --pred-dir work/preds/waste/v1 --card models/fte-waste/v1/model_card.json \
      2>&1 | tee work/logs/waste_select.log
    $PY -m fte.waste.export --pred-dir work/preds/waste/v1 --review-min 0.15 --review-limit 600 \
      2>&1 | tee work/logs/waste_export.log
    ;;
  canopy)
    INIT="${INIT:-models/fte-canopy/v1b/last.pt}"
    INIT_ARGS=(); [ -f "$INIT" ] && INIT_ARGS=(--init-weights "$INIT")
    $PY -m fte.canopy.train --device cuda --amp "${INIT_ARGS[@]}" --epochs "${EPOCHS:-60}" --patches-per-epoch 1000 \
      --batch "${BATCH:-8}" --lr "${LR:-2e-4}" --encoder-lr "${ENC_LR:-7e-5}" --encoder resnet34 \
      --time-box-min "${TBOX:-25}" --patience "${PATIENCE:-8}" --workers "$NW" --seed 2 \
      --out models/fte-canopy/v2 2>&1 | tee work/logs/canopy_train_cuda.log
    for w in best_contact best_mask; do
      $PY -m fte.canopy.infer --weights "models/fte-canopy/v2/$w.pt" --tiles vineyard --tta 4 \
        --out "work/preds/canopy/v2_$w" 2>&1 | tee "work/logs/canopy_infer_cuda_$w.log"
    done
    $PY -m fte.canopy.partition_eval --contact-dir work/preds/canopy/v2_best_contact/c1 \
      2>&1 | tee work/logs/partition_eval_v2.log
    ;;
  pack)
    mkdir -p work/bundle
    tar -cf work/bundle/fte_results.tar --exclude 'smoke' --exclude 'epoch9_last.pt' \
      models work/preds work/reports work/logs
    ls -lh work/bundle/fte_results.tar
    ;;
  *) echo "usage: $0 waste|canopy|pack"; exit 2 ;;
esac
