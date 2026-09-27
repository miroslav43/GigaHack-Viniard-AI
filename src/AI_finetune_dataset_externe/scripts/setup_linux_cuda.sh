#!/usr/bin/env bash
# Laptop (Linux + NVIDIA) setup. Run from the extracted bundle:
#   bash "~/Vin Gigahack/src/AI_finetune_dataset_externe/scripts/setup_linux_cuda.sh"
# Creates src/AI_finetune_dataset_externe/.venv-cuda. Reuses the system torch+CUDA when the system
# python is >= 3.12 (vineyard uses PEP 695 syntax); otherwise builds a 3.12 env with uv + CUDA torch.
set -euo pipefail
FTE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$FTE/../.." && pwd)"
VENV="$FTE/.venv-cuda"
PY="${PYTHON:-python3}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "WARN: nvidia-smi failed"
ok312=$($PY -c 'import sys; print(int(sys.version_info >= (3, 12)))')
has_torch=$($PY -c 'import torch; print(int(torch.cuda.is_available()))' 2>/dev/null || echo 0)
echo "system python: $($PY --version) | >=3.12: $ok312 | torch+cuda: $has_torch"
if [ "$ok312" = "1" ] && [ "$has_torch" = "1" ]; then
  $PY -m venv --system-site-packages "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q -r "$FTE/scripts/requirements-cuda.txt"
else
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv -p 3.12 "$VENV"
  CU="${CUDA_INDEX:-cu126}"
  uv pip install -p "$VENV/bin/python" torch torchvision --index-url "https://download.pytorch.org/whl/$CU"
  uv pip install -p "$VENV/bin/python" -r "$FTE/scripts/requirements-cuda.txt"
fi
# encoder weights shipped in the bundle -> HF cache
if [ -d "$REPO/hf_cache/hub" ]; then
  mkdir -p "$HOME/.cache/huggingface/hub"
  cp -rn "$REPO/hf_cache/hub/"* "$HOME/.cache/huggingface/hub/" || true
fi
PYTHONPATH="$REPO/src/AI:$FTE" "$VENV/bin/python" - <<'PY'
import torch, segmentation_models_pytorch as smp, rasterio, geopandas
import vineyard, fte
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
m = smp.Unet("resnet34", encoder_weights="imagenet", classes=3)
print("smp", smp.__version__, "resnet34 imagenet OK; vineyard + fte import OK")
PY
echo "setup done: $VENV"
