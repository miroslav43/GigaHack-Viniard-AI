"""Model -> probability heatmap for an image of any size (sliding windows, optional hflip TTA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fte.waste.windows import ceil_to, pad_to, ramp_weight, window_starts


@dataclass(frozen=True)
class PredictSpec:
    window: int = 1024
    overlap: int = 128
    tta_hflip: bool = True


def _to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(rgb)).to(device=device, dtype=torch.float32)
    return x.permute(2, 0, 1).unsqueeze(0) / 255.0


@torch.no_grad()
def _forward(model: nn.Module, x: torch.Tensor, tta: bool) -> torch.Tensor:
    p = torch.sigmoid(model(x))[:, 0]
    if tta:
        p = 0.5 * (p + torch.sigmoid(model(torch.flip(x, dims=[3])))[:, 0].flip(dims=[2]))
    return p[0]


def predict_heat(model: nn.Module, rgb: np.ndarray, device: torch.device, spec: PredictSpec) -> np.ndarray:
    """float32 (H, W) probabilities of the waste head."""
    h, w = rgb.shape[:2]
    win = spec.window  # fixed input shape: MPS caches kernels per shape
    ph, pw = max(win, ceil_to(h)), max(win, ceil_to(w))
    img = pad_to(rgb, ph, pw)
    if ph <= win and pw <= win:
        return _forward(model, _to_tensor(img, device), spec.tta_hflip).cpu().numpy()[:h, :w].copy()
    weight = ramp_weight(win, spec.overlap)
    acc = np.zeros((ph, pw), dtype=np.float32)
    norm = np.zeros((ph, pw), dtype=np.float32)
    for y0 in window_starts(ph, win, spec.overlap):
        for x0 in window_starts(pw, win, spec.overlap):
            tile = img[y0:y0 + win, x0:x0 + win]
            p = _forward(model, _to_tensor(tile, device), spec.tta_hflip).cpu().numpy()
            acc[y0:y0 + win, x0:x0 + win] += p * weight
            norm[y0:y0 + win, x0:x0 + win] += weight
    return (acc / np.maximum(norm, 1e-6))[:h, :w].copy()
