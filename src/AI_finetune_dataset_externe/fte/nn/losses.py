"""Per-channel masked losses; label value 255 means "ignore" for that channel only.

Labels are (B, K, H, W) uint8 in {0, 1, 255}; logits are (B, K, H, W). Each head gets BCE (or focal)
plus soft Dice over its labelled pixels, and the heads are combined with fixed weights. A head with no
labelled pixel in the batch contributes 0 (no NaN).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE = 255


@dataclass(frozen=True)
class HeadLoss:
    weight: float = 1.0
    dice_weight: float = 1.0
    pos_weight: float = 1.0
    focal_gamma: float = 0.0  # 0 -> plain BCE
    label_smoothing: float = 0.0


def _masks(labels: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (labels != IGNORE).to(dtype)
    target = (labels == 1).to(dtype)
    return target, valid


def head_loss(logits: torch.Tensor, labels: torch.Tensor, spec: HeadLoss,
              pixel_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Loss of one head: logits/labels (B, H, W)."""
    target, valid = _masks(labels, logits.dtype)
    if pixel_weight is not None:
        valid = valid * pixel_weight
    n = valid.sum()
    if n <= 0:
        return logits.sum() * 0.0
    eps = spec.label_smoothing
    soft = target * (1.0 - eps) + 0.5 * eps
    pw = torch.as_tensor(spec.pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, soft, reduction="none", pos_weight=pw)
    if spec.focal_gamma > 0:
        p = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, p, 1.0 - p)
        bce = bce * (1.0 - pt).pow(spec.focal_gamma)
    bce = (bce * valid).sum() / n
    prob = torch.sigmoid(logits)
    inter = (prob * target * valid).sum()
    denom = (prob * valid).sum() + (target * valid).sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return spec.weight * (bce + spec.dice_weight * dice)


def multi_head_loss(logits: torch.Tensor, labels: torch.Tensor, specs: Sequence[HeadLoss],
                    pixel_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, list[float]]:
    """Weighted sum over heads; returns (total, per-head floats for logging)."""
    if logits.shape != labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}")
    if logits.shape[1] != len(specs):
        raise ValueError(f"{logits.shape[1]} heads but {len(specs)} loss specs")
    parts = [head_loss(logits[:, k], labels[:, k], s, None if pixel_weight is None else pixel_weight[:, k])
             for k, s in enumerate(specs)]
    return torch.stack(parts).sum(), [float(p.detach()) for p in parts]
