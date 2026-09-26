"""Masked BCE with label smoothing + soft Dice (A§4.6, design 03 §3 N4); label 255 is ignored.

m = (label != ignore_index). BCE uses the smoothed target y·(1-ε) + ε/2 averaged over m; Dice uses the
hard target over m (smooth = 1). A batch with no labelled pixel gives 0 and zero gradients, not NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from vineyard.config import TrainConfig

IGNORE_INDEX: Final = 255
POSITIVE: Final = 1
DICE_SMOOTH: Final = 1.0
HALF: Final = 0.5


@dataclass(frozen=True)
class LossParams:
    bce_weight: float
    dice_weight: float
    label_smoothing: float
    ignore_index: int = IGNORE_INDEX
    dice_smooth: float = DICE_SMOOTH

    def __post_init__(self) -> None:
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.bce_weight < 0 or self.dice_weight < 0:
            raise ValueError(f"loss weights must be >= 0, got bce={self.bce_weight} dice={self.dice_weight}")

    @classmethod
    def from_config(cls, train: TrainConfig) -> LossParams:
        return cls(bce_weight=train.bce_weight, dice_weight=train.dice_weight, label_smoothing=train.label_smoothing)


@dataclass(frozen=True)
class LossValue:
    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor
    n_valid: int


def smooth_targets(y: torch.Tensor, eps: float) -> torch.Tensor:
    """1 -> 1 - ε/2, 0 -> ε/2."""
    return y * (1.0 - eps) + eps * HALF


def _prepare(logits: torch.Tensor, labels: torch.Tensor, p: LossParams) -> tuple[torch.Tensor, ...]:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"logits must be (B, 1, H, W) with one channel, got {tuple(logits.shape)}")
    if labels.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError(f"label shape {tuple(labels.shape)} does not match logits {tuple(logits.shape)}")
    lbl = labels.to(device=logits.device, dtype=torch.long)
    mask = (lbl != p.ignore_index).unsqueeze(1).to(logits.dtype)
    y = (lbl == POSITIVE).unsqueeze(1).to(logits.dtype)
    return logits, y, mask


def _bce_map(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, smooth_targets(y, eps), reduction="none") * mask


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, p: LossParams) -> torch.Tensor:
    """Mean smoothed BCE over the labelled pixels of the batch."""
    logits, y, mask = _prepare(logits, labels, p)
    return _bce_map(logits, y, mask, p.label_smoothing).sum() / mask.sum().clamp(min=1.0)


def _dice(prob: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...], smooth: float) -> torch.Tensor:
    inter = (prob * y * mask).sum(dim=dims)
    denom = (prob * mask).sum(dim=dims) + (y * mask).sum(dim=dims)
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


def masked_soft_dice(logits: torch.Tensor, labels: torch.Tensor, p: LossParams) -> torch.Tensor:
    """Soft Dice loss over the labelled pixels of the whole batch (hard targets)."""
    logits, y, mask = _prepare(logits, labels, p)
    return _dice(torch.sigmoid(logits), y, mask, (0, 1, 2, 3), p.dice_smooth)


def dice_bce_loss(logits: torch.Tensor, labels: torch.Tensor, p: LossParams) -> LossValue:
    """bce_weight·BCE + dice_weight·Dice; both parts are returned for logging."""
    logits, y, mask = _prepare(logits, labels, p)
    n_valid = mask.sum()
    bce = _bce_map(logits, y, mask, p.label_smoothing).sum() / n_valid.clamp(min=1.0)
    dice = _dice(torch.sigmoid(logits), y, mask, (0, 1, 2, 3), p.dice_smooth)
    return LossValue(total=p.bce_weight * bce + p.dice_weight * dice, bce=bce, dice=dice, n_valid=int(n_valid.item()))


def per_sample_loss(logits: torch.Tensor, labels: torch.Tensor, p: LossParams) -> torch.Tensor:
    """(B,) loss per patch (drop-noisiest ranking); a fully ignored patch scores 0."""
    logits, y, mask = _prepare(logits, labels, p)
    n = mask.sum(dim=(1, 2, 3))
    bce = _bce_map(logits, y, mask, p.label_smoothing).sum(dim=(1, 2, 3)) / n.clamp(min=1.0)
    dice = _dice(torch.sigmoid(logits), y, mask, (1, 2, 3), p.dice_smooth)
    return p.bce_weight * bce + p.dice_weight * dice


class DiceBceLoss(nn.Module):
    """nn.Module wrapper returning the total loss."""

    def __init__(self, params: LossParams) -> None:
        super().__init__()
        self.params = params

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return dice_bce_loss(logits, labels, self.params).total
