"""Training schedule (A§4.6, design 03 §3 N5): AdamW + per-epoch cosine LR, early stopping as immutable
state transitions, and the optional drop-noisiest index selection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from vineyard.config import TrainConfig

MIN_LR: Final = 0.0
FLOOR_EPS: Final = 1e-9
NO_EPOCH: Final = 0


@dataclass(frozen=True)
class EarlyStopState:
    best_score: float = -math.inf
    best_epoch: int = NO_EPOCH
    bad_epochs: int = 0
    last_epoch: int = NO_EPOCH
    improved: bool = False
    stop: bool = False


def early_stop_update(state: EarlyStopState, epoch: int, score: float, patience: int,
                      min_delta: float = 0.0) -> EarlyStopState:
    """New state after validation `score` (higher is better) of `epoch` (1-based, increasing)."""
    if not math.isfinite(score):
        raise ValueError(f"validation score must be finite, got {score} at epoch {epoch}")
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if epoch <= state.last_epoch:
        raise ValueError(f"epoch {epoch} is not after the last one ({state.last_epoch})")
    if score > state.best_score + min_delta:
        return EarlyStopState(best_score=score, best_epoch=epoch, bad_epochs=0, last_epoch=epoch, improved=True)
    bad = state.bad_epochs + 1
    return EarlyStopState(best_score=state.best_score, best_epoch=state.best_epoch, bad_epochs=bad, last_epoch=epoch,
                          improved=False, stop=bad >= patience)


def cosine_lr(base_lr: float, epoch: int, epochs: int, min_lr: float = MIN_LR) -> float:
    """LR of 0-based `epoch`: base at epoch 0, decaying along half a cosine towards min_lr at `epochs`."""
    if not 0 <= epoch < epochs:
        raise ValueError(f"epoch {epoch} outside [0, {epochs})")
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))


def build_optimizer(model: nn.Module, train: TrainConfig) -> torch.optim.AdamW:
    """AdamW(lr=nn.train.lr, weight_decay=nn.train.weight_decay) over the trainable parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=train.lr, weight_decay=train.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-epoch cosine (call .step() once after each epoch); factor = cosine_lr(1, epoch, epochs)."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: cosine_lr(1.0, min(e, epochs - 1), epochs))


def drop_noisiest(losses: Sequence[float], frac: float) -> tuple[int, ...]:
    """Indices kept after removing the floor(frac·n) highest-loss patches (ties: the later index goes)."""
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"frac must be in [0, 1), got {frac}")
    values = np.asarray(losses, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("per-patch losses must be finite")
    n_drop = int(math.floor(frac * values.size + FLOOR_EPS))
    if n_drop == 0:
        return tuple(range(values.size))
    order = np.lexsort((-np.arange(values.size), -values))  # highest loss first, later index first on ties
    dropped = set(order[:n_drop].tolist())
    return tuple(i for i in range(values.size) if i not in dropped)
