"""Early stopping, cosine LR, AdamW factory and drop-noisiest (design 03 §3 N5, §5)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from vineyard.config import load_config
from vineyard.nn.schedule import (
    EarlyStopState,
    build_optimizer,
    build_scheduler,
    cosine_lr,
    drop_noisiest,
    early_stop_update,
)


def _run(scores: list[float], patience: int) -> list[EarlyStopState]:
    states = [EarlyStopState()]
    for epoch, score in enumerate(scores, start=1):
        states.append(early_stop_update(states[-1], epoch, score, patience))
    return states[1:]


def test_early_stopping_design_case() -> None:
    states = _run([0.70, 0.74, 0.73, 0.735, 0.72], patience=3)
    assert [s.stop for s in states] == [False, False, False, False, True]
    assert states[-1].best_epoch == 2 and states[-1].best_score == pytest.approx(0.74)
    assert states[-1].bad_epochs == 3


def test_early_stopping_is_immutable() -> None:
    first = EarlyStopState()
    second = early_stop_update(first, 1, 0.5, 3)
    assert first == EarlyStopState() and second.best_epoch == 1
    assert second.improved


def test_early_stopping_min_delta() -> None:
    s = early_stop_update(EarlyStopState(), 1, 0.70, 2, min_delta=0.01)
    s = early_stop_update(s, 2, 0.705, 2, min_delta=0.01)
    assert s.best_epoch == 1 and s.bad_epochs == 1 and not s.improved


def test_early_stopping_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        early_stop_update(EarlyStopState(), 1, math.nan, 3)
    with pytest.raises(ValueError, match="patience"):
        early_stop_update(EarlyStopState(), 1, 0.5, 0)
    with pytest.raises(ValueError, match="epoch"):
        early_stop_update(early_stop_update(EarlyStopState(), 2, 0.5, 3), 2, 0.6, 3)


def test_cosine_lr() -> None:
    assert cosine_lr(3e-4, 0, 20) == pytest.approx(3e-4)
    assert cosine_lr(3e-4, 19, 20) < 3e-4 * 0.01
    assert cosine_lr(3e-4, 10, 20) == pytest.approx(1.5e-4)
    values = [cosine_lr(3e-4, e, 20) for e in range(20)]
    assert values == sorted(values, reverse=True)
    with pytest.raises(ValueError, match="epoch"):
        cosine_lr(3e-4, 20, 20)


def test_optimizer_and_scheduler_follow_the_config() -> None:
    train = load_config().nn.train
    model = nn.Linear(2, 1)
    opt = build_optimizer(model, train)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    sched = build_scheduler(opt, epochs=4)
    lrs = []
    for _ in range(4):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs == pytest.approx([cosine_lr(3e-4, e, 4) for e in range(4)])


def test_drop_noisiest_removes_the_single_worst() -> None:
    losses = [0.1, 0.2, 0.9, 0.3, 0.2, 0.1, 0.4, 0.5, 0.3, 0.2]
    kept = drop_noisiest(losses, 0.1)
    assert kept == tuple(i for i in range(10) if i != 2)


def test_drop_noisiest_edge_cases() -> None:
    assert drop_noisiest([0.5, 0.4], 0.1) == (0, 1)  # floor(0.2) = 0 dropped
    assert drop_noisiest([], 0.5) == ()
    with pytest.raises(ValueError, match="frac"):
        drop_noisiest([0.1], 1.0)
    with pytest.raises(ValueError, match="finite"):
        drop_noisiest([0.1, math.inf], 0.5)
