"""Masked Dice + BCE with label smoothing (design 03 §3 N4, §5)."""

from __future__ import annotations

import pytest
import torch

from vineyard.config import load_config
from vineyard.nn.losses import (
    DiceBceLoss,
    LossParams,
    dice_bce_loss,
    masked_bce,
    masked_soft_dice,
    per_sample_loss,
    smooth_targets,
)

IGN = 255


def _params(**kw: float) -> LossParams:
    base = dict(bce_weight=1.0, dice_weight=1.0, label_smoothing=0.05)
    return LossParams(**{**base, **kw})  # type: ignore[arg-type]


def _labels() -> torch.Tensor:
    lbl = torch.zeros(2, 8, 8, dtype=torch.int64)
    lbl[:, 2:6, 2:6] = 1
    lbl[:, 0, :] = IGN
    return lbl


def test_smoothing_targets() -> None:
    y = smooth_targets(torch.tensor([0.0, 1.0]), 0.05)
    assert torch.allclose(y, torch.tensor([0.025, 0.975]))


def test_params_from_config() -> None:
    p = LossParams.from_config(load_config().nn.train)
    assert (p.bce_weight, p.dice_weight, p.label_smoothing, p.ignore_index) == (1.0, 1.0, 0.05, 255)


def test_params_validation() -> None:
    with pytest.raises(ValueError, match="label_smoothing"):
        _params(label_smoothing=1.5)
    with pytest.raises(ValueError, match="weight"):
        _params(bce_weight=-1.0)


def test_all_ignored_gives_zero_finite_loss_with_a_graph() -> None:
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    lbl = torch.full((2, 8, 8), IGN, dtype=torch.int64)
    value = dice_bce_loss(logits, lbl, _params())
    assert float(value.total.detach()) == 0.0 and bool(torch.isfinite(value.total.detach()))
    assert value.n_valid == 0
    value.total.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) == 0.0


def test_perfect_prediction_beats_inverted() -> None:
    lbl = _labels()
    good = torch.where(lbl == 1, 8.0, -8.0).unsqueeze(1).float()
    bad = -good
    p = _params()
    assert float(dice_bce_loss(good, lbl, p).total) < float(dice_bce_loss(bad, lbl, p).total)


def test_gradient_is_zero_at_ignored_pixels() -> None:
    lbl = _labels()
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    dice_bce_loss(logits, lbl, _params()).total.backward()
    grad = logits.grad
    assert grad is not None
    assert float(grad[:, 0, 0, :].abs().sum()) == 0.0
    assert float(grad[:, 0, 1:, :].abs().sum()) > 0.0


def test_parts_and_weights() -> None:
    lbl = _labels()
    logits = torch.randn(2, 1, 8, 8)
    value = dice_bce_loss(logits, lbl, _params(bce_weight=2.0, dice_weight=0.5))
    expected = 2.0 * masked_bce(logits, lbl, _params()) + 0.5 * masked_soft_dice(logits, lbl, _params())
    assert torch.allclose(value.total, expected)
    assert value.n_valid == 2 * 7 * 8


def test_bce_uses_smoothed_targets() -> None:
    lbl = torch.ones(1, 1, 1, dtype=torch.int64)
    logit = torch.logit(torch.tensor([[[[0.975]]]]))
    smooth = masked_bce(logit, lbl, _params(label_smoothing=0.05))
    hard = masked_bce(logit, lbl, _params(label_smoothing=0.0))
    # 0.975 is the minimiser of the smoothed BCE, so the gradient there is ~0
    logit.requires_grad_(True)
    masked_bce(logit, lbl, _params()).backward()
    assert logit.grad is not None and abs(float(logit.grad)) < 1e-6
    # at p = 0.975 the smoothed target still pays the entropy of (0.975, 0.025), the hard one only -log 0.975
    assert float(hard) == pytest.approx(-torch.log(torch.tensor(0.975)).item(), rel=1e-4)
    assert float(smooth) > float(hard)


def test_per_sample_loss_shape_and_ignore() -> None:
    lbl = _labels()
    lbl[1] = IGN
    logits = torch.randn(2, 1, 8, 8)
    per = per_sample_loss(logits, lbl, _params())
    assert per.shape == (2,) and float(per[1]) == 0.0 and float(per[0]) > 0.0


def test_module_wrapper_matches_function() -> None:
    lbl = _labels()
    logits = torch.randn(2, 1, 8, 8)
    assert torch.allclose(DiceBceLoss(_params())(logits, lbl), dice_bce_loss(logits, lbl, _params()).total)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        dice_bce_loss(torch.zeros(2, 1, 8, 8), torch.zeros(2, 4, 4, dtype=torch.int64), _params())
    with pytest.raises(ValueError, match="channel"):
        dice_bce_loss(torch.zeros(2, 2, 8, 8), torch.zeros(2, 8, 8, dtype=torch.int64), _params())


def test_uint8_labels_are_accepted() -> None:
    lbl = _labels().to(torch.uint8)
    logits = torch.randn(2, 1, 8, 8)
    assert torch.allclose(dice_bce_loss(logits, lbl, _params()).total, dice_bce_loss(logits, lbl.long(), _params()).total)
