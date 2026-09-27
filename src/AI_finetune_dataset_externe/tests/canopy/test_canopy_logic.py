from __future__ import annotations

import numpy as np
import pytest
import torch

from fte.canopy.dataset import MixSpec, draw_centre, mosaic_2x2, window_at
from fte.canopy.infer import blend_weight, predict_image, window_starts
from fte.canopy.train import lr_factor
from fte.canopy.validate import MatchCounts, instances_from_probs, match_instances
from fte.convert.targets import IGNORE

# ------------------------------------------------------------------ sampling


def test_window_at_clamps_inside_tile() -> None:
    assert window_at(0, 0, 608) == (0, 0)
    assert window_at(2047, 2047, 608) == (2048 - 608, 2048 - 608)
    assert window_at(1000, 500, 600) == (700, 200)
    with pytest.raises(ValueError):
        window_at(0, 0, 4096)


def test_draw_centre_hits_weighted_cell() -> None:
    w = np.zeros((32, 32), np.float32)
    w[3, 7] = 1.0
    rng = np.random.default_rng(0)
    for _ in range(20):
        cx, cy = draw_centre(w, 64, 1.0, rng)
        assert 7 * 64 <= cx < 8 * 64 and 3 * 64 <= cy < 4 * 64


def test_draw_centre_uniform_when_no_weight() -> None:
    cx, cy = draw_centre(np.zeros((4, 4)), 64, 1.0, np.random.default_rng(1), tile_px=256)
    assert 0 <= cx < 256 and 0 <= cy < 256


def test_mosaic_pads_with_ignore() -> None:
    parts = [(np.full((10, 12, 3), i + 1, np.uint8), np.zeros((10, 12, 3), np.uint8)) for i in range(4)]
    img, lbl = mosaic_2x2(parts)
    assert img.shape == (24, 24, 3) and lbl.shape == (24, 24, 3)
    assert img[0, 0, 0] == 1 and img[0, 12, 0] == 2 and img[12, 0, 0] == 3 and img[12, 12, 0] == 4
    assert (lbl[10:12, :, :] == IGNORE).all() and lbl[0, 0, 0] == 0


def test_mix_probs_renormalise_missing_sources() -> None:
    np.testing.assert_allclose(MixSpec().probs(False, True), [2 / 3, 0, 1 / 3])
    with pytest.raises(ValueError):
        MixSpec(p_vine=0.0).probs(False, False)


# ------------------------------------------------------------------ sliding window


def test_window_starts_cover_and_overlap() -> None:
    s = window_starts(2048, 512, 96)
    assert s[0] == 0 and s[-1] == 2048 - 512
    assert all(b - a <= 512 - 96 for a, b in zip(s, s[1:], strict=False))
    assert window_starts(512, 512, 96) == [0]
    with pytest.raises(ValueError):
        window_starts(100, 512, 96)


def test_blend_weight_shape_and_centre() -> None:
    w = blend_weight(64, 16)
    assert w.shape == (64, 64) and w[32, 32] == pytest.approx(1.0) and w[0, 0] < 0.01 and w.min() > 0


class _ConstModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logit = 4 * (R channel - 0.5)
        return torch.cat([4 * (x[:, :1] - 0.5), torch.zeros_like(x[:, :1])], dim=1)


def test_predict_image_blends_seamlessly_and_zeroes_nodata() -> None:
    rgb = np.full((96, 160, 3), 200, np.uint8)
    rgb[:8, :8] = 0
    p = predict_image(_ConstModel(), rgb, torch.device("cpu"), tta=4, batch=3, win=64, overlap=16)
    assert p.shape == (2, 96, 160)
    expected = 1 / (1 + np.exp(-4 * (200 / 255 - 0.5)))
    np.testing.assert_allclose(p[0, 8:, 8:], expected, rtol=1e-5)
    assert (p[:, :8, :8] == 0).all()


# ------------------------------------------------------------------ instance F1


def test_instances_split_by_contact() -> None:
    c0 = np.zeros((20, 40), np.float32)
    c0[5:15, 5:35] = 0.9
    c1 = np.zeros_like(c0)
    c1[:, 19:21] = 0.9
    inst = instances_from_probs(c0, c1, min_px=20)
    assert inst.max() == 2 and inst[10, 10] != inst[10, 30]


def test_match_instances_counts() -> None:
    gt = np.zeros((20, 40), np.uint16)
    gt[5:15, 5:15] = 1
    gt[5:15, 25:35] = 2
    pred = np.zeros((20, 40), np.int32)
    pred[5:15, 5:15] = 1  # perfect match
    pred[0:4, 20:40] = 2  # false positive
    m = match_instances(pred, gt, np.ones((20, 40), bool), min_px=20)
    assert m == MatchCounts(1, 1, 1) and m.f1 == pytest.approx(0.5)


def test_match_ignores_predictions_outside_valid() -> None:
    gt = np.zeros((10, 10), np.uint16)
    pred = np.zeros((10, 10), np.int32)
    pred[:, :5] = 1
    valid = np.zeros((10, 10), bool)
    valid[:, 8:] = True
    assert match_instances(pred, gt, valid) == MatchCounts(0, 0, 0)


def test_lr_factor_warmup_then_cosine() -> None:
    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100) == pytest.approx(0.02)
    assert lr_factor(55, 10, 100) < lr_factor(20, 10, 100)
