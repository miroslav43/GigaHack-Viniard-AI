from __future__ import annotations

import numpy as np
import pytest

from fte.convert.coco import StoreImage, StoreObject
from fte.waste.dataset import EpochSampler, MixSpec, crop_around, pad_with_ignore, unannotated_far


def test_pad_with_ignore_label_255_and_flag_0() -> None:
    img = np.full((10, 20, 3), 7, np.uint8)
    lbl = np.ones((10, 20), np.uint8)
    down = np.ones((10, 20), np.uint8)
    pi, pl, pd = pad_with_ignore(img, lbl, 30, down)
    assert pi.shape == (30, 30, 3) and pl.shape == (30, 30)
    assert (pl == 255).sum() == 30 * 30 - 200 and (pl == 1).sum() == 200
    assert pd.sum() == 200


def test_crop_around_contains_centre() -> None:
    rng = np.random.default_rng(0)
    lbl = np.zeros((100, 100), np.uint8)
    lbl[90, 5] = 1
    img = np.zeros((100, 100, 3), np.uint8)
    for _ in range(20):
        _, cl, _ = crop_around(img, lbl, np.zeros_like(lbl), (5.0, 90.0), 40, rng)
        assert cl.shape == (40, 40) and (cl == 1).sum() == 1


def test_unannotated_far_excludes_margin() -> None:
    obj = StoreObject("waste", (10, 10, 5, 5), ())
    rec = StoreImage(1, "a.jpg", "s", "pos", 50, 50, (obj,))
    lbl = np.zeros((50, 50), np.uint8)
    lbl[10:15, 10:15] = 1
    flag = unannotated_far(rec, lbl, 4)
    assert flag[12, 12] == 0 and flag[12, 17] == 0 and flag[40, 40] == 1


def test_mix_probs_normalised() -> None:
    p = MixSpec().probs()
    assert p.sum() == pytest.approx(1.0) and p[0] == pytest.approx(0.30)
    with pytest.raises(ValueError):
        MixSpec(paste=-1).probs()


def test_epoch_sampler_disjoint_slices() -> None:
    s = EpochSampler(5)
    s.set_epoch(0)
    a = sorted(s)
    s.set_epoch(2)
    b = sorted(s)
    assert a == [0, 1, 2, 3, 4] and b == [10, 11, 12, 13, 14]
