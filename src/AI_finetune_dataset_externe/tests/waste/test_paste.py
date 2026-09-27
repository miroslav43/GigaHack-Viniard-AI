from __future__ import annotations

import numpy as np

from fte.convert.coco import StoreImage, StoreObject
from fte.convert.paste import (
    Instance,
    PasteSpec,
    composite,
    image_instances,
    luminance_match,
    paste_many,
    transform_instance,
)


def _inst(side: int = 12) -> Instance:
    rgb = np.full((side, side, 3), 250, np.uint8)
    mask = np.zeros((side, side), bool)
    mask[2:-2, 2:-2] = True
    return Instance(rgb, mask, "t")


def test_transform_alpha_range_and_shape() -> None:
    obj, alpha = transform_instance(_inst(), 45.0, 1.2)
    assert obj.shape[:2] == alpha.shape
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert alpha.max() > 0.9


def test_composite_label_and_inputs_untouched() -> None:
    img = np.zeros((50, 50, 3), np.uint8)
    lbl = np.zeros((50, 50), np.uint8)
    lbl[0:5, 0:5] = 255
    alpha = np.zeros((10, 10), np.float32)
    alpha[2:8, 2:8] = 1.0
    obj = np.full((10, 10, 3), 200, np.uint8)
    new_img, new_lbl, box = composite(img, lbl, obj, alpha, 20, 30)
    assert img.max() == 0 and lbl.max() == 255 and (lbl == 1).sum() == 0
    assert box == (22.0, 32.0, 28.0, 38.0)
    assert (new_lbl == 1).sum() == 36
    assert (new_lbl[0:5, 0:5] == 255).all()
    assert new_img[35, 25].tolist() == [200, 200, 200]


def test_composite_clips_at_border() -> None:
    img = np.zeros((20, 20, 3), np.uint8)
    lbl = np.zeros((20, 20), np.uint8)
    alpha = np.ones((10, 10), np.float32)
    _, new_lbl, box = composite(img, lbl, np.full((10, 10, 3), 9, np.uint8), alpha, -5, 15)
    assert box == (0.0, 15.0, 5.0, 20.0)
    assert (new_lbl == 1).sum() == 25


def test_paste_many_boxes_inside_and_labelled() -> None:
    rng = np.random.default_rng(0)
    img = np.full((96, 96, 3), 90, np.uint8)
    lbl = np.zeros((96, 96), np.uint8)
    new_img, new_lbl, boxes = paste_many(img, lbl, [_inst(16)], PasteSpec(), rng, n=3)
    assert boxes and set(np.unique(new_lbl)) <= {0, 1}
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= 96 and 0 <= y0 < y1 <= 96
    assert img.mean() == 90 and lbl.max() == 0


def test_luminance_match_moves_toward_background() -> None:
    obj = np.full((4, 4, 3), 200, np.uint8)
    alpha = np.ones((4, 4), np.float32)
    bg = np.full((4, 4, 3), 100, np.uint8)
    out = luminance_match(obj, alpha, bg, 0.5)
    assert 100 < out.mean() < 200
    assert obj.mean() == 200


def test_image_instances_area_filter() -> None:
    rgb = np.zeros((100, 100, 3), np.uint8)
    big = StoreObject("waste", (10, 10, 20, 20), ((10, 10, 30, 10, 30, 30, 10, 30),))
    tiny = StoreObject("waste", (60, 60, 2, 2), ((60, 60, 62, 60, 62, 62),))
    img = StoreImage(1, "a.jpg", "s", "pos", 100, 100, (big, tiny))
    inst = image_instances(rgb, img, "t")
    assert len(inst) == 1 and inst[0].mask.sum() >= 400
