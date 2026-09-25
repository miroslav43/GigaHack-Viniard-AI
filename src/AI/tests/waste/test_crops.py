from __future__ import annotations

import numpy as np
import pytest

from vineyard.perception.waste.crops import (
    CropWindow,
    box_to_crop,
    box_to_tile,
    crop_window,
    extract_crop,
    sam_window,
)
from vineyard.perception.waste.types import BoxPx


def test_small_box_gets_min_side_centred() -> None:
    w = crop_window(BoxPx(100, 100, 120, 110), 0.5, 64, 2048)
    assert w == CropWindow(78, 73, 64)


def test_context_side() -> None:
    assert crop_window(BoxPx(500, 500, 560, 540), 0.5, 64, 2048).side == 90


def test_crop_near_edges_is_shifted_inside() -> None:
    w = crop_window(BoxPx(2040, 0, 2048, 5), 0.5, 64, 2048)
    assert (w.x0, w.y0, w.side) == (1984, 0, 64)
    big = crop_window(BoxPx(0, 0, 2048, 2048), 0.5, 64, 2048)
    assert big == CropWindow(0, 0, 2048)


def test_sam_window_clamped() -> None:
    w = sam_window((10.0, 2040.0), 512, 2048)
    assert w.as_xyxy() == (0.0, 1536.0, 512.0, 2048.0)


def test_box_round_trip() -> None:
    w = CropWindow(78, 73, 64)
    b = (100.0, 100.0, 120.0, 110.0)
    for out in (None, 224):
        assert box_to_tile(box_to_crop(b, w, out), w, out) == pytest.approx(b)
    assert box_to_crop(b, w) == (22.0, 27.0, 42.0, 37.0)
    assert box_to_crop(b, w, 128) == (44.0, 54.0, 84.0, 74.0)


def test_extract_crop_shape_and_content() -> None:
    img = np.zeros((256, 256, 3), np.uint8)
    img[100:110, 100:120] = 255
    crop = extract_crop(img, crop_window(BoxPx(100, 100, 120, 110), 0.5, 64, 256), 224)
    assert crop.shape == (224, 224, 3) and crop.dtype == np.uint8
    assert crop[112, 112].tolist() == [255, 255, 255]
    assert crop[0, 0].tolist() == [0, 0, 0]
    small = extract_crop(img, CropWindow(0, 0, 256), 64)
    assert small.shape == (64, 64, 3)


def test_extract_crop_outside_raises() -> None:
    with pytest.raises(ValueError, match="outside"):
        extract_crop(np.zeros((32, 32, 3), np.uint8), CropWindow(0, 0, 64), 64)


def test_bad_parameters_raise() -> None:
    with pytest.raises(ValueError):
        crop_window(BoxPx(0, 0, 1, 1), -0.1, 64, 2048)
