import numpy as np
import pytest

from fte.convert.gsd import degrade_to_target, resize_mask, scaled_size


def test_scaled_size():
    assert scaled_size(640, 0.02, 0.025) == 512
    assert scaled_size(500, 0.014, 0.025) == 280
    with pytest.raises(ValueError):
        scaled_size(10, 0.0, 0.025)


def test_degrade_shapes_and_dtype():
    img = np.full((500, 502, 3), 128, np.uint8)
    out = degrade_to_target(img, 0.014)
    assert out.shape == (280, 281, 3) and out.dtype == np.uint8
    assert img.sum() == 128 * img.size  # input untouched


def test_degrade_from_coarser_source_upsamples():
    img = np.zeros((100, 100, 3), np.uint8)
    assert degrade_to_target(img, 0.05).shape == (200, 200, 3)


def test_resize_mask_keeps_labels():
    m = np.zeros((10, 10), np.uint8)
    m[:5] = 255
    m[5:, 5:] = 1
    out = resize_mask(m, (20, 20))
    assert set(np.unique(out)) == {0, 1, 255}
