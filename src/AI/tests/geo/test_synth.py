"""Self-checks of tests/helpers/synth.py (other packages build their tests on it)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers.synth import (
    arc_mask,
    blobs_mask,
    rgb_from_mask,
    row_axes,
    striped_mask,
    with_nodata_corner,
)
from vineyard.geo.raster import valid_mask


def test_striped_mask_fraction_and_orientation() -> None:
    m = striped_mask((400, 400), angle_deg=0.0, spacing_px=40.0, width_px=10.0)
    assert m.mean() == pytest.approx(0.25, abs=0.01)
    assert (m == m[:, :1]).all()  # horizontal rows: constant along u
    tilted = striped_mask((400, 400), angle_deg=35.0, spacing_px=40.0, width_px=10.0, offset_px=7.0)
    assert tilted.mean() == pytest.approx(0.25, abs=0.02)


def test_row_axes_are_parallel_and_inside() -> None:
    axes = row_axes((200, 300), angle_deg=20.0, spacing_px=25.0, offset_px=3.0)
    assert len(axes) >= 10
    for a in axes:
        assert a.shape == (2, 2)
        assert a.min() >= -1e-9 and a[:, 0].max() <= 300 + 1e-9 and a[:, 1].max() <= 200 + 1e-9
        d = a[1] - a[0]
        assert abs(((np.degrees(np.arctan2(d[1], d[0])) - 20.0) + 90) % 180 - 90) < 1e-6


def test_row_axes_match_stripe_centres() -> None:
    m = striped_mask((100, 100), angle_deg=0.0, spacing_px=20.0, width_px=4.0, offset_px=10.0)
    axes = row_axes((100, 100), angle_deg=0.0, spacing_px=20.0, offset_px=10.0)
    for a in axes:
        v = int(a[0, 1])
        assert m[v, 50] or m[v - 1, 50]


def test_arc_and_blobs() -> None:
    arcs = arc_mask((200, 200), centre=(100.0, 100.0), r0_px=30.0, spacing_px=20.0, width_px=6.0, n_rows=3)
    assert arcs[100, 130] and arcs[100, 150] and arcs[100, 170] and not arcs[100, 190]
    blobs = blobs_mask((50, 50), [(10.0, 10.0), (40.0, 40.0)], 5.0)
    assert blobs[9, 9] and blobs[39, 39] and not blobs[25, 25]
    assert blobs.sum() == pytest.approx(2 * np.pi * 25, rel=0.1)


def test_rgb_from_mask_and_nodata_corner() -> None:
    m = striped_mask((64, 64), angle_deg=0.0, spacing_px=16.0, width_px=4.0)
    rgb = rgb_from_mask(m, noise_sigma=2.0, seed=4)
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 64, 3)
    assert np.array_equal(rgb, rgb_from_mask(m, noise_sigma=2.0, seed=4))
    black = with_nodata_corner(rgb_from_mask(m), 40, "br", ring_px=2)
    assert (black[-40:, -40:] == 0).all() and (black[-42, -42] == 5).all()
    valid = valid_mask(black, max_rgb=10, min_area_px=100, close_px=2, dilate_px=2)
    assert not valid[-42:, -42:].any() and valid[:10, :10].all()
