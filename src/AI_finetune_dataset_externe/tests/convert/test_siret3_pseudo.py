from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, MultiLineString

from fte.convert.siret3_pseudo import (
    PseudoSpec,
    TileKind,
    axis_distance_px,
    lines_to_px,
    pseudo_labels,
    sampling_weights,
)
from fte.convert.targets import IGNORE

H = W = 128


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vertical row axis at x = 64; canopy block x in [58, 70), y in [20, 60); green grass at x >= 100."""
    rgb = np.full((H, W, 3), 120, np.uint8)
    rgb[:, 100:] = (60, 150, 60)
    rgb[20:60, 58:70] = (50, 140, 50)
    canopy = np.zeros((H, W), np.uint8)
    canopy[20:60, 58:70] = 1
    dist = axis_distance_px([np.array([[64.5, 0.0], [64.5, 128.0]])], (H, W))
    return rgb, np.ones((H, W), bool), canopy, dist


def test_axis_distance_zero_on_line_and_inf_without_lines() -> None:
    d = axis_distance_px([np.array([[10.5, 0.0], [10.5, 50.0]])], (50, 50))
    assert d[25, 10] == 0 and abs(d[25, 30] - 20) < 0.5
    assert np.isinf(axis_distance_px([], (4, 4))).all()


def test_lines_to_px_handles_multilines() -> None:
    geoms = [LineString([(0, 0), (1, 1)]), MultiLineString([[(0, 0), (2, 0)], [(3, 3), (4, 4)]])]
    out = lines_to_px(geoms, lambda xy: xy * 10)
    assert len(out) == 3 and out[0][1].tolist() == [10, 10]


def test_vineyard_c0_bands() -> None:
    rgb, valid, canopy, dist = _scene()
    lbl = pseudo_labels(rgb, valid, canopy, dist, TileKind.VINEYARD)
    c0 = lbl[..., 0]
    assert c0[40, 64] == 1  # inside canopy
    assert c0[40, 58] == IGNORE  # polygon edge band
    assert c0[80, 64] == 0  # row gap, on the axis
    assert c0[80, 64 + 16] == IGNORE  # 0.40 m from the axis: ignore band
    assert c0[80, 64 + 30] == 0  # 0.75 m: background
    assert (lbl[..., 1] == IGNORE).all()


def test_vineyard_c2_ground_veg_outside_corridor_only() -> None:
    rgb, valid, canopy, dist = _scene()
    c2 = pseudo_labels(rgb, valid, canopy, dist, TileKind.VINEYARD)[..., 2]
    assert c2[40, 110] == 1  # green far from the row
    assert c2[40, 64] == IGNORE  # vine canopy is not ground vegetation
    assert c2[40, 30] == 0  # grey soil


def test_partial_tile_far_field_ignored() -> None:
    rgb, valid, canopy, dist = _scene()
    lbl = pseudo_labels(rgb, valid, canopy, dist, TileKind.PARTIAL)
    assert lbl[80, 64 + 30, 0] == IGNORE and lbl[40, 64, 0] == 1
    assert lbl[40, 110, 2] == IGNORE and lbl[40, 30, 2] == 0


def test_negative_and_nodata() -> None:
    rgb, valid, canopy, dist = _scene()
    rgb[:10] = 0
    valid[:, :5] = False
    lbl = pseudo_labels(rgb, valid, np.zeros_like(canopy), dist, TileKind.NEGATIVE)
    assert (lbl[:10] == IGNORE).all() and (lbl[:, :5] == IGNORE).all()
    assert lbl[40, 64, 0] == 0 and lbl[40, 110, 2] == 1 and lbl[40, 30, 2] == 0


def test_sampling_weights_follow_corridor() -> None:
    rgb, valid, canopy, dist = _scene()
    lbl = pseudo_labels(rgb, valid, canopy, dist, TileKind.VINEYARD, PseudoSpec())
    w = sampling_weights(lbl, dist, TileKind.VINEYARD, cell_px=64)
    assert w.shape == (2, 2) and w[0, 1] > 0 and w[0, 0] > 0
    assert w.sum() > 0
