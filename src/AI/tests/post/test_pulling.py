"""String pulling between anchors on the eroded domain (arch §4.11.5, design 04 §3.9)."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Polygon, box

from vineyard.route.pulling import string_pull

CORRIDOR = box(0.0, -2.0, 40.0, 2.0)
ERODED = CORRIDOR.buffer(-0.3, join_style="mitre")
shapely.prepare(ERODED)


def _zigzag() -> np.ndarray:
    xs = np.arange(1.0, 39.0, 1.0)
    ys = np.where(np.arange(len(xs)) % 2 == 0, -1.0, 1.0)
    return np.column_stack([xs, ys])


def test_zigzag_is_shortened_keeping_anchors_and_domain() -> None:
    xy = _zigzag()
    anchors = (0, 20, len(xy) - 1)
    out, new_anchors = string_pull(xy, anchors, ERODED, lookahead=200)
    assert LineString(out).length < LineString(xy).length * 0.6
    assert [tuple(out[a]) for a in new_anchors] == [tuple(xy[a]) for a in anchors]
    assert shapely.covered_by(LineString(out), ERODED)
    assert len(out) == 3


def test_corner_of_an_l_corridor_is_not_cut() -> None:
    ell = Polygon([(0, 0), (20, 0), (20, 20), (16, 20), (16, 4), (0, 4)])
    eroded = ell.buffer(-0.3, join_style="mitre")
    xy = np.array([[1, 2], [5, 2], [10, 2], [18, 2], [18, 10], [18, 19]], float)
    out, anchors = string_pull(xy, (0, len(xy) - 1), eroded, lookahead=50)
    assert shapely.covered_by(LineString(out), eroded)
    assert len(out) == 3 and tuple(out[1]) == (18.0, 2.0)
    assert anchors == (0, 2)


def test_lookahead_one_keeps_the_line() -> None:
    xy = _zigzag()
    out, anchors = string_pull(xy, (0, len(xy) - 1), ERODED, lookahead=1)
    assert np.array_equal(out, xy) and anchors == (0, len(xy) - 1)


def test_out_and_back_tip_that_is_not_an_anchor_collapses() -> None:
    xy = np.array([[1, 0], [11, 0], [21, 0], [11, 0], [1, 0]], float)
    out, anchors = string_pull(xy, (0, 4), ERODED, lookahead=10)
    assert out.tolist() == [[1, 0]] and anchors == (0, 0)


def test_out_and_back_tip_anchor_is_kept() -> None:
    xy = np.array([[1, 0], [11, 0], [21, 0], [11, 0], [1, 0]], float)
    out, anchors = string_pull(xy, (0, 2, 4), ERODED, lookahead=10)
    assert out.tolist() == [[1, 0], [21, 0], [1, 0]]
    assert anchors == (0, 1, 2)
    assert not LineString(out).is_simple


def test_outside_detour_is_pulled_back_inside() -> None:
    xy = np.array([[1, 0], [1, 1.9], [3, 1.9], [3, 0]], float)  # y=1.9 is outside eroded (1.7)
    out, _ = string_pull(xy, (0, 3), ERODED, lookahead=10)
    assert out.tolist() == [[1, 0], [3, 0]]


def test_vertex_outside_eroded_is_left_in_place() -> None:
    xy = np.array([[1, 0], [5, 1.9], [9, 0]], float)
    out, _ = string_pull(xy, (0, 1, 2), ERODED, lookahead=10)
    assert out.tolist() == xy.tolist()


def test_rejects_bad_anchors() -> None:
    with pytest.raises(ValueError, match="anchors"):
        string_pull(_zigzag(), (3, 1), ERODED, lookahead=5)
    with pytest.raises(ValueError, match="lookahead"):
        string_pull(_zigzag(), (0, 5), ERODED, lookahead=0)
