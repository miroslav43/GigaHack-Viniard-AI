"""Unit-agnostic geometry ops (the P0 subset; clip/notch are P1 stubs)."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from vineyard.geo.ops import (
    clip_box,
    clip_line,
    clip_polygonal,
    drop_consecutive_duplicates,
    make_valid_polygonal,
    notch_holes,
    orient_ccw,
    split_multi,
)

BOWTIE = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])


def test_orient_ccw_exterior_ccw_holes_cw() -> None:
    cw = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)], [[(2, 2), (4, 2), (4, 4), (2, 4)]])
    assert not cw.exterior.is_ccw and cw.interiors[0].is_ccw
    out = orient_ccw(cw)
    assert out.exterior.is_ccw
    assert not out.interiors[0].is_ccw
    assert out.equals(cw)
    assert not cw.exterior.is_ccw  # input untouched


def test_make_valid_polygonal_bowtie_two_triangles() -> None:
    parts = make_valid_polygonal(BOWTIE)
    assert len(parts) == 2
    assert all(isinstance(p, Polygon) and p.is_valid for p in parts)
    assert sum(p.area for p in parts) == pytest.approx(2.0)


def test_make_valid_polygonal_valid_passthrough_and_drops_non_polygonal() -> None:
    sq = box(0, 0, 1, 1)
    assert make_valid_polygonal(sq) == [sq]
    mixed = GeometryCollection([sq, LineString([(0, 0), (5, 5)]), Point(3, 3)])
    assert make_valid_polygonal(mixed) == [sq]
    assert make_valid_polygonal(Polygon()) == []
    collapsed = Polygon([(0, 0), (1, 1), (2, 2), (0, 0)])
    assert make_valid_polygonal(collapsed) == []


def test_make_valid_polygonal_multipolygon() -> None:
    mp = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    assert [p.bounds for p in make_valid_polygonal(mp)] == [(0, 0, 1, 1), (2, 2, 3, 3)]


def test_split_multi_flattens_nested() -> None:
    gc = GeometryCollection([MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), Point(9, 9), Polygon()])
    parts = split_multi(gc)
    assert [p.geom_type for p in parts] == ["Polygon", "Polygon", "Point"]
    assert split_multi(Point(1, 1)) == [Point(1, 1)]
    assert split_multi(Polygon()) == []


def test_drop_consecutive_duplicates_open_and_closed() -> None:
    line = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [2, 2]], float)
    np.testing.assert_array_equal(drop_consecutive_duplicates(line, closed=False), [[0, 0], [1, 1], [2, 2]])
    ring = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [0, 0]], float)
    np.testing.assert_array_equal(drop_consecutive_duplicates(ring, closed=True), [[0, 0], [1, 0], [1, 1]])
    assert drop_consecutive_duplicates(np.zeros((0, 2)), closed=True).shape == (0, 2)
    same = np.array([[3, 3], [3, 3]], float)
    np.testing.assert_array_equal(drop_consecutive_duplicates(same, closed=True), [[3, 3]])
    with pytest.raises(ValueError):
        drop_consecutive_duplicates(np.zeros((3,)), closed=False)


def test_drop_consecutive_duplicates_does_not_mutate() -> None:
    ring = np.array([[0, 0], [0, 0], [1, 1]], float)
    before = ring.copy()
    drop_consecutive_duplicates(ring, closed=False)
    np.testing.assert_array_equal(ring, before)


def test_p1_stubs_raise_not_implemented() -> None:
    sq = box(0, 0, 1, 1)
    with pytest.raises(NotImplementedError):
        clip_polygonal(sq, sq)
    with pytest.raises(NotImplementedError):
        clip_line(LineString([(0, 0), (1, 1)]), sq)
    with pytest.raises(NotImplementedError):
        clip_box((0, 0, 1, 1), (0, 0, 2, 2))
    with pytest.raises(NotImplementedError):
        notch_holes(sq, 1.0)
