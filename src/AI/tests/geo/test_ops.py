"""Unit-agnostic geometry ops: orientation/validity helpers (P0) and clip/notch (P1-GEO)."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from vineyard.geo import ops
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
SEED = 20260925
N_CASES = 60


def _assert_ring_clean(ring: np.ndarray) -> None:
    assert np.array_equal(ring[0], ring[-1])  # closed exactly once
    open_ring = ring[:-1]
    assert len(open_ring) >= 3
    np.testing.assert_array_equal(drop_consecutive_duplicates(open_ring, closed=True), open_ring)


def _assert_clean(polys: list[Polygon]) -> None:
    for p in polys:
        assert isinstance(p, Polygon) and p.is_valid and not p.is_empty and p.area > 0
        assert p.exterior.is_ccw
        assert all(not ring.is_ccw for ring in p.interiors)
        for ring in (p.exterior, *p.interiors):
            _assert_ring_clean(np.asarray(ring.coords))


# ---------------------------------------------------------------- P0 helpers


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


# ---------------------------------------------------------------- clip_polygonal


def test_clip_polygonal_across_tile_edge_keeps_inner_part() -> None:
    out = clip_polygonal(box(-5, 0, 5, 10), box(0, 0, 100, 100))
    assert len(out) == 1 and out[0].equals(box(0, 0, 5, 10))
    _assert_clean(out)


def test_clip_polygonal_multipolygon_result_is_exploded() -> None:
    u_shape = Polygon([(0, 0), (10, 0), (10, 10), (8, 10), (8, 2), (2, 2), (2, 10), (0, 10)])
    out = clip_polygonal(u_shape, box(-1, 5, 11, 11))
    assert len(out) == 2
    assert sorted(p.area for p in out) == pytest.approx([10.0, 10.0])
    _assert_clean(out)


def test_clip_polygonal_invalid_inputs_are_repaired() -> None:
    out = clip_polygonal(BOWTIE, box(-1, -1, 3, 3))
    assert len(out) == 2 and sum(p.area for p in out) == pytest.approx(2.0)
    _assert_clean(out)
    assert sum(p.area for p in clip_polygonal(box(0, 0, 2, 2), BOWTIE)) == pytest.approx(2.0)


def test_clip_polygonal_keeps_holes_ccw_exterior_cw_holes() -> None:
    cw_holed = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)], [[(4, 4), (6, 4), (6, 6), (4, 6)]])
    out = clip_polygonal(cw_holed, box(-1, -1, 11, 11))
    assert len(out) == 1 and len(out[0].interiors) == 1
    _assert_clean(out)


def test_clip_polygonal_non_polygonal_or_empty_results() -> None:
    sq = box(0, 0, 1, 1)
    assert clip_polygonal(sq, box(1, 0, 2, 1)) == []  # shared edge only
    assert clip_polygonal(sq, box(5, 5, 6, 6)) == []
    assert clip_polygonal(LineString([(0, 0), (1, 1)]), sq) == []
    assert clip_polygonal(Polygon(), sq) == []
    assert clip_polygonal(sq, Polygon()) == []
    assert clip_polygonal(sq, LineString([(0, 0), (1, 1)])) == []


def test_clip_polygonal_property_random_polygons() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(N_CASES):
        pts = rng.uniform(0, 100, size=(int(rng.integers(3, 12)), 2))
        geom = Polygon(pts)  # frequently self-intersecting
        x0, y0 = rng.uniform(0, 60, size=2)
        clip = box(x0, y0, x0 + rng.uniform(5, 60), y0 + rng.uniform(5, 60))
        out = clip_polygonal(geom, clip)
        _assert_clean(out)
        expected = shapely.make_valid(geom, method="structure").intersection(clip).area
        assert sum(p.area for p in out) == pytest.approx(expected, rel=1e-9, abs=1e-9)
        assert all(clip.buffer(1e-9).covers(p) for p in out)


# ---------------------------------------------------------------- clip_line


def test_clip_line_across_tile_edge_keeps_direction() -> None:
    out = clip_line(LineString([(-10, 5), (10, 5)]), box(0, 0, 20, 20))
    assert out is not None and list(out.coords) == [(0.0, 5.0), (10.0, 5.0)]
    back = clip_line(LineString([(10, 5), (-10, 5)]), box(0, 0, 20, 20))
    assert back is not None and list(back.coords) == [(10.0, 5.0), (0.0, 5.0)]


def test_clip_line_across_nodata_hole_gives_one_line_first_to_last_valid() -> None:
    tile = box(0, 0, 20, 20)
    clip = tile.difference(box(8, -1, 12, 21))  # nodata band splits the tile
    line = LineString([(-5, 5), (3, 6), (17, 8), (25, 9)])
    out = clip_line(line, clip)
    assert isinstance(out, LineString)
    expected = line.intersection(tile)
    assert out.length == pytest.approx(expected.length)
    assert out.coords[0] == pytest.approx(expected.coords[0])
    assert out.coords[-1] == pytest.approx(expected.coords[-1])
    assert (3.0, 6.0) in list(out.coords) and (17.0, 8.0) in list(out.coords)


def test_clip_line_black_corner_drops_the_part_over_black_at_the_end() -> None:
    clip = box(0, 0, 20, 20).difference(box(-1, 12, 8, 21))  # black top-left corner
    out = clip_line(LineString([(-5, 25), (25, -5)]), clip)
    assert out is not None
    np.testing.assert_allclose(np.asarray(out.coords), [[8.0, 12.0], [20.0, 0.0]], atol=1e-9)


def test_clip_line_leaving_and_reentering_takes_longest_run_inside() -> None:
    tile = box(0, 0, 10, 10)
    line = LineString([(1, 5), (4, 5), (4, 15), (6, 15), (6, 5), (9.5, 5)])
    out = clip_line(line, tile)
    assert out is not None
    assert tile.buffer(1e-9).covers(out)
    assert out.length == pytest.approx(8.5)  # (6,10)->(6,5)->(9.5,5), longer than the 8.0 first run


def test_clip_line_parts_touching_in_a_point_are_merged() -> None:
    clip = MultiPolygon([box(0, 0, 5, 5), box(5, 5, 10, 10)])  # corner-touching cells
    out = clip_line(LineString([(-1, -1), (11, 11)]), clip)
    assert out is not None
    np.testing.assert_allclose(np.asarray(out.coords)[[0, -1]], [[0.0, 0.0], [10.0, 10.0]], atol=1e-9)


def test_clip_line_outside_touching_and_degenerate_return_none() -> None:
    sq = box(0, 0, 10, 10)
    assert clip_line(LineString([(20, 0), (30, 0)]), sq) is None
    assert clip_line(LineString([(-5, -5), (0, 0)]), sq) is None  # touches in one point
    assert clip_line(LineString([(-5, 0), (15, 0)]), Polygon()) is None
    assert clip_line(LineString([(3, 3), (3, 3)]), sq) is None
    assert clip_line(LineString(), sq) is None


def test_line_or_none_rejects_points_and_collapsed_lines() -> None:
    assert ops._line_or_none(Point(1, 1)) is None
    assert ops._line_or_none(LineString([(2, 2), (2, 2)])) is None


def test_clip_line_drops_duplicate_vertices_and_rejects_non_lines() -> None:
    out = clip_line(LineString([(1, 1), (1, 1), (5, 1), (5, 1), (8, 1)]), box(0, 0, 10, 10))
    assert out is not None and list(out.coords) == [(1.0, 1.0), (5.0, 1.0), (8.0, 1.0)]
    with pytest.raises(TypeError):
        clip_line(MultiLineString([[(0, 0), (1, 1)]]), box(0, 0, 10, 10))  # type: ignore[arg-type]


def test_clip_line_property_straight_lines_through_holed_clip() -> None:
    rng = np.random.default_rng(SEED)
    tile = box(0, 0, 100, 100)
    for _ in range(N_CASES):
        hx, hy = rng.uniform(10, 80, size=2)
        clip = tile.difference(box(hx, hy, hx + rng.uniform(1, 15), hy + rng.uniform(1, 15)))
        a, b = rng.uniform(-50, 150, size=(2, 2))
        line = LineString([a, b])
        out = clip_line(line, clip)
        in_clip = line.intersection(clip).length
        if in_clip == 0:
            assert out is None
            continue
        assert out is not None and out.is_valid and len(out.coords) >= 2
        assert in_clip - 1e-6 <= out.length <= line.intersection(tile).length + 1e-6
        assert tile.buffer(1e-6).covers(out)
        assert out.coords[0] == pytest.approx(line.interpolate(line.project(Point(out.coords[0]))).coords[0])


# ---------------------------------------------------------------- clip_box


def test_clip_box_inside_partial_outside() -> None:
    bounds = (0.0, 0.0, 2048.0, 2048.0)
    assert clip_box((10, 20, 30, 40), bounds) == (10, 20, 30, 40)
    assert clip_box((-10, 2000, 30, 2100), bounds) == (0.0, 2000, 30, 2048.0)
    assert clip_box((3000, 0, 3100, 10), bounds) is None
    assert clip_box((2048, 0, 2100, 10), bounds) is None  # zero width after clipping


def test_clip_box_rejects_inverted_boxes() -> None:
    with pytest.raises(ValueError, match="xyxy"):
        clip_box((5, 0, 1, 1), (0, 0, 10, 10))
    with pytest.raises(ValueError, match="bounds"):
        clip_box((0, 0, 1, 1), (0, 10, 10, 0))
    with pytest.raises(ValueError, match="finite"):
        clip_box((0, 0, float("nan"), 1), (0, 0, 10, 10))


# ---------------------------------------------------------------- notch_holes

SQUARE_WITH_HOLE = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(2, 4), (4, 4), (4, 6), (2, 6)]])


def test_notch_holes_no_hole_returns_ccw_copy() -> None:
    cw = Polygon([(0, 0), (0, 5), (5, 5), (5, 0)])
    out = notch_holes(cw, 0.5)
    assert len(out) == 1 and out[0].equals(cw) and out[0].exterior.is_ccw
    assert notch_holes(Polygon(), 0.5) == []


def test_notch_holes_square_hole_single_polygon_area_loss_bounded() -> None:
    width = 0.5
    out = notch_holes(SQUARE_WITH_HOLE, width)
    assert len(out) == 1 and len(out[0].interiors) == 0
    _assert_clean(out)
    distance = 2.0  # hole to the left edge; the notch is centred on the hole's facing edge
    loss = SQUARE_WITH_HOLE.area - out[0].area
    assert loss == pytest.approx(width * distance)
    assert SQUARE_WITH_HOLE.covers(out[0])
    assert out[0].intersection(box(-1, 4.75, 2, 5.25)).area == pytest.approx(0.0)


def test_notch_holes_several_holes_and_reflex_vertex() -> None:
    holes = [
        [(2, 2), (3, 2), (3, 3), (2, 3)],
        [(6, 6), (7, 6), (7, 7), (6, 7)],
        [(4.5, 4.5), (5, 4.5), (5, 5)],
    ]
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes)
    out = notch_holes(poly, 0.2)
    _assert_clean(out)
    assert all(len(p.interiors) == 0 for p in out)
    arrow = Polygon(
        [(0, 0), (10, 0), (10, 10), (5, 5.5), (0, 10)], [[(4.8, 3), (5.2, 3), (5.2, 4.5), (4.8, 4.5)]]
    )
    out_arrow = notch_holes(arrow, 0.3)
    _assert_clean(out_arrow)
    assert len(out_arrow) == 1 and not out_arrow[0].interiors


def test_notch_holes_full_width_hole_gives_two_polygons() -> None:
    full = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)], [[(5, 0), (6, 2), (5, 4), (4, 2)]])
    assert not full.is_valid  # the hole touches both long sides
    out = notch_holes(full, 0.5)
    assert len(out) == 2
    _assert_clean(out)
    assert sum(p.area for p in out) == pytest.approx(40.0 - 4.0)


def test_notch_holes_hole_touching_exterior_in_one_point() -> None:
    touching = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)], [[(0, 2), (8, 1), (8, 3)]])
    assert touching.is_valid
    out = notch_holes(touching, 0.5)
    assert len(out) == 1 and not out[0].interiors
    _assert_clean(out)


def test_notch_holes_rejects_bad_arguments() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="notch_width"):
            notch_holes(SQUARE_WITH_HOLE, bad)
    with pytest.raises(TypeError):
        notch_holes(MultiPolygon([box(0, 0, 1, 1)]), 0.5)  # type: ignore[arg-type]


def test_notch_holes_raises_when_a_cut_fails_to_open_the_hole(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops, "_notch", lambda *_args: Polygon())
    with pytest.raises(ValueError, match="holes remain"):
        notch_holes(SQUARE_WITH_HOLE, 0.5)


def test_notch_holes_is_deterministic_and_does_not_mutate() -> None:
    before = SQUARE_WITH_HOLE.wkb
    first, second = notch_holes(SQUARE_WITH_HOLE, 0.5), notch_holes(SQUARE_WITH_HOLE, 0.5)
    assert [p.wkb for p in first] == [p.wkb for p in second]
    assert SQUARE_WITH_HOLE.wkb == before


def _round_trip(poly: Polygon, decimals: int) -> Polygon:
    ring = drop_consecutive_duplicates(np.round(np.asarray(poly.exterior.coords), decimals), closed=True)
    return Polygon(ring)


def test_notch_holes_property_round_trip_through_tenth_pixel_stays_valid() -> None:
    rng = np.random.default_rng(SEED)
    notch_px = 1.0
    for _ in range(N_CASES):
        w, h = rng.uniform(30, 300, size=2)
        x0, y0 = rng.uniform(0, 1500, size=2)
        outer = Polygon([(x0, y0), (x0 + w, y0 + rng.uniform(-5, 5)), (x0 + w, y0 + h), (x0, y0 + h)])
        hx = x0 + rng.uniform(2, w - 12)
        hy = y0 + rng.uniform(7, h - 17)
        hole = [(hx, hy), (hx + rng.uniform(2, 10), hy), (hx + 5, hy + rng.uniform(2, 10))]
        poly = Polygon(outer.exterior.coords, [hole])
        if not poly.is_valid:
            continue
        out = notch_holes(poly, notch_px)
        _assert_clean(out)
        for part in out:
            assert not part.interiors
            rounded = _round_trip(part, 1)
            assert rounded.is_valid and rounded.exterior.is_ccw and not rounded.interiors
            assert rounded.area == pytest.approx(part.area, rel=0.02)
