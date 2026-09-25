"""Walking domain: union, seam closing, subtraction, erosion (design 04 §3.6, arch §4.11.1)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, box

from tests.post.passable_factories import START_XY, mini_vineyard, route_inputs
from vineyard.geo.ops import split_multi
from vineyard.route.cells import CellIndex, local_apply, quadtree_cells
from vineyard.route.domain import (
    DomainParams,
    DomainSet,
    build_domain,
    close_seams,
    domain_from_raw,
    passable_parts,
)

PARAMS = DomainParams(grid_size_m=0.001, inner_buffer_m=0.05, eroded_buffer_m=0.3, seam_close_m=0.05,
                      subtract_canopies=True)


def test_params_from_config_matches_default_yaml() -> None:
    from vineyard.config import load_config

    cfg = load_config()
    assert DomainParams.from_config(cfg.route.domain) == PARAMS


def test_params_reject_eroded_smaller_than_inner() -> None:
    with pytest.raises(ValueError, match="eroded_buffer_m"):
        DomainParams(grid_size_m=0.001, inner_buffer_m=0.3, eroded_buffer_m=0.05, seam_close_m=0.0,
                     subtract_canopies=False)


def test_real_passages_two_polygons_and_area() -> None:
    inputs = route_inputs()
    dom = build_domain((), inputs.passages, None, (), PARAMS)
    assert isinstance(dom.raw, MultiPolygon)
    assert dom.n_components == 2
    assert 39_950.0 < dom.raw.area < 40_001.0


def test_real_start_inside_inner_and_eroded() -> None:
    inputs = route_inputs()
    dom = build_domain((), inputs.passages, inputs.forbidden, (), PARAMS)
    start = Point(START_XY)
    assert start.coords[0] == pytest.approx(inputs.start_xy)
    assert shapely.contains(dom.inner, start)
    assert shapely.contains(dom.eroded, start)
    assert 0.85 <= dom.raw.boundary.distance(start) <= 0.90


def test_real_forbidden_only_touches_passages() -> None:
    # Vector check: the ~19 m2 raster overlap of design 04 §3.0 is a rasterisation artefact.
    inputs = route_inputs()
    plain = build_domain((), inputs.passages, None, (), PARAMS)
    minus = build_domain((), inputs.passages, inputs.forbidden, (), PARAMS)
    assert inputs.passages.distance(inputs.forbidden) == 0.0
    assert abs(plain.raw.area - minus.raw.area) < 1e-3
    assert minus.n_components == 2


def test_seam_gap_of_1cm_closes_but_20cm_stays_open() -> None:
    left = box(0, 0, 10, 2)
    closed = build_domain((left, box(10.01, 0, 20, 2)), MultiPolygon(), None, (), PARAMS)
    opened = build_domain((left, box(10.2, 0, 20, 2)), MultiPolygon(), None, (), PARAMS)
    assert closed.n_components == 1
    assert opened.n_components == 2


def test_seam_closing_keeps_outer_corners_square() -> None:
    dom = build_domain((box(0, 0, 10, 2),), MultiPolygon(), None, (), PARAMS)
    assert dom.raw.area == pytest.approx(20.0, abs=1e-3)


def test_bowtie_is_repaired() -> None:
    bowtie = Polygon([(0, 0), (4, 4), (4, 0), (0, 4), (0, 0)])
    assert not bowtie.is_valid
    dom = build_domain((bowtie,), MultiPolygon(), None, (), PARAMS)
    assert dom.raw.is_valid
    assert dom.raw.area == pytest.approx(8.0, abs=1e-3)


def test_canopy_and_forbidden_are_subtracted() -> None:
    strip = box(0, 0, 20, 3)
    canopy = box(5, 1, 6, 2)
    forbidden = box(15, -1, 25, 4)
    dom = build_domain((strip,), MultiPolygon(), forbidden, (canopy,), PARAMS)
    assert dom.raw.area == pytest.approx(15 * 3 - 1.0, abs=1e-3)
    assert not dom.raw.contains(Point(5.5, 1.5))


def test_canopies_kept_when_subtraction_disabled() -> None:
    params = dataclasses.replace(PARAMS, subtract_canopies=False)
    dom = build_domain((box(0, 0, 20, 3),), MultiPolygon(), None, (box(5, 1, 6, 2),), params)
    assert dom.raw.area == pytest.approx(60.0, abs=1e-3)


def test_inner_and_eroded_are_buffers_and_prepared() -> None:
    dom = build_domain((box(0, 0, 10, 4),), MultiPolygon(), None, (), PARAMS)
    assert dom.inner.area == pytest.approx(9.9 * 3.9, abs=1e-3)
    assert dom.eroded.area == pytest.approx(9.4 * 3.4, abs=1e-3)
    assert shapely.is_prepared(dom.inner)
    assert shapely.is_prepared(dom.eroded)
    assert isinstance(dom.inner, MultiPolygon) and isinstance(dom.eroded, MultiPolygon)


def test_thin_part_vanishes_from_eroded_but_not_raw() -> None:
    dom = build_domain((box(0, 0, 10, 0.4), box(0, 5, 10, 9)), MultiPolygon(), None, (), PARAMS)
    assert dom.n_components == 2
    assert len(dom.eroded.geoms) == 1


def test_domain_from_raw_and_empty_input() -> None:
    dom = domain_from_raw(box(0, 0, 5, 5), PARAMS)
    assert isinstance(dom, DomainSet)
    assert dom.eroded.area == pytest.approx(4.4 * 4.4, abs=1e-3)
    empty = domain_from_raw(Polygon(), PARAMS)
    assert empty.n_components == 0 and empty.raw.is_empty and empty.eroded.is_empty


def test_domain_rejects_non_polygonal_raw() -> None:
    with pytest.raises(ValueError, match="polygonal"):
        domain_from_raw(Point(0, 0).buffer(1).boundary, PARAMS)


def test_mini_vineyard_domain_is_connected_when_interrows_touch_passages() -> None:
    mini = mini_vineyard(end_gap_m=0.0)
    dom = build_domain(tuple(mini.interrow_pieces.geometry), mini.passages, None, (), PARAMS)
    assert dom.n_components == 1
    gapped = mini_vineyard(end_gap_m=2.0)
    dom2 = build_domain(tuple(gapped.interrow_pieces.geometry), gapped.passages, None, (), PARAMS)
    assert dom2.n_components == 2 + 3


def test_passable_parts_per_source_object() -> None:
    mini = mini_vineyard()
    dom = build_domain(tuple(mini.interrow_pieces.geometry), mini.passages, None, (), PARAMS)
    parts = passable_parts(dom, tuple(mini.interrow_pieces.geometry), tuple(mini.interrow_pieces.piece_id),
                           mini.passages)
    kinds = [p.kind for p in parts]
    assert kinds.count("interrow") == 3 and kinds.count("passage") == 2
    assert [p.part_id for p in parts] == [f"PP{k:05d}" for k in range(1, 6)]
    assert parts[0].ref_id == "siret3_r010_c010:I001"
    assert all(p.area_m2 > 0 and p.geom.is_valid for p in parts)
    total = shapely.union_all([p.geom for p in parts]).area
    assert total == pytest.approx(dom.raw.area, rel=1e-6)


def _comb(n_teeth: int = 40) -> Polygon:
    """Many-vertex polygon: a 400 m bar with `n_teeth` notches, so the quadtree must subdivide."""
    bar = box(0, 0, 400, 20)
    notches = [box(10 * k + 2, 15, 10 * k + 4, 20) for k in range(n_teeth)]
    return bar.difference(shapely.union_all(notches))


def test_cell_index_splits_and_preserves_area() -> None:
    comb = _comb()
    idx = CellIndex.build(comb, max_coords=32, min_cell_m=8.0)
    assert idx.n_cells > 1
    assert sum(c.area for c in idx.cells) == pytest.approx(comb.area, rel=1e-9)


def test_cell_index_inside_length_matches_overlay() -> None:
    comb = _comb()
    idx = CellIndex.build(comb, max_coords=32, min_cell_m=8.0)
    lines = [shapely.LineString([(1, 17), (399, 17)]), shapely.LineString([(-10, 5), (50, 5)]),
             shapely.LineString([(500, 5), (510, 5)])]
    got = idx.inside_length(lines)
    want = [shapely.intersection(ln, comb).length for ln in lines]
    assert np.allclose(got, want, atol=1e-6)
    assert idx.inside_length([]).shape == (0,)


def test_cell_index_clip_line_merges_across_cells() -> None:
    comb = _comb()
    idx = CellIndex.build(comb, max_coords=32, min_cell_m=8.0)
    parts = idx.clip_line(shapely.LineString([(-10, 5), (410, 5)]))
    assert len(parts) == 1 and parts[0].length == pytest.approx(400.0, abs=1e-5)
    notched = idx.clip_line(shapely.LineString([(0, 17), (25, 17)]))
    assert len(notched) == 4   # [0,2] [4,12] [14,22] [24,25]
    assert [round(p.length, 6) for p in notched] == [2.0, 8.0, 8.0, 1.0]
    assert idx.clip_line(shapely.LineString([(500, 5), (510, 5)])) == []


def test_cell_index_empty_region() -> None:
    idx = CellIndex.build(MultiPolygon())
    assert idx.n_cells == 0
    assert idx.clip_line(shapely.LineString([(0, 0), (1, 0)])) == []
    assert idx.inside_length([shapely.LineString([(0, 0), (1, 0)])]).tolist() == [0.0]


def test_components_sorted_by_area_desc() -> None:
    dom = domain_from_raw(MultiPolygon([box(0, 0, 1, 1), box(5, 5, 9, 9)]), PARAMS)
    areas = [p.area for p in dom.components()]
    assert areas == sorted(areas, reverse=True)
    assert np.isclose(areas[0], 16.0)


def _holey(n: int = 30) -> Polygon:
    """400 x 40 m bar with n x 3 small square holes (canopies) and notched edges."""
    holes = [box(12 * k + 3, y, 12 * k + 4, y + 1) for k in range(n) for y in (8, 18, 28)]
    return _comb(n).union(box(0, 20, 400, 40)).difference(shapely.union_all(holes))


def test_quadtree_cells_validation_and_empty() -> None:
    with pytest.raises(ValueError, match="max_coords"):
        quadtree_cells(box(0, 0, 1, 1), max_coords=0)
    assert quadtree_cells(Polygon()) == ()


@pytest.mark.parametrize("dist", [-0.05, -0.3, -1.0])
def test_local_buffer_equals_global_buffer(dist: float) -> None:
    region = _holey()
    local = local_apply(region, lambda g: shapely.buffer(g, dist), abs(dist), 1e-6, max_coords=64, min_cell_m=8.0)
    whole = shapely.buffer(region, dist)
    assert shapely.symmetric_difference(local, whole).area < 1e-4
    assert local.is_valid


def test_local_closing_matches_global_closing() -> None:
    parts = shapely.union_all([box(10 * k, 0, 10 * k + 9.98, 3) for k in range(40)])   # 2 cm seams
    local = close_seams(parts, 0.05, 0.001)
    assert len(split_multi(local)) == 1 and local.area == pytest.approx(399.98 * 3, abs=0.01)
