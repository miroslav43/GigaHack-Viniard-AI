"""Passage skeleton: raster 0.25 m -> skeletonize -> sknw -> UTM polylines, spur pruning (arch §4.11.2)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from tests.post.passable_factories import corridor, route_inputs
from vineyard.route.domain import DomainParams, build_domain
from vineyard.route.skeleton import (
    Branch,
    SkeletonParams,
    prune_spurs,
    rasterize,
    skeleton_lines,
)

PARAMS = SkeletonParams(res_m=0.25, erode_m=0.3, spur_min_m=2.0, simplify_tol_m=0.25)


def _total(lines: tuple[LineString, ...]) -> float:
    return float(sum(ln.length for ln in lines))


def test_params_from_config() -> None:
    from vineyard.config import load_config

    assert SkeletonParams.from_config(load_config().route.graph) == PARAMS


def test_params_validation() -> None:
    with pytest.raises(ValueError, match="res_m"):
        SkeletonParams(res_m=0.0, erode_m=0.3, spur_min_m=2.0, simplify_tol_m=0.25)
    with pytest.raises(ValueError, match="max_prune_passes"):
        dataclasses.replace(PARAMS, max_prune_passes=0)


def test_rasterize_pixel_centres_and_padding() -> None:
    mask, (minx, maxy) = rasterize(box(10.0, 20.0, 12.0, 21.0), 0.25)
    assert mask.dtype == bool
    assert mask.sum() == 8 * 4
    assert not mask[0].any() and not mask[-1].any() and not mask[:, 0].any() and not mask[:, -1].any()
    assert minx == pytest.approx(10.0 - 0.25) and maxy == pytest.approx(21.0 + 0.25)


def test_straight_corridor_axis_and_coordinates() -> None:
    lines = skeleton_lines(corridor(3.0, [(0, 100), (30, 100)]), PARAMS)
    assert len(lines) == 1
    xy = np.asarray(lines[0].coords)
    assert np.all(np.abs(xy[:, 1] - 100.0) <= 0.25)
    # medial axis of the eroded 2.4 m corridor ends one half-width (1.2 m) inside the eroded caps
    assert lines[0].length == pytest.approx(30.0 - 2 * (0.3 + 1.2), abs=0.75)


def test_l_corridor_length_matches_eroded_centreline() -> None:
    region = corridor(3.0, [(0, 0), (20, 0), (20, 15)])
    lines = skeleton_lines(region, PARAMS)
    expected = (20.0 - 1.5) + (15.0 - 1.5)
    assert len(lines) == 1
    assert _total(lines) == pytest.approx(expected, abs=0.5)
    assert shapely.covered_by(lines[0], region.buffer(-0.3))


def test_short_stub_pruned_long_stub_kept() -> None:
    main = corridor(3.0, [(0, 0), (40, 0)])
    short_stub = corridor(3.0, [(10, 0), (10, 2.5)])
    long_stub = corridor(3.0, [(30, 0), (30, 7.0)])
    lines = skeleton_lines(shapely.union_all([main, short_stub, long_stub]), PARAMS)
    ends = shapely.get_coordinates(shapely.boundary(shapely.line_merge(shapely.MultiLineString(lines))))
    tops = ends[ends[:, 1] > 1.0]
    assert len(tops) == 1
    assert tops[0, 0] == pytest.approx(30.0, abs=0.5)


def test_ring_road_around_hole_keeps_both_sides() -> None:
    outer = box(0, 0, 30, 20)
    ring = Polygon(outer.exterior.coords, [box(3, 3, 27, 17).exterior.coords])
    lines = skeleton_lines(ring, PARAMS)
    merged = shapely.line_merge(shapely.MultiLineString(lines))
    # a closed loop along the ring's mid-line (1.5 m in from the outer edge): 2 * (27 + 17) = 88 m
    assert _total(lines) == pytest.approx(88.0, rel=0.03)
    assert merged.geom_type == "LineString" and merged.is_ring


def test_parallel_paths_between_two_junctions_both_kept() -> None:
    # Two junctions joined by both sides of a hole: sknw multi=False would drop one side.
    outer = box(0, 0, 40, 20)
    hole = box(15, 3, 25, 17)
    region = shapely.union_all([Polygon(outer.exterior.coords, [hole.exterior.coords]),
                                corridor(3.0, [(-20, 10), (0.5, 10)]), corridor(3.0, [(39.5, 10), (60, 10)])])
    lines = skeleton_lines(region, PARAMS)
    probe_top, probe_bottom = shapely.Point(20, 18.5), shapely.Point(20, 1.5)
    assert min(ln.distance(probe_top) for ln in lines) < 0.5
    assert min(ln.distance(probe_bottom) for ln in lines) < 0.5


def test_thin_region_vanishes_and_empty_input() -> None:
    assert skeleton_lines(box(0, 0, 10, 0.5), PARAMS) == ()
    assert skeleton_lines(Polygon(), PARAMS) == ()
    assert skeleton_lines(MultiPolygon(), PARAMS) == ()


def test_isolated_short_segment_is_kept() -> None:
    lines = skeleton_lines(box(0, 0, 3.5, 2.0), dataclasses.replace(PARAMS, erode_m=0.0))
    assert len(lines) >= 1


def test_simplify_guard_keeps_lines_inside_guard() -> None:
    region = corridor(3.0, [(0, 0), (20, 0), (20, 15)])
    guard = region.buffer(-0.3)
    lines = skeleton_lines(region, PARAMS, guard=guard)
    assert all(shapely.covered_by(ln, guard) for ln in lines)
    coarse = skeleton_lines(region, dataclasses.replace(PARAMS, simplify_tol_m=0.0))
    assert sum(len(ln.coords) for ln in lines) <= sum(len(ln.coords) for ln in coarse)


def test_prune_spurs_pure() -> None:
    xy = lambda *pts: np.asarray(pts, dtype=float)  # noqa: E731
    branches = (
        Branch(0, 1, xy((0, 0), (10, 0))),
        Branch(1, 2, xy((10, 0), (20, 0))),
        Branch(1, 3, xy((10, 0), (10, 1))),      # 1 m spur off a junction: pruned
        Branch(2, 4, xy((20, 0), (20, 1.5))),    # terminal but node 2 becomes degree 2 only after pass 1
        Branch(5, 6, xy((50, 0), (51, 0))),      # isolated short segment: kept
    )
    kept = prune_spurs(branches, spur_min=2.0, max_passes=3)
    assert (1, 3) not in [(b.u, b.v) for b in kept]
    assert (5, 6) in [(b.u, b.v) for b in kept]
    assert (2, 4) in [(b.u, b.v) for b in kept]


@pytest.mark.slow
def test_real_passages_two_components() -> None:
    inputs = route_inputs()
    dom = build_domain((), inputs.passages, inputs.forbidden, (),
                       DomainParams(0.001, 0.05, 0.3, 0.05, True))
    lines = skeleton_lines(shapely.intersection(inputs.passages, dom.raw), PARAMS, guard=dom.eroded)
    merged = shapely.union_all(lines).buffer(0.01)
    comps = sorted(shapely.get_parts(merged), key=lambda g: g.area)
    assert len(comps) == 2
    minx, miny, maxx, maxy = comps[0].bounds
    assert minx >= 629913.0 and maxx <= 630293.0 and miny >= 5219226.0 and maxy <= 5219540.0
    assert all(shapely.covered_by(ln, dom.eroded) for ln in lines)
    start = shapely.Point(inputs.start_xy)
    assert min(ln.distance(start) for ln in lines) < 3.0
