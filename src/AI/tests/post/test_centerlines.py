"""Interrow centerlines: geometric mean of consecutive row axes, clipped to domain.eroded (arch §4.11.2)."""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, box

from tests.post.passable_factories import interrow_strip, mini_vineyard
from vineyard.geo.tiling import CRS_EPSG
from vineyard.route.centerlines import (
    CenterlineParams,
    fallback_centerlines,
    interrow_centerlines,
    midline,
    row_pairs,
)
from vineyard.route.domain import DomainParams, build_domain
from vineyard.route.skeleton import SkeletonParams

PARAMS = CenterlineParams(step_m=2.0, min_len_m=1.0, fallback_near_m=0.5)
DPARAMS = DomainParams(grid_size_m=0.001, inner_buffer_m=0.05, eroded_buffer_m=0.3, seam_close_m=0.05,
                       subtract_canopies=True)
SKEL = SkeletonParams(res_m=0.25, erode_m=0.3, spur_min_m=2.0, simplify_tol_m=0.25)


def _rows(lines: list[LineString], ids: list[str] | None = None, index: list[int] | None = None,
          vid: str = "V01") -> gpd.GeoDataFrame:
    n = len(lines)
    return gpd.GeoDataFrame(
        {"row_id": ids or [f"{vid}-R{k:03d}" for k in range(1, n + 1)], "vineyard_id": [vid] * n,
         "row_index": np.asarray(index or list(range(1, n + 1)), dtype=np.int16)},
        geometry=lines, crs=CRS_EPSG)


def test_params_from_config() -> None:
    from vineyard.config import load_config

    params = CenterlineParams.from_config(load_config().route.graph)
    assert params.step_m == 2.0 and params.min_len_m == 1.0 and params.fallback_near_m == 0.5


def test_params_validation() -> None:
    with pytest.raises(ValueError, match="step_m"):
        CenterlineParams(step_m=0.0, min_len_m=1.0, fallback_near_m=0.5)


def test_midline_overlap_only_and_midway() -> None:
    a = LineString([(0, 2.6), (50, 2.6)])
    b = LineString([(5, 0), (45, 0)])
    mid = midline(a, b, 2.0)
    assert mid is not None
    assert mid.length == pytest.approx(40.0, abs=1e-6)
    xy = np.asarray(mid.coords)
    assert np.allclose(xy[:, 1], 1.3)
    assert xy[0, 0] == pytest.approx(5.0) and xy[-1, 0] == pytest.approx(45.0)
    assert np.all(np.diff(xy[:, 0]) > 0) and np.max(np.diff(xy[:, 0])) <= 2.0 + 1e-9


def test_midline_none_without_overlap() -> None:
    assert midline(LineString([(0, 2.6), (10, 2.6)]), LineString([(20, 0), (30, 0)]), 2.0) is None


def test_midline_fan_two_degrees_equidistant() -> None:
    a = LineString([(0, 0), (60, 0)])
    ang = math.radians(2.0)
    b = LineString([(0, 2.6), (60 * math.cos(ang), 2.6 + 60 * math.sin(ang))])
    mid = midline(a, b, 2.0)
    pts = [Point(c) for c in mid.coords]
    diffs = [abs(a.distance(p) - b.distance(p)) for p in pts]
    assert max(diffs) < 0.01


def test_rows_2_6_apart_lengths_50_and_40() -> None:
    a = LineString([(0, 2.6), (50, 2.6)])
    b = LineString([(5, 0), (45, 0)])
    dom = build_domain((interrow_strip(a, b),), MultiPolygon(), None, (), DPARAMS)
    lines = interrow_centerlines(_rows([a, b]), dom.eroded, PARAMS)
    assert len(lines) == 1
    cl = lines[0]
    assert cl.geom.length == pytest.approx(39.4, abs=0.01)
    assert cl.interrow_id == "V01-I001" and cl.row_left_id == "V01-R001" and cl.row_right_id == "V01-R002"
    assert cl.piece == 1 and cl.source == "midline"
    for c in cl.geom.coords:
        assert a.distance(Point(c)) == pytest.approx(1.3, abs=1e-6)
        assert b.distance(Point(c)) == pytest.approx(1.3, abs=1e-6)


def test_tree_hole_gives_two_pieces() -> None:
    a = LineString([(0, 2.6), (40, 2.6)])
    b = LineString([(0, 0), (40, 0)])
    hole = box(19.0, 0.2, 21.0, 2.4)
    dom = build_domain((interrow_strip(a, b),), MultiPolygon(), None, (hole,), DPARAMS)
    lines = interrow_centerlines(_rows([a, b]), dom.eroded, PARAMS)
    assert [c.piece for c in lines] == [1, 2]
    assert lines[0].geom.coords[0][0] < lines[1].geom.coords[0][0]
    assert all(shapely.covered_by(c.geom, dom.eroded) for c in lines)


def test_short_pieces_dropped() -> None:
    a = LineString([(0, 2.6), (40, 2.6)])
    b = LineString([(0, 0), (40, 0)])
    hole = box(1.2, 0.2, 3.0, 2.4)   # leaves a 0.6 m stub at the west end
    dom = build_domain((interrow_strip(a, b),), MultiPolygon(), None, (hole,), DPARAMS)
    lines = interrow_centerlines(_rows([a, b]), dom.eroded, PARAMS)
    assert len(lines) == 1 and lines[0].geom.length > 30


def test_ids_follow_geometric_row_index_not_row_id_suffix() -> None:
    lines = [LineString([(0, y), (30, y)]) for y in (7.8, 5.2, 2.6, 0.0)]
    rows = _rows(lines, ids=["V02-R09", "V02-R03", "V02-R01", "V02-R07"], index=[1, 2, 3, 4], vid="V02")
    shuffled = rows.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    pairs = row_pairs(shuffled)
    assert [(p.row_left_id, p.row_right_id, p.interrow_id) for p in pairs] == [
        ("V02-R09", "V02-R03", "V02-I001"), ("V02-R03", "V02-R01", "V02-I002"), ("V02-R01", "V02-R07", "V02-I003"),
    ]


def test_pairs_per_block_and_index_gaps() -> None:
    v1 = _rows([LineString([(0, y), (30, y)]) for y in (5.2, 2.6, 0.0)], index=[1, 2, 4])
    v2 = _rows([LineString([(100, y), (130, y)]) for y in (2.6, 0.0)], vid="V02")
    pairs = row_pairs(gpd.GeoDataFrame(gpd.pd.concat([v2, v1], ignore_index=True), crs=CRS_EPSG))
    assert [p.interrow_id for p in pairs] == ["V01-I001", "V01-I002", "V02-I001"]


def test_row_pairs_rejects_missing_columns() -> None:
    bad = gpd.GeoDataFrame({"row_id": ["V01-R001"]}, geometry=[LineString([(0, 0), (1, 0)])], crs=CRS_EPSG)
    with pytest.raises(ValueError, match="row_index"):
        row_pairs(bad)


def test_mini_vineyard_three_interrows() -> None:
    mini = mini_vineyard()
    dom = build_domain(tuple(mini.interrow_pieces.geometry), mini.passages, None, (), DPARAMS)
    lines = interrow_centerlines(mini.rows, dom.eroded, PARAMS)
    assert [c.interrow_id for c in lines] == ["V01-I001", "V01-I002", "V01-I003"]
    # the eroded domain continues into the touching headland passages, so the midline spans the full overlap
    assert all(c.geom.length == pytest.approx(60.0, abs=0.01) for c in lines)


def test_fallback_skeleton_for_unlinked_piece() -> None:
    mini = mini_vineyard()
    extra = box(1000.0, 2000.0 - 2.3 - 0.3, 1060.0, 2000.0 - 0.3)   # strip south of the last row
    pieces = [*mini.interrow_pieces.geometry, extra]
    dom = build_domain(tuple(pieces), mini.passages, None, (), DPARAMS)
    lines = interrow_centerlines(mini.rows, dom.eroded, PARAMS)
    ids = [*mini.interrow_pieces.piece_id, "siret3_r010_c010:I004"]
    refs = [*mini.interrow_pieces.interrow_id, None]
    extra_lines = fallback_centerlines(pieces, ids, refs, lines, dom.eroded, SKEL, PARAMS)
    assert len(extra_lines) == 1
    fb = extra_lines[0]
    assert fb.source == "skeleton" and fb.interrow_id == "siret3_r010_c010:I004"
    assert fb.geom.length > 40.0
    assert shapely.covered_by(fb.geom, dom.eroded.buffer(1e-6))
