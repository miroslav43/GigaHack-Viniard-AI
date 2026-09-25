"""Row graph primitives: pair geometry, edges (parallel / skip-one / collinear), adjacency, bands."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import LineString, box

from vineyard.config import load_config
from vineyard.errors import StageError
from vineyard.perception.blocks_graph import (
    EDGE_COLLINEAR,
    EDGE_PARALLEL,
    EDGE_SKIP_ONE,
    GraphSettings,
    adjacent_pairs,
    axial_diff_deg,
    candidate_pairs,
    components,
    line_frame,
    neighbour_pairs,
    pair_geometry,
    transverse_bands,
)

X0, Y0 = 629600.0, 5220100.0


@pytest.fixture(scope="module")
def settings() -> GraphSettings:
    cfg = load_config()
    return GraphSettings.from_config(cfg.blocks, cfg.rows.detect, cfg.rows.link)


def hrow(y: float, x0: float = 0.0, x1: float = 40.0) -> LineString:
    return LineString([(X0 + x0, Y0 + y), (X0 + x1, Y0 + y)])


def test_axial_diff_wraps() -> None:
    assert axial_diff_deg(179.0, 1.0) == pytest.approx(2.0)
    assert axial_diff_deg(10.0, 100.0) == pytest.approx(90.0)


def test_line_frame_rejects_closed_chord() -> None:
    with pytest.raises(StageError):
        line_frame(LineString([(0, 0), (1, 1), (0, 0)]))


def test_line_frame_angle_and_projection() -> None:
    fr = line_frame(LineString([(0, 0), (3, 4)]))
    assert fr.length == pytest.approx(5.0)
    assert fr.angle_deg == pytest.approx(math.degrees(math.atan2(4, 3)))
    assert fr.along(np.array([[3.0, 4.0]]))[0] == pytest.approx(5.0)
    assert fr.across(np.array([[-4.0, 3.0]]))[0] == pytest.approx(5.0)


def test_pair_geometry_parallel_overlap() -> None:
    g = pair_geometry(hrow(0.0, 0, 40), hrow(2.5, 10, 50))
    assert g.spacing_m == pytest.approx(2.5)
    assert (g.overlap_from_m, g.overlap_to_m) == pytest.approx((10.0, 40.0))
    assert g.gap_m == 0.0
    assert g.angle_diff_deg == pytest.approx(0.0)
    assert g.connector is not None and g.connector.length == pytest.approx(2.5)


def test_pair_geometry_collinear_gap() -> None:
    g = pair_geometry(hrow(0.0, 0, 40), hrow(0.1, 43, 60))
    assert g.gap_m == pytest.approx(3.0)
    assert g.lateral_m == pytest.approx(0.1)
    assert math.isnan(g.spacing_m)
    assert g.connector is not None and g.connector.length == pytest.approx(math.hypot(3.0, 0.1))


def test_ten_rows_give_nine_parallel_edges(settings: GraphSettings) -> None:
    lines = [hrow(2.5 * k) for k in range(10)]
    edges = neighbour_pairs(lines, settings)
    assert len(edges) == 9
    assert set(edges.kind) == {EDGE_PARALLEL}
    assert components(10, edges) == (tuple(range(10)),)
    assert len(adjacent_pairs(edges)) == 9


def test_missing_middle_row_gives_skip_one_edge(settings: GraphSettings) -> None:
    ys = [0.0, 2.5, 5.0, 10.0, 12.5, 15.0]
    edges = neighbour_pairs([hrow(y) for y in ys], settings)
    skip = edges[edges.kind == EDGE_SKIP_ONE]
    assert list(zip(skip.a, skip.b, strict=True)) == [(2, 3)]
    assert skip.spacing_m.iloc[0] == pytest.approx(5.0)
    assert components(len(ys), edges) == (tuple(range(6)),)
    assert len(adjacent_pairs(edges)) == 5


def test_skip_one_disabled_splits(settings: GraphSettings) -> None:
    from dataclasses import replace

    off = replace(settings, skip_one_enabled=False)
    ys = [0.0, 2.5, 5.0, 10.0, 12.5, 15.0]
    edges = neighbour_pairs([hrow(y) for y in ys], off)
    assert len(components(len(ys), edges)) == 2


def test_passage_cuts_edges(settings: GraphSettings) -> None:
    ys = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 17.5, 20.0, 22.5, 25.0]
    passage = box(X0 - 10, Y0 + 11.0, X0 + 60, Y0 + 14.0)
    edges = neighbour_pairs([hrow(y) for y in ys], settings, cut=passage)
    assert components(len(ys), edges) == ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))


def test_collinear_pieces_are_linked(settings: GraphSettings) -> None:
    edges = neighbour_pairs([hrow(0.0, 0, 20), hrow(0.2, 24, 50)], settings)
    assert list(edges.kind) == [EDGE_COLLINEAR]
    far = neighbour_pairs([hrow(0.0, 0, 20), hrow(0.2, 26, 50)], settings)
    assert len(far) == 0


def test_non_parallel_and_short_overlap_rejected(settings: GraphSettings) -> None:
    tilted = LineString([(X0, Y0 + 2.5), (X0 + 40, Y0 + 2.5 + 40 * math.tan(math.radians(8)))])
    assert len(neighbour_pairs([hrow(0.0), tilted], settings)) == 0
    assert len(neighbour_pairs([hrow(0.0, 0, 40), hrow(2.5, 38, 78)], settings)) == 0


def test_close_spacing_non_adjacent_pairs_excluded(settings: GraphSettings) -> None:
    lines = [hrow(1.9 * k) for k in range(4)]
    edges = neighbour_pairs(lines, settings)
    assert len(edges) == 5  # 3 adjacent at 1.9 m + 2 at 3.8 m
    adj = adjacent_pairs(edges)
    assert sorted(zip(adj.a, adj.b, strict=True)) == [(0, 1), (1, 2), (2, 3)]


def test_candidate_pairs_empty_and_sorted() -> None:
    assert candidate_pairs([hrow(0.0)], 5.0).shape == (0, 2)
    pairs = candidate_pairs([hrow(0.0), hrow(2.0), hrow(4.0)], 3.0)
    assert pairs.tolist() == [[0, 1], [1, 2]]


def test_components_without_edges() -> None:
    edges = neighbour_pairs([hrow(0.0), hrow(30.0)], GraphSettings(5, 0.2, 1.8, 4, 0.3, 5, True, (1.6, 2.3), 4, 3, True, 0.1, 1.0))
    assert components(2, edges) == ((0,), (1,))
    assert components(0, edges) == ()


def test_full_width_band_found(settings: GraphSettings) -> None:
    lines = [hrow(2.5 * k, 0, 60) for k in range(5)]
    gaps = [[(20.0, 26.0)] for _ in lines]
    bands = transverse_bands(lines, gaps, list(range(5)), settings)
    assert len(bands) == 1
    assert bands[0].width_m == pytest.approx(6.0, abs=0.25)
    assert bands[0].rows == (0, 1, 2, 3, 4)
    assert bands[0].polygon.intersects(lines[2])


def test_partial_band_not_found(settings: GraphSettings) -> None:
    lines = [hrow(2.5 * k, 0, 60) for k in range(6)]
    gaps = [[(20.0, 26.0)] if k < 3 else [] for k in range(6)]
    assert transverse_bands(lines, gaps, list(range(6)), settings) == ()


def test_narrow_band_and_small_component_ignored(settings: GraphSettings) -> None:
    lines = [hrow(2.5 * k, 0, 60) for k in range(5)]
    narrow = [[(20.0, 23.0)] for _ in lines]
    assert transverse_bands(lines, narrow, list(range(5)), settings) == ()
    assert transverse_bands(lines[:2], narrow[:2], [0, 1], settings) == ()


def test_partial_band_rule_when_not_full_width(settings: GraphSettings) -> None:
    from dataclasses import replace

    loose = replace(settings, band_full_width=False)
    lines = [hrow(2.5 * k, 0, 60) for k in range(6)]
    gaps = [[(20.0, 26.0)] if k < 3 else [] for k in range(6)]
    bands = transverse_bands(lines, gaps, list(range(6)), loose)
    assert len(bands) == 1 and bands[0].rows == (0, 1, 2)
