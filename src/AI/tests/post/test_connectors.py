"""Connectors: row ends, piece joins, skeleton snaps, START link, headland strategies (critic S5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, box

from vineyard.contracts.enums import EdgeKind
from vineyard.route.connectors import (
    INSIDE_TOL_M,
    ConnectorParams,
    HeadlandStrategy,
    outer_row_ends,
    piece_joins,
    plan_connectors,
    row_end_connectors,
    snap_joins,
    start_connector,
)
from vineyard.route.domain import CellIndex
from vineyard.route.graph_types import PolylineDraft

IR = EdgeKind.INTERROW_CENTERLINE
PS = EdgeKind.PASSAGE_CENTERLINE
PARAMS = ConnectorParams(max_len_m=6.0, outside_penalty=10.0, snap_join_m=0.5)
Y = 2001.3


def _inner(*polys: shapely.Geometry) -> shapely.Geometry:
    geom = shapely.union_all(list(polys))
    shapely.prepare(geom)
    return geom


def _lines(gap_m: float) -> tuple[list[PolylineDraft], shapely.Geometry]:
    """Passage skeleton at x=998 (passage 996..1000 - gap); interrow 1000..1060 at y=2001.3."""
    passage = box(996 - gap_m, 1995, 1000 - gap_m, 2013)
    strip = box(1000, 2000.3, 1060, 2002.3)
    lines = [PolylineDraft(LineString([(998 - gap_m, 1996), (998 - gap_m, 2012)]), PS, "P0001"),
             PolylineDraft(LineString([(1000, Y), (1060, Y)]), IR, "V01-I003")]
    return lines, _inner(passage, strip)


def test_params_from_config_and_allowed_outside() -> None:
    from vineyard.config import load_config

    params = ConnectorParams.from_config(load_config().route.graph)
    assert params.max_len_m == 6.0 and params.outside_penalty == 10.0 and params.snap_join_m == 0.5
    assert params.strategy == HeadlandStrategy.PENALTY and math.isinf(params.allowed_outside_m)
    limit = ConnectorParams.from_config(load_config().route.graph, "limit", 1.0)
    assert limit.allowed_outside_m == 1.0
    assert ConnectorParams(6.0, 10.0, 0.5, "inside_only").allowed_outside_m == INSIDE_TOL_M
    with pytest.raises(ValueError):
        ConnectorParams(0.0, 10.0, 0.5)


def test_outer_row_ends_skip_joined_ends() -> None:
    lines = [PolylineDraft(LineString([(0, 0), (10, 0)]), IR, "V01-I001"),
             PolylineDraft(LineString([(12, 0), (20, 0)]), IR, "V01-I001"),
             PolylineDraft(LineString([(0, 5), (20, 5)]), PS, "P1")]
    ends = outer_row_ends(lines, [(0, 1)])
    assert ends.refs == ("V01-I001:start", "V01-I001:end")
    assert ends.xy.tolist() == [[0.0, 0.0], [20.0, 0.0]]


def test_touching_end_connects_inside() -> None:
    lines, inner = _lines(gap_m=0.0)
    out, lonely = row_end_connectors(lines, [], inner, CellIndex.build(inner), PARAMS)
    assert len(out) == 1
    c = out[0]
    assert c.kind == "row_end" and c.ref == "V01-I003:start" and c.target == 0
    assert c.to_xy == pytest.approx((998.0, Y)) and c.length_m == pytest.approx(2.0)
    assert c.outside_m == 0.0 and c.kept
    assert c.along_m == pytest.approx(Y - 1996)
    assert [ref for ref, _, _ in lonely] == ["V01-I003:end"]


@pytest.mark.parametrize(("strategy", "kept"), [("penalty", True), ("limit", False), ("inside_only", False)])
def test_headland_gap_connector_by_strategy(strategy: str, kept: bool) -> None:
    lines, inner = _lines(gap_m=2.0)
    params = ConnectorParams(6.0, 10.0, 0.5, strategy, 1.0)
    out, _ = row_end_connectors(lines, [], inner, CellIndex.build(inner), params)
    assert len(out) == 1
    assert out[0].length_m == pytest.approx(4.0)
    assert out[0].outside_m == pytest.approx(2.0, abs=1e-6)
    assert out[0].kept is kept


def test_cheapest_candidate_prefers_inside_over_shorter_crossing() -> None:
    passage = box(994, 1995, 998, 2013)
    strips = [box(1000, 2000.3, 1060, 2002.3), box(996, 2002.9, 1060, 2004.9)]
    lines = [PolylineDraft(LineString([(996, 1996), (996, 2012)]), PS, "P0001"),
             PolylineDraft(LineString([(1000, Y), (1060, Y)]), IR, "V01-I003"),
             PolylineDraft(LineString([(996, 2003.9), (1060, 2003.9)]), IR, "V01-I002")]
    inner = _inner(passage, *strips, box(994, 2000.3, 1000, 2002.3))
    out, _ = row_end_connectors(lines, [], inner, CellIndex.build(inner), PARAMS)
    by_ref = {c.ref: c for c in out}
    assert by_ref["V01-I003:start"].target == 0          # 4 m inside beats 2.6 m across the row (cost 2.6 + 6)
    assert by_ref["V01-I003:start"].outside_m == 0.0


def test_piece_join_records_outside_length() -> None:
    lines = [PolylineDraft(LineString([(0, 1), (10, 1)]), IR, "V01-I001"),
             PolylineDraft(LineString([(12, 1), (20, 1)]), IR, "V01-I001")]
    inner = _inner(box(0, 0, 10.5, 2), box(11.5, 0, 20, 2))
    (join,) = piece_joins(lines, [(0, 1)], inner, CellIndex.build(inner), PARAMS)
    assert join.kind == "piece_join" and join.target == 1 and join.along_m == 0.0
    assert join.length_m == pytest.approx(2.0) and join.outside_m == pytest.approx(1.0)
    assert join.kept
    strict = ConnectorParams(6.0, 10.0, 0.5, "inside_only")
    assert not piece_joins(lines, [(0, 1)], inner, CellIndex.build(inner), strict)[0].kept
    assert piece_joins(lines, [], inner, CellIndex.build(inner), PARAMS) == ()


def test_snap_join_only_inside_and_short() -> None:
    lines = [PolylineDraft(LineString([(0, 0), (10, 0)]), PS, "P1"),
             PolylineDraft(LineString([(5, 0.3), (5, 8)]), PS, "P2"),
             PolylineDraft(LineString([(20, 0), (20, 8)]), PS, "P3")]
    inner = _inner(box(-1, -1, 21, 9))
    snaps = snap_joins(lines, inner, CellIndex.build(inner), PARAMS)
    assert [(s.ref, s.target) for s in snaps] == [("P2:start:snap", 0)]
    assert snaps[0].to_xy == pytest.approx((5.0, 0.0))
    assert snap_joins(lines, inner, CellIndex.build(inner), ConnectorParams(6.0, 10.0, 0.0)) == ()


def test_start_connector_inside_or_error() -> None:
    lines, inner = _lines(gap_m=0.0)
    index = CellIndex.build(inner)
    c = start_connector(lines, (997.0, 1997.0), inner, index, PARAMS)
    assert c.kind == "start" and c.target == 0 and c.to_xy == pytest.approx((998.0, 1997.0))
    with pytest.raises(ValueError, match="not inside"):
        start_connector(lines, (900.0, 1997.0), inner, index, PARAMS)
    with pytest.raises(ValueError, match="no walk-graph line"):
        start_connector(lines[1:], (997.0, 1996.0), inner, index, PARAMS)


def test_plan_drafts_and_attachments() -> None:
    lines, inner = _lines(gap_m=0.0)
    plan = plan_connectors(lines, [], (998.0, 1997.0), inner, CellIndex.build(inner), PARAMS)
    kinds = [c.kind for c in plan.connectors]
    assert kinds == ["start", "row_end"]
    # START lies on the skeleton: a zero-length join gives an attachment but no polyline
    assert [d.ref for d in plan.drafts()] == ["V01-I003:start"]
    assert [(a.draft, round(a.along_m, 6)) for a in plan.attachments()] == [(0, 1.0), (0, round(Y - 1996, 6))]
    assert len(plan.kept) == 2 and plan.dropped == ()
    assert np.isclose(plan.drafts()[0].geom.length, 2.0)
    assert [ref for ref, _, _ in plan.unconnected_ends] == ["V01-I003:end"]
