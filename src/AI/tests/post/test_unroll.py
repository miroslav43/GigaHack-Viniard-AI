"""Tour -> LineString through the Dijkstra legs (arch §4.11.5, design 04 §3.9)."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString

from vineyard.route.unroll import dedupe_xy, edge_index, unroll

from .route_factories import CO, IC, PC, graph_from_lines, mini_route

START = (0.0, 0.0)


def _t_graph(start_offset: float = 0.0):
    """START -- (10,0) -- (20,0) with a dead end (10,0) -> (10,8); a costly parallel detour 0 -> (10,0)."""
    start = (START[0] + start_offset, START[1])
    lines = [(LineString([start, (10.0, 0.0)]), PC, "a"), (LineString([(10.0, 0.0), (20.0, 0.0)]), PC, "b"),
             (LineString([(10.0, 8.0), (10.0, 0.0)]), IC, "dead"),
             (LineString([start, (5.0, 3.0), (10.0, 0.0)]), CO, "detour")]
    return graph_from_lines(lines, start)


def _node(g, xy: tuple[float, float]) -> int:
    return int(np.flatnonzero(np.all(np.isclose(g.node_xy, xy), axis=1))[0])


def test_starts_and_ends_exactly_at_start_without_duplicates() -> None:
    g = _t_graph()
    un = unroll(g, [g.start_node, _node(g, (20.0, 0.0))], START, chunk=4)
    xy = shapely.get_coordinates(un.line)
    assert tuple(xy[0]) == START and tuple(xy[-1]) == START
    assert np.all(np.hypot(*np.diff(xy, axis=0).T) > 0)
    assert un.line.length == pytest.approx(40.0)
    assert un.anchors == (0, 2, 4)


def test_out_and_back_dead_end_is_kept_although_not_simple() -> None:
    g = _t_graph()
    un = unroll(g, [g.start_node, _node(g, (10.0, 8.0))], START, chunk=4)
    assert not un.line.is_simple
    assert un.line.length == pytest.approx(36.0)
    assert tuple(un.xy[un.anchors[1]]) == (10.0, 8.0)
    assert un.leg_len_m == pytest.approx((18.0, 18.0))


def test_edges_are_oriented_along_the_walk() -> None:
    g = _t_graph()
    un = unroll(g, [g.start_node, _node(g, (10.0, 8.0))], START, chunk=4)
    xy = shapely.get_coordinates(un.line).tolist()
    assert xy == [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [10.0, 0.0], [0.0, 0.0]]


def test_cheapest_parallel_edge_geometry_is_used() -> None:
    g = _t_graph()
    lookup = edge_index(g, "cost")
    e = lookup[(g.start_node, _node(g, (10.0, 0.0)))]
    assert g.edge_ref[e] == "a"


def test_repeated_stop_gives_zero_leg_and_no_duplicate_vertex() -> None:
    g = _t_graph()
    tip = _node(g, (20.0, 0.0))
    un = unroll(g, [g.start_node, tip, tip], START, chunk=2)
    assert un.anchors[1] == un.anchors[2]
    assert un.leg_len_m[1] == 0.0
    assert np.all(np.hypot(*np.diff(un.xy, axis=0).T) > 0)


def test_start_node_off_start_is_joined_to_exact_start() -> None:
    g = _t_graph(start_offset=0.3)
    un = unroll(g, [g.start_node, _node(g, (20.0, 0.0))], START, chunk=4)
    xy = shapely.get_coordinates(un.line)
    assert tuple(xy[0]) == START and tuple(xy[-1]) == START
    assert un.line.length == pytest.approx(40.0)


def test_start_node_within_snap_is_replaced() -> None:
    g = _t_graph(start_offset=0.004)
    un = unroll(g, [g.start_node, _node(g, (20.0, 0.0))], START, chunk=4)
    assert len(un.xy) == 5 and tuple(un.xy[0]) == START


def test_single_stop_tour_is_rejected() -> None:
    g = _t_graph()
    with pytest.raises(ValueError, match="at least one stop"):
        unroll(g, [g.start_node], START, chunk=4)


def test_first_stop_must_be_start() -> None:
    g = _t_graph()
    with pytest.raises(ValueError, match="START"):
        unroll(g, [1, 2], START, chunk=4)


def test_dedupe_remaps_anchors() -> None:
    xy = np.array([[0, 0], [1, 0], [1, 0], [2, 0], [2, 0], [0, 0]], float)
    out, anchors = dedupe_xy(xy, (0, 2, 4, 5))
    assert out.tolist() == [[0, 0], [1, 0], [2, 0], [0, 0]]
    assert anchors == (0, 1, 2, 3)


def test_mini_route_unroll_follows_centerlines() -> None:
    m = mini_route()
    far = int(np.argmax(m.graph.node_xy[:, 0]))
    un = unroll(m.graph, [m.graph.start_node, far], m.start_xy, chunk=64)
    assert tuple(un.xy[0]) == m.start_xy
    assert un.line.length == pytest.approx(2 * sum(un.leg_len_m) / 2)
    assert shapely.covered_by(un.line, m.inner)
