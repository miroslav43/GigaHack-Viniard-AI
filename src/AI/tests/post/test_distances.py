"""Distances on the walking graph: chunked multi-source Dijkstra, cm costs, path extraction."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from vineyard.route.distances import (
    distance_matrix,
    expand_matrix,
    leg_paths,
    path_nodes,
    shortest_path_trees,
    to_cost_cm,
)
from vineyard.route.graph_types import make_walk_graph


def _path_graph() -> object:
    """0 -1m- 1 -2m- 2 -3m- 3, plus an isolated edge 4-5 and a parallel 0-1 edge of cost 5."""
    xy = np.array([[0, 0], [1, 0], [3, 0], [6, 0], [10, 10], [11, 10]], float)
    uv = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [1, 0]])
    geoms = [LineString([xy[u], xy[v]]) for u, v in uv]
    cost = np.array([1.0, 2.0, 3.0, 1.0, 5.0])
    return make_walk_graph(xy, ["start", "passage", "passage", "row_end", "passage", "passage"], [""] * 6, uv,
                           geoms, ["passage_centerline"] * 5, [""] * 5, np.ones(5), cost, start_node=0)


def test_distance_matrix_symmetric_and_chunked() -> None:
    g = _path_graph()
    full = distance_matrix(g.adjacency("cost"), [0, 2, 3], chunk=256)
    chunked = distance_matrix(g.adjacency("cost"), [0, 2, 3], chunk=1)
    assert np.allclose(full, [[0, 3, 6], [3, 0, 3], [6, 3, 0]])
    assert np.array_equal(full, chunked)


def test_distance_matrix_duplicate_nodes_and_unreachable() -> None:
    g = _path_graph()
    d = distance_matrix(g.adjacency("cost"), [0, 0, 4], chunk=2)
    assert d[0, 1] == 0.0
    assert np.isinf(d[0, 2]) and np.isinf(d[2, 1])


def test_parallel_edges_keep_cheapest() -> None:
    g = _path_graph()
    assert distance_matrix(g.adjacency("cost"), [0, 1], chunk=8)[0, 1] == pytest.approx(1.0)


def test_to_cost_cm_rounds_and_caps_unreachable() -> None:
    d = np.array([[0.0, 1.004], [np.inf, 0.0]])
    cm = to_cost_cm(d, cost_per_m=100, unreachable_cm=10**9)
    assert cm.dtype == np.int64
    assert cm[0, 1] == 100
    assert cm[1, 0] == 10**9


def test_to_cost_cm_rejects_negative() -> None:
    with pytest.raises(ValueError, match="negative"):
        to_cost_cm(np.array([[0.0, -1.0], [1.0, 0.0]]), cost_per_m=100, unreachable_cm=10**9)


def test_expand_matrix_duplicates_rows_and_columns() -> None:
    d = np.array([[0.0, 2.0], [2.0, 0.0]])
    out = expand_matrix(d, np.array([0, 1, 1]))
    assert out.tolist() == [[0.0, 2.0, 2.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


def test_path_nodes_from_trees() -> None:
    g = _path_graph()
    trees = shortest_path_trees(g.adjacency("cost"), [0, 3], chunk=1)
    assert path_nodes(trees[0], 3) == (0, 1, 2, 3)
    assert path_nodes(trees[3], 0) == (3, 2, 1, 0)
    assert path_nodes(trees[0], 0) == (0,)


def test_leg_paths_chunked_match_trees() -> None:
    g = _path_graph()
    legs = [(0, 3), (3, 0), (2, 2), (0, 1)]
    for chunk in (1, 2, 256):
        assert leg_paths(g.adjacency("cost"), legs, chunk) == ((0, 1, 2, 3), (3, 2, 1, 0), (2,), (0, 1))


def test_leg_paths_with_limits_match_unbounded() -> None:
    g = _path_graph()
    legs = [(0, 3), (3, 0), (2, 2), (0, 1)]
    assert leg_paths(g.adjacency("cost"), legs, 4, limits=[6.0, 6.0, 0.0, 1.0]) == (
        (0, 1, 2, 3), (3, 2, 1, 0), (2,), (0, 1))
    with pytest.raises(ValueError, match="limits"):
        leg_paths(g.adjacency("cost"), legs, 4, limits=[1.0])


def test_leg_paths_limit_too_small_raises() -> None:
    g = _path_graph()
    with pytest.raises(ValueError, match="no path"):
        leg_paths(g.adjacency("cost"), [(0, 3)], 4, limits=[2.0])


def test_leg_paths_unreachable_raises() -> None:
    g = _path_graph()
    with pytest.raises(ValueError, match="no path"):
        leg_paths(g.adjacency("cost"), [(0, 4)], chunk=8)


def test_path_nodes_unreachable_raises() -> None:
    g = _path_graph()
    trees = shortest_path_trees(g.adjacency("cost"), [0], chunk=4)
    with pytest.raises(ValueError, match="no path"):
        path_nodes(trees[0], 5)


def test_distance_matrix_rejects_bad_chunk() -> None:
    g = _path_graph()
    with pytest.raises(ValueError, match="chunk"):
        distance_matrix(g.adjacency("cost"), [0], chunk=0)
