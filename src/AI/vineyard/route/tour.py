"""One GTSP solve on the walking graph: candidate sets -> cm matrix -> tour -> pulled LineString.

TSP index space (design 04 §3.8): index 0 is START, then every node of every set in order. A graph node
that belongs to two sets appears twice (zero-cost duplicates), so each index is in exactly one set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from vineyard.config import RouteConfig, RouteSolverConfig
from vineyard.errors import RouteValidationError
from vineyard.route.distances import distance_matrix, to_cost_cm
from vineyard.route.graph_types import CandidateSet, WalkGraph
from vineyard.route.pulling import string_pull
from vineyard.route.solver import DEPOT, solve_gtsp, time_limit_for
from vineyard.route.unroll import unroll

# Never reached (every set node is in START's component, checked below); keeps the matrix integral.
UNREACHABLE_CM: Final = 10**12


@dataclass(frozen=True)
class TourParams:
    solver: RouteSolverConfig
    time_limit_s: int
    chunk: int
    cost_per_m: int
    lookahead: int
    start_xy: tuple[float, float]

    @classmethod
    def from_route_cfg(cls, cfg: RouteConfig, start_xy: tuple[float, float],
                       time_limit_s: int | None = None) -> TourParams:
        limit = time_limit_s if time_limit_s is not None else time_limit_for(cfg.solver)
        return cls(solver=cfg.solver, time_limit_s=limit, chunk=cfg.solver.dijkstra_chunk,
                   cost_per_m=cfg.solver.cost_per_m, lookahead=cfg.graph.pull_lookahead,
                   start_xy=(float(start_xy[0]), float(start_xy[1])))


@dataclass(frozen=True, eq=False)
class Tour:
    """Pulled closed route; `anchors` index START, every stop, and the return to START in `xy`."""

    xy: np.ndarray
    anchors: tuple[int, ...]
    stop_nodes: tuple[int, ...]
    stop_sets: tuple[int, ...]
    solver: str
    solve_time_s: float
    cost_cm: int
    fallback_reason: str = ""

    @property
    def line(self) -> LineString:
        return LineString(self.xy)

    @property
    def length_m(self) -> float:
        return float(np.hypot(*np.diff(self.xy, axis=0).T).sum())


@dataclass(frozen=True, eq=False)
class DistanceTable:
    """Shortest-path distances (cost weights) between a fixed node pool of one graph; built once, sliced often."""

    nodes: np.ndarray
    d_m: np.ndarray

    @classmethod
    def build(cls, g: WalkGraph, nodes: Sequence[int], chunk: int) -> DistanceTable:
        pool = np.unique(np.asarray(nodes, dtype=np.int64))
        return cls(nodes=pool, d_m=distance_matrix(g.adjacency("cost"), pool.tolist(), chunk))

    @classmethod
    def for_sets(cls, g: WalkGraph, sets: Sequence[CandidateSet], chunk: int) -> DistanceTable:
        return cls.build(g, [g.start_node, *(n for s in sets for n in s.nodes)], chunk)

    def matrix(self, nodes: np.ndarray) -> np.ndarray:
        idx = np.asarray(nodes, dtype=np.int64)
        pos = np.searchsorted(self.nodes, idx)
        missing = (pos >= len(self.nodes)) | (self.nodes[np.minimum(pos, len(self.nodes) - 1)] != idx)
        if np.any(missing):
            raise ValueError(f"nodes not in the distance table: {idx[missing][:5].tolist()}")
        return self.d_m[np.ix_(pos, pos)]


def index_space(g: WalkGraph, sets: Sequence[CandidateSet]) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """(graph node per TSP index, GTSP index sets, owning set per index; -1 for START)."""
    nodes, owner, gtsp = [g.start_node], [-1], []
    for k, cset in enumerate(sets):
        gtsp.append(list(range(len(nodes), len(nodes) + len(cset.nodes))))
        nodes += [int(n) for n in cset.nodes]
        owner += [k] * len(cset.nodes)
    return np.asarray(nodes, dtype=np.int64), gtsp, np.asarray(owner, dtype=np.int64)


def _check_reachable(g: WalkGraph, nodes: np.ndarray) -> None:
    off = nodes[g.component[nodes] != g.start_component]
    if len(off):
        raise RouteValidationError("candidate nodes outside START's component", n_nodes=len(off),
                                   examples=off[:5].tolist())


def solve_tour(g: WalkGraph, sets: Sequence[CandidateSet], eroded: BaseGeometry, params: TourParams,
               table: DistanceTable | None = None) -> Tour:
    """Best closed tour from START visiting one node of every set, unrolled and string-pulled."""
    if not sets:
        raise RouteValidationError("no candidate sets to route")
    nodes, gtsp, owner = index_space(g, sets)
    _check_reachable(g, nodes)
    d_m = (table if table is not None else DistanceTable.build(g, nodes, params.chunk)).matrix(nodes)
    d_cm = to_cost_cm(d_m, params.cost_per_m, UNREACHABLE_CM)
    res = solve_gtsp(d_cm, gtsp, params.solver, time_limit_s=params.time_limit_s)
    order = [p for p in res.order if p != DEPOT]
    stop_nodes = tuple(int(nodes[p]) for p in order)
    seq = [DEPOT, *order, DEPOT]
    limits = [float(d_m[a, b]) for a, b in zip(seq[:-1], seq[1:], strict=True)]
    un = unroll(g, [g.start_node, *stop_nodes], params.start_xy, chunk=params.chunk, leg_limits=limits)
    xy, anchors = string_pull(un.xy, un.anchors, eroded, params.lookahead)
    xy.setflags(write=False)
    return Tour(xy=xy, anchors=anchors, stop_nodes=stop_nodes, stop_sets=tuple(int(owner[p]) for p in order),
                solver=res.solver, solve_time_s=res.solve_time_s, cost_cm=res.cost_cm,
                fallback_reason=res.fallback_reason)


__all__ = ["UNREACHABLE_CM", "DistanceTable", "Tour", "TourParams", "index_space", "solve_tour"]
