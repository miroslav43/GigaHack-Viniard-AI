"""Frozen walking-graph types shared by the passable and route stages (design 04 §2, arch §4.11.2).

A `WalkGraph` is an undirected multigraph in EPSG:32635 metres. Arrays are read-only copies, so a
graph can be shared between stages and threads; every change builds a new graph through
`make_walk_graph` (candidates splitting edges, target spurs, ...).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString

from vineyard.contracts.enums import EdgeKind

# Edges shorter than this still get a strictly positive weight: csgraph drops explicit zeros in
# some code paths, which would silently disconnect coincident nodes.
MIN_EDGE_WEIGHT_M: Final = 1e-6
# Edge geometry ends must sit on their nodes; 1 mm is the domain grid size (arch §4.11.1).
ENDPOINT_TOL_M: Final = 1e-3
NO_COMPONENT: Final = -1

Weight = Literal["cost", "length"]


class NodeKind(StrEnum):
    START = "start"
    ROW_END = "row_end"          # end of an interrow centerline piece
    INTERROW = "interrow"        # intermediate interrow centerline node (every centerline_step_m)
    PASSAGE = "passage"          # passage skeleton node (end or every passage_node_step_m)
    JUNCTION = "junction"        # skeleton junction (degree >= 3)
    ATTACH = "attach"            # point inserted on an edge by a connector or snap join
    TARGET = "target"            # candidate projection added by the route stage


@dataclass(frozen=True)
class PolylineDraft:
    """A walkable polyline before node snapping: centerline piece, skeleton branch or connector."""

    geom: LineString
    kind: EdgeKind
    ref: str
    outside_len_m: float = 0.0

    def __post_init__(self) -> None:
        if self.geom.is_empty or len(self.geom.coords) < 2:
            raise ValueError(f"PolylineDraft {self.ref!r}: geometry needs >= 2 vertices")
        if self.outside_len_m < 0.0 or not np.isfinite(self.outside_len_m):
            raise ValueError(f"PolylineDraft {self.ref!r}: bad outside_len_m {self.outside_len_m}")
        object.__setattr__(self, "kind", EdgeKind(self.kind))


def _frozen(arr: np.ndarray, dtype: type | np.dtype, name: str) -> np.ndarray:
    out = np.array(arr, dtype=dtype, copy=True)
    if out.dtype.kind == "f" and not np.all(np.isfinite(out)):
        raise ValueError(f"WalkGraph.{name} contains non-finite values")
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class WalkGraph:
    """Undirected walking graph. Arrays are read-only; use `make_walk_graph` to build one."""

    node_xy: np.ndarray                 # (N, 2) float64 UTM
    node_kind: tuple[str, ...]          # NodeKind values
    node_ref: tuple[str, ...]           # interrow_id / passage branch / target_id / ""
    edge_uv: np.ndarray                 # (E, 2) int64, u != v
    edge_geom: tuple[LineString, ...]   # oriented u -> v
    edge_len_m: np.ndarray              # (E,) float64
    edge_cost: np.ndarray               # (E,) float64 = length + penalty * outside length
    edge_inside_frac: np.ndarray        # (E,) float64 in [0, 1], against domain.inner
    edge_kind: tuple[str, ...]          # EdgeKind values
    edge_ref: tuple[str, ...]
    start_node: int
    component: np.ndarray               # (N,) int32 connected-component label

    def __post_init__(self) -> None:
        n, e = len(self.node_xy), len(self.edge_uv)
        if self.node_xy.shape != (n, 2) or len(self.node_kind) != n or len(self.node_ref) != n:
            raise ValueError(f"WalkGraph: node arrays disagree (xy {self.node_xy.shape}, kinds "
                             f"{len(self.node_kind)}, refs {len(self.node_ref)})")
        edge_lens = {len(self.edge_geom), len(self.edge_len_m), len(self.edge_cost),
                     len(self.edge_inside_frac), len(self.edge_kind), len(self.edge_ref)}
        if self.edge_uv.shape != (e, 2) or edge_lens != {e}:
            raise ValueError(f"WalkGraph: edge arrays disagree (uv {self.edge_uv.shape}, lengths {edge_lens})")
        if self.component.shape != (n,):
            raise ValueError(f"WalkGraph: component shape {self.component.shape} != ({n},)")
        if not 0 <= self.start_node < n:
            raise ValueError(f"WalkGraph: start_node {self.start_node} out of range [0, {n})")

    @property
    def n_nodes(self) -> int:
        return len(self.node_xy)

    @property
    def n_edges(self) -> int:
        return len(self.edge_uv)

    @property
    def start_component(self) -> int:
        return int(self.component[self.start_node])

    def edge_weights(self, weight: Weight = "cost") -> np.ndarray:
        if weight == "cost":
            return self.edge_cost
        if weight == "length":
            return self.edge_len_m
        raise ValueError(f"unknown edge weight {weight!r} (expected 'cost' or 'length')")

    def adjacency(self, weight: Weight = "cost") -> sp.csr_matrix:
        """Symmetric CSR matrix; parallel edges keep the cheapest weight."""
        return adjacency_matrix(self.n_nodes, self.edge_uv, self.edge_weights(weight))

    def outside_len_m(self) -> np.ndarray:
        return self.edge_len_m * (1.0 - self.edge_inside_frac)

    def equals(self, other: WalkGraph, tol: float = 1e-9) -> bool:
        """Structural equality (geometry compared vertex-wise within `tol`)."""
        same_meta = (self.node_kind == other.node_kind and self.node_ref == other.node_ref
                     and self.edge_kind == other.edge_kind and self.edge_ref == other.edge_ref
                     and self.start_node == other.start_node and self.n_edges == other.n_edges)
        if not same_meta or self.node_xy.shape != other.node_xy.shape:
            return False
        arrays = ((self.node_xy, other.node_xy), (self.edge_len_m, other.edge_len_m),
                  (self.edge_cost, other.edge_cost), (self.edge_inside_frac, other.edge_inside_frac))
        if not all(np.allclose(a, b, atol=tol, rtol=0.0) for a, b in arrays):
            return False
        if not np.array_equal(self.edge_uv, other.edge_uv):
            return False
        return all(ga.equals_exact(gb, tol) for ga, gb in zip(self.edge_geom, other.edge_geom, strict=True))


def adjacency_matrix(n_nodes: int, edge_uv: np.ndarray, weights: np.ndarray) -> sp.csr_matrix:
    """Symmetric (n, n) CSR adjacency with min weight over parallel edges, weights >= MIN_EDGE_WEIGHT_M."""
    uv = np.asarray(edge_uv, dtype=np.int64).reshape(-1, 2)
    w = np.maximum(np.asarray(weights, dtype=np.float64), MIN_EDGE_WEIGHT_M)
    if len(uv) == 0:
        return sp.csr_matrix((n_nodes, n_nodes), dtype=np.float64)
    rows = np.concatenate([uv[:, 0], uv[:, 1]])
    cols = np.concatenate([uv[:, 1], uv[:, 0]])
    data = np.concatenate([w, w])
    key = rows * n_nodes + cols
    order = np.lexsort((data, key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    keep = order[first]
    return sp.csr_matrix((data[keep], (rows[keep], cols[keep])), shape=(n_nodes, n_nodes))


def component_labels(n_nodes: int, edge_uv: np.ndarray) -> np.ndarray:
    """Connected-component label per node (scipy numbering, deterministic for a given graph)."""
    if n_nodes == 0:
        return np.zeros(0, dtype=np.int32)
    adj = adjacency_matrix(n_nodes, edge_uv, np.ones(len(np.asarray(edge_uv).reshape(-1, 2))))
    _, labels = connected_components(adj, directed=False)
    return labels.astype(np.int32)


def _check_edges(node_xy: np.ndarray, edge_uv: np.ndarray, edge_geom: Sequence[LineString]) -> None:
    n = len(node_xy)
    if len(edge_uv) and (edge_uv.min() < 0 or edge_uv.max() >= n):
        raise ValueError(f"WalkGraph: edge endpoint index out of range [0, {n})")
    loops = np.flatnonzero(edge_uv[:, 0] == edge_uv[:, 1]) if len(edge_uv) else np.zeros(0, int)
    if len(loops):
        raise ValueError(f"WalkGraph: self-loop edges {loops[:5].tolist()}")
    for i, geom in enumerate(edge_geom):
        coords = np.asarray(geom.coords)
        u, v = edge_uv[i]
        if (np.hypot(*(coords[0, :2] - node_xy[u])) > ENDPOINT_TOL_M
                or np.hypot(*(coords[-1, :2] - node_xy[v])) > ENDPOINT_TOL_M):
            raise ValueError(f"WalkGraph: edge {i} ({u}->{v}) geometry ends are not on its nodes")


def make_walk_graph(
    node_xy: np.ndarray,
    node_kind: Sequence[str],
    node_ref: Sequence[str],
    edge_uv: np.ndarray,
    edge_geom: Sequence[LineString],
    edge_kind: Sequence[str],
    edge_ref: Sequence[str],
    edge_inside_frac: np.ndarray,
    edge_cost: np.ndarray | None = None,
    *,
    start_node: int,
) -> WalkGraph:
    """Validated, read-only WalkGraph. Lengths come from the geometries; cost defaults to length."""
    xy = _frozen(np.asarray(node_xy, dtype=np.float64).reshape(-1, 2), np.float64, "node_xy")
    uv = _frozen(np.asarray(edge_uv, dtype=np.int64).reshape(-1, 2), np.int64, "edge_uv")
    geoms = tuple(edge_geom)
    _check_edges(xy, uv, geoms)
    lengths = _frozen(np.array([g.length for g in geoms], dtype=np.float64), np.float64, "edge_len_m")
    inside = _frozen(np.clip(np.asarray(edge_inside_frac, dtype=np.float64), 0.0, 1.0), np.float64,
                     "edge_inside_frac")
    cost = _frozen(lengths if edge_cost is None else np.asarray(edge_cost, dtype=np.float64), np.float64,
                   "edge_cost")
    if np.any(cost < 0.0):
        raise ValueError("WalkGraph: negative edge cost")
    return WalkGraph(
        node_xy=xy, node_kind=tuple(str(NodeKind(k)) for k in node_kind), node_ref=tuple(map(str, node_ref)),
        edge_uv=uv, edge_geom=geoms, edge_len_m=lengths, edge_cost=cost, edge_inside_frac=inside,
        edge_kind=tuple(str(EdgeKind(k)) for k in edge_kind), edge_ref=tuple(map(str, edge_ref)),
        start_node=int(start_node), component=_frozen(component_labels(len(xy), uv), np.int32, "component"),
    )


def connector_cost(length_m: float | np.ndarray, outside_len_m: float | np.ndarray,
                   penalty: float) -> float | np.ndarray:
    """Cost of an edge that leaves the domain for `outside_len_m` (arch §4.11.2 penalised connectors)."""
    return length_m + penalty * outside_len_m


@dataclass(frozen=True)
class CandidateSet:
    """GTSP disjunction: the graph nodes from which the targets in `target_ids` count as visited."""

    target_ids: tuple[str, ...]
    nodes: tuple[int, ...]
    role: Literal["must", "optional"]


@dataclass(frozen=True)
class Reach:
    """Final graph-based reachability of one target."""

    target_id: str
    reachable: bool
    note: str
    n_candidates: int
    snap_dist_m: float


__all__ = [
    "ENDPOINT_TOL_M", "MIN_EDGE_WEIGHT_M", "NO_COMPONENT", "CandidateSet", "EdgeKind", "NodeKind",
    "PolylineDraft", "Reach", "WalkGraph", "Weight", "adjacency_matrix", "component_labels", "connector_cost",
    "make_walk_graph",
]
