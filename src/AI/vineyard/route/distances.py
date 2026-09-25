"""Graph distances for the route solver (arch §4.11.4): chunked multi-source Dijkstra on a CSR graph.

Only the k x k block between the TSP nodes is kept, so memory stays at chunk x N floats per pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

# scipy marks "no predecessor" with this value.
NO_PREDECESSOR: Final = -9999
# A bounded Dijkstra must still reach a target at exactly the known distance despite float rounding.
LIMIT_REL_MARGIN: Final = 1e-9
LIMIT_ABS_MARGIN: Final = 1e-6


@dataclass(frozen=True, eq=False)
class PathTree:
    """Single-source shortest-path tree: distances and predecessors over every node."""

    source: int
    dist: np.ndarray
    pred: np.ndarray


def _unique_sources(nodes: Sequence[int]) -> np.ndarray:
    return np.unique(np.asarray(nodes, dtype=np.int64))


def _check_chunk(chunk: int) -> None:
    if chunk < 1:
        raise ValueError(f"dijkstra chunk must be >= 1, got {chunk}")


def distance_matrix(adj: sp.csr_matrix, nodes: Sequence[int], chunk: int) -> np.ndarray:
    """(k, k) shortest-path distances between `nodes` (duplicates allowed); inf when unreachable."""
    _check_chunk(chunk)
    idx = np.asarray(nodes, dtype=np.int64)
    uniq = _unique_sources(idx)
    block = np.empty((len(uniq), len(uniq)), dtype=np.float64)
    for lo in range(0, len(uniq), chunk):
        part = uniq[lo:lo + chunk]
        rows = dijkstra(adj, directed=False, indices=part)
        block[lo:lo + len(part)] = np.atleast_2d(rows)[:, uniq]
    pos = np.searchsorted(uniq, idx)
    return block[np.ix_(pos, pos)]


def shortest_path_trees(adj: sp.csr_matrix, sources: Sequence[int], chunk: int) -> Mapping[int, PathTree]:
    """Shortest-path trees for each distinct source, computed `chunk` sources at a time."""
    _check_chunk(chunk)
    uniq = _unique_sources(sources)
    trees: dict[int, PathTree] = {}
    for lo in range(0, len(uniq), chunk):
        part = uniq[lo:lo + chunk]
        dist, pred = dijkstra(adj, directed=False, indices=part, return_predecessors=True)
        dist, pred = np.atleast_2d(dist), np.atleast_2d(pred)
        for row, src in enumerate(part):
            trees[int(src)] = PathTree(int(src), dist[row].copy(), pred[row].copy())
    return MappingProxyType(trees)


def path_nodes(tree: PathTree, target: int) -> tuple[int, ...]:
    """Node sequence source -> target along `tree`."""
    if not np.isfinite(tree.dist[target]):
        raise ValueError(f"no path from node {tree.source} to node {target}")
    out = [int(target)]
    while out[-1] != tree.source:
        prev = int(tree.pred[out[-1]])
        if prev == NO_PREDECESSOR:
            raise ValueError(f"broken predecessor chain from {tree.source} to {target} at {out[-1]}")
        out.append(prev)
    return tuple(reversed(out))


def _limited_paths(adj: sp.csr_matrix, legs: Sequence[tuple[int, int]], by_source: Mapping[int, list[int]],
                   limits: Sequence[float]) -> list[tuple[int, ...]]:
    """One bounded Dijkstra per source: nothing beyond its longest leg (plus a margin) is explored."""
    out: list[tuple[int, ...]] = [()] * len(legs)
    for src, ks in sorted(by_source.items()):
        bound = max(float(limits[k]) for k in ks)
        bound = bound * (1.0 + LIMIT_REL_MARGIN) + LIMIT_ABS_MARGIN if np.isfinite(bound) else np.inf
        dist, pred = dijkstra(adj, directed=False, indices=src, return_predecessors=True, limit=bound)
        tree = PathTree(int(src), np.asarray(dist), np.asarray(pred))
        for k in ks:
            out[k] = path_nodes(tree, int(legs[k][1]))
    return out


def leg_paths(adj: sp.csr_matrix, legs: Sequence[tuple[int, int]], chunk: int,
              limits: Sequence[float] | None = None) -> tuple[tuple[int, ...], ...]:
    """Node path of every (source, target) leg.

    With `limits` (the known leg distances) every source runs a bounded Dijkstra; otherwise trees are
    built `chunk` sources at a time and dropped.
    """
    _check_chunk(chunk)
    by_source: dict[int, list[int]] = {}
    for k, (src, _) in enumerate(legs):
        by_source.setdefault(int(src), []).append(k)
    if limits is not None:
        if len(limits) != len(legs):
            raise ValueError(f"{len(limits)} leg limits for {len(legs)} legs")
        return tuple(_limited_paths(adj, legs, by_source, limits))
    sources = np.array(sorted(by_source), dtype=np.int64)
    out: list[tuple[int, ...]] = [()] * len(legs)
    for lo in range(0, len(sources), chunk):
        part = sources[lo:lo + chunk]
        dist, pred = dijkstra(adj, directed=False, indices=part, return_predecessors=True)
        dist, pred = np.atleast_2d(dist), np.atleast_2d(pred)
        for row, src in enumerate(part):
            tree = PathTree(int(src), dist[row], pred[row])
            for k in by_source[int(src)]:
                out[k] = path_nodes(tree, int(legs[k][1]))
    return tuple(out)


def expand_matrix(d: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Matrix over `index` positions of `d` (rows and columns duplicated as needed)."""
    idx = np.asarray(index, dtype=np.int64)
    return np.asarray(d)[np.ix_(idx, idx)]


def to_cost_cm(d_m: np.ndarray, cost_per_m: int, unreachable_cm: int) -> np.ndarray:
    """Integer costs (OR-Tools needs int): round(d * cost_per_m); inf -> `unreachable_cm`."""
    d = np.asarray(d_m, dtype=np.float64)
    finite = np.isfinite(d)
    if np.any(d[finite] < 0.0):
        raise ValueError("distance matrix has negative entries")
    out = np.full(d.shape, unreachable_cm, dtype=np.int64)
    out[finite] = np.minimum(np.rint(d[finite] * cost_per_m), unreachable_cm).astype(np.int64)
    return out


__all__ = [
    "NO_PREDECESSOR", "PathTree", "distance_matrix", "expand_matrix", "leg_paths", "path_nodes",
    "shortest_path_trees", "to_cost_cm",
]
