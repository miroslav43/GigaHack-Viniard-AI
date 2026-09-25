"""Tour -> LineString (arch §4.11.5, design 04 §3.9).

Each leg between consecutive stops follows its Dijkstra path; the oriented edge geometries are
concatenated, START is the exact first and last vertex, and consecutive duplicates are dropped. Anchor
indices (vertex index of START, of every stop, and of the return to START) survive every step, so string
pulling and validation can refer to legs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString

from vineyard.route.distances import leg_paths
from vineyard.route.graph_types import WalkGraph, Weight

# A START node closer than this to START is moved onto it; farther, START is joined by a segment.
START_SNAP_M: Final = 0.005


@dataclass(frozen=True, eq=False)
class Unrolled:
    """Route vertices (read-only) with one anchor per stop: START, stops..., START."""

    xy: np.ndarray
    anchors: tuple[int, ...]
    leg_len_m: tuple[float, ...]

    @property
    def line(self) -> LineString:
        return LineString(self.xy)


def edge_index(g: WalkGraph, weight: Weight = "cost") -> Mapping[tuple[int, int], int]:
    """(u, v) and (v, u) -> the cheapest edge between them (ties: lowest edge index)."""
    w = g.edge_weights(weight)
    best: dict[tuple[int, int], int] = {}
    for e in np.lexsort((np.arange(g.n_edges), w)):
        u, v = (int(n) for n in g.edge_uv[e])
        best.setdefault((u, v), int(e))
        best.setdefault((v, u), int(e))
    return best


def _path_coords(g: WalkGraph, nodes: Sequence[int], lookup: Mapping[tuple[int, int], int]) -> tuple[np.ndarray, float]:
    """Vertices after the first node of `nodes`, and the walked length."""
    parts, length = [], 0.0
    for u, v in zip(nodes[:-1], nodes[1:], strict=True):
        e = lookup[(u, v)]
        coords = shapely.get_coordinates(g.edge_geom[e])
        parts.append((coords if int(g.edge_uv[e, 0]) == u else coords[::-1])[1:])
        length += float(g.edge_len_m[e])
    return (np.vstack(parts) if parts else np.zeros((0, 2))), length


def dedupe_xy(xy: np.ndarray, anchors: Sequence[int]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Drop consecutive duplicate vertices; each anchor moves to the vertex that replaces it."""
    pts = np.asarray(xy, dtype=np.float64)
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
    new_index = np.cumsum(keep) - 1
    return pts[keep], tuple(int(new_index[a]) for a in anchors)


def _pin_start(xy: np.ndarray, anchors: list[int], start: np.ndarray) -> tuple[np.ndarray, list[int]]:
    out, shift = xy.copy(), 0
    if np.hypot(*(out[0] - start)) <= START_SNAP_M:
        out[0] = start
    else:
        out, shift = np.vstack([start, out]), 1
    anchors = [0] + [a + shift for a in anchors[1:]]
    if np.hypot(*(out[-1] - start)) <= START_SNAP_M:
        out[-1] = start
    else:
        out = np.vstack([out, start])
    anchors[-1] = len(out) - 1
    return out, anchors


def unroll(g: WalkGraph, stops: Sequence[int], start_xy: tuple[float, float], *, chunk: int,
           weight: Weight = "cost", leg_limits: Sequence[float] | None = None) -> Unrolled:
    """Closed walk START -> stops[1] -> ... -> stops[-1] -> START along shortest paths.

    `leg_limits` (the known shortest distance of every leg, return leg last) bounds each Dijkstra.
    """
    if len(stops) < 2:
        raise ValueError(f"a tour needs START and at least one stop, got {list(stops)}")
    if int(stops[0]) != g.start_node:
        raise ValueError(f"tour must begin at the START node {g.start_node}, got {stops[0]}")
    seq = [int(s) for s in stops] + [g.start_node]
    legs = list(zip(seq[:-1], seq[1:], strict=True))
    paths = leg_paths(g.adjacency(weight), legs, chunk, leg_limits)
    lookup = edge_index(g, weight)
    chunks, anchors, lens = [g.node_xy[g.start_node][None, :]], [0], []
    count = 1
    for path in paths:
        coords, length = _path_coords(g, path, lookup)
        chunks.append(coords)
        count += len(coords)
        anchors.append(count - 1)
        lens.append(length)
    xy, pinned = _pin_start(np.vstack(chunks), anchors, np.asarray(start_xy, dtype=np.float64))
    xy, final = dedupe_xy(xy, pinned)
    xy.setflags(write=False)
    return Unrolled(xy=xy, anchors=final, leg_len_m=tuple(lens))


__all__ = ["START_SNAP_M", "Unrolled", "dedupe_xy", "edge_index", "unroll"]
