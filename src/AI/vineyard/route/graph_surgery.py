"""Immutable graph edits for the route stage: split edges at candidate points, add target spurs.

A split piece inherits its parent's kind, ref and inside fraction; its cost is the parent cost scaled
by the piece length, so the total cost of the graph is unchanged by splitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.ops import substring

from vineyard.contracts.enums import EdgeKind
from vineyard.route.graph_types import NodeKind, WalkGraph, make_walk_graph

NodeKey = tuple[str, int, float]   # ("node", node, 0.0) or ("split", edge, t)
# Two split parameters closer than this on one edge are the same node (avoids zero-length pieces).
MIN_PIECE_M: Final = 1e-6
FULL_INSIDE: Final = 1.0


@dataclass(frozen=True)
class SpurDraft:
    attach: NodeKey
    end_xy: tuple[float, float]


@dataclass(frozen=True)
class SplitPlan:
    keys: Mapping[str, tuple[NodeKey, ...]]
    spurs: Mapping[str, SpurDraft] = field(default_factory=dict)
    xy: Mapping[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Request:
    edge: int
    t: float
    target_id: str
    strict: bool   # spur attach points never move by more than the base merge distance


def _requests(plan: SplitPlan) -> list[_Request]:
    out = [_Request(k[1], k[2], tid, False) for tid, keys in plan.keys.items() for k in keys if k[0] == "split"]
    out += [_Request(s.attach[1], s.attach[2], tid, True) for tid, s in plan.spurs.items()
            if s.attach[0] == "split"]
    return sorted(out, key=lambda r: (r.edge, r.t, r.target_id))


def _fits(g: WalkGraph, req: _Request, t_rep: float, plan: SplitPlan, keep_within_m: float,
          strict_m: float) -> bool:
    if req.strict:
        return abs(req.t - t_rep) <= strict_m
    pt = shapely.get_coordinates(shapely.line_interpolate_point(g.edge_geom[req.edge], t_rep))[0]
    return float(np.hypot(*(pt - plan.xy[req.target_id]))) <= keep_within_m


def _clusters(g: WalkGraph, reqs: Sequence[_Request], plan: SplitPlan, merge_m: float,
              keep_within_m: float, strict_m: float) -> list[tuple[float, list[_Request]]]:
    """Greedy runs of requests within `merge_m`; members that would drift too far stay alone."""
    runs: list[list[_Request]] = []
    for req in reqs:
        if runs and req.t - runs[-1][0].t <= merge_m:
            runs[-1].append(req)
        else:
            runs.append([req])
    out: list[tuple[float, list[_Request]]] = []
    for run in runs:
        t_rep = float(np.mean([r.t for r in run]))
        kept = [r for r in run if _fits(g, r, t_rep, plan, keep_within_m, strict_m)]
        out += [(t_rep, kept)] if kept else []
        out += [(r.t, [r]) for r in run if r not in kept]
    out.sort(key=lambda c: c[0])
    merged: list[tuple[float, list[_Request]]] = []
    for t, members in out:
        if merged and t - merged[-1][0] < MIN_PIECE_M:
            merged[-1][1].extend(members)
        else:
            merged.append((t, list(members)))
    return merged


@dataclass
class _Builder:
    """Mutable scratch space local to `apply_splits`; the result is a new frozen WalkGraph."""

    xy: list[tuple[float, float]]
    kind: list[str]
    ref: list[str]
    uv: list[tuple[int, int]] = field(default_factory=list)
    geom: list[LineString] = field(default_factory=list)
    ekind: list[str] = field(default_factory=list)
    eref: list[str] = field(default_factory=list)
    inside: list[float] = field(default_factory=list)
    cost: list[float] = field(default_factory=list)

    def add_node(self, xy: np.ndarray | tuple[float, float], kind: str, ref: str) -> int:
        self.xy.append((float(xy[0]), float(xy[1])))
        self.kind.append(kind)
        self.ref.append(ref)
        return len(self.xy) - 1

    def add_edge(self, u: int, v: int, geom: LineString, kind: str, ref: str, inside: float, cost: float) -> None:
        coords = shapely.get_coordinates(geom)
        coords[0], coords[-1] = self.xy[u], self.xy[v]
        self.uv.append((u, v))
        self.geom.append(LineString(coords))
        self.ekind.append(kind)
        self.eref.append(ref)
        self.inside.append(inside)
        self.cost.append(cost)


def _split_edge(g: WalkGraph, b: _Builder, e: int, clusters: list[tuple[float, list[_Request]]],
                where: dict[tuple[int, float, str], int]) -> None:
    geom, length = g.edge_geom[e], float(g.edge_len_m[e])
    ts = [c[0] for c in clusters]
    pts = shapely.get_coordinates(shapely.line_interpolate_point(geom, ts))
    chain = [int(g.edge_uv[e, 0])]
    for (_, members), pt in zip(clusters, pts, strict=True):
        node = b.add_node(pt, NodeKind.TARGET, min(r.target_id for r in members))
        chain.append(node)
        for r in members:
            where[(e, r.t, r.target_id)] = node
    chain.append(int(g.edge_uv[e, 1]))
    cuts = [0.0, *ts, length]
    for (a, c), (u, v) in zip(zip(cuts[:-1], cuts[1:], strict=True), zip(chain[:-1], chain[1:], strict=True),
                              strict=True):
        piece = substring(geom, a, c)
        b.add_edge(u, v, piece, g.edge_kind[e], g.edge_ref[e], float(g.edge_inside_frac[e]),
                   float(g.edge_cost[e]) * (c - a) / length)


def _resolve(key: NodeKey, tid: str, where: Mapping[tuple[int, float, str], int]) -> int:
    return key[1] if key[0] == "node" else where[(key[1], key[2], tid)]


def apply_splits(g: WalkGraph, plan: SplitPlan, *, merge_m: float,
                 keep_within_m: float) -> tuple[WalkGraph, dict[str, tuple[int, ...]]]:
    """New graph with every requested split and spur; returns it with the candidate nodes per target."""
    reqs = _requests(plan)
    by_edge: dict[int, list[_Request]] = {}
    for r in reqs:
        by_edge.setdefault(r.edge, []).append(r)
    b = _Builder(xy=[tuple(p) for p in g.node_xy.tolist()], kind=list(g.node_kind), ref=list(g.node_ref))
    where: dict[tuple[int, float, str], int] = {}
    strict_m = min(merge_m, keep_within_m)
    for e in range(g.n_edges):
        if e in by_edge:
            _split_edge(g, b, e, _clusters(g, by_edge[e], plan, merge_m, keep_within_m, strict_m), where)
        else:
            u, v = (int(n) for n in g.edge_uv[e])
            b.add_edge(u, v, g.edge_geom[e], g.edge_kind[e], g.edge_ref[e], float(g.edge_inside_frac[e]),
                       float(g.edge_cost[e]))
    nodes: dict[str, tuple[int, ...]] = {}
    for tid, keys in plan.keys.items():
        nodes[tid] = tuple(dict.fromkeys(_resolve(k, tid, where) for k in keys))
    for tid, spur in sorted(plan.spurs.items()):
        attach = _resolve(spur.attach, tid, where)
        end = b.add_node(spur.end_xy, NodeKind.TARGET, tid)
        geom = LineString([b.xy[attach], b.xy[end]])
        b.add_edge(attach, end, geom, EdgeKind.TARGET_SPUR, tid, FULL_INSIDE, geom.length)
        nodes[tid] = (end,)
    graph = make_walk_graph(np.array(b.xy), b.kind, b.ref, np.array(b.uv, dtype=np.int64).reshape(-1, 2), b.geom,
                            b.ekind, b.eref, np.array(b.inside), np.array(b.cost), start_node=g.start_node)
    return graph, {k: v for k, v in nodes.items() if v}


def restrict_edges(g: WalkGraph, keep: np.ndarray, cost: np.ndarray | None = None) -> WalkGraph:
    """Same nodes, only the edges where `keep` is True (optionally with new costs); components recomputed."""
    mask = np.asarray(keep, dtype=bool)
    if mask.shape != (g.n_edges,):
        raise ValueError(f"restrict_edges: mask shape {mask.shape} != ({g.n_edges},)")
    idx = np.flatnonzero(mask)
    new_cost = g.edge_cost if cost is None else np.asarray(cost, dtype=np.float64)
    return make_walk_graph(g.node_xy, g.node_kind, g.node_ref, g.edge_uv[idx], [g.edge_geom[i] for i in idx],
                           [g.edge_kind[i] for i in idx], [g.edge_ref[i] for i in idx], g.edge_inside_frac[idx],
                           new_cost[idx], start_node=g.start_node)


__all__ = ["MIN_PIECE_M", "NodeKey", "SplitPlan", "SpurDraft", "apply_splits", "restrict_edges"]
