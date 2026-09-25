"""GTSP candidate sets (arch §4.11.3, design 04 §3.8).

For every reachable target, each graph edge within `radius_m` contributes its closest point (edge
endpoints within `node_snap_m` are reused, other points split the edge). Candidates are grouped by
side (sign of the cross product with the nearest edge's direction), so both neighbouring interrows
of a row target stay available, and at most `max_per_side` are kept per side. A target with no edge
within `radius_m` but one within `max_snap_m` gets a `target_spur` edge ending `spur_standoff_m` short
of it, when the spur lies in `inner`. Targets whose candidate nodes coincide share one CandidateSet.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.config import RouteConfig
from vineyard.route.graph_surgery import SplitPlan, SpurDraft, apply_splits
from vineyard.route.graph_types import CandidateSet, Reach, WalkGraph

ROLE_MUST: Final = "must"
ROLE_OPTIONAL: Final = "optional"
NOTE_TOO_FAR: Final = "too_far"
NOTE_DISCONNECTED: Final = "disconnected"
NOTE_OUTSIDE_ONLY: Final = "outside_only"
# Priorities 1-2 are routed as "must" when the targets layer carries no route_role (design 04 §3.5).
MUST_MAX_PRIORITY: Final = 2
# CONFIG-REQUEST: route.graph.candidate_node_snap_m = 0.05
CANDIDATE_NODE_SNAP_M: Final = 0.05
# Half-length of the probe that reads an edge's local direction at a projection.
SIDE_PROBE_M: Final = 0.1
# |cross| below this means the target lies on the candidate's edge (its own side, 0).
SIDE_EPS_M: Final = 1e-3
REQUIRED_TARGET_COLUMNS: Final = ("target_id",)


@dataclass(frozen=True)
class TargetPoint:
    target_id: str
    x: float
    y: float
    role: str
    reachable: bool = True
    note: str = ""
    priority: int = MUST_MAX_PRIORITY
    gap_length_m: float = math.nan


@dataclass(frozen=True)
class CandidateParams:
    radius_m: float
    max_per_side: int
    max_snap_m: float
    spur_standoff_m: float
    node_snap_m: float = CANDIDATE_NODE_SNAP_M
    merge_m: float = 0.0

    def __post_init__(self) -> None:
        if self.max_per_side < 1 or self.radius_m <= 0.0 or self.max_snap_m < self.radius_m:
            raise ValueError(f"bad candidate params: {self}")
        if not 0.0 < self.spur_standoff_m < self.radius_m:
            raise ValueError(f"spur_standoff_m must be in (0, radius_m), got {self.spur_standoff_m}")

    @classmethod
    def from_route_cfg(cls, cfg: RouteConfig) -> CandidateParams:
        return cls(radius_m=cfg.candidate_radius_m, max_per_side=cfg.max_candidates_per_side,
                   max_snap_m=cfg.max_snap_m, spur_standoff_m=cfg.graph.target_spur_standoff_m)

    def with_changes(self, **changes: float | int) -> CandidateParams:
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True, eq=False)
class Augmented:
    graph: WalkGraph
    sets: tuple[CandidateSet, ...]
    reach: tuple[Reach, ...]
    target_nodes: Mapping[str, tuple[int, ...]]

    def nodes_of(self, target_id: str) -> tuple[int, ...]:
        return self.target_nodes.get(target_id, ())


@dataclass(frozen=True)
class _Hit:
    edge: int
    t: float
    dist: float
    side: int


def _column(frame: gpd.GeoDataFrame, name: str, default: object) -> list[object]:
    return list(frame[name]) if name in frame.columns else [default] * len(frame)


def _role_of(role: object, priority: int) -> str:
    if isinstance(role, str) and role in (ROLE_MUST, ROLE_OPTIONAL):
        return role
    return ROLE_MUST if priority <= MUST_MAX_PRIORITY else ROLE_OPTIONAL


def target_points(targets: gpd.GeoDataFrame) -> tuple[TargetPoint, ...]:
    """Plain target records from a `targets` layer (geometry wins over x/y)."""
    missing = [c for c in REQUIRED_TARGET_COLUMNS if c not in targets.columns]
    if missing:
        raise ValueError(f"targets layer lacks columns {missing}; has {list(targets.columns)}")
    xy = shapely.get_coordinates(targets.geometry.values)
    prios = [int(p) for p in _column(targets, "priority", MUST_MAX_PRIORITY)]
    roles = _column(targets, "route_role", None)
    reach = _column(targets, "reachable", True)
    notes = _column(targets, "reach_note", "")
    gaps = _column(targets, "gap_length_m", math.nan)
    return tuple(
        TargetPoint(target_id=str(tid), x=float(xy[i, 0]), y=float(xy[i, 1]), role=_role_of(roles[i], prios[i]),
                    reachable=bool(reach[i]), note="" if notes[i] is None else str(notes[i]), priority=prios[i],
                    gap_length_m=float(gaps[i]) if gaps[i] is not None else math.nan)
        for i, tid in enumerate(targets["target_id"])
    )


def _direction(geom: BaseGeometry, t: float) -> np.ndarray:
    lo, hi = max(t - SIDE_PROBE_M, 0.0), min(t + SIDE_PROBE_M, geom.length)
    a, b = shapely.line_interpolate_point(geom, [lo, hi])
    return np.array([b.x - a.x, b.y - a.y])


def _hits(g: WalkGraph, pt: np.ndarray, edges: np.ndarray, dists: np.ndarray) -> list[_Hit]:
    geoms = np.asarray(g.edge_geom, dtype=object)[edges]
    ts = shapely.line_locate_point(geoms, shapely.points(np.broadcast_to(pt, (len(edges), 2))))
    proj = shapely.get_coordinates(shapely.line_interpolate_point(geoms, ts))
    order = np.lexsort((ts, edges, dists))
    ref = _direction(geoms[order[0]], float(ts[order[0]]))
    ref = ref / max(float(np.hypot(*ref)), SIDE_EPS_M)
    rel = pt - proj
    cross = ref[0] * rel[:, 1] - ref[1] * rel[:, 0]
    sides = np.where(np.abs(cross) < SIDE_EPS_M, 0, np.sign(cross)).astype(int)
    return [_Hit(int(edges[k]), float(ts[k]), float(dists[k]), int(sides[k])) for k in order]


def _node_key(g: WalkGraph, hit: _Hit, snap: float) -> tuple[str, int, float]:
    length = float(g.edge_len_m[hit.edge])
    u, v = (int(n) for n in g.edge_uv[hit.edge])
    if hit.t <= snap:
        return ("node", u, 0.0)
    if length - hit.t <= snap:
        return ("node", v, 0.0)
    return ("split", hit.edge, hit.t)


def _pick(g: WalkGraph, hits: Sequence[_Hit], params: CandidateParams) -> list[tuple[str, int, float]]:
    """Nearest distinct candidate points, at most `max_per_side` on every side."""
    chosen: list[tuple[str, int, float]] = []
    per_side: dict[int, int] = {}
    for hit in hits:
        key = _node_key(g, hit, params.node_snap_m)
        dup = any(k[0] == key[0] and k[1] == key[1] and abs(k[2] - key[2]) <= params.node_snap_m for k in chosen)
        if dup or per_side.get(hit.side, 0) >= params.max_per_side:
            continue
        per_side[hit.side] = per_side.get(hit.side, 0) + 1
        chosen.append(key)
    return chosen


def _spur(g: WalkGraph, pt: np.ndarray, hit: _Hit, inner: BaseGeometry | None,
          params: CandidateParams) -> SpurDraft | None:
    if inner is None:
        return None
    proj = shapely.get_coordinates(shapely.line_interpolate_point(g.edge_geom[hit.edge], hit.t))[0]
    away = (proj - pt) / hit.dist
    end = pt + away * params.spur_standoff_m
    if not shapely.covered_by(shapely.linestrings([proj, end]), inner):
        return None
    return SpurDraft(attach=_node_key(g, hit, params.node_snap_m), end_xy=(float(end[0]), float(end[1])))


@dataclass(frozen=True)
class _Resolved:
    keys: tuple[tuple[str, int, float], ...] = ()
    spur: SpurDraft | None = None
    note: str = ""
    snap_dist_m: float = math.nan


def _resolve(g: WalkGraph, t: TargetPoint, near: tuple[np.ndarray, np.ndarray], snap_dist: float,
             inner: BaseGeometry | None, params: CandidateParams) -> _Resolved:
    if not t.reachable:
        return _Resolved(note=t.note, snap_dist_m=snap_dist)
    edges, dists = near
    in_comp = g.component[g.edge_uv[edges, 0]] == g.start_component
    if not np.any(in_comp):
        note = NOTE_DISCONNECTED if len(edges) else NOTE_TOO_FAR
        return _Resolved(note=note, snap_dist_m=snap_dist)
    pt = np.array([t.x, t.y])
    edges, dists = edges[in_comp], dists[in_comp]
    close = dists <= params.radius_m
    if np.any(close):
        keys = _pick(g, _hits(g, pt, edges[close], dists[close]), params)
        return _Resolved(keys=tuple(keys), snap_dist_m=snap_dist)
    spur = _spur(g, pt, _hits(g, pt, edges, dists)[0], inner, params)
    return _Resolved(spur=spur, note="" if spur else NOTE_TOO_FAR, snap_dist_m=snap_dist)


def _near_edges(g: WalkGraph, targets: Sequence[TargetPoint], max_m: float) -> tuple[list, np.ndarray]:
    pts = shapely.points(np.array([[t.x, t.y] for t in targets], dtype=np.float64).reshape(-1, 2))
    geoms = np.asarray(g.edge_geom, dtype=object)
    tree = shapely.STRtree(geoms)
    ti, ei = tree.query(pts, predicate="dwithin", distance=max_m)
    dist = shapely.distance(geoms[ei], pts[ti]) if len(ti) else np.zeros(0)
    near = [(ei[ti == k], dist[ti == k]) for k in range(len(targets))]
    _, nearest_d = tree.query_nearest(pts, return_distance=True, all_matches=False)
    return near, np.asarray(nearest_d, dtype=np.float64)


def _sets(target_nodes: Mapping[str, tuple[int, ...]], roles: Mapping[str, str]) -> tuple[CandidateSet, ...]:
    groups: dict[tuple[int, ...], list[str]] = {}
    for tid in sorted(target_nodes):
        nodes = tuple(sorted(set(target_nodes[tid])))
        if nodes:
            groups.setdefault(nodes, []).append(tid)
    out = [CandidateSet(target_ids=tuple(tids), nodes=nodes,
                        role=ROLE_MUST if any(roles[t] == ROLE_MUST for t in tids) else ROLE_OPTIONAL)
           for nodes, tids in groups.items()]
    return tuple(sorted(out, key=lambda s: s.target_ids[0]))


def _reach(t: TargetPoint, res: _Resolved, nodes: tuple[int, ...]) -> Reach:
    ok = bool(nodes)
    note = "" if ok else (res.note or NOTE_TOO_FAR)
    return Reach(target_id=t.target_id, reachable=ok, note=note, n_candidates=len(nodes),
                 snap_dist_m=float(res.snap_dist_m))


def augment_with_targets(g: WalkGraph, targets: Sequence[TargetPoint], inner: BaseGeometry | None,
                         params: CandidateParams) -> Augmented:
    """Graph with candidate nodes split in, the GTSP sets and the final reachability of every target."""
    ids = [t.target_id for t in targets]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate target_id in route targets")
    if not targets or g.n_edges == 0:
        reach = tuple(Reach(t.target_id, False, t.note or NOTE_TOO_FAR, 0, math.nan) for t in targets)
        return Augmented(g, (), reach, MappingProxyType({}))
    near, nearest = _near_edges(g, targets, params.max_snap_m)
    resolved = [_resolve(g, t, near[k], float(nearest[k]), inner, params) for k, t in enumerate(targets)]
    plan = SplitPlan(keys={t.target_id: r.keys for t, r in zip(targets, resolved, strict=True)},
                     spurs={t.target_id: r.spur for t, r in zip(targets, resolved, strict=True) if r.spur},
                     xy={t.target_id: (t.x, t.y) for t in targets})
    graph, target_nodes = apply_splits(g, plan, merge_m=max(params.merge_m, params.node_snap_m),
                                       keep_within_m=params.radius_m + params.node_snap_m)
    roles = {t.target_id: t.role for t in targets}
    reach = tuple(_reach(t, r, target_nodes.get(t.target_id, ())) for t, r in zip(targets, resolved, strict=True))
    return Augmented(graph, _sets(target_nodes, roles), reach, MappingProxyType(dict(target_nodes)))


__all__ = [
    "CANDIDATE_NODE_SNAP_M", "MUST_MAX_PRIORITY", "NOTE_DISCONNECTED", "NOTE_OUTSIDE_ONLY", "NOTE_TOO_FAR",
    "ROLE_MUST", "ROLE_OPTIONAL", "Augmented", "CandidateParams", "TargetPoint", "augment_with_targets",
    "target_points",
]
