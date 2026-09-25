"""Connectors between walkable polylines (arch §4.11.2, design 04 §3.7, critic S5 headland risk).

* row end -> cheapest point on another line within `max_len_m` (cost = length + penalty * outside);
* consecutive pieces of one interrow (split by a canopy intruding into the corridor);
* passage skeleton loose ends -> another line within `snap_join_m`, only when fully inside;
* START -> nearest line point with the segment inside `domain.inner` (else the stage must stop).

Interrows stop at the shorter row end; the headland beyond is neither interrow nor passage, so a
row-end connector can cross outside the domain. `HeadlandStrategy` decides which of those survive.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import EdgeKind
from vineyard.route.cells import CellIndex
from vineyard.route.graph import NODE_MERGE_M, Attachment, outside_lengths
from vineyard.route.graph_types import PolylineDraft

if TYPE_CHECKING:
    from vineyard.config.sections_post import RouteGraphConfig

ConnectorKind = Literal["row_end", "piece_join", "snap", "start"]
LooseEnd = tuple[str, float, float]   # (end ref, x, y)


class HeadlandStrategy(StrEnum):
    PENALTY = "penalty"          # keep every cheapest row-end connector; outside metres cost `penalty` each
    LIMIT = "limit"              # drop connectors with more than `outside_max_m` outside the domain
    INSIDE_ONLY = "inside_only"  # enter interrows only where they touch the domain; dead ends out and back


# CONFIG-REQUEST: route.graph.headland_strategy = "penalty"
DEFAULT_STRATEGY: Final = HeadlandStrategy.PENALTY
# CONFIG-REQUEST: route.graph.connector_outside_max_m = 1.0
DEFAULT_OUTSIDE_MAX_M: Final = 1.0
# Outside length at or below the 1 mm domain grid is overlay noise and counts as inside
# (same tolerance as route.policies.INSIDE_TOL_M, so its `inside_only` policy matches INSIDE_ONLY here).
INSIDE_TOL_M: Final = 1e-3
START_REF: Final = "START"


@dataclass(frozen=True)
class ConnectorParams:
    max_len_m: float
    outside_penalty: float
    snap_join_m: float
    strategy: HeadlandStrategy = DEFAULT_STRATEGY
    outside_max_m: float = DEFAULT_OUTSIDE_MAX_M

    def __post_init__(self) -> None:
        if self.max_len_m <= 0.0:
            raise ValueError(f"max_len_m must be > 0, got {self.max_len_m}")
        if min(self.outside_penalty, self.snap_join_m, self.outside_max_m) < 0.0:
            raise ValueError("outside_penalty, snap_join_m and outside_max_m must be >= 0")
        object.__setattr__(self, "strategy", HeadlandStrategy(self.strategy))

    @classmethod
    def from_config(cls, cfg: RouteGraphConfig, strategy: HeadlandStrategy | str = DEFAULT_STRATEGY,
                    outside_max_m: float = DEFAULT_OUTSIDE_MAX_M) -> ConnectorParams:
        return cls(max_len_m=cfg.connector_max_m, outside_penalty=cfg.connector_outside_penalty,
                   snap_join_m=cfg.snap_join_m, strategy=HeadlandStrategy(strategy), outside_max_m=outside_max_m)

    @property
    def allowed_outside_m(self) -> float:
        if self.strategy == HeadlandStrategy.PENALTY:
            return math.inf
        if self.strategy == HeadlandStrategy.LIMIT:
            return max(self.outside_max_m, INSIDE_TOL_M)
        return INSIDE_TOL_M


@dataclass(frozen=True)
class Connector:
    kind: ConnectorKind
    ref: str
    from_xy: tuple[float, float]
    to_xy: tuple[float, float]
    target: int          # draft index the connector ends on
    along_m: float       # position on the target draft
    length_m: float
    outside_m: float
    kept: bool


@dataclass(frozen=True, eq=False)
class ConnectorPlan:
    connectors: tuple[Connector, ...]
    unconnected_ends: tuple[LooseEnd, ...]   # row ends with no line within max_len_m

    @property
    def kept(self) -> tuple[Connector, ...]:
        return tuple(c for c in self.connectors if c.kept)

    @property
    def dropped(self) -> tuple[Connector, ...]:
        return tuple(c for c in self.connectors if not c.kept)

    def drafts(self) -> tuple[PolylineDraft, ...]:
        """Connector polylines to append after the line drafts (zero-length joins need none)."""
        return tuple(PolylineDraft(LineString([c.from_xy, c.to_xy]), EdgeKind.CONNECTOR, c.ref, c.outside_m)
                     for c in self.kept if c.length_m >= NODE_MERGE_M)

    def attachments(self) -> tuple[Attachment, ...]:
        return tuple(Attachment(c.target, c.along_m, c.ref) for c in self.kept)

    def restricted(self, max_outside_m: float) -> ConnectorPlan:
        """Same plan with connectors of more than `max_outside_m` outside no longer kept."""
        limit = max(max_outside_m, INSIDE_TOL_M)
        return ConnectorPlan(tuple(replace(c, kept=c.kept and c.outside_m <= limit) for c in self.connectors),
                             self.unconnected_ends)


@dataclass(frozen=True, eq=False)
class _Ends:
    xy: np.ndarray            # (m, 2)
    draft: np.ndarray         # (m,) draft index the end belongs to
    refs: tuple[str, ...]     # connector ref per end


def _end_ref(ref: str, is_start: bool) -> str:
    return f"{ref}:{'start' if is_start else 'end'}"


def outer_row_ends(lines: Sequence[PolylineDraft], joins: Sequence[tuple[int, int]]) -> _Ends:
    """Ends of interrow drafts that no piece join uses (end of i / start of j for each join (i, j))."""
    inner_ends = {(i, False) for i, _ in joins} | {(j, True) for _, j in joins}
    xy, draft, refs = [], [], []
    for i, d in enumerate(lines):
        if d.kind != EdgeKind.INTERROW_CENTERLINE:
            continue
        for is_start, coord in ((True, d.geom.coords[0]), (False, d.geom.coords[-1])):
            if (i, is_start) not in inner_ends:
                xy.append(coord[:2])
                draft.append(i)
                refs.append(_end_ref(d.ref, is_start))
    return _Ends(np.asarray(xy, float).reshape(-1, 2), np.asarray(draft, np.int64), tuple(refs))


@dataclass(frozen=True, eq=False)
class _Candidates:
    end: np.ndarray
    line: np.ndarray
    along: np.ndarray
    to_xy: np.ndarray
    length: np.ndarray
    outside: np.ndarray


def _candidates(ends: _Ends, lines: Sequence[PolylineDraft], radius: float, inner: BaseGeometry,
                index: CellIndex) -> _Candidates:
    geoms = np.asarray([d.geom for d in lines], dtype=object)
    pts = shapely.points(ends.xy) if len(ends.xy) else np.zeros(0, dtype=object)
    if len(pts) == 0 or len(geoms) == 0:
        empty = np.zeros(0)
        return _Candidates(empty.astype(np.int64), empty.astype(np.int64), empty, empty.reshape(0, 2), empty, empty)
    ei, li = shapely.STRtree(geoms).query(pts, predicate="dwithin", distance=radius)
    refs = np.asarray([d.ref for d in lines], dtype=object)
    # never connect a row end to its own interrow: consecutive pieces are joined by `piece_joins`
    own = ends.draft[ei]
    same = (li == own) | ((own >= 0) & (refs[li] == refs[np.maximum(own, 0)]))
    ei, li = ei[~same], li[~same]
    along = shapely.line_locate_point(geoms[li], pts[ei])
    to_xy = shapely.get_coordinates(shapely.line_interpolate_point(geoms[li], along))
    length = np.hypot(*(to_xy - ends.xy[ei]).T) if len(ei) else np.zeros(0)
    segs = shapely.linestrings(np.stack([ends.xy[ei], to_xy], axis=1)) if len(ei) else np.zeros(0, dtype=object)
    outside = outside_lengths(segs, inner, index) if len(ei) else np.zeros(0)
    return _Candidates(ei, li, along, to_xy.reshape(-1, 2), length, outside)


def _effective_outside(c: _Candidates) -> np.ndarray:
    return np.where(c.outside > INSIDE_TOL_M, c.outside, 0.0)


def _cheapest(c: _Candidates, penalty: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Index into `c` of the cheapest candidate per end among `mask` (ties: lower line index)."""
    idx = np.arange(len(c.end)) if mask is None else np.flatnonzero(mask)
    if len(idx) == 0:
        return np.zeros(0, dtype=np.int64)
    cost = np.round(c.length[idx] + penalty * _effective_outside(c)[idx], 6)
    order = idx[np.lexsort((c.line[idx], cost, c.end[idx]))]
    first = np.concatenate([[True], c.end[order][1:] != c.end[order][:-1]])
    return order[first]


def _connector(kind: ConnectorKind, ref: str, from_xy: np.ndarray, c: _Candidates, k: int,
               allowed: float) -> Connector:
    outside = float(_effective_outside(c)[k])
    return Connector(kind, ref, (float(from_xy[0]), float(from_xy[1])), (float(c.to_xy[k, 0]), float(c.to_xy[k, 1])),
                     int(c.line[k]), float(c.along[k]), float(c.length[k]), outside, outside <= allowed)


def _tiers(params: ConnectorParams) -> list[float]:
    """Outside-length tiers up to the strategy's limit: inside, `outside_max_m`, unlimited."""
    tiers = sorted({INSIDE_TOL_M, max(params.outside_max_m, INSIDE_TOL_M), math.inf})
    return [t for t in tiers if t <= params.allowed_outside_m]


def _row_end_choice(c: _Candidates, lines: Sequence[PolylineDraft], params: ConnectorParams) -> np.ndarray:
    """Per end and per outside tier: the cheapest connector overall and the cheapest one to a passage line.

    The cheapest overall is usually the headland turn into the neighbouring interrow; without the
    passage one, a block whose ends all stop short of the passages would form an island. Taking the
    best of every stricter tier too makes the PENALTY graph a superset of the LIMIT and INSIDE_ONLY
    graphs, so `graph.restrict_outside` switches strategy without rebuilding. Ends with no allowed
    candidate also get their (rejected, kept=False) cheapest ones, for the report.
    """
    is_passage = np.asarray([lines[i].kind == EdgeKind.PASSAGE_CENTERLINE for i in c.line], dtype=bool)
    out = _effective_outside(c)
    pen = params.outside_penalty
    picks = [p for t in _tiers(params) for p in (_cheapest(c, pen, out <= t), _cheapest(c, pen, (out <= t) & is_passage))]
    allowed = np.unique(np.concatenate(picks)).astype(np.int64) if picks else np.zeros(0, np.int64)
    has_ok = set(c.end[allowed].tolist())
    lacking = np.asarray([int(e) not in has_ok for e in c.end], dtype=bool)
    rejected = np.concatenate([_cheapest(c, pen, lacking), _cheapest(c, pen, lacking & is_passage)])
    return np.unique(np.concatenate([allowed, rejected])).astype(np.int64)


def row_end_connectors(lines: Sequence[PolylineDraft], joins: Sequence[tuple[int, int]], inner: BaseGeometry,
                       index: CellIndex, params: ConnectorParams) -> tuple[tuple[Connector, ...], tuple[LooseEnd, ...]]:
    """Connectors of every outer row end (see `_row_end_choice`) and the ends with no line in reach."""
    ends = outer_row_ends(lines, joins)
    cand = _candidates(ends, lines, params.max_len_m, inner, index)
    chosen = sorted(_row_end_choice(cand, lines, params).tolist(), key=lambda k: (int(cand.end[k]), k))
    out = tuple(_connector("row_end", ends.refs[int(cand.end[k])], ends.xy[int(cand.end[k])], cand, int(k),
                           params.allowed_outside_m) for k in chosen)
    served = set(cand.end.tolist())
    lonely = tuple((ends.refs[i], float(ends.xy[i, 0]), float(ends.xy[i, 1]))
                   for i in range(len(ends.xy)) if i not in served)
    return out, lonely


def piece_joins(lines: Sequence[PolylineDraft], joins: Sequence[tuple[int, int]], inner: BaseGeometry,
                index: CellIndex, params: ConnectorParams) -> tuple[Connector, ...]:
    """End of draft i -> start of draft j for each (i, j): the gap a canopy leaves in one interrow."""
    if not joins:
        return ()
    a = np.asarray([lines[i].geom.coords[-1][:2] for i, _ in joins], float)
    b = np.asarray([lines[j].geom.coords[0][:2] for _, j in joins], float)
    segs = shapely.linestrings(np.stack([a, b], axis=1))
    outside = outside_lengths(segs, inner, index)
    lengths = np.hypot(*(b - a).T)
    cand = _Candidates(np.arange(len(joins)), np.asarray([j for _, j in joins], np.int64), np.zeros(len(joins)),
                       b, lengths, outside)
    return tuple(_connector("piece_join", f"{lines[i].ref}:join{k + 1}", a[k], cand, k, params.allowed_outside_m)
                 for k, (i, _) in enumerate(joins))


def _loose_passage_ends(lines: Sequence[PolylineDraft]) -> _Ends:
    xy, draft, refs = [], [], []
    for i, d in enumerate(lines):
        for is_start, coord in ((True, d.geom.coords[0]), (False, d.geom.coords[-1])):
            xy.append(coord[:2])
            draft.append(i)
            refs.append(_end_ref(d.ref, is_start))
    pts = np.asarray(xy, float).reshape(-1, 2)
    if len(pts) == 0:
        return _Ends(pts, np.zeros(0, np.int64), ())
    counts = np.asarray([len(n) for n in cKDTree(pts).query_ball_point(pts, NODE_MERGE_M)])
    loose = [k for k in range(len(pts)) if counts[k] == 1 and lines[draft[k]].kind == EdgeKind.PASSAGE_CENTERLINE]
    return _Ends(pts[loose], np.asarray([draft[k] for k in loose], np.int64), tuple(refs[k] for k in loose))


def snap_joins(lines: Sequence[PolylineDraft], inner: BaseGeometry, index: CellIndex,
               params: ConnectorParams) -> tuple[Connector, ...]:
    """Loose passage-skeleton ends -> nearest other line within `snap_join_m`, fully inside only."""
    if params.snap_join_m <= 0.0:
        return ()
    ends = _loose_passage_ends(lines)
    cand = _candidates(ends, lines, params.snap_join_m, inner, index)
    best = [int(k) for k in _cheapest(cand, params.outside_penalty) if cand.outside[k] <= INSIDE_TOL_M]
    return tuple(_connector("snap", f"{ends.refs[int(cand.end[k])]}:snap", ends.xy[int(cand.end[k])], cand, k,
                            INSIDE_TOL_M) for k in best)


def start_connector(lines: Sequence[PolylineDraft], start_xy: tuple[float, float], inner: BaseGeometry,
                    index: CellIndex, params: ConnectorParams) -> Connector:
    """Nearest line point whose segment from START is inside `inner`; raises ValueError if there is none."""
    if not shapely.covered_by(Point(start_xy), inner):
        raise ValueError(f"START {start_xy} is not inside domain.inner")
    ends = _Ends(np.asarray([start_xy], float), np.asarray([-1], np.int64), (START_REF,))
    cand = _candidates(ends, lines, params.max_len_m, inner, index)
    inside = np.flatnonzero(cand.outside <= INSIDE_TOL_M)
    if len(inside) == 0:
        raise ValueError(f"START {start_xy}: no walk-graph line within {params.max_len_m} m reachable "
                         f"by a segment inside domain.inner ({len(cand.end)} candidates, all crossing outside)")
    k = int(inside[np.lexsort((cand.line[inside], np.round(cand.length[inside], 6)))[0]])
    return _connector("start", START_REF, ends.xy[0], cand, k, INSIDE_TOL_M)


def plan_connectors(
    lines: Sequence[PolylineDraft],
    joins: Sequence[tuple[int, int]],
    start_xy: tuple[float, float],
    inner: BaseGeometry,
    index: CellIndex,
    params: ConnectorParams,
) -> ConnectorPlan:
    """Every connector of the walk graph; `kept` follows `params.strategy` (START and snaps always inside)."""
    ends, lonely = row_end_connectors(lines, joins, inner, index, params)
    joined = piece_joins(lines, joins, inner, index, params)
    snaps = snap_joins(lines, inner, index, params)
    start = start_connector(lines, start_xy, inner, index, params)
    return ConnectorPlan((start, *joined, *ends, *snaps), lonely)


__all__ = [
    "DEFAULT_OUTSIDE_MAX_M", "DEFAULT_STRATEGY", "INSIDE_TOL_M", "Connector", "ConnectorParams", "ConnectorPlan",
    "HeadlandStrategy", "outer_row_ends", "piece_joins", "plan_connectors", "row_end_connectors", "snap_joins",
    "start_connector",
]
