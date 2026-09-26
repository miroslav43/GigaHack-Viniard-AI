"""Route stops (contract §2.5.13 `route_stops`) and per-target visits (`target_visits`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString

from vineyard.route.candidates import TargetPoint
from vineyard.route.graph_types import Reach

NOTE_TRUNCATED: Final = "truncated"
# Optional target left out because route.solver.include_optional is off.
NOTE_NOT_ROUTED: Final = "not_routed"
# Optional target whose cheapest insertion into the must tour costs more than route.solver.optional_max_detour_m.
NOTE_OPTIONAL_DETOUR: Final = "optional_detour"


@dataclass(frozen=True)
class Stop:
    seq: int
    target_id: str | None
    x: float
    y: float
    cum_dist_m: float
    leg_m: float


@dataclass(frozen=True)
class TargetVisit:
    target_id: str
    x: float
    y: float
    reachable_final: bool
    reach_note: str
    n_candidates: int
    covered: bool
    visit_dist_m: float
    route_role: str


def cumulative_lengths(xy: np.ndarray) -> np.ndarray:
    """Distance along the route at every vertex."""
    pts = np.asarray(xy, dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])


def build_stops(xy: np.ndarray, anchors: Sequence[int], stop_targets: Sequence[Sequence[str]]) -> tuple[Stop, ...]:
    """START (seq 0), one row per visited target in tour order, then the return to START."""
    if len(anchors) != len(stop_targets) + 2:
        raise ValueError(f"{len(anchors)} anchors for {len(stop_targets)} stops (expected stops + 2)")
    cum = cumulative_lengths(xy)
    rows = [Stop(0, None, float(xy[0, 0]), float(xy[0, 1]), 0.0, 0.0)]
    prev = 0.0
    for anchor, tids in zip(anchors[1:-1], stop_targets, strict=True):
        here = float(cum[anchor])
        for k, tid in enumerate(tids):
            leg = here - prev if k == 0 else 0.0
            rows.append(Stop(len(rows), tid, float(xy[anchor, 0]), float(xy[anchor, 1]), here, leg))
        prev = here
    end = int(anchors[-1])
    rows.append(Stop(len(rows), None, float(xy[end, 0]), float(xy[end, 1]), float(cum[end]), float(cum[end]) - prev))
    return tuple(rows)


def build_visits(line: LineString, targets: Sequence[TargetPoint], reach: Mapping[str, Reach],
                 radius_m: float) -> tuple[TargetVisit, ...]:
    """Coverage of every target (reachable or not) by the final line."""
    if not targets:
        return ()
    pts = shapely.points(np.array([[t.x, t.y] for t in targets], dtype=np.float64))
    dist = shapely.distance(line, pts)
    out = []
    for t, d in zip(targets, dist, strict=True):
        r = reach.get(t.target_id)
        out.append(TargetVisit(
            target_id=t.target_id, x=t.x, y=t.y, reachable_final=bool(r.reachable) if r else False,
            reach_note=r.note if r else NOTE_TRUNCATED, n_candidates=r.n_candidates if r else 0,
            covered=bool(d <= radius_m), visit_dist_m=float(d), route_role=t.role))
    return tuple(out)


__all__ = ["NOTE_NOT_ROUTED", "NOTE_OPTIONAL_DETOUR", "NOTE_TRUNCATED", "Stop", "TargetVisit", "build_stops",
           "build_visits", "cumulative_lengths"]
