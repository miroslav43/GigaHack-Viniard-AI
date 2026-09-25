"""Naive baselines for the pitch (arch §4.11.8, design 04 §3.9).

- `id_order_m`: START -> targets in target_id order (first candidate of each set) -> START, on the graph;
- `all_interrows_m`: every interrow centerline walked once (a lower bound of any all-interrow walk);
- `serpentine_est_m`: interrows in interrow_id order, each walked end to end, joined by straight hops
  between the nearest chain ends (hops are Euclidean, so this estimate flatters the baseline).
Savings are reported in km, % and minutes at `walking_speed_kmh`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from vineyard.contracts.enums import EdgeKind
from vineyard.route.distances import distance_matrix
from vineyard.route.graph_types import CandidateSet, WalkGraph
from vineyard.route.policies import interrow_ends

M_PER_KM: Final = 1000.0
MIN_PER_H: Final = 60.0
PERCENT: Final = 100.0


@dataclass(frozen=True)
class Baseline:
    route_m: float
    id_order_m: float | None
    all_interrows_m: float
    serpentine_est_m: float | None
    walking_speed_kmh: float

    def minutes(self, length_m: float) -> float:
        return length_m / M_PER_KM / self.walking_speed_kmh * MIN_PER_H

    def _saving(self, other: float | None) -> dict[str, float] | None:
        if other is None or other <= 0.0:
            return None
        saved = other - self.route_m
        return {"baseline_m": round(other, 2), "saved_km": round(saved / M_PER_KM, 3),
                "saved_pct": round(saved / other * PERCENT, 2), "saved_min": round(self.minutes(saved), 1)}

    def to_json(self) -> dict[str, Any]:
        return {
            "route_m": round(self.route_m, 2), "route_min": round(self.minutes(self.route_m), 1),
            "walking_speed_kmh": self.walking_speed_kmh,
            "id_order": self._saving(self.id_order_m), "all_interrows": self._saving(self.all_interrows_m),
            "serpentine_est": self._saving(self.serpentine_est_m),
        }


def id_order_length(g: WalkGraph, sets: Sequence[CandidateSet], chunk: int) -> float | None:
    """Closed walk through the first candidate of every set, sets in target_id order (length weights)."""
    if not sets:
        return None
    ordered = sorted(sets, key=lambda s: s.target_ids[0])
    nodes = [g.start_node, *(int(s.nodes[0]) for s in ordered)]
    d = distance_matrix(g.adjacency("length"), nodes, chunk)
    idx = np.arange(len(nodes))
    total = float(d[idx, np.roll(idx, -1)].sum())
    return total if np.isfinite(total) else None


def interrow_length(g: WalkGraph) -> float:
    mask = np.array([k == EdgeKind.INTERROW_CENTERLINE for k in g.edge_kind], dtype=bool)
    return float(g.edge_len_m[mask].sum()) if len(mask) else 0.0


def _chains(g: WalkGraph) -> list[tuple[str, float, np.ndarray]]:
    """(interrow ref, walked length, the two farthest-apart chain ends) per interrow ref."""
    ends = set(interrow_ends(g).tolist())
    groups: dict[str, list[int]] = {}
    for e, (ref, kind) in enumerate(zip(g.edge_ref, g.edge_kind, strict=True)):
        if kind == EdgeKind.INTERROW_CENTERLINE:
            groups.setdefault(ref, []).append(e)
    out = []
    for ref in sorted(groups):
        idx = groups[ref]
        nodes = sorted({int(n) for e in idx for n in g.edge_uv[e]} & ends)
        if len(nodes) < 2:
            continue
        pts = g.node_xy[nodes]
        gap = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
        i, j = np.unravel_index(int(np.argmax(gap)), gap.shape)
        out.append((ref, float(g.edge_len_m[idx].sum()), pts[[i, j]]))
    return out


def serpentine_estimate(g: WalkGraph, start_xy: tuple[float, float]) -> float | None:
    chains = _chains(g)
    if not chains:
        return None
    here = np.asarray(start_xy, dtype=np.float64)
    total = 0.0
    for _, length, ends in chains:
        near = int(np.argmin(np.hypot(*(ends - here).T)))
        total += float(np.hypot(*(ends[near] - here))) + length
        here = ends[1 - near]
    return total + float(np.hypot(*(here - np.asarray(start_xy, dtype=np.float64))))


def compute_baseline(g: WalkGraph, sets: Sequence[CandidateSet], route_m: float, start_xy: tuple[float, float],
                     *, chunk: int, walking_speed_kmh: float) -> Baseline:
    return Baseline(route_m=route_m, id_order_m=id_order_length(g, sets, chunk), all_interrows_m=interrow_length(g),
                    serpentine_est_m=serpentine_estimate(g, start_xy), walking_speed_kmh=walking_speed_kmh)


__all__ = ["Baseline", "compute_baseline", "id_order_length", "interrow_length", "serpentine_estimate"]
