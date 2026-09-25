"""Outside-domain policies for the headland risk (critic S5) and the headland report.

Interrow centerlines stop at the shorter row end and headlands are neither interrow nor passage, so a
connector into an interrow may cross outside the domain. The planner tries these policies in order and
keeps the first route whose outside share passes publish (<= route.max_outside_frac_publish):

- `penalty`:       graph as built (cost = length + penalty * outside length);
- `penalty_x10`:   the outside penalty times 10 (design 04 §3.9 re-solve);
- `drop_optional`: x10, then optional stops carrying the most outside length are dropped (re-solved)
                   until the share fits (`reachable=false`, `outside_budget`);
- `must_only`:     x10 without the optional phase (optional targets are reported, not routed);
- `drop_outside`:  must only, then must stops are dropped like in `drop_optional`;
- `cap_outside`:   drops every edge with more than `CAP_OUTSIDE_M` outside;
- `inside_only`:   drops every edge that leaves the domain, so interrows are entered only from ends that
                   touch a passage and dead ends are walked out and back.
Targets whose candidates all disappear with the removed edges become `reachable=false` (`outside_only`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import EdgeKind
from vineyard.route.graph_surgery import restrict_edges
from vineyard.route.graph_types import CandidateSet, WalkGraph

# CONFIG-REQUEST: route.graph.outside_cap_m = 1.0
CAP_OUTSIDE_M: Final = 1.0
# Outside lengths at or below the domain grid size are overlay noise, not a real excursion.
INSIDE_TOL_M: Final = 1e-3
# CONFIG-REQUEST: route.graph.strict_penalty_mult = 10.0
STRICT_PENALTY_MULT: Final = 10.0
# One key for both headland reports (this one and walk_report's): route.graph.near_passage_m.
# CONFIG-REQUEST: route.graph.near_passage_m = 0.5
HEADLAND_NEAR_M: Final = 0.5


DropMode = Literal["none", "optional", "any"]


@dataclass(frozen=True)
class OutsidePolicy:
    name: str
    penalty_mult: float
    max_edge_outside_m: float | None
    drop: DropMode = "none"
    optional: bool = True


# CONFIG-REQUEST: route.graph.outside_policies = [penalty, penalty_x10, drop_optional, must_only, drop_outside,
#                 cap_outside, inside_only]
DEFAULT_POLICIES: Final[tuple[OutsidePolicy, ...]] = (
    OutsidePolicy("penalty", 1.0, None),
    OutsidePolicy("penalty_x10", STRICT_PENALTY_MULT, None),
    OutsidePolicy("drop_optional", STRICT_PENALTY_MULT, None, "optional"),
    OutsidePolicy("must_only", STRICT_PENALTY_MULT, None, "none", optional=False),
    OutsidePolicy("drop_outside", STRICT_PENALTY_MULT, None, "any", optional=False),
    OutsidePolicy("cap_outside", STRICT_PENALTY_MULT, CAP_OUTSIDE_M),
    OutsidePolicy("inside_only", STRICT_PENALTY_MULT, INSIDE_TOL_M),
)


def apply_policy(g: WalkGraph, policy: OutsidePolicy, base_penalty: float) -> WalkGraph:
    """Graph with the policy's outside penalty and edge cap; nodes (and their indices) are unchanged."""
    if policy.penalty_mult == 1.0 and policy.max_edge_outside_m is None:
        return g
    outside = g.outside_len_m()
    cost = g.edge_cost + (policy.penalty_mult - 1.0) * base_penalty * outside
    keep = np.ones(g.n_edges, dtype=bool) if policy.max_edge_outside_m is None \
        else outside <= policy.max_edge_outside_m
    return restrict_edges(g, keep, cost)


def restrict_sets(g: WalkGraph, sets: Sequence[CandidateSet]) -> tuple[tuple[CandidateSet, ...], tuple[str, ...]]:
    """Sets reduced to nodes in START's component; target ids whose set became empty."""
    kept, lost = [], []
    for cset in sets:
        nodes = tuple(n for n in cset.nodes if g.component[n] == g.start_component)
        if nodes:
            kept.append(CandidateSet(cset.target_ids, nodes, cset.role))
        else:
            lost.extend(cset.target_ids)
    return tuple(kept), tuple(sorted(lost))


def interrow_ends(g: WalkGraph) -> np.ndarray:
    """Nodes with exactly one incident interrow-centerline edge (the ends of every centerline chain)."""
    mask = np.array([k == EdgeKind.INTERROW_CENTERLINE for k in g.edge_kind], dtype=bool)
    counts = np.bincount(g.edge_uv[mask].ravel(), minlength=g.n_nodes) if np.any(mask) \
        else np.zeros(g.n_nodes, dtype=np.int64)
    return np.flatnonzero(counts == 1)


def headland_report(g: WalkGraph, passages: BaseGeometry | None, near_m: float = HEADLAND_NEAR_M) -> dict[str, Any]:
    """Share of interrow ends within `near_m` of a passage, and the graph's outside connectors."""
    ends = interrow_ends(g)
    near = np.zeros(len(ends), dtype=bool)
    if passages is not None and not passages.is_empty and len(ends):
        near = shapely.distance(shapely.points(g.node_xy[ends]), passages) <= near_m
    outside = g.outside_len_m()
    connectors = np.array([k == EdgeKind.CONNECTOR for k in g.edge_kind], dtype=bool)
    leaving = connectors & (outside > INSIDE_TOL_M)
    return {
        "n_interrow_ends": int(len(ends)),
        "n_interrow_ends_near_passage": int(near.sum()),
        "share_interrow_ends_near_passage": float(near.mean()) if len(ends) else None,
        "near_m": near_m,
        "n_connectors": int(connectors.sum()),
        "n_connectors_outside": int(leaving.sum()),
        "connector_outside_len_m": round(float(outside[connectors].sum()), 3),
        "connector_outside_max_m": round(float(outside[leaving].max()), 3) if np.any(leaving) else 0.0,
    }


__all__ = [
    "CAP_OUTSIDE_M", "DEFAULT_POLICIES", "HEADLAND_NEAR_M", "INSIDE_TOL_M", "STRICT_PENALTY_MULT", "DropMode",
    "OutsidePolicy", "apply_policy", "headland_report", "interrow_ends", "restrict_sets",
]
