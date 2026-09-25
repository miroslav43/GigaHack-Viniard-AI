"""Headland-risk metrics of a walk graph and the strategy choice (critic S5, 00_PLAN "Headland strategy").

Interrow centerlines stop where the shorter row ends; if the headland beyond is not a passage, every
entry into the interrow crosses outside the domain. The proxy measured here:

* the share of outer interrow ends within `near_m` (0.5 m) of `in_passages`;
* the outside length of the connectors in START's component, relative to the interrow length that
  START can reach (`outside_share_proxy`, a route-independent stand-in for the route's outside_share).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import EdgeKind
from vineyard.route.connectors import HeadlandStrategy, outer_row_ends
from vineyard.route.policies import HEADLAND_NEAR_M
from vineyard.route.walk_build import LineSet, PassableParams, PassableResult, restrict_result

NEAR_PASSAGE_M: Final = HEADLAND_NEAR_M  # the route stage's headland report uses the same distance
STRATEGY_ORDER: Final = (HeadlandStrategy.PENALTY, HeadlandStrategy.LIMIT, HeadlandStrategy.INSIDE_ONLY)
_EPS: Final = 1e-12


@dataclass(frozen=True)
class HeadlandMetrics:
    strategy: str
    n_row_ends: int
    n_row_ends_near_passage: int
    near_passage_share: float
    n_unconnected_ends: int
    n_connectors_kept: int
    n_connectors_dropped: int
    n_connectors_outside: int
    connector_outside_total_m: float
    connector_outside_max_m: float
    n_components: int
    graph_length_m: float
    start_component_length_m: float
    n_interrows: int
    n_interrows_reachable: int
    centerline_length_m: float
    centerline_reachable_m: float
    start_outside_m: float
    outside_share_proxy: float
    vineyards_reachable: Mapping[str, bool]

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "vineyards_reachable": dict(sorted(self.vineyards_reachable.items()))}


def ends_near_passages(lines: LineSet, passages: BaseGeometry, near_m: float = NEAR_PASSAGE_M) -> tuple[int, int]:
    """(outer interrow ends within `near_m` of the passages, all outer interrow ends)."""
    ends = outer_row_ends(lines.drafts, lines.joins)
    if len(ends.xy) == 0 or passages.is_empty:
        return 0, len(ends.xy)
    shapely.prepare(passages)
    return int(shapely.dwithin(shapely.points(ends.xy), passages, near_m).sum()), len(ends.xy)


def _vineyards(lines: LineSet, reachable: set[str]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for c in lines.centerlines:
        if c.vineyard_id:
            out[c.vineyard_id] = out.get(c.vineyard_id, False) or c.interrow_id in reachable
    return out


def _connector_stats(result: PassableResult) -> tuple[int, int, int, float, float]:
    kept = [c for c in result.plan.kept if c.kind in ("row_end", "piece_join")]
    outside = np.asarray([c.outside_m for c in kept], dtype=np.float64)
    n_out = int((outside > 0.0).sum())
    return (len(kept), len(result.plan.dropped), n_out, float(outside.sum()) if len(outside) else 0.0,
            float(outside.max()) if len(outside) else 0.0)


def headland_metrics(result: PassableResult, passages: BaseGeometry, near_m: float = NEAR_PASSAGE_M) -> HeadlandMetrics:
    g = result.graph
    near, n_ends = ends_near_passages(result.lines, passages, near_m)
    in_start = g.component[g.edge_uv[:, 0]] == g.start_component if g.n_edges else np.zeros(0, bool)
    is_ir = np.asarray([k == EdgeKind.INTERROW_CENTERLINE for k in g.edge_kind], dtype=bool)
    reach_ids = {g.edge_ref[i] for i in np.flatnonzero(in_start & is_ir)}
    all_ids = {c.interrow_id for c in result.lines.centerlines}
    reach_len = float(g.edge_len_m[in_start & is_ir].sum())
    start_out = float(g.outside_len_m()[in_start].sum())
    n_kept, n_drop, n_out, out_total, out_max = _connector_stats(result)
    return HeadlandMetrics(
        strategy=str(result.strategy), n_row_ends=n_ends, n_row_ends_near_passage=near,
        near_passage_share=near / n_ends if n_ends else 0.0, n_unconnected_ends=len(result.plan.unconnected_ends),
        n_connectors_kept=n_kept, n_connectors_dropped=n_drop, n_connectors_outside=n_out,
        connector_outside_total_m=out_total, connector_outside_max_m=out_max,
        n_components=len(result.components), graph_length_m=float(g.edge_len_m.sum()),
        start_component_length_m=float(g.edge_len_m[in_start].sum()), n_interrows=len(all_ids),
        n_interrows_reachable=len(reach_ids & all_ids), centerline_length_m=float(g.edge_len_m[is_ir].sum()),
        centerline_reachable_m=reach_len, start_outside_m=start_out,
        outside_share_proxy=start_out / max(reach_len, _EPS) if reach_len > 0.0 else 0.0,
        vineyards_reachable=_vineyards(result.lines, reach_ids),
    )


def compare_strategies(base: PassableResult, params: PassableParams, passages: BaseGeometry,
                       strategies: Sequence[HeadlandStrategy] = STRATEGY_ORDER,
                       ) -> tuple[tuple[HeadlandMetrics, PassableResult], ...]:
    """Metrics of every strategy, each obtained by restricting the PENALTY build `base`."""
    if base.strategy != HeadlandStrategy.PENALTY:
        raise ValueError(f"compare_strategies needs a PENALTY base build, got {base.strategy!r}")
    out = []
    for strategy in strategies:
        result = restrict_result(base, params.with_strategy(strategy).connector)
        out.append((headland_metrics(result, passages), result))
    return tuple(out)


def choose_strategy(metrics: Sequence[HeadlandMetrics], max_outside_share: float) -> HeadlandStrategy:
    """Most reachable interrows among strategies whose proxy share <= the limit (order breaks ties).

    When none qualifies, INSIDE_ONLY: its connectors are inside, so the route stays inside and the
    interrows it cannot enter make their targets `reachable=false` instead of pushing outside_share up.
    """
    ok = [m for m in metrics if m.outside_share_proxy <= max_outside_share]
    if not ok:
        return HeadlandStrategy.INSIDE_ONLY
    rank = {str(s): i for i, s in enumerate(STRATEGY_ORDER)}
    best = max(ok, key=lambda m: (m.n_interrows_reachable, -rank.get(m.strategy, len(rank))))
    return HeadlandStrategy(best.strategy)


__all__ = [
    "NEAR_PASSAGE_M", "STRATEGY_ORDER", "HeadlandMetrics", "choose_strategy", "compare_strategies",
    "ends_near_passages", "headland_metrics",
]
