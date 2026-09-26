"""Route stage outputs as frames and JSON documents (contract §2.5.13, §5.1; design 04 layer table).

Pure builders: the stage module does the I/O. Every JSON document goes through `json_safe`, because
the canonical writer rejects NaN and infinities.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from vineyard.geo.tiling import CRS_EPSG
from vineyard.route.baseline import Baseline
from vineyard.route.planner import RoutePlan
from vineyard.route.stops import Stop, TargetVisit
from vineyard.route.validate import RouteValidation

ROUTE_ID_FMT: Final = "RT-{run_id}"
FULL_CONFIDENCE: Final = 1.0
M_PER_KM: Final = 1000.0
MIN_PER_H: Final = 60.0
_RUN_STAMP: Final = re.compile(r"^(\d{8}T\d{4})-")


@dataclass(frozen=True)
class RouteProvenance:
    source: str
    run_id: str
    model_version: str
    confidence: float = FULL_CONFIDENCE

    def columns(self, n: int) -> dict[str, list[Any]]:
        return {"source": [self.source] * n, "run_id": [self.run_id] * n,
                "model_version": [self.model_version] * n, "confidence": [self.confidence] * n,
                "qa_flags": [""] * n}


def json_safe(obj: Any) -> Any:
    """Non-finite floats -> None, tuples -> lists, numpy scalars -> Python, recursively."""
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def walking_minutes(length_m: float, speed_kmh: float) -> float:
    return length_m / M_PER_KM / speed_kmh * MIN_PER_H


def created_at(run_id: str, tz: str) -> str | None:
    """ISO 8601 time of the run (from its id, so a rerun of the same run writes the same bytes)."""
    match = _RUN_STAMP.match(run_id)
    if match is None:
        return None
    stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M").replace(tzinfo=ZoneInfo(tz))
    return stamp.isoformat()


def baseline_length(baseline: Baseline) -> float | None:
    """The naive walk quoted in route.geojson: serpentine through every interrow, else targets in id order."""
    return baseline.serpentine_est_m if baseline.serpentine_est_m is not None else baseline.id_order_m


def route_props(plan: RoutePlan, v: RouteValidation, baseline: Baseline, prov: RouteProvenance, *,
                speed_kmh: float, tz: str) -> dict[str, Any]:
    minutes = round(walking_minutes(v.length_m, speed_kmh), 1)
    base = baseline_length(baseline)
    props = {
        "route_id": ROUTE_ID_FMT.format(run_id=prov.run_id), "run_id": prov.run_id, "source": prov.source,
        "n_targets": v.n_targets, "n_visited_est": v.n_visited_est, "coverage_est": round(v.coverage_est, 4),
        "outside_frac_est": round(v.outside_frac, 5), "outside_share": round(v.outside_frac, 5),
        "closure_m": round(v.closure_m, 3), "walking_time_min": minutes, "duration_min": minutes,
        "baseline_length_m": None if base is None else round(base, 2),
        "baseline_id_order_m": None if baseline.id_order_m is None else round(baseline.id_order_m, 2),
        "policy": plan.policy, "solver": plan.solver, "created_at": created_at(prov.run_id, tz),
    }
    return json_safe(props)


def route_frame(line: LineString, v: RouteValidation, plan: RoutePlan, prov: RouteProvenance,
                speed_kmh: float) -> gpd.GeoDataFrame:
    data = {"route_id": [ROUTE_ID_FMT.format(run_id=prov.run_id)], "length_m": [float(line.length)],
            "n_targets": [v.n_targets], "n_visited_est": [v.n_visited_est], "coverage_est": [v.coverage_est],
            "outside_frac_est": [v.outside_frac], "closure_m": [v.closure_m],
            "walking_time_min": [walking_minutes(line.length, speed_kmh)], "solver": [plan.solver],
            "solve_time_s": [plan.solve_time_s]}
    return gpd.GeoDataFrame(data | prov.columns(1), geometry=[line], crs=CRS_EPSG)


def stops_frame(stops: Sequence[Stop], prov: RouteProvenance) -> gpd.GeoDataFrame:
    route_id = ROUTE_ID_FMT.format(run_id=prov.run_id)
    data = {"route_id": [route_id] * len(stops), "seq": [s.seq for s in stops],
            "target_id": [s.target_id for s in stops], "cum_dist_m": [s.cum_dist_m for s in stops],
            "leg_m": [s.leg_m for s in stops]}
    return gpd.GeoDataFrame(data | prov.columns(len(stops)), geometry=[Point(s.x, s.y) for s in stops],
                            crs=CRS_EPSG)


def visits_frame(visits: Sequence[TargetVisit], prov: RouteProvenance) -> gpd.GeoDataFrame:
    data = {"target_id": [t.target_id for t in visits], "reachable_final": [t.reachable_final for t in visits],
            "reach_note": [t.reach_note for t in visits], "n_candidates": [t.n_candidates for t in visits],
            "covered": [t.covered for t in visits], "visit_dist_m": [t.visit_dist_m for t in visits],
            "route_role": [t.route_role for t in visits]}
    return gpd.GeoDataFrame(data | prov.columns(len(visits)), geometry=[Point(t.x, t.y) for t in visits],
                            crs=CRS_EPSG)


def validation_doc(plan: RoutePlan, v: RouteValidation, file_checks: Sequence[Any],
                   headland: Mapping[str, Any]) -> dict[str, Any]:
    doc = v.to_json() | {
        "file_checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in file_checks],
        "policy": plan.policy, "policy_accepted": plan.accepted, "plan_outside_limit": plan.plan_outside_limit,
        "policies": [r.to_json() for r in plan.reports], "solver": plan.solver,
        "solve_time_s": plan.solve_time_s, "fallback_reason": plan.fallback_reason,
        "cover_iterations_kept": plan.iterations, "must_length_m": plan.must_length_m,
        "optional_delta_m": plan.optional_delta_m, "ladder_level": plan.ladder_level,
        "n_candidate_sets": len(plan.sets), "headland": dict(headland),
        "dropped_outside_budget": list(plan.dropped),
        "n_unreachable": sum(1 for t in plan.visits if not t.reachable_final),
        "unreachable_notes": _note_counts(plan.visits),
    }
    return json_safe(doc)


def _note_counts(visits: Sequence[TargetVisit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in visits:
        if not t.reachable_final:
            counts[t.reach_note] = counts.get(t.reach_note, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "ROUTE_ID_FMT", "RouteProvenance", "baseline_length", "created_at", "json_safe", "route_frame", "route_props",
    "stops_frame", "validation_doc", "visits_frame", "walking_minutes",
]
