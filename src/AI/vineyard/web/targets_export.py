"""targets.geojson and route.geojson of the web data bundle (src/Web/CLAUDE.md §6.3).

Targets are renumbered T001.. in route order (visited first, by position along the route), then the
unvisited ones by their source id. A target counts as visited when the route stage says so
(`target_visits.covered`), else when it is a route stop or lies within the visit radius of the line.

Every unvisited target gets a `skip_reason`: the route stage's reach note (`target_visits.reach_note`, else
the targets layer's), else `not_covered` / `no_route`. `reachable` is false only when there is no walkable
way to the target (UNREACHABLE_NOTES); targets the planner left out (PLANNER_SKIP_NOTES) stay reachable, and
any other note keeps the stages' own reachable flag.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import TargetKind
from vineyard.errors import SchemaError
from vineyard.geo.vector_io import round_geometry
from vineyard.web.rows_export import features_frame, finite_or_none, natural_key, normalize_enum, text_or_none

TARGET_COLUMNS: Final = ("target_id", "type", "vineyard_id", "row_id", "waste_id", "route_order", "reachable",
                         "gap_length_m", "note", "kind", "tile", "source_target_id", "priority", "route_role",
                         "skip_reason")
ROUTE_COLUMNS: Final = ("length_m", "duration_min", "speed_kmh", "baseline_length_m", "outside_share")
TARGET_KINDS: Final[tuple[str, ...]] = tuple(k.value for k in TargetKind)
WEB_GAP: Final = "gap"
WEB_MISSING: Final = "missing"
WEB_WASTE: Final = "waste"
# The web enum is gap | missing | waste; `other` is shown as a row problem and keeps its kind in `note`.
WEB_TYPE_OF_KIND: Final[Mapping[str, str]] = MappingProxyType({
    TargetKind.ROW_GAP.value: WEB_GAP, TargetKind.ROW_END_SHORT.value: WEB_GAP,
    TargetKind.MISSING_ROW.value: WEB_GAP, TargetKind.MISSING_PLANT.value: WEB_MISSING,
    TargetKind.SPARSE.value: WEB_MISSING, TargetKind.WASTE.value: WEB_WASTE, TargetKind.OTHER.value: WEB_GAP,
})
TARGET_ID_FORMAT: Final = "T{:03d}"
NOTE_SEPARATOR: Final = "; "
# Reach notes of the route stage (vineyard.route.candidates / stops / budget; a test pins the values).
# No walkable way to the target: off the walk graph's START component, or farther than the snap distance.
UNREACHABLE_NOTES: Final = frozenset({"disconnected", "too_far"})
# Walkable, but left out by the planner: optional detour too long, TSP node budget, optional kinds off,
# outside-share budget, or only reachable through outside edges the chosen policy excludes.
PLANNER_SKIP_NOTES: Final = frozenset({"optional_detour", "truncated", "not_routed", "outside_budget",
                                       "outside_only"})
SKIP_NOT_COVERED: Final = "not_covered"  # no note, but the line passes farther than the visit radius
SKIP_NO_ROUTE: Final = "no_route"  # the bundle has no route
M_PER_KM: Final = 1000.0
MIN_PER_HOUR: Final = 60.0
MIN_ROUTE_VERTICES: Final = 2


@dataclass(frozen=True)
class RouteInfo:
    """The planned route (EPSG:32635) and the informative numbers the web shows next to it."""

    line: LineString
    baseline_length_m: float | None = None
    outside_share: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.line, LineString) or self.line.is_empty or len(self.line.coords) < MIN_ROUTE_VERTICES:
            raise SchemaError("route must be a LineString with >= 2 vertices", got=type(self.line).__name__)
        share = self.outside_share
        if share is not None and not (math.isfinite(share) and 0.0 <= share <= 1.0):
            raise SchemaError("outside_share must be within [0, 1]", outside_share=share)
        base = self.baseline_length_m
        if base is not None and not (math.isfinite(base) and base >= 0.0):
            raise SchemaError("baseline_length_m must be finite and >= 0", baseline_length_m=base)


def route_features(route: RouteInfo, *, speed_kmh: float, decimals: int) -> gpd.GeoDataFrame:
    """One-feature frame; length_m is measured on the coordinates as written (rounded to `decimals`)."""
    if not (math.isfinite(speed_kmh) and speed_kmh > 0):
        raise SchemaError("walking speed must be > 0", speed_kmh=speed_kmh)
    line = round_geometry(route.line, decimals)
    length = float(line.length)
    record = {"length_m": length, "duration_min": length / (speed_kmh * M_PER_KM / MIN_PER_HOUR),
              "speed_kmh": float(speed_kmh), "baseline_length_m": route.baseline_length_m,
              "outside_share": route.outside_share}
    return features_frame([record], [line], ROUTE_COLUMNS)


# ------------------------------------------------------------------ targets


@dataclass(frozen=True)
class _Visit:
    covered: bool
    reachable: bool | None
    note: str | None
    route_role: str | None


def _stop_positions(stops: pd.DataFrame | None) -> dict[str, float]:
    """target_id -> distance along the route of its first stop."""
    if stops is None or stops.empty:
        return {}
    positions: dict[str, float] = {}
    for tid, dist in zip(stops["target_id"], stops["cum_dist_m"], strict=True):
        key = text_or_none(tid)
        if key is not None:
            positions[key] = min(float(dist), positions.get(key, math.inf))
    return positions


def _visits(visits: pd.DataFrame | None) -> dict[str, _Visit]:
    if visits is None or visits.empty:
        return {}
    out: dict[str, _Visit] = {}
    for rec in visits.to_dict("records"):
        reachable = rec.get("reachable_final")
        out[str(rec["target_id"])] = _Visit(covered=bool(rec["covered"]),
                                            reachable=None if pd.isna(reachable) else bool(reachable),
                                            note=text_or_none(rec.get("reach_note")),
                                            route_role=text_or_none(rec.get("route_role")))
    return out


def _point_of(rec: Mapping[str, Any]) -> BaseGeometry:
    geom = rec.get("geometry")
    if geom is not None and not geom.is_empty:
        return geom
    return Point(float(rec["x"]), float(rec["y"]))


def _known(value: object, known: Collection[str] | None, what: str) -> tuple[str | None, str | None]:
    """(id kept for the web, note) — ids missing from the bundle are nulled so references stay valid."""
    text = text_or_none(value)
    if text is None or known is None or text in known:
        return text, None
    return None, f"{what}={text}"


@dataclass(frozen=True)
class _Draft:
    record: dict[str, Any]
    geometry: BaseGeometry
    visited: bool
    position: float


def _flag(value: object) -> bool:
    """A stage's reachable flag; a missing one (None, NA, NaN) counts as false."""
    missing = value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))
    return False if missing else bool(value)


def _priority(value: object) -> int | None:
    number = finite_or_none(value)
    return None if number is None else int(number)


def _route_role(visit: _Visit | None, rec: Mapping[str, Any]) -> str | None:
    """The role the route stage planned with (`target_visits`), else the targets layer's; lower-cased."""
    role = (visit.route_role if visit is not None else None) or text_or_none(rec.get("route_role"))
    return None if role is None else role.lower()


def _reach(visit: _Visit | None, rec: Mapping[str, Any], visited: bool, has_route: bool) -> dict[str, Any]:
    """`reachable` and `skip_reason` (module docstring); the route stage's verdict wins over the targets
    layer's preliminary one."""
    if visited:
        return {"reachable": True, "skip_reason": None}
    flag = visit.reachable if visit is not None and visit.reachable is not None else rec.get("reachable")
    reason = visit.note if visit is not None else text_or_none(rec.get("reach_note"))
    reachable = reason in PLANNER_SKIP_NOTES or (reason not in UNREACHABLE_NOTES and _flag(flag))
    return {"reachable": reachable, "skip_reason": reason or (SKIP_NOT_COVERED if has_route else SKIP_NO_ROUTE)}


def _draft(rec: Mapping[str, Any], route: RouteInfo | None, stops: Mapping[str, float],
           visits: Mapping[str, _Visit], radius_m: float, known_rows: Collection[str] | None,
           known_waste: Collection[str] | None) -> _Draft:
    sid = str(rec["target_id"])
    point = _point_of(rec)
    kind = normalize_enum(rec.get("kind"), TARGET_KINDS, TargetKind.OTHER.value)
    visit = visits.get(sid)
    near = route is not None and route.line.distance(point) <= radius_m
    visited = visit.covered if visit is not None else (sid in stops or near)
    position = stops.get(sid, route.line.project(point) if route is not None else math.inf)
    row_id, row_note = _known(rec.get("row_id"), known_rows, "row")
    waste_id, waste_note = _known(rec.get("waste_id"), known_waste, "waste")
    reach_note = visit.note if visit is not None and visit.note else text_or_none(rec.get("reach_note"))
    notes = [kind, text_or_none(rec.get("reason")), reach_note, row_note, waste_note]
    record = {"type": WEB_TYPE_OF_KIND[kind], "vineyard_id": text_or_none(rec.get("vineyard_id")),
              "row_id": row_id, "waste_id": waste_id, "gap_length_m": finite_or_none(rec.get("gap_length_m")),
              "note": NOTE_SEPARATOR.join(n for n in notes if n), "kind": kind,
              "tile": text_or_none(rec.get("tile_id")), "source_target_id": sid,
              "priority": _priority(rec.get("priority")), "route_role": _route_role(visit, rec)}
    reach = _reach(visit, rec, visited, has_route=route is not None or bool(stops))
    return _Draft(record | reach, point, visited, float(position))


def _ordered(drafts: list[_Draft]) -> list[_Draft]:
    visited = sorted((d for d in drafts if d.visited),
                     key=lambda d: (d.position, natural_key(d.record["source_target_id"])))
    rest = sorted((d for d in drafts if not d.visited), key=lambda d: natural_key(d.record["source_target_id"]))
    return visited + rest


def target_features(
    targets: gpd.GeoDataFrame | None,
    *,
    route: RouteInfo | None,
    visit_radius_m: float,
    stops: pd.DataFrame | None = None,
    visits: pd.DataFrame | None = None,
    known_rows: Collection[str] | None = None,
    known_waste: Collection[str] | None = None,
) -> gpd.GeoDataFrame:
    """Web targets: T001.. in route order, `type` from TargetKind, `route_order` null when not visited."""
    if targets is None or targets.empty:
        return features_frame([], [], TARGET_COLUMNS)
    stop_pos, visit_rows = _stop_positions(stops), _visits(visits)
    drafts = [_draft(rec, route, stop_pos, visit_rows, visit_radius_m, known_rows, known_waste)
              for rec in targets.to_dict("records")]
    ordered = _ordered(drafts)
    n_visited = sum(d.visited for d in ordered)
    records = [{"target_id": TARGET_ID_FORMAT.format(k), "route_order": k if k <= n_visited else None} | d.record
               for k, d in enumerate(ordered, start=1)]
    return features_frame(records, [d.geometry for d in ordered], TARGET_COLUMNS)
