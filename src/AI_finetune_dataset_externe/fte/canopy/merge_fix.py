"""Merge-fix (optional, default off): re-join fragments of one plant that the vegetation mask split apart.

Two canopies of the same tile and row, consecutive along the axis, closer than `max_gap_m` and together
no longer than `max_len_m` become one polygon: their union plus a thin bridge (`bridge_w_m` wide) across
the gap, so the canopy union grows only by the bridge area. With a contact map, pairs whose bridge has
mean contact probability >= `max_contact` (two plants touching) are kept apart.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points

from fte.canopy.partition import (
    ContactFn,
    _as_frame,
    _axis_lookup,
    _pick_axis,
    _renumber,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeParams:
    max_gap_m: float = 0.15
    max_len_m: float = 1.4
    min_gap_m: float = 1e-6  # touching pieces (e.g. partition cuts) are never re-joined
    bridge_w_m: float = 0.05
    max_contact: float = 0.5
    clump_area_m2: float = 2.0

    def __post_init__(self) -> None:
        if self.max_gap_m < 0 or self.max_len_m <= 0 or self.bridge_w_m <= 0:
            raise ValueError("merge params must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _along(poly: Polygon, axis: LineString | None) -> tuple[float, float]:
    xy = np.asarray(poly.exterior.coords)
    if axis is None:
        return float(xy[:, 0].min()), float(xy[:, 0].max())
    s = shapely.line_locate_point(axis, shapely.points(xy))
    return float(s.min()), float(s.max())


def bridge(a: Polygon, b: Polygon, width_m: float) -> Polygon | None:
    """a ∪ b ∪ thin bridge between their nearest points; None if the result is not one polygon."""
    p, q = nearest_points(a, b)
    parts = [a, b]
    if p.distance(q) > 0:
        parts.append(LineString([p, q]).buffer(width_m / 2, cap_style="square"))
    merged = shapely.union_all(parts)
    return merged if isinstance(merged, Polygon) and merged.is_valid else None


def _contact_ok(p: Polygon, q: Polygon, contact: np.ndarray | None, x0: float, y0: float,
                params: MergeParams) -> bool:
    if contact is None:
        return True
    a, b = nearest_points(p, q)
    n = max(int(a.distance(b) / 0.025) + 1, 2)
    xs, ys = np.linspace(a.x, b.x, n), np.linspace(a.y, b.y, n)
    cols = np.clip(((xs - x0) / 0.025).astype(int), 0, contact.shape[1] - 1)
    rows = np.clip(((y0 - ys) / 0.025).astype(int), 0, contact.shape[0] - 1)
    vals = contact[rows, cols].astype(float) / (255.0 if contact.dtype == np.uint8 else 1.0)
    return float(vals.mean()) < params.max_contact


def _merge_row(recs: list[dict[str, Any]], axis: LineString | None, params: MergeParams,
               contact: np.ndarray | None, x0: float, y0: float) -> tuple[list[dict[str, Any]], int]:
    spans = [_along(r["geometry"], axis) for r in recs]
    order = sorted(range(len(recs)), key=lambda k: spans[k][0])
    out: list[dict[str, Any]] = []
    n_merged = 0
    for k in order:
        rec, (lo, hi) = recs[k], spans[k]
        if out:
            prev = out[-1]
            merged = _try_merge(prev, rec, lo, hi, params, contact, x0, y0)
            if merged is not None:
                out[-1] = merged
                n_merged += 1
                continue
        out.append({**rec, "_lo": lo, "_hi": hi})
    return out, n_merged


def _try_merge(prev: dict[str, Any], rec: dict[str, Any], lo: float, hi: float, params: MergeParams,
               contact: np.ndarray | None, x0: float, y0: float) -> dict[str, Any] | None:
    a, b = prev["geometry"], rec["geometry"]
    gap = a.distance(b)
    if max(hi, prev["_hi"]) - min(lo, prev["_lo"]) > params.max_len_m or not params.min_gap_m <= gap <= params.max_gap_m:
        return None
    if not _contact_ok(a, b, contact, x0, y0, params):
        return None
    geom = bridge(a, b, params.bridge_w_m)
    if geom is None:
        return None
    geom = shapely.orient_polygons(geom)
    lo2, hi2 = min(lo, prev["_lo"]), max(hi, prev["_hi"])
    return {**prev, "geometry": geom, "area_m2": float(geom.area), "n_vertices": len(geom.exterior.coords) - 1,
            "along_m": hi2 - lo2, "is_clump": bool(geom.area > params.clump_area_m2),
            "touches_edge": bool(prev["touches_edge"] or rec["touches_edge"]), "_lo": lo2, "_hi": hi2}


def _merge_tile(group: gpd.GeoDataFrame, axes: dict, params: MergeParams, contact: np.ndarray | None,
                tile_id: str) -> tuple[list[dict[str, Any]], int]:
    from vineyard.geo.tiling import tile_ref

    ref = tile_ref(tile_id)
    out: list[dict[str, Any]] = []
    n_total = 0
    records = group.to_dict("records")
    by_row: dict[Any, list[dict[str, Any]]] = {}
    for rec in records:
        by_row.setdefault(rec.get("row_id"), []).append(rec)
    for row_id, recs in by_row.items():
        if not isinstance(row_id, str) or len(recs) < 2:
            out.extend(recs)
            continue
        axis = _pick_axis(axes, tile_id, row_id, shapely.union_all([r["geometry"] for r in recs]))
        merged, n = _merge_row(recs, axis, params, contact, float(ref.x0), float(ref.y0))
        out.extend({k: v for k, v in r.items() if not k.startswith("_")} for r in merged)
        n_total += n
    return out, n_total


def merge_fragments(canopies: gpd.GeoDataFrame, row_pieces: gpd.GeoDataFrame, params: MergeParams,
                    contact: ContactFn | None = None) -> gpd.GeoDataFrame:
    """New canopies frame with close same-row fragments merged; inputs are not modified."""
    axes = _axis_lookup(row_pieces)
    records: list[dict[str, Any]] = []
    n_total = 0
    for tile_id, group in canopies.groupby("tile_id", sort=False):
        prob = contact(str(tile_id)) if contact is not None else None
        tile_recs, n = _merge_tile(group, axes, params, prob, str(tile_id))
        records.extend(_renumber(tile_recs, str(tile_id)) if n else tile_recs)
        n_total += n
    _log.info("merge-fix: %d merges, %d -> %d canopies", n_total, len(canopies), len(records))
    return _as_frame(records, canopies)
