"""build_targets: global rows + canopies + waste -> `targets` + `target_extents` (arch §4.10, contract §2.5.11).

Pipeline: one `RowContext` per global row (gaps from the injected gap engine, unknown spans from the
coverage), the pure rules of `target_rules` / `target_waste`, then: drop drafts over nodata, dedupe
(waste never takes part), ids per kind sorted by (vineyard_id, row_id, position), priority/route_role,
and a preliminary reachability against the walking domain (the route stage decides the final one).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from vineyard.config import AppConfig, TargetsConfig
from vineyard.contracts.enums import Severity, Source, TargetKind
from vineyard.contracts.ids import format_target_id
from vineyard.contracts.qa import QaIssue
from vineyard.contracts.schema_defs import MIN_LINE_LENGTH_M
from vineyard.contracts.schemas import coerce_layer, empty_layer
from vineyard.errors import SchemaError
from vineyard.geo.tiling import CRS_EPSG, tile_of_point
from vineyard.route.target_gaps import GapFn, row_gaps, unknown_intervals
from vineyard.route.target_rules import (
    PRIORITY_NORMAL,
    RowContext,
    TargetDraft,
    end_extension_drafts,
    end_gap_drafts,
    gap_drafts,
    missing_plant_drafts,
    missing_row_drafts,
    sparse_drafts,
)
from vineyard.route.target_waste import waste_drafts

# A row end closer than this to the coverage boundary (tile edge / nodata) is "on the boundary": the
# row may continue in unseen imagery, so no row_end_short target is raised there.
# CONFIG-REQUEST: targets.end_boundary_tol_m = 0.5
END_BOUNDARY_TOL_M: Final = 0.5
ROLE_MUST: Final = "must"
ROLE_OPTIONAL: Final = "optional"
NOTE_OK: Final = ""
NOTE_UNCHECKED: Final = "unchecked"
NOTE_FORBIDDEN: Final = "in_forbidden"
NOTE_TOO_FAR: Final = "too_far"
UNREACHABLE_CODE: Final = "target_unreachable"
TARGET_CONFIDENCE: Final = 1.0
ROW_COLUMNS: Final = ("row_id", "vineyard_id", "row_index")
EXTRA_COLUMNS: Final = ("reason", "route_role", "along_m")
KIND_RANK: Final[Mapping[str, int]] = MappingProxyType({str(k): i for i, k in enumerate(TargetKind)})


@dataclass(frozen=True)
class TargetProvenance:
    source: Source
    run_id: str
    model_version: str


@dataclass(frozen=True)
class TargetSettings:
    targets: TargetsConfig
    corridor_half_m: float
    reach_radius_m: float
    end_boundary_tol_m: float = END_BOUNDARY_TOL_M

    def __post_init__(self) -> None:
        if min(self.corridor_half_m, self.reach_radius_m, self.end_boundary_tol_m) <= 0.0:
            raise ValueError("TargetSettings: corridor_half_m, reach_radius_m and end_boundary_tol_m must be > 0")

    @classmethod
    def from_config(cls, cfg: AppConfig) -> TargetSettings:
        return cls(targets=cfg.targets, corridor_half_m=cfg.canopy.corridor_half_m,
                   reach_radius_m=cfg.route.candidate_radius_m)

    @property
    def engine_min_length_m(self) -> float:
        """Shortest gap any enabled rule needs (sparse needs every gap to measure occupancy)."""
        t = self.targets
        if t.include_sparse:
            return 0.0
        floor = min(t.gap_min_m, t.end_short_min_m)
        return min(floor, t.missing_min_m) if t.include_missing else floor


@dataclass(frozen=True)
class TargetInputs:
    """rows = derive's global `rows`; coverage = imaged area (tile_valid union); optional reach domain."""

    rows: gpd.GeoDataFrame
    canopies: gpd.GeoDataFrame
    waste: gpd.GeoDataFrame
    coverage: BaseGeometry
    reach_domain: BaseGeometry | None = None
    forbidden: BaseGeometry | None = None


@dataclass(frozen=True)
class TargetsResult:
    targets: gpd.GeoDataFrame
    extents: gpd.GeoDataFrame
    issues: tuple[QaIssue, ...]
    counts: Mapping[str, int]


# ------------------------------------------------------------------ row contexts


def _check_rows(rows: gpd.GeoDataFrame) -> None:
    missing = [c for c in ROW_COLUMNS if c not in rows.columns]
    if missing:
        raise SchemaError("rows layer lacks columns needed for targets", columns=", ".join(missing))
    bad = [str(rid) for rid, g in zip(rows["row_id"], rows.geometry, strict=True) if not isinstance(g, LineString)]
    if bad:
        raise SchemaError("rows layer geometries must be LineStrings", row_ids=", ".join(bad[:5]))


def _ends_on_boundary(axes: Sequence[LineString], coverage: BaseGeometry, tol_m: float) -> np.ndarray:
    """(n, 2) bool: head / tail end within tol_m of the coverage boundary (or outside it)."""
    if not axes:
        return np.zeros((0, 2), dtype=bool)
    ends = np.array([[a.coords[0], a.coords[-1]] for a in axes], dtype=np.float64).reshape(-1, 2)
    disks = shapely.buffer(shapely.points(ends), tol_m)
    return ~shapely.covers(coverage, disks).reshape(-1, 2)


def _row_unknown(axis: LineString, coverage: BaseGeometry, half_m: float) -> BaseGeometry | None:
    corridor = axis.buffer(half_m, cap_style="flat")
    if shapely.covers(coverage, corridor):
        return None
    unknown = corridor.difference(coverage)
    return None if unknown.is_empty else unknown


def _row_index(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def row_contexts(inputs: TargetInputs, settings: TargetSettings, gap_fn: GapFn) -> tuple[RowContext, ...]:
    """One RowContext per global row, ordered by (vineyard_id, row_index, row_id)."""
    rows = inputs.rows
    _check_rows(rows)
    order = sorted(range(len(rows)), key=lambda i: (str(rows["vineyard_id"].iloc[i]),
                                                     _row_index(rows["row_index"].iloc[i]) or 0,
                                                     str(rows["row_id"].iloc[i])))
    axes = [rows.geometry.iloc[i] for i in order]
    flags = _ends_on_boundary(axes, inputs.coverage, settings.end_boundary_tol_m)
    geoms = inputs.canopies.geometry.to_numpy()
    tree = shapely.STRtree(geoms)
    out = []
    for pos, i in enumerate(order):
        axis, row_id = axes[pos], str(rows["row_id"].iloc[i])
        near = geoms[np.sort(tree.query(axis, predicate="dwithin", distance=settings.corridor_half_m))]
        unknown = _row_unknown(axis, inputs.coverage, settings.corridor_half_m)
        gaps = row_gaps(gap_fn, axis, list(near), unknown, min_length_m=settings.engine_min_length_m,
                        row_id=row_id)
        out.append(RowContext(row_id=row_id, vineyard_id=str(rows["vineyard_id"].iloc[i]),
                              row_index=_row_index(rows["row_index"].iloc[i]), axis=axis, gaps=gaps,
                              unknown=unknown_intervals(axis, unknown), head_on_boundary=bool(flags[pos, 0]),
                              tail_on_boundary=bool(flags[pos, 1])))
    return tuple(out)


# ------------------------------------------------------------------ drafts


def _neighbours(ctx: RowContext, by_index: Mapping[int, RowContext]) -> tuple[RowContext, ...]:
    if ctx.row_index is None:
        return ()
    return tuple(by_index[k] for k in (ctx.row_index - 1, ctx.row_index + 1) if k in by_index)


def _block_drafts(block: Sequence[RowContext], cfg: TargetsConfig) -> list[TargetDraft]:
    by_index = {c.row_index: c for c in block if c.row_index is not None}
    drafts: list[TargetDraft] = []
    for ctx in block:
        drafts += [*gap_drafts(ctx, cfg), *end_gap_drafts(ctx, cfg),
                   *end_extension_drafts(ctx, _neighbours(ctx, by_index), cfg)]
        if cfg.include_missing:
            drafts += missing_plant_drafts(ctx, cfg)
        if cfg.include_sparse:
            drafts += sparse_drafts(ctx, cfg)
    return drafts + list(missing_row_drafts(block, cfg))


def row_drafts(contexts: Sequence[RowContext], cfg: TargetsConfig) -> tuple[TargetDraft, ...]:
    """All row-based drafts (GAP, END, MSP, SPR per row; MRW per block)."""
    blocks: dict[str, list[RowContext]] = defaultdict(list)
    for ctx in contexts:
        blocks[ctx.vineyard_id].append(ctx)
    return tuple(d for vid in sorted(blocks) for d in _block_drafts(blocks[vid], cfg))


def _on_coverage(drafts: Sequence[TargetDraft], coverage: BaseGeometry) -> tuple[TargetDraft, ...]:
    if not drafts:
        return ()
    inside = shapely.covers(coverage, shapely.points([(d.x, d.y) for d in drafts]))
    return tuple(d for d, ok in zip(drafts, inside, strict=True) if ok)


def _sort_key(d: TargetDraft) -> tuple[Any, ...]:
    along = d.along_m if math.isfinite(d.along_m) else 0.0
    return (d.vineyard_id, d.row_id or d.sort_ref, along, d.sort_ref, round(d.x, 6), round(d.y, 6))


def dedupe(drafts: Sequence[TargetDraft], dedupe_m: float) -> tuple[tuple[TargetDraft, ...], int]:
    """Greedy by (priority, kind, key): a draft within dedupe_m of a kept one is dropped. Waste is exempt."""
    waste = [d for d in drafts if d.kind is TargetKind.WASTE]
    rest = sorted((d for d in drafts if d.kind is not TargetKind.WASTE),
                  key=lambda d: (d.priority, KIND_RANK[d.kind], _sort_key(d)))
    if dedupe_m <= 0.0 or len(rest) < 2:
        return (*rest, *waste), 0
    pts = shapely.points([(d.x, d.y) for d in rest])
    left, right = shapely.STRtree(pts).query(pts, predicate="dwithin", distance=dedupe_m)
    near: dict[int, list[int]] = defaultdict(list)
    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        if a != b:
            near[a].append(b)
    kept = np.zeros(len(rest), dtype=bool)
    for i in range(len(rest)):
        kept[i] = not any(kept[j] for j in near[i])
    return (*(d for d, k in zip(rest, kept, strict=True) if k), *waste), int((~kept).sum())


# ------------------------------------------------------------------ reachability


@dataclass(frozen=True)
class Reachability:
    reachable: np.ndarray
    note: tuple[str, ...]
    snap_dist_m: np.ndarray


def reachability(xy: np.ndarray, domain: BaseGeometry | None, forbidden: BaseGeometry | None,
                 radius_m: float) -> Reachability:
    """Preliminary: False in `forbidden` or farther than radius_m from `domain`; unchecked without domain."""
    pts = shapely.points(np.asarray(xy, dtype=np.float64).reshape(-1, 2))
    n = len(pts)
    if domain is None or domain.is_empty:
        snap, too_far, base = np.full(n, np.nan), np.zeros(n, dtype=bool), NOTE_UNCHECKED
    else:
        snap = shapely.distance(domain, pts).astype(np.float64)
        too_far, base = snap > radius_m, NOTE_OK
    banned = shapely.intersects(forbidden, pts) if forbidden is not None and not forbidden.is_empty \
        else np.zeros(n, dtype=bool)
    notes = tuple(NOTE_FORBIDDEN if b else NOTE_TOO_FAR if f else base for b, f in zip(banned, too_far, strict=True))
    return Reachability(reachable=~(banned | too_far), note=notes, snap_dist_m=snap)


# ------------------------------------------------------------------ layers


def _ordered_with_ids(drafts: Sequence[TargetDraft]) -> list[tuple[str, TargetDraft]]:
    ordered = sorted(drafts, key=lambda d: (KIND_RANK[d.kind], _sort_key(d)))
    counters: dict[TargetKind, int] = defaultdict(int)
    out = []
    for d in ordered:
        counters[d.kind] += 1
        out.append((format_target_id(d.kind, counters[d.kind]), d))
    return out


def _provenance(prov: TargetProvenance, n: int) -> dict[str, list[Any]]:
    return {"source": [Source(prov.source).value] * n, "run_id": [prov.run_id] * n,
            "model_version": [prov.model_version] * n, "confidence": [TARGET_CONFIDENCE] * n, "qa_flags": [""] * n}


def _target_frame(items: Sequence[tuple[str, TargetDraft]], reach: Reachability,
                  prov: TargetProvenance) -> gpd.GeoDataFrame:
    if not items:
        return empty_layer("targets").assign(reason=pd.Series(dtype=str), route_role=pd.Series(dtype=str),
                                             along_m=pd.Series(dtype=np.float64))
    drafts = [d for _, d in items]
    data = {
        "target_id": [tid for tid, _ in items], "kind": [d.kind.value for d in drafts],
        "vineyard_id": [d.vineyard_id for d in drafts],
        "tile_id": [tile_of_point(d.x, d.y).tile_id for d in drafts], "row_id": [d.row_id for d in drafts],
        "interrow_id": [d.interrow_id for d in drafts], "waste_id": [d.waste_id for d in drafts],
        "x": [d.x for d in drafts], "y": [d.y for d in drafts], "gap_length_m": [d.gap_length_m for d in drafts],
        "priority": [d.priority for d in drafts], "reachable": list(reach.reachable), "reach_note": list(reach.note),
        "snap_dist_m": list(reach.snap_dist_m), **_provenance(prov, len(drafts)),
        "reason": [d.reason for d in drafts],
        "route_role": [ROLE_MUST if d.priority <= PRIORITY_NORMAL else ROLE_OPTIONAL for d in drafts],
        "along_m": [d.along_m for d in drafts],
    }
    geoms = gpd.GeoSeries([Point(d.x, d.y) for d in drafts], crs=CRS_EPSG)
    return coerce_layer(gpd.GeoDataFrame(data, geometry=geoms, crs=CRS_EPSG), "targets")


def _extent_frame(items: Sequence[tuple[str, TargetDraft]], prov: TargetProvenance) -> gpd.GeoDataFrame:
    kept = [(tid, d.kind.value, d.extent) for tid, d in items
            if isinstance(d.extent, LineString) and d.extent.length >= MIN_LINE_LENGTH_M]
    if not kept:
        return empty_layer("target_extents")
    data = {"target_id": [tid for tid, _, _ in kept], "kind": [kind for _, kind, _ in kept],
            "length_m": [line.length for _, _, line in kept], **_provenance(prov, len(kept))}
    geoms = gpd.GeoSeries([line for _, _, line in kept], crs=CRS_EPSG)
    return coerce_layer(gpd.GeoDataFrame(data, geometry=geoms, crs=CRS_EPSG), "target_extents")


def _issues(targets: gpd.GeoDataFrame) -> tuple[QaIssue, ...]:
    bad = targets[~targets["reachable"].astype(bool)]
    return tuple(QaIssue(Severity.WARNING, UNREACHABLE_CODE, str(t.tile_id), str(t.target_id),
                         f"target {t.kind} unreachable: {t.reach_note}", float(t.x), float(t.y))
                 for t in bad.itertuples())


def _counts(targets: gpd.GeoDataFrame, n_deduped: int, n_nodata: int) -> Mapping[str, int]:
    per_kind = {k.value: int((targets["kind"] == k.value).sum()) for k in TargetKind}
    extra = {"deduped": n_deduped, "nodata_dropped": n_nodata,
             "unreachable": int((~targets["reachable"].astype(bool)).sum()), "total": len(targets)}
    return MappingProxyType(per_kind | extra)


def build_targets(inputs: TargetInputs, settings: TargetSettings, prov: TargetProvenance,
                  gap_fn: GapFn) -> TargetsResult:
    """Targets and their extents from the global rows, canopies and waste of one AnnSet."""
    cfg = settings.targets
    shapely.prepare(inputs.coverage)
    candidates = row_drafts(row_contexts(inputs, settings, gap_fn), cfg)
    on_data = _on_coverage(candidates, inputs.coverage)
    waste = waste_drafts(inputs.waste) if cfg.include_waste else ()
    drafts, n_deduped = dedupe((*on_data, *waste), cfg.dedupe_m)
    items = _ordered_with_ids(drafts)
    xy = np.array([(d.x, d.y) for _, d in items], dtype=np.float64).reshape(-1, 2)
    reach = reachability(xy, inputs.reach_domain, inputs.forbidden, settings.reach_radius_m)
    targets = _target_frame(items, reach, prov)
    return TargetsResult(targets=targets, extents=_extent_frame(items, prov), issues=_issues(targets),
                         counts=_counts(targets, n_deduped, len(candidates) - len(on_data)))


__all__ = [
    "END_BOUNDARY_TOL_M", "ROLE_MUST", "ROLE_OPTIONAL", "Reachability", "TargetInputs", "TargetProvenance",
    "TargetSettings", "TargetsResult", "build_targets", "dedupe", "reachability", "row_contexts", "row_drafts",
]
