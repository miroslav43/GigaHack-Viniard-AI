"""build_targets: global rows + canopies + waste -> `targets` + `target_extents` (arch §4.10, contract §2.5.11).

Pipeline: one `RowContext` per global row (gaps from the injected gap engine, unknown spans from the
coverage), the pure rules of `target_rules` / `target_waste`, then: drop drafts over nodata, drop the
edge artefacts (row-derived drafts within `targets.edge_margin_m` of the coverage boundary; no
row_end_short on the outermost rows of a block), dedupe (waste never takes part), ids per kind sorted by
(vineyard_id, row_id, position), priority, route_role (`must` for `targets.must_kinds`, else `optional`)
and a preliminary reachability against the walking domain (the route stage decides the final one).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
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
from vineyard.route.cross_paths import subtract_paths
from vineyard.route.target_gaps import GapFn, row_gaps, unknown_intervals
from vineyard.route.target_rules import (
    END_SKIP_BOUNDARY,
    END_SKIP_NOT_LATERAL,
    END_SKIP_UNOBSERVED,
    RowContext,
    TargetDraft,
    end_boundary_skips,
    end_extensions,
    end_gap_drafts,
    end_samples,
    extension_drafts,
    gap_drafts,
    missing_plant_drafts,
    missing_row_drafts,
    sparse_drafts,
)
from vineyard.route.target_waste import waste_drafts

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
# counts keys of the dropped artefacts (targets.json)
COUNT_EDGE: Final = "edge_dropped"
COUNT_OUTER: Final = "end_outer_row"
COUNT_ILL_DEFINED: Final = "end_ill_defined"
COUNT_END_BOUNDARY: Final = "end_on_boundary"
# prefix of the counts of what the cross-paths removed (targets.json): drafts per kind, rows, gap centimetres
COUNT_CROSS_PATH: Final = "cross_path"
_SKIP_COUNT: Final[Mapping[str, str]] = MappingProxyType({
    END_SKIP_BOUNDARY: COUNT_END_BOUNDARY, END_SKIP_NOT_LATERAL: COUNT_ILL_DEFINED,
    END_SKIP_UNOBSERVED: COUNT_ILL_DEFINED})
ARTEFACT_COUNTS: Final = (COUNT_EDGE, COUNT_OUTER, COUNT_ILL_DEFINED, COUNT_END_BOUNDARY)


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

    def __post_init__(self) -> None:
        if min(self.corridor_half_m, self.reach_radius_m, self.targets.edge_margin_m) <= 0.0:
            raise ValueError("TargetSettings: corridor_half_m, reach_radius_m and edge_margin_m must be > 0")

    @classmethod
    def from_config(cls, cfg: AppConfig) -> TargetSettings:
        return cls(targets=cfg.targets, corridor_half_m=cfg.canopy.corridor_half_m,
                   reach_radius_m=cfg.route.candidate_radius_m)

    @property
    def edge_margin_m(self) -> float:
        """Row ends and row-derived targets this close to the coverage boundary are edge artefacts."""
        return self.targets.edge_margin_m

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
    # strips of the tracks across the rows (cross_paths): their stretch of every gap is no missing vine
    cross_paths: BaseGeometry | None = None


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
    flags = _ends_on_boundary(axes, inputs.coverage, settings.edge_margin_m)
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


@dataclass(frozen=True)
class RowDrafts:
    """Row-based drafts and the number of END targets not emitted, per artefact count key."""

    drafts: tuple[TargetDraft, ...]
    skipped: Mapping[str, int]


def _outer_indices(block: Sequence[RowContext]) -> frozenset[int]:
    indices = [c.row_index for c in block if c.row_index is not None]
    return frozenset((min(indices), max(indices))) if indices else frozenset()


def _end_drafts(ctx: RowContext, neighbours: Sequence[RowContext], outer: bool, cfg: TargetsConfig,
                coverage: BaseGeometry | None) -> tuple[tuple[TargetDraft, ...], Counter[str]]:
    """END (a) + well-defined END (b) of one row, and the END targets left out per count key."""
    exts = end_extensions(ctx, neighbours, cfg, coverage)
    ends = (*end_gap_drafts(ctx, cfg), *(d for ext in exts if not ext.skip for d in extension_drafts(ctx, ext, cfg)))
    skipped = Counter({COUNT_END_BOUNDARY: end_boundary_skips(ctx, cfg)})
    skipped += Counter(k for ext in exts if ext.skip for k in [_SKIP_COUNT[ext.skip]] * end_samples(ext.length_m, cfg))
    if outer and cfg.end_skip_outer_rows:
        return (), skipped + Counter({COUNT_OUTER: len(ends)})
    return ends, skipped


def _row_level_drafts(ctx: RowContext, cfg: TargetsConfig) -> tuple[TargetDraft, ...]:
    """GAP (+ MSP, SPR when enabled) of one row."""
    return (*gap_drafts(ctx, cfg), *(missing_plant_drafts(ctx, cfg) if cfg.include_missing else ()),
            *(sparse_drafts(ctx, cfg) if cfg.include_sparse else ()))


def _block_drafts(block: Sequence[RowContext], cfg: TargetsConfig,
                  coverage: BaseGeometry | None) -> tuple[tuple[TargetDraft, ...], Counter[str]]:
    by_index = {c.row_index: c for c in block if c.row_index is not None}
    outer = _outer_indices(block)
    ends = [_end_drafts(ctx, _neighbours(ctx, by_index), ctx.row_index in outer, cfg, coverage) for ctx in block]
    # draft order is irrelevant: dedupe and id assignment sort by (priority, kind, position)
    drafts = tuple(d for ctx, (end, _) in zip(block, ends, strict=True) for d in (*_row_level_drafts(ctx, cfg), *end))
    return (*drafts, *missing_row_drafts(block, cfg)), sum((c for _, c in ends), Counter())


def row_drafts(contexts: Sequence[RowContext], cfg: TargetsConfig, coverage: BaseGeometry | None = None) -> RowDrafts:
    """All row-based drafts (GAP, END, MSP, SPR per row; MRW per block) and the END drafts left out."""
    blocks: dict[str, list[RowContext]] = defaultdict(list)
    for ctx in contexts:
        blocks[ctx.vineyard_id].append(ctx)
    per_block = [_block_drafts(blocks[vid], cfg, coverage) for vid in sorted(blocks)]
    skipped = sum((c for _, c in per_block), Counter())
    return RowDrafts(tuple(d for drafts, _ in per_block for d in drafts),
                     MappingProxyType({k: skipped[k] for k in ARTEFACT_COUNTS if k != COUNT_EDGE}))


def _on_coverage(drafts: Sequence[TargetDraft], coverage: BaseGeometry) -> tuple[TargetDraft, ...]:
    if not drafts:
        return ()
    inside = shapely.covers(coverage, shapely.points([(d.x, d.y) for d in drafts]))
    return tuple(d for d, ok in zip(drafts, inside, strict=True) if ok)


def off_edge(drafts: Sequence[TargetDraft], coverage: BaseGeometry, margin_m: float) -> tuple[TargetDraft, ...]:
    """Drafts farther than `margin_m` from the coverage boundary (tile edges, nodata); waste always kept."""
    if not drafts:
        return ()
    near = shapely.dwithin(coverage.boundary, shapely.points([(d.x, d.y) for d in drafts]), margin_m)
    return tuple(d for d, edge in zip(drafts, near, strict=True) if d.kind is TargetKind.WASTE or not edge)


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


def route_role(kind: TargetKind, must_kinds: Sequence[str]) -> str:
    """`must` for the configured must-visit kinds (targets.must_kinds), `optional` for every other kind."""
    return ROLE_MUST if kind.value in must_kinds else ROLE_OPTIONAL


def _target_frame(items: Sequence[tuple[str, TargetDraft]], reach: Reachability, prov: TargetProvenance,
                  must_kinds: Sequence[str]) -> gpd.GeoDataFrame:
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
        "route_role": [route_role(d.kind, must_kinds) for d in drafts],
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


def _counts(targets: gpd.GeoDataFrame, dropped: Mapping[str, int]) -> Mapping[str, int]:
    per_kind = {k.value: int((targets["kind"] == k.value).sum()) for k in TargetKind}
    roles = {f"role_{r}": int((targets["route_role"] == r).sum()) for r in (ROLE_MUST, ROLE_OPTIONAL)}
    extra = {"unreachable": int((~targets["reachable"].astype(bool)).sum()), "total": len(targets)}
    return MappingProxyType(per_kind | roles | dict(dropped) | extra)


def _without_cross_paths(contexts: Sequence[RowContext], cfg: TargetsConfig, cross: BaseGeometry | None,
                         coverage: BaseGeometry) -> tuple[tuple[RowContext, ...], dict[str, int]]:
    """Contexts with the cross-path stretches cut out of their gaps, and what that removed (draft counts
    per kind before dedupe / edge filtering, rows touched, gap metres)."""
    if cross is None or cross.is_empty:
        return tuple(contexts), {}
    shapely.prepare(cross)
    cut = [subtract_paths(c, cross) for c in contexts]
    trimmed = tuple(c for c, _ in cut)
    before = Counter(d.kind.value for d in row_drafts(contexts, cfg, coverage).drafts)
    after = Counter(d.kind.value for d in row_drafts(trimmed, cfg, coverage).drafts)
    dropped = {f"{COUNT_CROSS_PATH}_{k}": before[k] - after[k] for k in sorted(before) if before[k] != after[k]}
    return trimmed, {**dropped, f"{COUNT_CROSS_PATH}_rows": sum(1 for _, m in cut if m > 0.0),
                     f"{COUNT_CROSS_PATH}_gap_cm": round(100 * sum(m for _, m in cut))}


def _kept_drafts(inputs: TargetInputs, settings: TargetSettings, gap_fn: GapFn) -> tuple[list[TargetDraft], dict]:
    """Row drafts off nodata and off the edge margin, plus waste, deduped; with the dropped counts."""
    cfg = settings.targets
    contexts, crossed = _without_cross_paths(row_contexts(inputs, settings, gap_fn), cfg, inputs.cross_paths,
                                             inputs.coverage)
    rows = row_drafts(contexts, cfg, inputs.coverage)
    on_data = _on_coverage(rows.drafts, inputs.coverage)
    inner = off_edge(on_data, inputs.coverage, settings.edge_margin_m)
    waste = waste_drafts(inputs.waste) if cfg.include_waste else ()
    drafts, n_deduped = dedupe((*inner, *waste), cfg.dedupe_m)
    dropped = {"deduped": n_deduped, "nodata_dropped": len(rows.drafts) - len(on_data),
               COUNT_EDGE: len(on_data) - len(inner), **rows.skipped, **crossed}
    return list(drafts), dropped


def build_targets(inputs: TargetInputs, settings: TargetSettings, prov: TargetProvenance,
                  gap_fn: GapFn) -> TargetsResult:
    """Targets and their extents from the global rows, canopies and waste of one AnnSet."""
    shapely.prepare(inputs.coverage)
    drafts, dropped = _kept_drafts(inputs, settings, gap_fn)
    items = _ordered_with_ids(drafts)
    xy = np.array([(d.x, d.y) for _, d in items], dtype=np.float64).reshape(-1, 2)
    reach = reachability(xy, inputs.reach_domain, inputs.forbidden, settings.reach_radius_m)
    targets = _target_frame(items, reach, prov, settings.targets.must_kinds)
    return TargetsResult(targets=targets, extents=_extent_frame(items, prov), issues=_issues(targets),
                         counts=_counts(targets, dropped))


__all__ = [
    "ARTEFACT_COUNTS", "COUNT_EDGE", "COUNT_END_BOUNDARY", "COUNT_ILL_DEFINED", "COUNT_OUTER", "ROLE_MUST",
    "ROLE_OPTIONAL", "Reachability", "RowDrafts", "TargetInputs", "TargetProvenance", "TargetSettings",
    "TargetsResult", "build_targets", "dedupe", "off_edge", "reachability", "route_role", "row_contexts",
    "row_drafts",
]
