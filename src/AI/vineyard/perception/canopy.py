"""Canopy extraction (02 §3.7 with plan S1/S4): mask ∧ corridor labels -> CC -> raw contours -> UTM polygons.

S1: raw findContours convention (integer vertices, no +0.5, no outset) unless the eval-sweep options
say otherwise; no vector clip to the corridor by default. S4: pieces flagged `row_interpolated` get
canopies only when their corridor carries enough vegetation. Optional: the local pieces are first moved
onto the canopy mask (perception.axis_refine), and components too small for a canopy but at least
`evidence_min_area_m2` are returned apart as plant evidence for the row-structure gaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import cv2
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.ids import format_canopy_id, row_index_of
from vineyard.geo.ops import make_valid_polygonal, orient_ccw
from vineyard.geo.tiling import GSD_M, TileRef, px_to_utm
from vineyard.perception.axis_refine import AxisRefineOptions, refine_pieces
from vineyard.perception.corridor import corridor_label_raster, corridor_polygon
from vineyard.perception.types import I32, BoolMask

if TYPE_CHECKING:
    from vineyard.config.sections_perception import CanopyConfig

LABEL_CONVENTIONS: Final = ("continuous", "index")
INTERPOLATED_FLAG: Final = "row_interpolated"
INTERP_TILES_COLUMN: Final = "interp_tile_ids"
INTERP_TILES_SEP: Final = ","
FLAG_SEPARATORS: Final = (";", ",", " ")
CANOPY_COLUMNS: Final = (
    "canopy_id", "tile_id", "vineyard_id", "row_id", "area_m2", "n_vertices", "along_m", "is_clump",
    "touches_edge", "confidence", "qa_flags",
)
EVIDENCE_COLUMNS: Final = ("tile_id", "row_id", "area_m2")
TOUCH_TOL_PX: Final = 1.0  # raw contours stop at index 2047, one pixel short of the 2048 edge
_AREA_EPS_PX: Final = 1e-9
_JOIN: Final = "mitre"


@dataclass(frozen=True)
class CanopyOptions:
    """Vectorization and filtering parameters (from CanopyConfig plus the S1/S4 flags)."""

    corridor_half_m: float
    min_area_m2: float
    connectivity: int
    approx_eps_px: float
    offset_px: float
    outset_px: float
    clump_area_m2: float
    clip_to_corridor: bool
    interpolated_min_veg_frac: float
    label_convention: str  # "index": raw vertices stay inside ±half_m
    axis_refine: AxisRefineOptions | None = None  # None: corridors on the given axes
    evidence_min_area_m2: float | None = None  # None: no gap evidence

    def __post_init__(self) -> None:
        if self.label_convention not in LABEL_CONVENTIONS:
            raise ValueError(f"canopy label_convention {self.label_convention!r} not in {LABEL_CONVENTIONS}")
        if self.evidence_min_area_m2 is not None and not 0 < self.evidence_min_area_m2 < self.min_area_m2:
            raise ValueError(f"evidence_min_area_m2 {self.evidence_min_area_m2} must be in (0, {self.min_area_m2})")

    @classmethod
    def from_config(cls, cfg: CanopyConfig, *, clip_to_corridor: bool | None = None,
                    interpolated_min_veg_frac: float | None = None,
                    label_convention: str | None = None) -> CanopyOptions:
        """Options from `canopy`; explicit arguments override the config (eval sweeps)."""
        return cls(
            corridor_half_m=cfg.corridor_half_m, min_area_m2=cfg.min_area_m2, connectivity=cfg.connectivity,
            approx_eps_px=cfg.simplify_px, offset_px=cfg.vector_offset_px, outset_px=cfg.vector_outset_px,
            clump_area_m2=cfg.clump_area_m2,
            clip_to_corridor=_pick(clip_to_corridor, cfg.clip_to_corridor),
            interpolated_min_veg_frac=_pick(interpolated_min_veg_frac, cfg.interpolated_min_veg_frac),
            label_convention=_pick(label_convention, cfg.corridor_label_convention),
            axis_refine=_refine_options(cfg), evidence_min_area_m2=_evidence_area(cfg),
        )

    @property
    def min_component_px(self) -> int:
        return _min_px(self.min_area_m2)

    @property
    def floor_area_m2(self) -> float:
        """Smallest vector area kept at all (canopies or evidence)."""
        return self.min_area_m2 if self.evidence_min_area_m2 is None else self.evidence_min_area_m2


def _min_px(area_m2: float) -> int:
    return int(math.ceil(area_m2 / (GSD_M * GSD_M) - _AREA_EPS_PX))


def _refine_options(cfg: CanopyConfig) -> AxisRefineOptions | None:
    if cfg.axis_refine_band_m <= 0:
        return None
    return AxisRefineOptions(band_m=cfg.axis_refine_band_m, iterations=cfg.axis_refine_iters,
                             max_shift_m=cfg.axis_refine_max_m, min_px=cfg.axis_refine_min_px)


def _evidence_area(cfg: CanopyConfig) -> float | None:
    area = cfg.gap_evidence_min_area_m2
    return area if 0 < area < cfg.min_area_m2 else None


def _pick(explicit: Any, configured: Any) -> Any:
    return configured if explicit is None else explicit


@dataclass(frozen=True)
class CanopyStats:
    corridor_veg_frac: float
    n_components: int
    n_small_dropped: int
    n_pieces: int
    n_interpolated_skipped: int


@dataclass(frozen=True, eq=False)
class CanopyResult:
    canopies: gpd.GeoDataFrame
    stats: CanopyStats
    evidence: gpd.GeoDataFrame  # EVIDENCE_COLUMNS: plant pieces below the canopy minimum (row-structure gaps)


@dataclass(frozen=True)
class _Component:
    label: int
    piece: int  # 0-based index into pieces
    bbox: tuple[int, int, int, int]  # x, y, w, h
    overlap: float
    n_px: int


# ------------------------------------------------------------------ masks


def interpolated_mask(pieces: gpd.GeoDataFrame) -> np.ndarray:
    """Bool per piece: a `row_interpolated` bool column, else the piece's tile listed in the row's
    `interp_tile_ids` (rows layer from blocks), else a `row_interpolated` token in qa_flags."""
    if INTERPOLATED_FLAG in pieces.columns:
        return pieces[INTERPOLATED_FLAG].fillna(False).to_numpy(dtype=bool)
    if INTERP_TILES_COLUMN in pieces.columns and "tile_id" in pieces.columns:
        return np.array([str(t) in str(ids or "").split(INTERP_TILES_SEP)
                         for t, ids in zip(pieces["tile_id"], pieces[INTERP_TILES_COLUMN], strict=True)], dtype=bool)
    if "qa_flags" not in pieces.columns:
        return np.zeros(len(pieces), dtype=bool)
    return np.array([_has_flag(str(f or ""), INTERPOLATED_FLAG) for f in pieces["qa_flags"]], dtype=bool)


def _has_flag(flags: str, flag: str) -> bool:
    tokens = [flags]
    for sep in FLAG_SEPARATORS:
        tokens = [t for tok in tokens for t in tok.split(sep)]
    return any(t.strip().split(":")[0] == flag for t in tokens)


def label_veg_fractions(mask: BoolMask, labels: I32, n_labels: int, valid: BoolMask | None = None) -> np.ndarray:
    """(n_labels,) fraction of mask pixels among valid pixels of each corridor label (NaN when empty)."""
    inside = labels > 0 if valid is None else (labels > 0) & valid
    lab = labels[inside]
    total = np.bincount(lab, minlength=n_labels + 1)[1:].astype(np.float64)
    hits = np.bincount(lab, weights=mask[inside].astype(np.float64), minlength=n_labels + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, hits / np.maximum(total, 1.0), np.nan)


def eligible_labels(interp: np.ndarray, veg_frac: np.ndarray, min_veg_frac: float) -> np.ndarray:
    """(n+1,) bool lookup (index 0 = background): supported pieces, or interpolated with veg evidence."""
    ok = ~interp | (np.nan_to_num(veg_frac, nan=0.0) >= min_veg_frac)
    return np.concatenate(([False], ok))


def vine_mask(veg_or_fused: BoolMask, labels: I32, tree: BoolMask | None) -> I32:
    """Corridor label where the mask is set (and no tree), 0 elsewhere (NN pseudo-labels)."""
    keep = veg_or_fused & (labels > 0)
    if tree is not None:
        keep = keep & ~tree
    return np.where(keep, labels, 0).astype(np.int32)


# ------------------------------------------------------------------ components


def _components(mask: np.ndarray, labels: I32, n_labels: int, connectivity: int,
                min_px: int) -> tuple[np.ndarray, list[_Component], int]:
    """(component raster, components >= min_px, number of all components)."""
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    if n <= 1:
        return comp, [], 0
    area = stats[1:, cv2.CC_STAT_AREA]
    fg = comp > 0
    pairs = comp[fg].astype(np.int64) * (n_labels + 1) + labels[fg]
    counts = np.bincount(pairs, minlength=n * (n_labels + 1)).reshape(n, n_labels + 1)[1:, 1:]
    best = counts.argmax(axis=1)
    keep = np.flatnonzero(area >= min_px)
    out = [
        _Component(label=int(i + 1), piece=int(best[i]),
                   bbox=tuple(int(v) for v in stats[i + 1, :4]),  # type: ignore[arg-type]
                   overlap=float(counts[i, best[i]] / area[i]), n_px=int(area[i]))
        for i in keep
    ]
    return comp, out, int(len(area))


def component_ring_px(comp: np.ndarray, c: _Component, approx_eps_px: float) -> np.ndarray | None:
    """Raw external contour (index coords) of one component, approxPolyDP-simplified."""
    x, y, w, h = c.bbox
    crop = (comp[y : y + h, x : x + w] == c.label).astype(np.uint8)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    ring = cv2.approxPolyDP(contour, approx_eps_px, True) if approx_eps_px > 0 else contour
    pts = ring.reshape(-1, 2).astype(np.float64) + (x, y)
    return pts if len(pts) >= 3 else None


def ring_to_polygons(ring_px: np.ndarray, tile: TileRef, clip: BaseGeometry, opts: CanopyOptions,
                     corridor: BaseGeometry | None = None, *, min_area_m2: float | None = None) -> list[Polygon]:
    """px ring -> (+offset, outset) -> UTM -> make_valid -> ∩ clip (∩ corridor) -> parts >= min area, CCW."""
    floor = opts.min_area_m2 if min_area_m2 is None else min_area_m2
    shape: BaseGeometry = Polygon(ring_px + opts.offset_px)
    if opts.outset_px > 0:
        shape = shapely.make_valid(shape).buffer(opts.outset_px, join_style=_JOIN)
    utm = shapely.transform(shape, lambda uv: px_to_utm(tile, uv))
    parts = make_valid_polygonal(utm)
    limit = clip if corridor is None else clip.intersection(corridor)
    out: list[Polygon] = []
    for part in parts:
        pieces = [part] if limit.contains(part) else make_valid_polygonal(part.intersection(limit))
        out.extend(orient_ccw(p) for p in pieces if p.area >= floor)
    return out


# ------------------------------------------------------------------ attributes


def _along_span(axis: LineString, poly: Polygon) -> tuple[float, float]:
    along = shapely.line_locate_point(axis, shapely.points(np.asarray(poly.exterior.coords)))
    return float(np.min(along)), float(np.max(along))


def _touches(poly: Polygon, boundary: BaseGeometry) -> bool:
    return bool(poly.distance(boundary) <= TOUCH_TOL_PX * GSD_M + 1e-9)


def _row_index(piece: object, row_id: str) -> int:
    value = getattr(piece, "row_index", None)
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        return int(value)
    idx = row_index_of(row_id)
    return idx if idx is not None else 0


def _records(polys: list[tuple[Polygon, int, float]], pieces: gpd.GeoDataFrame, interp: np.ndarray,
             clip: BaseGeometry, opts: CanopyOptions) -> list[dict[str, object]]:
    boundary = clip.boundary
    recs: list[dict[str, object]] = []
    for poly, k, overlap in polys:
        piece = pieces.iloc[k]
        row_id = str(piece["row_id"])
        lo, hi = _along_span(piece.geometry, poly)
        recs.append({
            "vineyard_id": str(piece["vineyard_id"]), "row_id": row_id, "_row_index": _row_index(piece, row_id),
            "_along_from": lo, "area_m2": float(poly.area), "n_vertices": len(poly.exterior.coords) - 1,
            "along_m": hi - lo, "is_clump": bool(poly.area > opts.clump_area_m2),
            "touches_edge": _touches(poly, boundary), "confidence": overlap,
            "qa_flags": INTERPOLATED_FLAG if interp[k] else "", "geometry": poly,
        })
    return recs


def _frame(recs: list[dict[str, object]], tile: TileRef, crs: object) -> gpd.GeoDataFrame:
    ordered = sorted(recs, key=lambda r: (r["vineyard_id"], r["_row_index"], r["_along_from"], r["area_m2"]))
    rows = [
        {"canopy_id": format_canopy_id(tile.tile_id, i + 1), "tile_id": tile.tile_id,
         **{k: v for k, v in r.items() if not k.startswith("_")}}
        for i, r in enumerate(ordered)
    ]
    if not rows:
        return gpd.GeoDataFrame({c: [] for c in CANOPY_COLUMNS}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)[[*CANOPY_COLUMNS, "geometry"]]


# ------------------------------------------------------------------ main entry


def extract_canopies(
    mask: BoolMask,
    pieces: gpd.GeoDataFrame,
    tile: TileRef,
    clip: BaseGeometry,
    opts: CanopyOptions,
    *,
    valid: BoolMask | None = None,
    tree: BoolMask | None = None,
) -> CanopyResult:
    """Canopy polygons of one tile from a vegetation (or fused) mask and the tile's row pieces (UTM).

    `pieces` come from corridor.clip_rows_to_tile (with the rows margin) and need row_id and
    vineyard_id; `clip` = tile_box ∩ tile_valid. Inputs are not modified.
    """
    axes = refine_pieces(pieces, mask, tile, opts.axis_refine)
    labels = corridor_label_raster(axes, tile, opts.corridor_half_m, convention=opts.label_convention)
    n_labels = len(axes)
    interp = interpolated_mask(axes)
    veg_frac = label_veg_fractions(mask, labels, n_labels, valid)
    eligible = eligible_labels(interp, veg_frac, opts.interpolated_min_veg_frac)
    keep = vine_mask(mask, labels, tree)
    keep = np.where(eligible[keep], keep, 0)
    comp, comps, n_all = _components(keep > 0, labels, n_labels, opts.connectivity, _min_px(opts.floor_area_m2))
    shapely.prepare(clip)
    polys, evidence = _vectorize(comp, comps, axes, tile, clip, opts)
    n_big = sum(1 for c in comps if c.n_px >= opts.min_component_px)
    inside = labels > 0 if valid is None else (labels > 0) & valid
    stats = CanopyStats(
        corridor_veg_frac=float(mask[inside].mean()) if inside.any() else 0.0,
        n_components=n_big, n_small_dropped=n_all - n_big, n_pieces=n_labels,
        n_interpolated_skipped=int((~eligible[1:]).sum()),
    )
    return CanopyResult(canopies=_frame(_records(polys, axes, interp, clip, opts), tile, axes.crs), stats=stats,
                        evidence=_evidence_frame(evidence, axes, tile))


def _vectorize(comp: np.ndarray, comps: list[_Component], axes: gpd.GeoDataFrame, tile: TileRef,
               clip: BaseGeometry, opts: CanopyOptions
               ) -> tuple[list[tuple[Polygon, int, float]], list[tuple[Polygon, int]]]:
    """Canopies (component >= min px and part >= min area) and gap evidence (the other parts >= floor)."""
    polys: list[tuple[Polygon, int, float]] = []
    evidence: list[tuple[Polygon, int]] = []
    for c in comps:
        ring = component_ring_px(comp, c, opts.approx_eps_px)
        if ring is None:
            continue
        corr = corridor_polygon(axes.geometry.iloc[c.piece], opts.corridor_half_m) if opts.clip_to_corridor else None
        big = c.n_px >= opts.min_component_px
        for part in ring_to_polygons(ring, tile, clip, opts, corr, min_area_m2=opts.floor_area_m2):
            if big and part.area >= opts.min_area_m2:
                polys.append((part, c.piece, c.overlap))
            else:
                evidence.append((part, c.piece))
    return polys, evidence


def _evidence_frame(evidence: list[tuple[Polygon, int]], axes: gpd.GeoDataFrame, tile: TileRef) -> gpd.GeoDataFrame:
    recs = sorted(({"tile_id": tile.tile_id, "row_id": str(axes["row_id"].iloc[k]), "area_m2": float(p.area),
                    "geometry": p} for p, k in evidence),
                  key=lambda r: (r["row_id"], r["geometry"].bounds, r["area_m2"]))
    if not recs:
        return gpd.GeoDataFrame({c: [] for c in EVIDENCE_COLUMNS}, geometry=gpd.GeoSeries([], crs=axes.crs),
                                crs=axes.crs)
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=axes.crs)[[*EVIDENCE_COLUMNS, "geometry"]]
