"""Canopy extraction (02 §3.7 with plan S1/S4): mask ∧ corridor labels -> CC -> raw contours -> UTM polygons.

S1: raw findContours convention (integer vertices, no +0.5, no outset) unless the eval-sweep options
say otherwise; no vector clip to the corridor by default. S4: pieces flagged `row_interpolated` get
canopies only when their corridor carries enough vegetation.
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

    def __post_init__(self) -> None:
        if self.label_convention not in LABEL_CONVENTIONS:
            raise ValueError(f"canopy label_convention {self.label_convention!r} not in {LABEL_CONVENTIONS}")

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
        )

    @property
    def min_component_px(self) -> int:
        return int(math.ceil(self.min_area_m2 / (GSD_M * GSD_M) - _AREA_EPS_PX))


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


@dataclass(frozen=True)
class _Component:
    label: int
    piece: int  # 0-based index into pieces
    bbox: tuple[int, int, int, int]  # x, y, w, h
    overlap: float


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


def _components(mask: np.ndarray, labels: I32, n_labels: int, opts: CanopyOptions) -> tuple[np.ndarray, list[_Component], int]:
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=opts.connectivity)
    if n <= 1:
        return comp, [], 0
    area = stats[1:, cv2.CC_STAT_AREA]
    fg = comp > 0
    pairs = comp[fg].astype(np.int64) * (n_labels + 1) + labels[fg]
    counts = np.bincount(pairs, minlength=n * (n_labels + 1)).reshape(n, n_labels + 1)[1:, 1:]
    best = counts.argmax(axis=1)
    keep = np.flatnonzero(area >= opts.min_component_px)
    out = [
        _Component(label=int(i + 1), piece=int(best[i]),
                   bbox=tuple(int(v) for v in stats[i + 1, :4]),  # type: ignore[arg-type]
                   overlap=float(counts[i, best[i]] / area[i]))
        for i in keep
    ]
    return comp, out, int(len(area) - len(keep))


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
                     corridor: BaseGeometry | None = None) -> list[Polygon]:
    """px ring -> (+offset, outset) -> UTM -> make_valid -> ∩ clip (∩ corridor) -> parts >= min area, CCW."""
    shape: BaseGeometry = Polygon(ring_px + opts.offset_px)
    if opts.outset_px > 0:
        shape = shapely.make_valid(shape).buffer(opts.outset_px, join_style=_JOIN)
    utm = shapely.transform(shape, lambda uv: px_to_utm(tile, uv))
    parts = make_valid_polygonal(utm)
    limit = clip if corridor is None else clip.intersection(corridor)
    out: list[Polygon] = []
    for part in parts:
        pieces = [part] if limit.contains(part) else make_valid_polygonal(part.intersection(limit))
        out.extend(orient_ccw(p) for p in pieces if p.area >= opts.min_area_m2)
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
    labels = corridor_label_raster(pieces, tile, opts.corridor_half_m, convention=opts.label_convention)
    n_labels = len(pieces)
    interp = interpolated_mask(pieces)
    veg_frac = label_veg_fractions(mask, labels, n_labels, valid)
    eligible = eligible_labels(interp, veg_frac, opts.interpolated_min_veg_frac)
    keep = vine_mask(mask, labels, tree)
    keep = np.where(eligible[keep], keep, 0)
    comp, comps, n_small = _components(keep > 0, labels, n_labels, opts)
    shapely.prepare(clip)
    polys: list[tuple[Polygon, int, float]] = []
    for c in comps:
        ring = component_ring_px(comp, c, opts.approx_eps_px)
        if ring is None:
            continue
        corr = corridor_polygon(pieces.geometry.iloc[c.piece], opts.corridor_half_m) if opts.clip_to_corridor else None
        polys.extend((p, c.piece, c.overlap) for p in ring_to_polygons(ring, tile, clip, opts, corr))
    inside = labels > 0 if valid is None else (labels > 0) & valid
    stats = CanopyStats(
        corridor_veg_frac=float(mask[inside].mean()) if inside.any() else 0.0,
        n_components=len(comps), n_small_dropped=n_small, n_pieces=n_labels,
        n_interpolated_skipped=int((~eligible[1:]).sum()),
    )
    return CanopyResult(canopies=_frame(_records(polys, pieces, interp, clip, opts), tile, pieces.crs), stats=stats)
