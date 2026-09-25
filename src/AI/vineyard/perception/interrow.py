"""Interrow bands and per-tile pieces (02 §3.8, plan S3/S5): offset curves ±0.30 m trimmed to the shorter
row, forbidden holes as notches, per-tile cut, cover classification from veg / vis rasters."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import shapely
from rasterio.transform import Affine
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import InterrowCover
from vineyard.contracts.ids import format_interrow_id, format_interrow_piece_id
from vineyard.contracts.ordering import angle_deg_utm, mean_axial_angle_deg, order_by_normal
from vineyard.geo import ops
from vineyard.geo.ops import make_valid_polygonal, orient_ccw, split_multi
from vineyard.geo.tiling import TILE_PX, TileRef, utm_to_px
from vineyard.perception.types import U8, VIS_NODATA, VIS_OVEREXPOSED, VIS_SHADOW, BoolMask

if TYPE_CHECKING:
    from vineyard.config.sections_perception import InterrowConfig

INTERROW_COLUMNS: Final = (
    "interrow_id", "vineyard_id", "row_left_id", "row_right_id", "area_m2", "width_mean_m", "length_m",
    "k", "angle_deg", "n_notches",
)
PIECE_COLUMNS: Final = (
    "piece_id", "interrow_id", "vineyard_id", "tile_id", "row_left_id", "row_right_id", "interrow_cover",
    "veg_frac", "shadow_frac", "area_m2", "width_mean_m", "n_notches", "qa_flags",
)
BORDERLINE_FLAG: Final = "cover_borderline"
_FAR_M: Final = 1.0e4  # half-size of the cutting strip; far larger than any row
_SHADOW_CODES: Final = (VIS_SHADOW, VIS_OVEREXPOSED)
_FRAC_EPS: Final = 1e-9  # a fraction exactly at threshold ± margin counts as borderline


@dataclass(frozen=True)
class CoverStats:
    veg_frac: float
    shadow_frac: float
    n_px: int


# ------------------------------------------------------------------ geometry helpers


def _unit(line: LineString) -> np.ndarray:
    xy = np.asarray(line.coords)
    d = xy[-1] - xy[0]
    norm = float(np.hypot(*d))
    if norm <= 0:
        raise ValueError(f"row axis has zero extent: {line.wkt[:80]}")
    return d / norm


def _span(line: LineString, u: np.ndarray, origin: np.ndarray) -> tuple[float, float]:
    t = (np.asarray(line.coords) - origin) @ u
    return float(t.min()), float(t.max())


def _strip(origin: np.ndarray, u: np.ndarray, lo: float, hi: float) -> Polygon:
    n = np.array([-u[1], u[0]])
    corners = [origin + u * a + n * b for a, b in ((lo, -_FAR_M), (hi, -_FAR_M), (hi, _FAR_M), (lo, _FAR_M))]
    return Polygon(corners)


def _longest_line(geom: BaseGeometry) -> LineString | None:
    lines = [g for g in split_multi(shapely.line_merge(geom) if not geom.is_empty else geom)
             if isinstance(g, LineString)]
    return max(lines, key=lambda g: g.length) if lines else None


def _oriented_offset(line: LineString, towards: np.ndarray, offset_m: float) -> LineString:
    u = _unit(line)
    n = np.array([-u[1], u[0]])
    side = 1.0 if float((towards - np.asarray(line.coords)[0]) @ n) >= 0 else -1.0
    off = line.offset_curve(side * offset_m, join_style="mitre")
    merged = _longest_line(off) if not isinstance(off, LineString) else off
    if merged is None or merged.is_empty:
        raise ValueError(f"offset curve collapsed for {line.wkt[:80]}")
    xy = np.asarray(merged.coords)
    return LineString(xy if float((xy[-1] - xy[0]) @ u) >= 0 else xy[::-1])


def band_polygon(left: LineString, right: LineString, offset_m: float) -> Polygon | None:
    """Band from `left` + offset to `right` − offset, cut to the common along-interval (shorter row)."""
    u = _unit(left)
    right = right if float(_unit(right) @ u) >= 0 else LineString(np.asarray(right.coords)[::-1])
    origin = np.asarray(left.coords)[0]
    (a0, a1), (b0, b1) = _span(left, u, origin), _span(right, u, origin)
    lo, hi = max(a0, b0), min(a1, b1)
    if hi <= lo:
        return None
    mid_r = np.asarray(right.interpolate(0.5, normalized=True).coords)[0]
    mid_l = np.asarray(left.interpolate(0.5, normalized=True).coords)[0]
    strip = _strip(origin, u, lo, hi)
    lo_line = _longest_line(_oriented_offset(left, mid_r, offset_m).intersection(strip))
    ro_line = _longest_line(_oriented_offset(right, mid_l, offset_m).intersection(strip))
    if lo_line is None or ro_line is None:
        return None
    ring = np.vstack([np.asarray(lo_line.coords), np.asarray(ro_line.coords)[::-1]])
    parts = make_valid_polygonal(Polygon(ring))
    return orient_ccw(max(parts, key=lambda p: p.area)) if parts else None


def punch_holes(band: Polygon, forbidden: BaseGeometry | None, min_hole_m2: float,
                notch_width_m: float) -> tuple[list[Polygon], int]:
    """Subtract forbidden parts >= min_hole_m2; interior holes become notches (geo.ops.notch_holes)."""
    if forbidden is None or forbidden.is_empty or not band.intersects(forbidden):
        return [band], 0
    holes = [h for h in make_valid_polygonal(band.intersection(forbidden)) if h.area >= min_hole_m2]
    if not holes:
        return [band], 0
    out: list[Polygon] = []
    n_notches = 0
    for part in make_valid_polygonal(band.difference(shapely.union_all(holes))):
        if part.interiors:
            n_notches += len(part.interiors)
            out.extend(ops.notch_holes(part, notch_width_m))
        else:
            out.append(part)
    return [orient_ccw(p) for p in out], n_notches


# ------------------------------------------------------------------ pairs and numbering


def geometric_positions(rows: gpd.GeoDataFrame) -> dict[str, int]:
    """row_id -> 1-based position inside its block, ordered by n·c descending (contract §1.5)."""
    pos: dict[str, int] = {}
    for _, block in rows.groupby("vineyard_id", sort=True):
        lines = list(block.geometry)
        angle = mean_axial_angle_deg([angle_deg_utm(g.coords[0], g.coords[-1]) for g in lines])
        cents = np.array([[g.centroid.x, g.centroid.y] for g in lines])
        for p, i in enumerate(order_by_normal(angle, cents)):
            pos[str(block["row_id"].iloc[int(i)])] = p + 1
    return pos


def row_positions(rows: gpd.GeoDataFrame) -> dict[str, int]:
    """row_id -> position in its block: the blocks' `row_index` when complete and unique per block,
    else geometric_positions (so I k always sits between R k and R k+1 of the rows layer)."""
    if "row_index" not in rows.columns:
        return geometric_positions(rows)
    idx = pd.to_numeric(rows["row_index"], errors="coerce")
    if idx.isna().any() or pd.DataFrame({"v": rows["vineyard_id"], "k": idx}).duplicated().any():
        return geometric_positions(rows)
    return {str(r): int(k) for r, k in zip(rows["row_id"], idx, strict=True)}


def consecutive_pairs(rows: gpd.GeoDataFrame) -> pd.DataFrame:
    """row_pairs-like frame (vineyard_id, row_a, row_b) of neighbours (row_positions order) in each block."""
    pos = row_positions(rows)
    recs = []
    for vid, block in rows.groupby("vineyard_id", sort=True):
        ids = sorted((str(r) for r in block["row_id"]), key=pos.__getitem__)
        recs.extend({"vineyard_id": vid, "row_a": a, "row_b": b} for a, b in zip(ids[:-1], ids[1:], strict=True))
    return pd.DataFrame(recs, columns=["vineyard_id", "row_a", "row_b"])


def _numbered_pairs(pairs: pd.DataFrame, pos: dict[str, int]) -> list[tuple[str, str, str, int]]:
    """(vineyard_id, left, right, k) with left = lower geometric position; k unique per block."""
    items = []
    for vid, a, b in zip(pairs["vineyard_id"], pairs["row_a"], pairs["row_b"], strict=True):
        unknown = [str(r) for r in (a, b) if str(r) not in pos]
        if unknown:
            raise ValueError(f"row pair ({a}, {b}) of {vid} references rows missing from the rows layer: {unknown}")
        left, right = (str(a), str(b)) if pos[str(a)] <= pos[str(b)] else (str(b), str(a))
        items.append((str(vid), left, right, pos[left]))
    items.sort(key=lambda it: (it[0], it[3], pos[it[2]]))
    used: dict[str, set[int]] = {}
    out = []
    for vid, left, right, k in items:
        taken = used.setdefault(vid, set())
        kk = k if k not in taken else max(taken) + 1
        taken.add(kk)
        out.append((vid, left, right, kk))
    return out


# ------------------------------------------------------------------ bands (global)


def build_interrow_bands(rows: gpd.GeoDataFrame, pairs: pd.DataFrame, forbidden: BaseGeometry | None,
                         cfg: InterrowConfig, notch_width_m: float) -> gpd.GeoDataFrame:
    """Contract `interrows` (+ k, angle_deg, n_notches): one (Multi)Polygon per neighbour pair."""
    geoms = dict(zip((str(r) for r in rows["row_id"]), rows.geometry, strict=True))
    pos = row_positions(rows)
    recs = []
    for vid, left, right, k in _numbered_pairs(pairs, pos):
        band = band_polygon(geoms[left], geoms[right], cfg.offset_m)
        if band is None:
            continue
        parts, n_notches = punch_holes(band, forbidden, cfg.hole_min_area_m2, notch_width_m)
        if not parts:
            continue
        geom = parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)
        u = _unit(geoms[left])
        length_m = _span_of_poly(band, u)
        recs.append({
            "interrow_id": format_interrow_id(vid, k), "vineyard_id": vid, "row_left_id": left,
            "row_right_id": right, "area_m2": float(geom.area), "width_mean_m": float(geom.area / length_m),
            "length_m": float(length_m), "k": k, "angle_deg": angle_deg_utm((0.0, 0.0), (u[0], u[1])),
            "n_notches": n_notches, "geometry": geom,
        })
    crs = rows.crs
    if not recs:
        return gpd.GeoDataFrame({c: [] for c in INTERROW_COLUMNS}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=crs).sort_values("interrow_id", kind="stable") \
        .reset_index(drop=True)


def _span_of_poly(poly: BaseGeometry, u: np.ndarray) -> float:
    t = shapely.get_coordinates(poly) @ u
    return float(t.max() - t.min()) if len(t) else 0.0


# ------------------------------------------------------------------ per-tile pieces and cover


def cover_stats(veg: BoolMask, vis: U8, piece: Polygon, tile: TileRef) -> CoverStats:
    """veg_frac over visible-data pixels and shadow_frac (vis ∈ {shadow, overexposed}) of the piece."""
    m, (x0, y0) = _piece_mask(piece, tile)
    h, w = m.shape
    vis_c = vis[y0 : y0 + h, x0 : x0 + w][m]
    veg_c = veg[y0 : y0 + h, x0 : x0 + w][m]
    data = vis_c != VIS_NODATA
    n_data = int(data.sum())
    veg_frac = float(veg_c[data].mean()) if n_data else math.nan
    shadow = float(np.isin(vis_c[data], _SHADOW_CODES).mean()) if n_data else math.nan
    return CoverStats(veg_frac=veg_frac, shadow_frac=shadow, n_px=int(m.sum()))


def _piece_mask(piece: Polygon, tile: TileRef) -> tuple[np.ndarray, tuple[int, int]]:
    px = shapely.transform(piece, lambda xy: utm_to_px(tile, xy))
    minx, miny, maxx, maxy = px.bounds
    x0, y0 = max(int(math.floor(minx)), 0), max(int(math.floor(miny)), 0)
    x1, y1 = min(int(math.ceil(maxx)), TILE_PX), min(int(math.ceil(maxy)), TILE_PX)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), dtype=bool), (x0, y0)
    local = affinity.translate(px, -x0, -y0)
    m = rasterio.features.rasterize([(local, 1)], out_shape=(y1 - y0, x1 - x0), transform=Affine.identity(),
                                    fill=0, dtype=np.uint8, all_touched=False)
    return m > 0, (x0, y0)


def classify_cover(s: CoverStats, width_mean_m: float, cfg: InterrowConfig) -> tuple[InterrowCover, tuple[str, ...]]:
    """unassessable (shadow / narrow / no data), bare_soil, mixed or vegetation; plus borderline flag."""
    if s.n_px == 0 or math.isnan(s.veg_frac) or s.shadow_frac > cfg.shadow_unassessable \
            or width_mean_m < cfg.min_width_assessable_m:
        return InterrowCover.UNASSESSABLE, ()
    near = min(abs(s.veg_frac - cfg.cover_bare_max), abs(s.veg_frac - cfg.cover_veg_min))
    flags = (BORDERLINE_FLAG,) if near <= cfg.cover_borderline_margin + _FRAC_EPS else ()
    if s.veg_frac < cfg.cover_bare_max:
        return InterrowCover.BARE_SOIL, flags
    if s.veg_frac > cfg.cover_veg_min:
        return InterrowCover.VEGETATION, flags
    return InterrowCover.MIXED, flags


def _sorted_parts(geom: BaseGeometry, clip: BaseGeometry, u: np.ndarray, min_piece_m2: float) -> list[Polygon]:
    parts = [p for p in make_valid_polygonal(geom.intersection(clip)) if p.area >= min_piece_m2]
    return sorted((orient_ccw(p) for p in parts), key=lambda p: (float(np.asarray(p.centroid.coords)[0] @ u),
                                                                 -p.area))


def cut_interrows_to_tile(interrows: gpd.GeoDataFrame, tile: TileRef, clip: BaseGeometry,
                          min_piece_m2: float) -> gpd.GeoDataFrame:
    """Per-tile pieces: band ∩ clip -> explode -> >= min_piece_m2; ids <interrow>@<tile>[#k] (k along)."""
    recs = []
    cand = interrows[interrows.intersects(clip)] if len(interrows) else interrows
    for rec in cand.itertuples(index=False):
        u = _direction(float(rec.angle_deg))
        for j, part in enumerate(_sorted_parts(rec.geometry, clip, u, min_piece_m2)):
            span = _span_of_poly(part, u)
            recs.append({
                "piece_id": format_interrow_piece_id(rec.interrow_id, tile.tile_id, j + 1),
                "interrow_id": rec.interrow_id, "vineyard_id": rec.vineyard_id, "tile_id": tile.tile_id,
                "row_left_id": rec.row_left_id, "row_right_id": rec.row_right_id,
                "area_m2": float(part.area), "width_mean_m": float(part.area / span) if span > 0 else 0.0,
                "n_notches": int(rec.n_notches), "geometry": part,
            })
    return _pieces_frame(recs, interrows.crs)


def _direction(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])


def _pieces_frame(recs: Sequence[dict[str, object]], crs: object) -> gpd.GeoDataFrame:
    base = [c for c in PIECE_COLUMNS if c not in ("interrow_cover", "veg_frac", "shadow_frac", "qa_flags")]
    if not recs:
        return gpd.GeoDataFrame({c: [] for c in base}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    return gpd.GeoDataFrame(list(recs), geometry="geometry", crs=crs).sort_values("piece_id", kind="stable") \
        .reset_index(drop=True)


def classify_pieces(pieces: gpd.GeoDataFrame, veg: BoolMask, vis: U8, tile: TileRef,
                    cfg: InterrowConfig) -> gpd.GeoDataFrame:
    """New frame with interrow_cover, veg_frac, shadow_frac and qa_flags for each piece."""
    covers, vfs, sfs, flags = [], [], [], []
    for poly, width in zip(pieces.geometry, pieces["width_mean_m"], strict=True):
        s = cover_stats(veg, vis, poly, tile)
        cover, fl = classify_cover(s, float(width), cfg)
        covers.append(cover.value)
        vfs.append(s.veg_frac)
        sfs.append(s.shadow_frac)
        flags.append(";".join(fl))
    out = pieces.assign(interrow_cover=covers, veg_frac=vfs, shadow_frac=sfs, qa_flags=flags)
    return out[[*PIECE_COLUMNS, out.geometry.name]]
