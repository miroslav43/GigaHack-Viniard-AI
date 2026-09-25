"""Row corridors (02 §2.2, §3.7): per-tile row pieces, flat-cap corridor polygons, corridor label raster.

Consumers (canopy, waste, derive, NN pseudo-labels) code against these signatures.
"""

from __future__ import annotations

from typing import Final

import geopandas as gpd
import numpy as np
import rasterio.features
import shapely
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from vineyard.contracts.ids import format_row_piece_id
from vineyard.contracts.schema_defs import MIN_LINE_LENGTH_M
from vineyard.geo.ops import make_valid_polygonal, split_multi
from vineyard.geo.tiling import TILE_PX, TileRef, utm_to_px
from vineyard.perception.types import I32

PIECE_COLUMNS: Final = ("tile_id", "piece_id", "along_from_m", "along_to_m")
_JOIN: Final = "mitre"
_CAP: Final = "flat"
_LABEL_DTYPE: Final = np.int32
# polygon shift (px) so that rasterio's pixel-centre rule tests the wanted point of each pixel
_SHIFT_PX: Final = {"continuous": 0.0, "index": 0.5}


def _grown_clip(clip: BaseGeometry, margin_m: float) -> BaseGeometry:
    if margin_m < 0:
        raise ValueError(f"margin_m must be >= 0, got {margin_m}")
    return clip.buffer(margin_m, join_style=_JOIN) if margin_m > 0 else clip


def _along_span(line: LineString, inter: BaseGeometry) -> tuple[float, float] | None:
    parts = [p for p in split_multi(inter) if isinstance(p, (LineString, MultiLineString, Point))]
    coords = [c for p in parts for c in np.asarray(shapely.get_coordinates(p))]
    if not coords:
        return None
    along = shapely.line_locate_point(line, shapely.points(np.asarray(coords)))
    lo, hi = float(np.min(along)), float(np.max(along))
    return (lo, hi) if hi - lo >= MIN_LINE_LENGTH_M else None


def _require_line(geom: BaseGeometry, row_id: object) -> LineString:
    if not isinstance(geom, LineString):
        raise TypeError(f"row {row_id!r}: expected a LineString, got {getattr(geom, 'geom_type', geom)!r}")
    return geom


def _piece_ids(row_ids: list[str], tile_id: str) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for rid in row_ids:
        seen[rid] = seen.get(rid, 0) + 1
        out.append(format_row_piece_id(rid, tile_id, seen[rid]))
    return out


def clip_rows_to_tile(
    rows: gpd.GeoDataFrame, tile: TileRef, clip: BaseGeometry, margin_m: float = 0.0
) -> gpd.GeoDataFrame:
    """One polyline per row per tile (first to last valid vertex), clipped to `clip` grown by margin_m.

    Returns a new frame with the input columns plus tile_id, piece_id (<row_id>@<tile>[#k]),
    along_from_m / along_to_m (piece position on the full row); rows that miss the tile are dropped.
    """
    grown = _grown_clip(clip, margin_m)
    keep: list[int] = []
    geoms: list[LineString] = []
    spans: list[tuple[float, float]] = []
    for i, (rid, geom) in enumerate(zip(rows.get("row_id", rows.index), rows.geometry, strict=True)):
        line = _require_line(geom, rid)
        span = _along_span(line, line.intersection(grown)) if line.intersects(grown) else None
        if span is None:
            continue
        keep.append(i)
        geoms.append(substring(line, span[0], span[1]))
        spans.append(span)
    out = rows.iloc[keep].reset_index(drop=True)
    row_ids = [str(r) for r in out["row_id"]] if "row_id" in out.columns else [str(i) for i in keep]
    return gpd.GeoDataFrame(
        out.drop(columns=out.geometry.name).assign(
            tile_id=tile.tile_id,
            piece_id=_piece_ids(row_ids, tile.tile_id),
            along_from_m=[s[0] for s in spans],
            along_to_m=[s[1] for s in spans],
        ),
        geometry=gpd.GeoSeries(geoms, crs=rows.crs),
        crs=rows.crs,
    )


def _end_direction(xy: np.ndarray) -> np.ndarray | None:
    """Unit vector pointing out of the line at xy[0], from the first distinct following vertex."""
    for other in xy[1:]:
        d = xy[0] - other
        norm = float(np.hypot(*d))
        if norm > 0:
            return d / norm
    return None


def _extend_line(line: LineString, at_head: bool, at_tail: bool, extend_m: float) -> LineString:
    xy = np.asarray(line.coords, dtype=float)
    head, tail = _end_direction(xy), _end_direction(xy[::-1])
    out = xy.copy()
    # the end vertex moves along its own segment, so no collinear vertex is added
    if at_head and head is not None:
        out[0] = xy[0] + head * extend_m
    if at_tail and tail is not None:
        out[-1] = xy[-1] + tail * extend_m
    return LineString(out)


def extend_censored_ends(rows: gpd.GeoDataFrame, boundary: BaseGeometry | None, *, extend_m: float,
                         tol_m: float) -> gpd.GeoDataFrame:
    """New frame whose row ends within tol_m of `boundary` (where the data stops, not the row) are
    extended by extend_m along their end segment, so bands and corridors reach the clip edge."""
    if extend_m < 0:
        raise ValueError(f"extend_m must be >= 0, got {extend_m}")
    if tol_m < 0:
        raise ValueError(f"tol_m must be >= 0, got {tol_m}")
    if extend_m == 0 or boundary is None or boundary.is_empty or rows.empty:
        return rows.copy()
    ids = rows["row_id"] if "row_id" in rows.columns else rows.index
    lines = [_require_line(g, rid) for rid, g in zip(ids, rows.geometry, strict=True)]
    heads = shapely.dwithin(boundary, shapely.points([ln.coords[0] for ln in lines]), tol_m)
    tails = shapely.dwithin(boundary, shapely.points([ln.coords[-1] for ln in lines]), tol_m)
    geoms = [_extend_line(ln, bool(h), bool(t), extend_m) for ln, h, t in zip(lines, heads, tails, strict=True)]
    return rows.set_geometry(gpd.GeoSeries(geoms, index=rows.index, crs=rows.crs))


def corridor_polygon(axis: LineString, half_m: float) -> Polygon:
    """Flat-capped corridor of half-width `half_m` around `axis`."""
    if not half_m > 0:
        raise ValueError(f"half_m must be > 0, got {half_m}")
    if axis.length <= 0:
        raise ValueError(f"axis has zero length: {axis.wkt[:80]}")
    parts = make_valid_polygonal(axis.buffer(half_m, cap_style=_CAP, join_style=_JOIN))
    if not parts:
        raise ValueError(f"empty corridor for axis {axis.wkt[:80]}")
    return max(parts, key=lambda p: p.area)


def corridor_px(axis: LineString, tile: TileRef, half_m: float) -> Polygon:
    """Corridor polygon of `axis` (UTM) in CVAT continuous pixel coordinates of `tile`."""
    return shapely.transform(corridor_polygon(axis, half_m), lambda xy: utm_to_px(tile, xy))


def corridor_labels(pieces: gpd.GeoDataFrame, tile: TileRef, half_m: float) -> I32:
    """int32 raster (tile px): label k+1 inside the corridor of piece k, 0 elsewhere.

    Continuous convention (pixel centre inside, same as geo.raster.rasterize_px); on overlaps the
    later piece wins.
    """
    return corridor_label_raster(pieces, tile, half_m, convention="continuous")


def corridor_label_raster(pieces: gpd.GeoDataFrame, tile: TileRef, half_m: float, *, convention: str) -> I32:
    """corridor_labels with an explicit pixel convention.

    "continuous": pixel (i, j) is labelled when its centre (i + 0.5, j + 0.5) is inside the corridor.
    "index": when its index point (i, j) is inside, i.e. where a raw findContours vertex of that pixel
    lands, so raw-convention canopy vertices never leave the corridor (plan S1).
    """
    if convention not in _SHIFT_PX:
        raise ValueError(f"unknown pixel convention {convention!r}; expected one of {sorted(_SHIFT_PX)}")
    shift = _SHIFT_PX[convention]
    shapes = [
        (shapely.transform(corridor_px(axis, tile, half_m), lambda uv: uv + shift), k + 1)
        for k, axis in enumerate(pieces.geometry)
        if axis is not None and not axis.is_empty and axis.length > 0
    ]
    if not shapes:
        return np.zeros((TILE_PX, TILE_PX), dtype=_LABEL_DTYPE)
    return rasterio.features.rasterize(
        shapes, out_shape=(TILE_PX, TILE_PX), transform=Affine.identity(), fill=0, dtype=_LABEL_DTYPE,
        all_touched=False,
    )
