"""Raster helpers: tile reading, nodata mask, mask <-> polygon, rasterization, resizing, PNG I/O.

Pixel conventions (contract §1.2/§1.4 + plan erratum R1):
- "continuous": a pixel is set when its centre (i+0.5, j+0.5) lies inside the polygon given in CVAT
  (u, v) coordinates (half-open, so adjacent polygons partition the pixels). Rows, interrows.
- "index": vertices are pixel indices and the boundary pixels are included. It is the exact inverse
  of raw cv2.findContours output, which is how the reference canopies are drawn.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

import cv2
import numpy as np
import rasterio
import rasterio.features
import shapely
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.errors import IngestError
from vineyard.geo.ops import make_valid_polygonal, orient_ccw, split_multi
from vineyard.geo.tiling import GSD_M, TILE_PX, TileRef, px_to_utm, tile_box, utm_to_px
from vineyard.pipeline.atomic import atomic_write_bytes

Convention = Literal["continuous", "index"]
RGB_BANDS: Final = 3
_U8_MAX: Final = 255
_TILE_SHAPE: Final = (TILE_PX, TILE_PX)
_PNG_BILEVEL: Final = [cv2.IMWRITE_PNG_BILEVEL, 1]
_VALID_OFFSET_PX: Final = 0.5  # valid polygon follows pixel squares, so a full tile is the tile box
_MITRE: Final = "mitre"


# ------------------------------------------------------------------ tiles


def read_tile(path: str | Path) -> np.ndarray:
    """Read a tile as an (H, W, 3) uint8 RGB C-contiguous array (GDAL decodes the JPEG/YCbCr)."""
    source = Path(path)
    with rasterio.open(source) as src:
        if src.count < RGB_BANDS:
            raise IngestError(f"{source.name}: expected 3 bands, found {src.count}")
        if any(dt != "uint8" for dt in src.dtypes[:RGB_BANDS]):
            raise IngestError(f"{source.name}: expected uint8 bands, found {src.dtypes}")
        bands = src.read([1, 2, 3])
    return np.ascontiguousarray(np.moveaxis(bands, 0, -1))


def _require_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != RGB_BANDS or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8 RGB, got {rgb.shape} {rgb.dtype}")


def _disk(radius_px: int) -> np.ndarray:
    size = 2 * radius_px + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _border_components(nd: np.ndarray, min_area_px: float) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(nd, connectivity=8)
    h, w = nd.shape
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    touches = (x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)
    keep = touches & (stats[:, cv2.CC_STAT_AREA] >= min_area_px)
    keep[0] = False  # label 0 is the background of `nd`
    return keep[labels] if n > 1 else np.zeros_like(nd, dtype=bool)


def valid_mask(rgb: np.ndarray, *, max_rgb: int, min_area_px: float, close_px: int, dilate_px: int) -> np.ndarray:
    """True = image. Nodata = near-black components touching the border (contract §1.7).

    close_px and dilate_px are radii of elliptical structuring elements.
    """
    _require_rgb(rgb)
    dark = (rgb.max(axis=2) <= max_rgb).astype(np.uint8)
    nodata = _border_components(dark, min_area_px).astype(np.uint8)
    if close_px > 0:
        nodata = cv2.morphologyEx(nodata, cv2.MORPH_CLOSE, _disk(close_px))
    if dilate_px > 0:
        nodata = cv2.dilate(nodata, _disk(dilate_px))
    return nodata == 0


def valid_polygon(valid: np.ndarray, t: TileRef, *, approx_eps_px: float) -> MultiPolygon:
    """Vector valid area in UTM, following pixel squares and clipped to the tile box."""
    polys = mask_to_polygons(
        valid, t, approx_eps_px=approx_eps_px, min_area_px=0.0, pixel_offset=_VALID_OFFSET_PX,
        outset_px=_VALID_OFFSET_PX,
    )
    clipped = shapely.intersection(MultiPolygon(polys), tile_box(t)) if polys else MultiPolygon()
    return MultiPolygon([orient_ccw(p) for p in make_valid_polygonal(clipped)])


# ------------------------------------------------------------------ mask -> polygons


def _as_u8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    return (np.asarray(mask) > 0).astype(np.uint8)


def _ring_px(contour: np.ndarray, eps: float, offset: float) -> np.ndarray:
    ring = cv2.approxPolyDP(contour, eps, True) if eps > 0 else contour
    return ring.reshape(-1, 2).astype(np.float64) + offset


def _outset(geom: BaseGeometry, outset_px: float) -> BaseGeometry:
    # keep_collapsed: 1-px lines/points must still grow into their pixel squares
    base = geom if geom.is_valid else shapely.make_valid(geom, method="structure", keep_collapsed=True)
    return base.buffer(outset_px, cap_style="square", join_style=_MITRE)


def _contour_shape(shell: np.ndarray, holes: list[np.ndarray], outset_px: float) -> BaseGeometry | None:
    if len(shell) < 3:
        if outset_px <= 0:
            return None  # a point or a segment has no area in the raw convention
        base = Point(shell[0]) if len(np.unique(shell, axis=0)) == 1 else LineString(shell)
        return _outset(base, outset_px)
    poly = Polygon(shell, [h for h in holes if len(h) >= 3])
    return _outset(poly, outset_px) if outset_px > 0 else poly


def _contour_shapes(
    contours: Sequence[np.ndarray], hierarchy: np.ndarray, eps: float, offset: float, outset_px: float
) -> list[BaseGeometry]:
    holes_of: dict[int, list[np.ndarray]] = {}
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent >= 0:
            holes_of.setdefault(int(parent), []).append(_ring_px(contours[i], eps, offset))
    shapes: list[BaseGeometry] = []
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent < 0:
            shape = _contour_shape(_ring_px(contours[i], eps, offset), holes_of.get(i, []), outset_px)
            if shape is not None and not shape.is_empty:
                shapes.append(shape)
    return shapes


def mask_to_polygons(
    mask: np.ndarray,
    t: TileRef,
    *,
    approx_eps_px: float,
    min_area_px: float,
    pixel_offset: float = 0.0,
    outset_px: float = 0.0,
) -> list[Polygon]:
    """Vectorize a mask to valid CCW UTM polygons with holes.

    findContours(RETR_CCOMP) -> approxPolyDP(eps) -> +pixel_offset -> buffer(outset_px) -> UTM ->
    make_valid(structure). Defaults (0, 0) = raw contour convention (reference canopies, R1).
    Parts whose vector area is below `min_area_px` (px²) are dropped.
    """
    m = _as_u8_mask(mask)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    shapes = _contour_shapes(contours, hierarchy[0], approx_eps_px, pixel_offset, outset_px)
    min_area_m2 = min_area_px * GSD_M * GSD_M
    polys: list[Polygon] = []
    for shape in shapes:
        utm = shapely.transform(shape, lambda uv: px_to_utm(t, uv))
        polys.extend(orient_ccw(p) for p in make_valid_polygonal(utm) if p.area >= min_area_m2)
    return polys


# ------------------------------------------------------------------ rasterization


def _out_dtype(value: int) -> type[np.integer]:
    return np.uint8 if 0 <= value <= _U8_MAX else np.int32


def _burn_continuous(geoms: Sequence[BaseGeometry], shape: tuple[int, int], value: int) -> np.ndarray:
    dtype = _out_dtype(value)
    shapes = [(g, value) for g in geoms if not g.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=dtype)
    return rasterio.features.rasterize(
        shapes, out_shape=shape, transform=Affine.identity(), fill=0, dtype=dtype, all_touched=False
    )


def _index_ring(ring: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(ring, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)


def _burn_holed(out: np.ndarray, shell: np.ndarray, holes: list[np.ndarray], value: int) -> None:
    h, w = out.shape
    lo = np.maximum(shell.reshape(-1, 2).min(axis=0), 0)
    hi = np.minimum(shell.reshape(-1, 2).max(axis=0) + 1, (w, h))
    if (hi <= lo).any():
        return
    scratch = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), dtype=np.uint8)
    local_holes = [ring - lo for ring in holes]
    cv2.fillPoly(scratch, [shell - lo], 1)
    cv2.fillPoly(scratch, local_holes, 0)
    cv2.polylines(scratch, local_holes, True, 1)  # hole rings run through foreground pixels
    out[lo[1] : hi[1], lo[0] : hi[0]][scratch > 0] = value


def _burn_index(polys: Sequence[Polygon], shape: tuple[int, int], value: int) -> np.ndarray:
    out = np.zeros(shape, dtype=_out_dtype(value))
    for poly in polys:
        shell = _index_ring(np.asarray(poly.exterior.coords))
        if poly.interiors:
            _burn_holed(out, shell, [_index_ring(np.asarray(r.coords)) for r in poly.interiors], value)
        else:
            cv2.fillPoly(out, [shell], value)
    return out


def _check_convention(convention: str) -> None:
    if convention not in ("continuous", "index"):
        raise ValueError(f"unknown pixel convention {convention!r}")


def rasterize_px(
    polys_uv: Sequence[np.ndarray],
    shape: tuple[int, int] = _TILE_SHAPE,
    value: int = 1,
    *,
    convention: Convention = "continuous",
) -> np.ndarray:
    """Burn (N,2) rings given in pixel coordinates; each ring is filled on its own (no holes).

    Output is uint8 (int32 when value > 255). Use rasterize_utm for polygons with holes.
    """
    _check_convention(convention)
    rings = [np.asarray(r, dtype=np.float64).reshape(-1, 2) for r in polys_uv]
    if convention == "index":
        out = np.zeros(shape, dtype=_out_dtype(value))
        for ring in rings:
            if len(ring):
                cv2.fillPoly(out, [_index_ring(ring)], value)
        return out
    return _burn_continuous([Polygon(r) for r in rings if len(r) >= 3], shape, value)


def _polygonal_parts(geom: BaseGeometry) -> list[Polygon]:
    parts = split_multi(geom)
    for part in parts:
        if not isinstance(part, Polygon):
            raise TypeError(f"rasterize_utm accepts polygonal geometries only, got {part.geom_type}")
    return parts  # type: ignore[return-value]


def rasterize_utm(
    geoms: Sequence[BaseGeometry],
    t: TileRef,
    shape: tuple[int, int] = _TILE_SHAPE,
    *,
    convention: Convention = "continuous",
) -> np.ndarray:
    """Burn UTM (Multi)Polygons of tile `t` into a uint8 0/1 mask (holes respected)."""
    _check_convention(convention)
    px_polys = [shapely.transform(p, lambda xy: utm_to_px(t, xy)) for g in geoms for p in _polygonal_parts(g)]
    if convention == "index":
        return _burn_index(px_polys, shape, 1)
    return _burn_continuous(px_polys, shape, 1)


# ------------------------------------------------------------------ resizing


def resize_aligned(arr: np.ndarray, out_px: int, *, mode: Literal["down", "up"]) -> np.ndarray:
    """Square resize keeping pixel edges aligned: INTER_AREA down, INTER_LINEAR up.

    Bool input is resized as float32 fractions.
    """
    h, w = arr.shape[:2]
    if h != w:
        raise ValueError(f"resize_aligned expects a square array, got {arr.shape}")
    if (mode == "down" and out_px > h) or (mode == "up" and out_px < h):
        raise ValueError(f"mode={mode!r} is inconsistent with {h} -> {out_px} px")
    src = arr.astype(np.float32) if arr.dtype == np.bool_ else arr
    interp = cv2.INTER_AREA if mode == "down" else cv2.INTER_LINEAR
    return cv2.resize(src, (out_px, out_px), interpolation=interp)


# ------------------------------------------------------------------ PNG I/O


def _encode_png(img: np.ndarray, params: list[int], path: Path) -> bytes:
    ok, buf = cv2.imencode(".png", img, params)
    if not ok:
        raise ValueError(f"PNG encoding failed for {path}")
    return buf.tobytes()


def _decode_png(path: Path) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PNG not found: {source}")
    img = cv2.imdecode(np.frombuffer(source.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 2:
        raise ValueError(f"{source.name}: not a single-channel PNG")
    return img


def write_mask_png(path: Path, mask: np.ndarray) -> Path:
    """Write a bool mask as a 1-bit PNG, atomically."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    img = np.where(mask, np.uint8(_U8_MAX), np.uint8(0))
    return atomic_write_bytes(Path(path), _encode_png(img, _PNG_BILEVEL, Path(path)))


def read_mask_png(path: Path) -> np.ndarray:
    """Read a mask PNG as bool (non-zero = True)."""
    return _decode_png(path) > 0


def write_u8_png(path: Path, arr: np.ndarray) -> Path:
    """Write a 2-D uint8 array (e.g. vis codes 0-3) as a grayscale PNG, atomically."""
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError(f"expected a 2-D uint8 array, got {arr.shape} {arr.dtype}")
    return atomic_write_bytes(Path(path), _encode_png(arr, [], Path(path)))


def read_u8_png(path: Path) -> np.ndarray:
    """Read a single-channel 8-bit PNG."""
    img = _decode_png(path)
    if img.dtype != np.uint8:
        raise ValueError(f"{Path(path).name}: expected 8-bit PNG, got {img.dtype}")
    return img
