"""Mask/label rasters -> UTM polygons (S1: raw findContours convention by default).

The low-level vectorizers live in geo.raster (frozen) and are re-exported here; this module adds
per-label vectorization on bounding-box crops, optional per-label vector clipping and a minimum
area measured on the final vector area.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.geo.ops import clip_polygonal
from vineyard.geo.raster import mask_to_polygons, valid_polygon
from vineyard.geo.tiling import GSD_M, TileRef

__all__ = ["mask_to_polygons", "polygonize_labels", "polygonize_mask", "valid_polygon"]

_CROP_PAD_PX: Final = 1  # a zero border keeps findContours identical to the full-raster result
_MASK_LABEL: Final = 1


def _check_labels(labels: np.ndarray) -> np.ndarray:
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {lab.shape}")
    if lab.dtype == np.bool_:
        return lab.astype(np.uint8)
    if not np.issubdtype(lab.dtype, np.integer):
        raise ValueError(f"labels must have an integer or bool dtype, got {lab.dtype}")
    if lab.size and lab.min() < 0:
        raise ValueError(f"labels must be non-negative (0 = background), got min {lab.min()}")
    return lab


def _padded(window: tuple[slice, slice], shape: tuple[int, ...]) -> tuple[slice, slice]:
    rows, cols = window
    return (
        slice(max(rows.start - _CROP_PAD_PX, 0), min(rows.stop + _CROP_PAD_PX, shape[0])),
        slice(max(cols.start - _CROP_PAD_PX, 0), min(cols.stop + _CROP_PAD_PX, shape[1])),
    )


def _crop_ref(t: TileRef, window: tuple[slice, slice]) -> TileRef:
    rows, cols = window
    return TileRef(t.tile_id, t.grid_row, t.grid_col, t.x0 + GSD_M * cols.start, t.y0 - GSD_M * rows.start)


def _label_polygons(
    mask: np.ndarray, t: TileRef, eps_px: float, offset_px: float, outset_px: float
) -> list[Polygon]:
    return mask_to_polygons(
        mask, t, approx_eps_px=eps_px, min_area_px=0.0, pixel_offset=offset_px, outset_px=outset_px
    )


def _finish(polys: list[Polygon], clip: BaseGeometry | None, min_area_m2: float) -> list[Polygon]:
    parts = polys if clip is None else [q for p in polys for q in clip_polygonal(p, clip)]
    return [p for p in parts if p.area >= min_area_m2]


def _check_params(min_area_m2: float, eps_px: float, outset_px: float) -> None:
    for name, value in (("min_area_m2", min_area_m2), ("eps_px", eps_px), ("outset_px", outset_px)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite number >= 0, got {value}")


def polygonize_labels(
    labels: np.ndarray,
    t: TileRef,
    *,
    min_area_m2: float,
    eps_px: float,
    offset_px: float = 0.0,
    outset_px: float = 0.0,
    clips: Mapping[int, BaseGeometry] | None = None,
) -> list[tuple[int, Polygon]]:
    """Vectorize every label > 0 of a tile raster into valid CCW UTM polygons (holes kept).

    Per label: findContours(RETR_CCOMP) -> approxPolyDP(eps_px) -> +offset_px -> outset_px ->
    UTM (geo.raster.mask_to_polygons on a padded bbox crop). With `clips`, each label's polygons are
    intersected with clips[label] (every label must have one). Parts whose final vector area is
    below `min_area_m2` are dropped. Output is sorted by label, then in contour order.
    """
    _check_params(min_area_m2, eps_px, outset_px)
    lab = _check_labels(labels)
    out: list[tuple[int, Polygon]] = []
    for index, window in enumerate(ndimage.find_objects(lab) if lab.size else []):
        if window is None:
            continue
        value = index + 1
        if clips is not None and value not in clips:
            raise ValueError(f"no clip geometry for label {value}")
        crop = _padded(window, lab.shape)
        polys = _label_polygons(lab[crop] == value, _crop_ref(t, crop), eps_px, offset_px, outset_px)
        clip = clips[value] if clips is not None else None
        out.extend((value, p) for p in _finish(polys, clip, min_area_m2))
    return out


def polygonize_mask(
    mask: np.ndarray,
    t: TileRef,
    *,
    min_area_m2: float,
    eps_px: float,
    offset_px: float = 0.0,
    outset_px: float = 0.0,
    clip: BaseGeometry | None = None,
) -> list[Polygon]:
    """Single-mask variant of polygonize_labels (non-zero = foreground), optional vector clip."""
    lab = (_check_labels(mask) > 0).astype(np.uint8)
    clips = None if clip is None else {_MASK_LABEL: clip}
    found = polygonize_labels(
        lab, t, min_area_m2=min_area_m2, eps_px=eps_px, offset_px=offset_px, outset_px=outset_px, clips=clips
    )
    return [p for _, p in found]
