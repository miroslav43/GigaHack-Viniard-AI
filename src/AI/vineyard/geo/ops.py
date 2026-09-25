"""Unit-agnostic geometry operations (work the same in pixels or metres).

P0 implements orientation, validity and flattening helpers; the clipping and hole-notching
functions are frozen signatures implemented in P1 (01 §2.2, §3.7-§3.8).
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient

_CCW = 1.0


def orient_ccw(poly: Polygon) -> Polygon:
    """New polygon with a CCW exterior and CW holes (contract §1.5)."""
    return orient(poly, _CCW)


def split_multi(geom: BaseGeometry) -> list[BaseGeometry]:
    """Flatten Multi*/GeometryCollection (recursively) into non-empty single parts."""
    if geom.is_empty:
        return []
    if isinstance(geom, BaseMultipartGeometry):
        return [part for sub in geom.geoms for part in split_multi(sub)]
    return [geom]


def make_valid_polygonal(geom: BaseGeometry) -> list[Polygon]:
    """make_valid(method="structure") and keep only the non-empty polygonal parts."""
    fixed = geom if shapely.is_valid(geom) else shapely.make_valid(geom, method="structure", keep_collapsed=False)
    return [p for p in split_multi(fixed) if isinstance(p, Polygon) and p.area > 0.0]


def drop_consecutive_duplicates(coords: np.ndarray, *, closed: bool) -> np.ndarray:
    """Remove vertices equal to their predecessor.

    closed=True treats `coords` as a ring: a trailing vertex equal to the first one is removed as
    well, and the result never repeats the first vertex at the end.
    """
    pts = np.asarray(coords)
    if pts.ndim != 2:
        raise ValueError(f"coords must be (N, D), got shape {pts.shape}")
    if len(pts) == 0:
        return pts.copy()
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
    out = pts[keep]
    if closed and len(out) > 1 and np.array_equal(out[-1], out[0]):
        out = out[:-1]
    return out


# ---------------------------------------------------------------- P1 (signatures frozen)


def clip_polygonal(geom: BaseGeometry, clip: BaseGeometry) -> list[Polygon]:
    """Intersection restricted to polygonal parts, exploded (contract §1.6)."""
    raise NotImplementedError("geo.ops.clip_polygonal is implemented in P1-GEO")


def clip_line(line: LineString, clip: BaseGeometry) -> LineString | None:
    """One polyline per row per tile: first to last valid vertex along the axis (contract §1.6)."""
    raise NotImplementedError("geo.ops.clip_line is implemented in P1-GEO")


def clip_box(
    xyxy: tuple[float, float, float, float], bounds: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    """Clip an axis-aligned box to bounds; None when nothing is left."""
    raise NotImplementedError("geo.ops.clip_box is implemented in P1-GEO")


def notch_holes(poly: Polygon, notch_width: float) -> list[Polygon]:
    """Cut each hole open towards the exterior with a notch of `notch_width` (01 §3.8)."""
    raise NotImplementedError("geo.ops.notch_holes is implemented in P1-GEO")
