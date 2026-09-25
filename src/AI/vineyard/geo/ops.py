"""Unit-agnostic geometry operations (work the same in pixels or metres).

Orientation, validity and flattening helpers, plus the clipping and hole-notching operations of
01 §2.2 / §3.7-§3.8 and contract §1.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient
from shapely.ops import substring

_CCW = 1.0
_REL_TOL: Final = 1e-9  # relative to the clip extent: absorbs float noise on shared boundaries
_CONTACT_REL_TOL: Final = 1e-3  # of notch_width: hole points this close to the minimum distance tie
_MAX_NOTCH_PASSES: Final = 3
_BOX_LEN: Final = 4


def _valid(geom: BaseGeometry) -> BaseGeometry:
    if shapely.is_valid(geom):
        return geom
    return shapely.make_valid(geom, method="structure", keep_collapsed=False)


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
    return [p for p in split_multi(_valid(geom)) if isinstance(p, Polygon) and p.area > 0.0]


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


# ---------------------------------------------------------------- polygons


def clip_polygonal(geom: BaseGeometry, clip: BaseGeometry) -> list[Polygon]:
    """Intersection restricted to polygonal parts, exploded, valid and CCW (contract §1.6).

    Invalid inputs are repaired with make_valid(structure) first; holes are kept.
    """
    if geom.is_empty or clip.is_empty:
        return []
    inter = shapely.intersection(_valid(geom), _valid(clip))
    return [orient_ccw(p) for p in make_valid_polygonal(inter)]


# ---------------------------------------------------------------- lines


@dataclass(frozen=True)
class _Run:
    start: float  # distance along the line
    end: float
    covered: float  # length of the line inside the clip within [start, end]


def _covered_intervals(line: LineString, clip: BaseGeometry, tol: float) -> list[tuple[float, float]]:
    parts = [p for p in split_multi(line.intersection(clip)) if isinstance(p, LineString) and p.length > 0]
    spans = sorted(
        (min(d0, d1), max(d0, d1))
        for d0, d1 in ((line.project(Point(p.coords[0])), line.project(Point(p.coords[-1]))) for p in parts)
    )
    merged: list[tuple[float, float]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1] + tol:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return [(lo, hi) for lo, hi in merged if hi - lo > tol]


def _runs(line: LineString, intervals: list[tuple[float, float]], hull: BaseGeometry) -> list[_Run]:
    runs = [_Run(*intervals[0], intervals[0][1] - intervals[0][0])]
    for lo, hi in intervals[1:]:
        last = runs[-1]
        if hull.covers(substring(line, last.end, lo)):
            runs[-1] = _Run(last.start, hi, last.covered + hi - lo)
        else:
            runs.append(_Run(lo, hi, hi - lo))
    return runs


def _line_or_none(geom: BaseGeometry) -> LineString | None:
    if not isinstance(geom, LineString) or geom.is_empty:
        return None
    pts = drop_consecutive_duplicates(np.asarray(geom.coords), closed=False)
    return LineString(pts) if len(pts) >= 2 else None


def clip_line(line: LineString, clip: BaseGeometry) -> LineString | None:
    """One polyline per row per tile (contract §1.6), in the direction of `line`.

    The line is cut to `clip`; when the clip splits it (nodata inside the tile) the result runs from
    the first to the last point inside the clip along the line, bridging the gaps that stay inside
    the clip's convex hull. A line that leaves the hull and comes back keeps its longest run.
    """
    if not isinstance(line, LineString):
        raise TypeError(f"clip_line expects a LineString, got {line.geom_type}")
    if line.is_empty or clip.is_empty or line.length <= 0:
        return None
    area = _valid(clip)
    minx, miny, maxx, maxy = area.bounds
    tol = _REL_TOL * max(maxx - minx, maxy - miny, line.length)
    intervals = _covered_intervals(line, area, tol)
    if not intervals:
        return None
    runs = _runs(line, intervals, area.convex_hull.buffer(tol))
    best = max(runs, key=lambda r: r.covered)  # max() keeps the first run on ties
    return _line_or_none(substring(line, best.start, best.end))


# ---------------------------------------------------------------- boxes


def clip_box(
    xyxy: tuple[float, float, float, float], bounds: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    """Clip an axis-aligned (min_x, min_y, max_x, max_y) box to bounds; None when no area is left."""
    for name, values in (("xyxy", xyxy), ("bounds", bounds)):
        if len(values) != _BOX_LEN or not all(math.isfinite(v) for v in values):
            raise ValueError(f"{name} must hold 4 finite numbers, got {values}")
        if values[0] > values[2] or values[1] > values[3]:
            raise ValueError(f"{name} must be (min_x, min_y, max_x, max_y), got {values}")
    x0, y0 = max(xyxy[0], bounds[0]), max(xyxy[1], bounds[1])
    x1, y1 = min(xyxy[2], bounds[2]), min(xyxy[3], bounds[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (float(x0), float(y0), float(x1), float(y1))


# ---------------------------------------------------------------- holes -> notches


def _contact_point(hole: BaseGeometry, exterior: BaseGeometry, notch_width: float) -> Point:
    """Middle of the longest stretch of the hole ring at (near-)minimum distance from the exterior."""
    nearest = Point(shapely.shortest_line(hole, exterior).coords[0])
    band = exterior.buffer(shapely.distance(hole, exterior) + notch_width * _CONTACT_REL_TOL)
    near = shapely.line_merge(shapely.intersection(hole, band))
    pieces = [p for p in split_multi(near) if isinstance(p, LineString) and p.length > 0 and not p.is_closed]
    if not pieces:
        return nearest
    best = max(pieces, key=lambda p: p.length)
    return best.interpolate(0.5, normalized=True)


def _notch(hole: BaseGeometry, exterior: BaseGeometry, notch_width: float) -> BaseGeometry:
    contact = _contact_point(hole, exterior, notch_width)
    a, b = np.asarray(shapely.shortest_line(contact, exterior).coords)
    dist = float(np.hypot(*(b - a)))
    if dist <= notch_width * _CONTACT_REL_TOL:  # the hole touches the exterior in a point
        return contact.buffer(notch_width / 2, cap_style="square")
    u = (b - a) / dist
    # extended past both rings so the flat caps never leave a sliver bridge
    cut = LineString([a - u * notch_width, b + u * notch_width])
    return cut.buffer(notch_width / 2, cap_style="flat")


def _notch_part(part: Polygon, notch_width: float, passes: int) -> list[Polygon]:
    if not part.interiors:
        return [orient_ccw(part)]
    if passes >= _MAX_NOTCH_PASSES:
        raise ValueError(f"notch_holes: holes remain after {passes} passes (bounds={part.bounds})")
    cuts = shapely.union_all([_notch(ring, part.exterior, notch_width) for ring in part.interiors])
    pieces = make_valid_polygonal(shapely.difference(part, cuts))
    return [q for piece in pieces for q in _notch_part(piece, notch_width, passes + 1)]


def notch_holes(poly: Polygon, notch_width: float) -> list[Polygon]:
    """Open every hole towards the nearest exterior edge with a cut of `notch_width` (01 §3.8).

    Returns valid CCW polygons without interiors. An invalid input is repaired first, so a hole
    spanning the full width yields two polygons. `notch_width` is in the units of `poly`; callers
    pass >= 1 px so 0.1 px vertex rounding cannot close the notch (05 C7).
    """
    if not isinstance(poly, Polygon):
        raise TypeError(f"notch_holes expects a Polygon, got {poly.geom_type}")
    if not (math.isfinite(notch_width) and notch_width > 0):
        raise ValueError(f"notch_width must be a finite positive number, got {notch_width}")
    return [q for part in make_valid_polygonal(poly) for q in _notch_part(part, notch_width, 0)]
