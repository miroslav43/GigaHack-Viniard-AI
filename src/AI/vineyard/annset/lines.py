"""Line geometry for derive/targets: PCA direction, projections, offsets, axial median, midline, extension.

Directions are unit vectors in UTM; the canonical direction of an axis is (cos θ, sin θ) with θ its axial
angle in [0, 180) (contract §1.5). The canonical block normal lives in `contracts.ordering`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
from shapely.geometry import LineString

from vineyard.contracts.ordering import AXIAL_PERIOD_DEG, angle_deg_utm, mean_axial_angle_deg
from vineyard.errors import SchemaError

HALF_PERIOD_DEG: Final = AXIAL_PERIOD_DEG / 2.0
# Two lines overlapping less than this along the common direction have no midline.
MIN_OVERLAP_M: Final = 1e-6


def _points(coords: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise SchemaError(f"coordinates must have shape (N, 2), got {pts.shape}")
    return pts


def direction_of_angle(angle_deg: float) -> np.ndarray:
    t = math.radians(float(angle_deg))
    return np.array([math.cos(t), math.sin(t)])


def axial_angle_of(direction: np.ndarray | Sequence[float]) -> float:
    """Axial angle in [0, 180) of a direction vector (a zero vector raises SchemaError)."""
    d = np.asarray(direction, dtype=np.float64)
    return angle_deg_utm((0.0, 0.0), (float(d[0]), float(d[1])))


def pca_direction(coords: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Canonical unit direction of the main axis of the points (largest-variance eigenvector)."""
    pts = _points(coords)
    if len(pts) < 2:
        raise SchemaError("PCA direction needs at least 2 points", n_points=len(pts))
    centred = pts - pts.mean(axis=0)
    if not np.any(centred):
        raise SchemaError("PCA direction of identical points", point=tuple(pts[0]))
    _, vectors = np.linalg.eigh(centred.T @ centred)
    return direction_of_angle(axial_angle_of(vectors[:, -1]))


def line_angle_deg(line: LineString) -> float:
    """Axial angle of the PCA axis of the line's vertices."""
    return axial_angle_of(pca_direction(np.asarray(line.coords)[:, :2]))


def project(coords: np.ndarray, direction: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
    """Scalar position of each point along `direction` (relative to `origin`, default (0, 0))."""
    pts = _points(coords)
    base = np.zeros(2) if origin is None else np.asarray(origin, dtype=np.float64)
    return (pts - base) @ np.asarray(direction, dtype=np.float64)


def oriented(line: LineString, direction: np.ndarray) -> LineString:
    """The line, reversed when its last vertex projects before its first along `direction`."""
    coords = np.asarray(line.coords)[:, :2]
    ends = project(coords[[0, -1]], direction)
    return LineString(coords[::-1]) if ends[1] < ends[0] else line


def endpoint_direction(line: LineString) -> np.ndarray:
    """Unit vector from the first to the last vertex."""
    coords = np.asarray(line.coords)[:, :2]
    delta = coords[-1] - coords[0]
    norm = float(np.hypot(*delta))
    if norm == 0.0:
        raise SchemaError("closed line has no endpoint direction", start=tuple(coords[0]))
    return delta / norm


def lateral_offset(point: np.ndarray, anchor: np.ndarray, direction: np.ndarray) -> float:
    """Distance from `point` to the infinite line through `anchor` with unit `direction`."""
    rel = np.asarray(point, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    return abs(float(d[0] * rel[1] - d[1] * rel[0]))


def signed_offset(point: np.ndarray, anchor: np.ndarray, normal: np.ndarray) -> float:
    """normal · (point - anchor)."""
    rel = np.asarray(point, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    return float(rel @ np.asarray(normal, dtype=np.float64))


def axial_median_deg(angles_deg: Sequence[float]) -> float:
    """Median of axial angles around their circular mean, in [0, 180)."""
    values = np.asarray(angles_deg, dtype=np.float64)
    if values.size == 0:
        raise SchemaError("axial median needs at least one angle")
    mean = mean_axial_angle_deg(values.tolist())
    deviation = np.mod(values - mean + HALF_PERIOD_DEG, AXIAL_PERIOD_DEG) - HALF_PERIOD_DEG
    return float(np.mod(mean + float(np.median(deviation)), AXIAL_PERIOD_DEG))


def along_range(line: LineString, direction: np.ndarray) -> tuple[float, float]:
    """(min, max) projection of the line's vertices on `direction`."""
    along = project(np.asarray(line.coords)[:, :2], direction)
    return float(along.min()), float(along.max())


def points_at(line: LineString, direction: np.ndarray, along: np.ndarray) -> np.ndarray:
    """Points of the line at the given projections on `direction` (line assumed monotone along it)."""
    coords = np.asarray(line.coords)[:, :2]
    proj = project(coords, direction)
    order = np.argsort(proj, kind="stable")
    s = np.asarray(along, dtype=np.float64)
    return np.column_stack([np.interp(s, proj[order], coords[order, 0]), np.interp(s, proj[order], coords[order, 1])])


def midline(a: LineString, b: LineString) -> LineString | None:
    """Mean of two roughly parallel lines over their common extent along their mean direction.

    Points of equal projection on the bisector direction are averaged, so the result is equidistant to
    both lines even when they fan out. None when the lines do not overlap along that direction.
    """
    direction = direction_of_angle(mean_axial_angle_deg([line_angle_deg(a), line_angle_deg(b)]))
    (a0, a1), (b0, b1) = along_range(a, direction), along_range(b, direction)
    lo, hi = max(a0, b0), min(a1, b1)
    if hi - lo <= MIN_OVERLAP_M:
        return None
    inner = np.concatenate([project(np.asarray(g.coords)[:, :2], direction) for g in (a, b)])
    s = np.unique(np.concatenate([[lo, hi], inner[(inner > lo) & (inner < hi)]]))
    return LineString((points_at(a, direction, s) + points_at(b, direction, s)) / 2.0)


def extend_line(line: LineString, *, before_m: float = 0.0, after_m: float = 0.0) -> LineString:
    """Line linearly extrapolated along its first / last segment by the given lengths."""
    if before_m < 0 or after_m < 0:
        raise SchemaError("extension lengths must be >= 0", before_m=before_m, after_m=after_m)
    coords = np.asarray(line.coords)[:, :2]
    head, tail = [], []
    if before_m > 0:
        head = [coords[0] - before_m * endpoint_direction(LineString(coords[:2]))]
    if after_m > 0:
        tail = [coords[-1] + after_m * endpoint_direction(LineString(coords[-2:]))]
    return LineString([*head, *coords, *tail])
