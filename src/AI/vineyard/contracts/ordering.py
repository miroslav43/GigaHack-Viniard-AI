"""Contract ordering rules (contract §1.5, §2.4): angles, canonical block normal, R001 and V01 order.

Angles are UTM, degrees, axial (mod 180) in [0, 180): east = 0, counter-clockwise positive.
"""

import math
from collections.abc import Sequence
from typing import Final

import numpy as np

from vineyard.errors import SchemaError

AXIAL_PERIOD_DEG: Final = 180.0
# Contract §1.5: rows within this |n_y| of exact N-S use n_x > 0 instead of n_y > 0.
NORMAL_NS_TOL: Final = 0.01


def _axial(angle_deg: float) -> float:
    a = math.fmod(angle_deg, AXIAL_PERIOD_DEG)
    a = a + AXIAL_PERIOD_DEG if a < 0 else a
    return 0.0 if a >= AXIAL_PERIOD_DEG else a


def angle_deg_utm(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """Axial direction of the segment p0→p1 in UTM, in [0, 180)."""
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    if dx == 0.0 and dy == 0.0:
        raise SchemaError(f"angle of a zero-length segment at {tuple(p0)}")
    return _axial(math.degrees(math.atan2(dy, dx)))


def angle_px_to_utm(angle_px_deg: float) -> float:
    """Pixel v points south, so the UTM angle is the negated pixel angle, mod 180."""
    return _axial(-float(angle_px_deg))


def canonical_normal(angle_deg: float) -> tuple[float, float]:
    """n = (-sin θ, cos θ), flipped so n_y > 0; when |n_y| < NORMAL_NS_TOL, flipped so n_x > 0."""
    theta = math.radians(float(angle_deg))
    nx, ny = -math.sin(theta), math.cos(theta)
    flip = nx < 0 if abs(ny) < NORMAL_NS_TOL else ny < 0
    return (-nx, -ny) if flip else (nx, ny)


def order_by_normal(angle_deg: float, centroids: np.ndarray) -> np.ndarray:
    """Indices sorting centroids by n·c descending (R001 = northernmost along the normal); stable."""
    pts = np.asarray(centroids, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=np.intp)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise SchemaError(f"centroids must have shape (N, 2), got {pts.shape}")
    nx, ny = canonical_normal(angle_deg)
    proj = pts[:, 0] * nx + pts[:, 1] * ny
    return np.argsort(-proj, kind="stable")


def block_sort_key(x: float, y: float) -> tuple[int, int]:
    """V01 = northernmost, then westernmost representative point: (-round(y), round(x))."""
    return (-round(float(y)), round(float(x)))


def order_blocks(points: np.ndarray) -> np.ndarray:
    """Indices of block representative points in V01, V02, ... order (stable on ties)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=np.intp)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise SchemaError(f"points must have shape (N, 2), got {pts.shape}")
    keys = [block_sort_key(x, y) for x, y in pts]
    return np.array(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.intp)


def mean_axial_angle_deg(angles_deg: Sequence[float]) -> float:
    """Circular mean of axial angles (doubled-angle method), in [0, 180)."""
    a = np.radians(np.asarray(angles_deg, dtype=np.float64)) * 2.0
    if a.size == 0:
        raise SchemaError("mean_axial_angle_deg needs at least one angle")
    mean = math.atan2(float(np.sin(a).mean()), float(np.cos(a).mean())) / 2.0
    return _axial(math.degrees(mean))
