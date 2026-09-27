"""Pure geometry of the canopy partition: axis frame, width / contact profiles, cut candidates, strip split.

A polygon is described in a local (s, t) frame: s along the row axis, t across it. Cuts are lines
s = const; a partition is the intersection of the polygon with the strips between consecutive cuts,
so the union of the pieces is the polygon itself (up to floating-point noise on the shared cut edges).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

GSD_M: Final = 0.025
TILE_PX: Final = 2048
AXIS_PROBE_M: Final = 1.0  # half-length of the axis chord used for the local direction
STRIP_PAD_M: Final = 1.0
MIN_PART_AREA_M2: Final = 1e-9
SAMPLE_STEP_M: Final = GSD_M  # contact samples along a cut segment: one per pixel


@dataclass(frozen=True)
class AxisFrame:
    """Local frame: origin (x, y), unit vector along the row (d) and across it (n)."""

    ox: float
    oy: float
    dx: float
    dy: float

    @property
    def nx(self) -> float:
        return -self.dy

    @property
    def ny(self) -> float:
        return self.dx

    def s_of(self, xy: np.ndarray) -> np.ndarray:
        return (xy[:, 0] - self.ox) * self.dx + (xy[:, 1] - self.oy) * self.dy

    def t_of(self, xy: np.ndarray) -> np.ndarray:
        return (xy[:, 0] - self.ox) * self.nx + (xy[:, 1] - self.oy) * self.ny

    def to_xy(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        s_arr, t_arr = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
        x = self.ox + s_arr * self.dx + t_arr * self.nx
        y = self.oy + s_arr * self.dy + t_arr * self.ny
        return np.stack([x, y], axis=-1)


@dataclass(frozen=True)
class Cut:
    s: float
    strength: float


def _unit(vx: float, vy: float) -> tuple[float, float] | None:
    norm = float(np.hypot(vx, vy))
    return None if norm < 1e-9 else (vx / norm, vy / norm)


def _rect_direction(poly: Polygon) -> tuple[float, float]:
    rect = shapely.minimum_rotated_rectangle(poly)
    coords = np.asarray(rect.exterior.coords) if isinstance(rect, Polygon) else np.asarray(poly.exterior.coords)
    edges = np.diff(coords, axis=0)
    longest = edges[int(np.argmax(np.hypot(edges[:, 0], edges[:, 1])))]
    return _unit(float(longest[0]), float(longest[1])) or (1.0, 0.0)


def axis_frame(poly: Polygon, axis: LineString | None) -> AxisFrame:
    """Frame at the polygon centroid, along the local direction of `axis` (fallback: min rotated rect)."""
    c = poly.centroid
    direction = None
    if axis is not None and not axis.is_empty and axis.length > 0:
        s_c = axis.project(c)
        p0 = axis.interpolate(max(s_c - AXIS_PROBE_M, 0.0))
        p1 = axis.interpolate(min(s_c + AXIS_PROBE_M, axis.length))
        direction = _unit(p1.x - p0.x, p1.y - p0.y)
    dx, dy = direction or _rect_direction(poly)
    return AxisFrame(float(c.x), float(c.y), dx, dy)


def s_extent(poly: Polygon, frame: AxisFrame) -> tuple[float, float, float]:
    """(s0, s1, max |t|) of the exterior ring in the frame."""
    xy = np.asarray(poly.exterior.coords)
    s = frame.s_of(xy)
    return float(s.min()), float(s.max()), float(np.abs(frame.t_of(xy)).max())


def sample_positions(s0: float, s1: float, step_m: float) -> np.ndarray:
    return np.arange(s0 + step_m, s1 - step_m * 0.5, step_m)


def cut_segments(frame: AxisFrame, ss: np.ndarray, half_len_m: float) -> np.ndarray:
    a = frame.to_xy(ss, -half_len_m)
    b = frame.to_xy(ss, half_len_m)
    return shapely.linestrings(np.stack([a, b], axis=1))


def width_profile(poly: Polygon, frame: AxisFrame, ss: np.ndarray, half_len_m: float) -> np.ndarray:
    """Length of polygon ∩ perpendicular segment at every s."""
    if len(ss) == 0:
        return np.zeros(0)
    return shapely.length(shapely.intersection(cut_segments(frame, ss, half_len_m), poly))


def smooth(values: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average of width k (edges replicated); a copy when k <= 1."""
    if k <= 1 or len(values) < k:
        return values.astype(float).copy()
    pad = k // 2
    padded = np.pad(values.astype(float), (pad, k - 1 - pad), mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def local_extrema(values: np.ndarray, *, minima: bool) -> np.ndarray:
    """Interior indices that are local minima (maxima) with at least one strict side."""
    if len(values) < 3:
        return np.zeros(0, dtype=int)
    v = values if minima else -values
    mid, left, right = v[1:-1], v[:-2], v[2:]
    ok = (mid <= left) & (mid <= right) & ((mid < left) | (mid < right))
    return np.nonzero(ok)[0] + 1


def contact_profile(poly: Polygon, frame: AxisFrame, ss: np.ndarray, t_max: float,
                    prob: np.ndarray, tile_x0: float, tile_y0: float) -> np.ndarray:
    """Mean contact probability (0..1) over the pixels of each cut segment inside the polygon."""
    if len(ss) == 0:
        return np.zeros(0)
    ts = np.arange(-t_max, t_max + SAMPLE_STEP_M * 0.5, SAMPLE_STEP_M)
    xy = frame.to_xy(ss[:, None], ts[None, :])  # (n_s, n_t, 2)
    inside = shapely.contains_xy(poly, xy[..., 0], xy[..., 1])
    col = np.clip(np.floor((xy[..., 0] - tile_x0) / GSD_M).astype(int), 0, prob.shape[1] - 1)
    row = np.clip(np.floor((tile_y0 - xy[..., 1]) / GSD_M).astype(int), 0, prob.shape[0] - 1)
    values = prob[row, col].astype(float)
    if prob.dtype == np.uint8:
        values = values / 255.0
    counts = inside.sum(axis=1)
    sums = np.where(inside, values, 0.0).sum(axis=1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)


def width_candidates(widths: np.ndarray, ss: np.ndarray, alpha: float) -> list[Cut]:
    positive = widths[widths > 0]
    if len(positive) == 0:
        return []
    med = float(np.median(positive))
    idx = [i for i in local_extrema(widths, minima=True) if widths[i] < alpha * med]
    return [Cut(float(ss[i]), float(1.0 - widths[i] / med)) for i in idx]


def contact_candidates(contact: np.ndarray, ss: np.ndarray, t_c: float) -> list[Cut]:
    idx = [i for i in local_extrema(contact, minima=False) if contact[i] > t_c]
    return [Cut(float(ss[i]), float(contact[i])) for i in idx]


def combined_candidates(widths: np.ndarray, contact: np.ndarray, ss: np.ndarray, alpha: float,
                        t_c: float) -> list[Cut]:
    """Width minima (below alpha x median) where the contact map says > t_c within one step."""
    out = []
    by_s = {round(c.s, 9): c for c in width_candidates(widths, ss, alpha)}
    for i, s in enumerate(ss):
        cut = by_s.get(round(float(s), 9))
        if cut is None:
            continue
        near = float(contact[max(i - 1, 0): i + 2].max())
        if near > t_c:
            out.append(Cut(cut.s, cut.strength + near))
    return out


def _strip(frame: AxisFrame, s_a: float, s_b: float, half: float) -> Polygon:
    corners = frame.to_xy(np.array([s_a, s_b, s_b, s_a]), np.array([-half, -half, half, half]))
    return Polygon(corners)


def split_by_cuts(poly: Polygon, frame: AxisFrame, cuts: Sequence[float], s0: float, s1: float,
                  t_max: float) -> list[Polygon] | None:
    """Pieces between consecutive cuts (ordered by s); None if a piece is not a single polygon."""
    half = t_max + STRIP_PAD_M
    bounds = [s0 - STRIP_PAD_M, *sorted(cuts), s1 + STRIP_PAD_M]
    pieces: list[Polygon] = []
    for s_a, s_b in zip(bounds[:-1], bounds[1:], strict=True):
        part = poly.intersection(_strip(frame, s_a, s_b, half))
        polys = [p for p in shapely.get_parts(part) if isinstance(p, Polygon) and p.area > MIN_PART_AREA_M2]
        if len(polys) != 1:
            return None
        pieces.append(polys[0])
    return pieces


def pieces_ok(pieces: Sequence[BaseGeometry] | None, min_area_m2: float) -> bool:
    return pieces is not None and all(p.area >= min_area_m2 for p in pieces)
