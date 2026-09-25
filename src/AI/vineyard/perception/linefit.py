"""Per-peak row line fit: slab pre-selection, iterated Huber fitLine, percentile ends, clip extension,
optional curved-row tracking (port of axes.detect_rows / clip_to_tile, 02 §3.4 f-h).

All geometry is in CVAT continuous pixel coordinates (u, v); lengths given in metres are converted
with GSD_M.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.geo.tiling import GSD_M, TILE_PX
from vineyard.perception.profile import F64, unit_vectors
from vineyard.perception.types import F32

if TYPE_CHECKING:
    from vineyard.config import RowsDetectConfig

REJECT_MIN_BAND_AREA: Final = "min_band_area"
REJECT_ANGLE_GATE: Final = "angle_gate"
HUBER_REPS: Final = 0.01
HUBER_AEPS: Final = 0.01
RESIDUAL_PCT: Final = 95.0  # residual_p95: percentile of the window-median centreline residuals
MIN_FIT_POINTS: Final = 2
# Huber fits use a deterministic stride subsample of at most this many band points (speed; ~0.01 deg).
FIT_MAX_POINTS: Final = 8000
EDGE_EPS_PX: Final = 1e-6
TRACK_MIN_POINTS: Final = 3  # a line fit needs points spread along the window
PX_AREA_M2: Final = GSD_M * GSD_M


@dataclass(frozen=True)
class BandFit:
    """A fitted row line: point + unit direction (u, v), along-extent [t_lo, t_hi] in px."""

    point: tuple[float, float]
    direction: tuple[float, float]
    angle_px_deg: float
    t_lo: float
    t_hi: float
    n_points: int
    residual_p95_px: float
    reject: str | None = None

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        p, d = np.asarray(self.point), np.asarray(self.direction)
        a, b = p + d * self.t_lo, p + d * self.t_hi
        return (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))

    def line(self) -> LineString:
        return LineString(self.endpoints())


@dataclass(frozen=True, eq=False)
class SlabIndex:
    """Points sorted by their offset along the normal of one angle, for fast slab queries."""

    points: F32
    offsets: F64
    order: np.ndarray
    sorted_offsets: F64

    @classmethod
    def build(cls, points: F32, angle_px_deg: float) -> SlabIndex:
        _, n = unit_vectors(angle_px_deg)
        off = points[:, 0].astype(np.float64) * n[0] + points[:, 1].astype(np.float64) * n[1]
        order = np.argsort(off, kind="stable")
        return cls(points=points, offsets=off, order=order, sorted_offsets=off[order])

    def slab(self, centre_px: float, half_px: float) -> np.ndarray:
        """Indices (into points) with |offset - centre| < half_px, ascending."""
        lo = np.searchsorted(self.sorted_offsets, centre_px - half_px, side="right")
        hi = np.searchsorted(self.sorted_offsets, centre_px + half_px, side="left")
        return np.sort(self.order[lo:hi])


def axial_diff_deg(a: float, b: float) -> float:
    """Smallest difference between two axial angles, in [0, 90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _fit_huber(q: F32) -> tuple[F64, F64]:
    step = max(1, math.ceil(len(q) / FIT_MAX_POINTS))
    sub = np.ascontiguousarray(q[::step], dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(sub, cv2.DIST_HUBER, 0, HUBER_REPS, HUBER_AEPS).ravel()
    return np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)


def _oriented(direction: F64, angle_px_deg: float) -> F64:
    d0, _ = unit_vectors(angle_px_deg)
    return direction if float(direction @ d0) >= 0 else -direction


def _nominal_fit(pts: F32, peak_px: float, angle_px_deg: float, pct: tuple[float, float],
                 reject: str) -> BandFit | None:
    d, n = unit_vectors(angle_px_deg)
    t = pts.astype(np.float64) @ d
    if len(pts) < MIN_FIT_POINTS or float(np.ptp(t)) <= 0:
        return None
    lo, hi = np.percentile(t, pct)
    base = n * peak_px
    return BandFit(point=(float(base[0]), float(base[1])), direction=(float(d[0]), float(d[1])),
                   angle_px_deg=float(angle_px_deg), t_lo=float(lo), t_hi=float(hi), n_points=len(pts),
                   residual_p95_px=float("nan"), reject=reject)


def _iterate(pts: F32, q: F32, cfg: RowsDetectConfig) -> tuple[F64, F64, np.ndarray]:
    band_px = cfg.band_m / GSD_M
    p0, v = np.zeros(2), np.array([1.0, 0.0])
    keep = np.zeros(len(pts), dtype=bool)
    for _ in range(cfg.fit_iterations):
        p0, v = _fit_huber(q)
        nu = np.array([-v[1], v[0]])
        r = np.abs((pts.astype(np.float64) - p0) @ nu)
        keep = r < band_px
        if int(keep.sum()) < MIN_FIT_POINTS:
            break
        q = pts[keep]
    return p0, v, keep


def centreline_residual_px(t: F64, r: F64, window_px: float) -> float:
    """p95 of |median signed residual| over along-windows of window_px: the centreline's bend, not the
    canopy's width (a straight row gives ~0 however wide its pixel cloud is)."""
    if len(t) == 0:
        return float("nan")
    bins = np.floor((t - t.min()) / window_px).astype(np.int64)
    order = np.argsort(bins, kind="stable")
    _, starts = np.unique(bins[order], return_index=True)
    groups = np.split(r[order], starts[1:])
    medians = [abs(float(np.median(g))) for g in groups if len(g) >= MIN_FIT_POINTS]
    return float(np.percentile(medians, RESIDUAL_PCT)) if medians else float("nan")


def _stub_fit(q: F32, angle_px_deg: float, pct: tuple[float, float], residual_px: float) -> BandFit:
    d, _ = unit_vectors(angle_px_deg)
    centre = np.median(q.astype(np.float64), axis=0)
    t = (q.astype(np.float64) - centre) @ d
    lo, hi = np.percentile(t, pct)
    return BandFit(point=(float(centre[0]), float(centre[1])), direction=(float(d[0]), float(d[1])),
                   angle_px_deg=float(angle_px_deg), t_lo=float(lo), t_hi=float(hi), n_points=len(q),
                   residual_p95_px=residual_px, reject=None)


def fit_band_line(points: F32, offsets: F64, peak_px: float, angle_px_deg: float,
                  cfg: RowsDetectConfig, *, stub_len_m: float = 0.0) -> BandFit | None:
    """Huber line through the slab `points` (|offset - peak| < band_max_drift) around one profile peak.

    Bands shorter than stub_len_m keep the profile angle (a stub's own angle is unreliable), so the
    angle gate never removes corner stubs. Returns a BandFit whose `reject` is min_band_area /
    angle_gate for failed bands, or None when too few points exist to place any line.
    """
    pct = (float(cfg.end_percentiles[0]), float(cfg.end_percentiles[1]))
    init = np.abs(offsets - peak_px) < cfg.band_init_m / GSD_M
    if int(init.sum()) * PX_AREA_M2 < cfg.min_band_area_m2:
        return _nominal_fit(points[init], peak_px, angle_px_deg, pct, REJECT_MIN_BAND_AREA)
    p0, v, keep = _iterate(points, points[init], cfg)
    q = points[keep]
    if int(keep.sum()) * PX_AREA_M2 < cfg.min_band_area_m2:
        return _nominal_fit(points[init], peak_px, angle_px_deg, pct, REJECT_MIN_BAND_AREA)
    v = _oriented(v, angle_px_deg)
    angle = math.degrees(math.atan2(v[1], v[0])) % 180.0
    rel = q.astype(np.float64) - p0
    t = rel @ v
    residual = centreline_residual_px(t, rel @ np.array([-v[1], v[0]]), float(cfg.track_window_px))
    lo, hi = np.percentile(t, pct)
    if (hi - lo) * GSD_M < stub_len_m:
        return _stub_fit(q, angle_px_deg, pct, residual)
    reject = REJECT_ANGLE_GATE if axial_diff_deg(angle, angle_px_deg) > cfg.angle_gate_deg else None
    return BandFit(point=(float(p0[0]), float(p0[1])), direction=(float(v[0]), float(v[1])), angle_px_deg=angle,
                   t_lo=float(lo), t_hi=float(hi), n_points=int(keep.sum()), residual_p95_px=residual,
                   reject=reject)


def _first_hit(origin: F64, direction: F64, boundary: BaseGeometry, reach: float) -> F64 | None:
    ray = LineString([origin, origin + direction * reach])
    hit = ray.intersection(boundary)
    if hit.is_empty:
        return None
    coords = np.asarray(shapely.get_coordinates(hit), dtype=np.float64)
    along = (coords - origin) @ direction
    return coords[int(np.argmin(along))]


def _snap_to_frame(p: F64) -> F64:
    out = p.copy()
    for edge in (0.0, float(TILE_PX)):
        out[np.abs(out - edge) < EDGE_EPS_PX] = edge
    return out


def _end_target(end: F64, unit: F64, seg_len: float, clip: BaseGeometry, max_dist_px: float,
                ray_max_factor: float) -> F64 | None:
    boundary = clip.boundary
    if not clip.covers(Point(end)):
        # A slanted wide band projects its end beyond the frame: pull the axis back to where it enters.
        return _first_hit(end, -unit, boundary, seg_len)
    if boundary.distance(Point(end)) >= max_dist_px:
        return None
    return _first_hit(end, unit, boundary, ray_max_factor * max_dist_px)


def extend_to_clip(line: LineString, clip: Polygon | BaseGeometry, max_dist_px: float, *,
                   ray_max_factor: float) -> LineString:
    """Extend each end lying within max_dist_px of the clip boundary along the line to that boundary;
    an end lying outside the clip is pulled back onto the boundary.

    The ray is cast outwards; the first boundary crossing is used (a nodata bay stops the extension).
    Extensions longer than ray_max_factor * max_dist_px (rows almost parallel to an edge) are refused
    (rows.detect.snap_ray_max_factor).
    """
    coords = np.asarray(line.coords, dtype=np.float64)
    ends = [(0, coords[0] - coords[1]), (-1, coords[-1] - coords[-2])]
    out = coords.copy()
    for idx, outward in ends:
        norm = float(np.hypot(*outward))
        if norm <= 0 or clip.is_empty:
            continue
        hit = _end_target(coords[idx], outward / norm, norm, clip, max_dist_px, ray_max_factor)
        if hit is not None:
            out[idx] = _snap_to_frame(hit)
    return LineString(out)


def near_edge(point: tuple[float, float], clip: BaseGeometry, max_dist_px: float) -> bool:
    """True if the point lies within max_dist_px of the clip boundary."""
    return bool(clip.boundary.distance(Point(point)) <= max_dist_px)


@dataclass(frozen=True)
class _Station:
    """Local centreline in the fit frame: along t, across r and slope dr/dt."""

    t: float
    r: float
    slope: float


def _local_line(t: F64, r: F64, guess: _Station, half_len: float, band: float) -> _Station | None:
    """Least-squares r(t) line of the band points in the window around guess.t, evaluated at guess.t
    (clamped to the window's data so truncated end windows land on the last points, not beyond)."""
    dt = t - guess.t
    sel = (np.abs(dt) <= half_len) & (np.abs(r - guess.r - guess.slope * dt) < band)
    if int(sel.sum()) < TRACK_MIN_POINTS or float(np.ptp(dt[sel])) <= 0:
        return None
    slope, intercept = np.polyfit(dt[sel], r[sel], 1)
    t_eval = float(np.clip(guess.t, t[sel].min(), t[sel].max()))
    return _Station(t=t_eval, r=float(intercept + slope * (t_eval - guess.t)), slope=float(slope))


def _march(t: F64, r: F64, seed: _Station, step: float, band: float) -> list[_Station]:
    """Stations from seed in steps of `step` (signed) while the window still holds band points."""
    out: list[_Station] = []
    cur = seed
    for _ in range(math.ceil(float(np.ptp(t)) / abs(step)) + 1):
        guess = _Station(t=cur.t + step, r=cur.r + cur.slope * step, slope=cur.slope)
        found = _local_line(t, r, guess, abs(step) / 2, band)
        if found is None or (found.t - cur.t) * step <= 0:
            break
        out.append(found)
        cur = found
    return out


def track_row(points: F32, fit: BandFit, cfg: RowsDetectConfig) -> LineString:
    """Curved-row tracking: local lines every track_window_px, marched from the fit centre both ways
    (each window predicted from the previous slope), then DP-simplified."""
    d, p0 = np.asarray(fit.direction), np.asarray(fit.point)
    nu = np.array([-d[1], d[0]])
    rel = points.astype(np.float64) - p0
    t, r = rel @ d, rel @ nu
    window, band = float(cfg.track_window_px), cfg.band_m / GSD_M
    seed = _local_line(t, r, _Station(t=0.5 * (fit.t_lo + fit.t_hi), r=0.0, slope=0.0), window / 2, band)
    if seed is None:
        return fit.line()
    stations = [*reversed(_march(t, r, seed, -window, band)), seed, *_march(t, r, seed, window, band)]
    if len(stations) < MIN_FIT_POINTS:
        return fit.line()
    verts = np.array([p0 + d * s.t + nu * s.r for s in stations])
    simplified = LineString(verts).simplify(cfg.dp_tolerance_m / GSD_M, preserve_topology=False)
    return simplified if simplified.length > 0 else fit.line()


def max_deviation_px(line: LineString, reference: LineString) -> float:
    """Largest distance from a vertex of `line` to `reference` (px)."""
    return max(reference.distance(Point(c)) for c in line.coords)
