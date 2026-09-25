"""Plant lattice along a row axis (A§4.9 rule 2a, design 03 W2).

White blobs near an axis sit at the plants (protection tubes), so their along-axis positions t are
periodic: pitch = mode of the pairwise differences inside `pitch_range_m` (histogram of `bin_m`,
refined by the median of the modal differences; a well-supported half pitch wins, so missing plants
never double the pitch); phase = circular mean of t mod pitch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString

SUBHARMONIC_MIN_FRAC: Final = 0.5  # half pitch wins when it has >= this share of the modal count
_TWO_PI: Final = 2.0 * math.pi
_MIN_AXIS_VERTICES: Final = 2


@dataclass(frozen=True)
class LatticeParams:
    pitch_range_m: tuple[float, float]
    min_points: int
    bin_m: float

    def __post_init__(self) -> None:
        lo, hi = self.pitch_range_m
        if not (0.0 < lo < hi):
            raise ValueError(f"pitch_range_m must satisfy 0 < low < high, got {self.pitch_range_m}")
        if self.min_points < 2:
            raise ValueError(f"min_points must be >= 2, got {self.min_points}")
        if not self.bin_m > 0.0:
            raise ValueError(f"bin_m must be > 0, got {self.bin_m}")


@dataclass(frozen=True)
class Lattice:
    pitch_m: float
    phase_m: float
    n_points: int


def _pairwise_in_range(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    diffs = np.abs(t[:, None] - t[None, :])[np.triu_indices(len(t), k=1)]
    return np.sort(diffs[(diffs >= lo) & (diffs <= hi)])


def _near(diffs: np.ndarray, centre: float, half_width: float) -> np.ndarray:
    return diffs[np.abs(diffs - centre) <= half_width]


def _modal_pitch(diffs: np.ndarray, p: LatticeParams) -> float:
    lo, hi = p.pitch_range_m
    edges = np.arange(lo, hi + p.bin_m, p.bin_m)
    counts, _ = np.histogram(diffs, bins=edges)
    k = int(np.argmax(counts))  # ties -> the smaller pitch
    centre = (edges[k] + edges[k + 1]) / 2.0
    modal = _near(diffs, centre, 1.5 * p.bin_m)
    pitch = float(np.median(modal))
    half = _near(diffs, pitch / 2.0, p.bin_m) if pitch / 2.0 >= lo else np.array([])
    if len(half) and len(half) >= SUBHARMONIC_MIN_FRAC * len(modal):
        return float(np.median(half))
    return pitch


def _circular_phase(t: np.ndarray, pitch: float) -> float:
    angles = _TWO_PI * t / pitch
    mean_angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    return (mean_angle / _TWO_PI * pitch) % pitch


def fit_lattice(t: np.ndarray, p: LatticeParams) -> Lattice | None:
    """Lattice of the along-axis positions `t` (m); None with < min_points or no in-range spacing."""
    pos = np.unique(np.asarray(t, dtype=np.float64))
    if len(pos) < p.min_points:
        return None
    diffs = _pairwise_in_range(pos, *p.pitch_range_m)
    if not len(diffs):
        return None
    pitch = _modal_pitch(diffs, p)
    return Lattice(pitch_m=pitch, phase_m=_circular_phase(pos, pitch), n_points=len(pos))


def phase_offset(t: float, lat: Lattice) -> float:
    """Signed along-axis distance (m) from `t` to the nearest predicted plant position."""
    half = lat.pitch_m / 2.0
    return ((t - lat.phase_m + half) % lat.pitch_m) - half


def is_periodic(t: float, lat: Lattice, tol_frac: float) -> bool:
    """True when `t` lies within tol_frac * pitch of a predicted plant position."""
    return abs(phase_offset(t, lat)) <= tol_frac * lat.pitch_m


def axis_projection(axis: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(along, distance) of each point to the polyline `axis` ((N,2), same units as `pts`)."""
    coords = np.asarray(axis, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < _MIN_AXIS_VERTICES:
        raise ValueError(f"axis must be an (N>=2, 2) array, got shape {coords.shape}")
    line = LineString(coords)
    points = shapely.points(np.asarray(pts, dtype=np.float64).reshape(-1, 2))
    return shapely.line_locate_point(line, points), shapely.distance(line, points)
