"""Perpendicular offset profile of vegetation points (port of axes.dominant_angle / detect_rows, 02 §3.3-3.4).

Points are CVAT continuous pixel coordinates (u, v) (pixel centres = index + 0.5). Angles are pixel
angles measured from +u towards +v in [0, 180); the row normal is n = (-sin a, cos a).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from vineyard.contracts.ordering import angle_px_to_utm
from vineyard.geo.tiling import GSD_M
from vineyard.perception.types import F32, BoolMask

if TYPE_CHECKING:
    from vineyard.config import RowsDetectConfig

F64 = NDArray[np.float64]

AXIAL_DEG: Final = 180.0
# Coarse local maxima refined at the fine step (guards against a coarse grid landing between two lobes).
COARSE_CANDIDATES: Final = 2
# The autocorrelation is searched up to this multiple of spacing_max_m so an orchard's true period
# (e.g. 5 m) is seen and folded back to its 2nd harmonic inside the vine range.
AUTOCORR_RANGE_FACTOR: Final = 2.0
FFT_PAD_FACTOR: Final = 4


@dataclass(frozen=True, eq=False)
class Profile:
    """Histogram of point offsets along the normal of `angle_px_deg` (bin 0 starts at origin_px)."""

    angle_px_deg: float
    origin_px: float
    bin_px: float
    counts: F64
    smooth: F64

    @property
    def bin_m(self) -> float:
        return self.bin_px * GSD_M

    def centre_px(self, index: int | F64) -> float | F64:
        return self.origin_px + (np.asarray(index, dtype=np.float64) + 0.5) * self.bin_px


@dataclass(frozen=True)
class SpacingEstimate:
    """Row spacing of a profile; ok=False when no autocorrelation maximum lies in the vine range."""

    spacing_m: float
    snr: float
    harmonic_flag: bool
    ok: bool


@dataclass(frozen=True)
class PeakOffset:
    offset_px: float
    height: float
    lattice_dev_frac: float
    on_lattice: bool


def unit_vectors(angle_px_deg: float) -> tuple[F64, F64]:
    """(direction, normal) unit vectors of a pixel angle."""
    t = math.radians(angle_px_deg)
    return np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])


def px_angle_to_utm(angle_px_deg: float) -> float:
    """UTM axial angle of a pixel angle: (-a) mod 180 (v points south)."""
    return angle_px_to_utm(angle_px_deg)


def mask_points(mask: BoolMask) -> F32:
    """(N, 2) float32 pixel-centre coordinates (u, v) of every True pixel, in row-major order."""
    rows, cols = np.nonzero(mask)
    return np.column_stack((cols.astype(np.float32) + 0.5, rows.astype(np.float32) + 0.5))


def subsample(points: F32, max_points: int, seed: int) -> F32:
    """Up to max_points rows of `points`, drawn without replacement with default_rng(seed), order kept."""
    if len(points) <= max_points:
        return points
    pick = np.sort(np.random.default_rng(seed).choice(len(points), max_points, replace=False))
    return points[pick]


def sample_points(mask: BoolMask, max_points: int, seed: int) -> F32:
    """Up to max_points pixel-centre points of `mask` (see subsample)."""
    return subsample(mask_points(mask), max_points, seed)


def _offsets(points: F32, angle_px_deg: float) -> F64:
    _, n = unit_vectors(angle_px_deg)
    return points[:, 0].astype(np.float64) * n[0] + points[:, 1].astype(np.float64) * n[1]


def _hist_variance(points: F32, angle_px_deg: float, bin_px: float) -> float:
    off = _offsets(points, angle_px_deg)
    idx = np.floor((off - off.min()) / bin_px).astype(np.int64)
    return float(np.bincount(idx).var())


def _coarse_peaks(angles: F64, scores: F64, k: int) -> list[float]:
    left, right = np.roll(scores, 1), np.roll(scores, -1)  # the axial search space is circular
    is_peak = (scores >= left) & (scores >= right)
    order = np.argsort(-np.where(is_peak, scores, -np.inf), kind="stable")
    return [float(angles[i]) for i in order[:k] if is_peak[i]] or [float(angles[int(np.argmax(scores))])]


def _row_band_power(points: F32, angle_px_deg: float, bin_px: float, spacing_min_m: float,
                    spacing_max_m: float) -> float:
    """Largest FFT power of the offset histogram at the row frequencies [1/spacing_max_m, 1/spacing_min_m]."""
    off = _offsets(points, angle_px_deg)
    counts = np.bincount(np.floor((off - off.min()) / bin_px).astype(np.int64)).astype(np.float64)
    nfft = int(2 ** math.ceil(math.log2(max(len(counts), 2) * FFT_PAD_FACTOR)))
    power = np.abs(np.fft.rfft(counts - counts.mean(), nfft)) ** 2
    freq = np.fft.rfftfreq(nfft, d=bin_px * GSD_M)
    band = (freq >= 1.0 / spacing_max_m) & (freq <= 1.0 / spacing_min_m)
    return float(power[band].max()) if band.any() else 0.0


def _best_angle(score: Callable[[float], float], coarse_step_deg: float, fine_step_deg: float) -> float:
    """Angle in [0, 180) maximizing `score` (coarse grid, then the best coarse peaks refined at the fine step)."""
    coarse = np.arange(0.0, AXIAL_DEG, coarse_step_deg)
    scores = np.array([score(float(a)) for a in coarse])
    best_angle, best_score = 0.0, -np.inf
    for centre in _coarse_peaks(coarse, scores, COARSE_CANDIDATES):
        fine = centre + np.arange(-coarse_step_deg, coarse_step_deg + fine_step_deg / 2, fine_step_deg)
        for a in fine:
            s = score(float(a) % AXIAL_DEG)
            if s > best_score:
                best_angle, best_score = float(a) % AXIAL_DEG, s
    return best_angle


def _need_points(points: F32, what: str) -> None:
    if len(points) < 2:
        raise ValueError(f"{what} needs >= 2 points, got {len(points)}")


def dominant_angle_px(points: F32, *, coarse_step_deg: float, fine_step_deg: float, bin_px: float) -> float:
    """Angle in [0, 180) maximizing the variance of the offset histogram (coarse grid, then fine refine)."""
    _need_points(points, "dominant_angle_px")
    return _best_angle(lambda a: _hist_variance(points, a, bin_px), coarse_step_deg, fine_step_deg)


def periodic_angle_px(points: F32, *, coarse_step_deg: float, fine_step_deg: float, bin_px: float,
                      spacing_min_m: float, spacing_max_m: float) -> float:
    """Angle in [0, 180) maximizing the offset-histogram power in the row-spacing band.

    The histogram variance also rewards large aperiodic structure (a tree belt, a sand/grass edge), which
    can outweigh the rows; the band power only sees periodicity at 1/spacing_max_m .. 1/spacing_min_m.
    """
    _need_points(points, "periodic_angle_px")
    return _best_angle(lambda a: _row_band_power(points, a, bin_px, spacing_min_m, spacing_max_m),
                       coarse_step_deg, fine_step_deg)


def offset_profile(points: F32, angle_px_deg: float, bin_px: float, sigma_bins: float,
                   weights: F32 | None = None) -> Profile:
    """Offset histogram (bin_px) and its Gaussian smoothing (sigma_bins, zero outside the data)."""
    if len(points) == 0:
        raise ValueError("offset_profile needs at least one point")
    off = _offsets(points, angle_px_deg)
    origin = float(np.floor(off.min() / bin_px) * bin_px)
    idx = np.floor((off - origin) / bin_px).astype(np.int64)
    counts = np.bincount(idx, weights=weights).astype(np.float64)
    smooth = gaussian_filter1d(counts, sigma_bins, mode="constant") if sigma_bins > 0 else counts.copy()
    return Profile(angle_px_deg=float(angle_px_deg), origin_px=origin, bin_px=float(bin_px), counts=counts,
                   smooth=smooth)


def _highpass(x: F64, sigma_bins: float) -> F64:
    return x - gaussian_filter1d(x, sigma_bins, mode="nearest")


def _autocorr(x: F64) -> F64:
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec), 2 * n)[:n]
    return ac / ac[0] if ac[0] > 0 else np.zeros(n)


def _parabolic(y: F64, i: int) -> float:
    if 0 < i < len(y) - 1:
        denom = y[i - 1] - 2 * y[i] + y[i + 1]
        if denom < 0:
            return i + 0.5 * (y[i - 1] - y[i + 1]) / denom
    return float(i)


def _best_lag(ac: F64, lo: int, hi: int) -> tuple[float, bool]:
    """Lag (bins) of the highest interior local maximum of ac in [lo, hi]; (argmax, False) if none."""
    idx = np.arange(max(lo, 1), min(hi, len(ac) - 2) + 1)
    if len(idx) == 0:
        return float(max(lo, 0)), False
    peak = (ac[idx] > ac[idx - 1]) & (ac[idx] >= ac[idx + 1]) & (ac[idx] > 0)
    if not peak.any():
        return float(idx[int(np.argmax(ac[idx]))]), False
    cand = idx[peak]
    best = int(cand[int(np.argmax(ac[cand]))])
    return _parabolic(ac, best), True


def _band_power(power: F64, freqs: F64, f0: float, rel_width: float) -> float:
    sel = np.abs(freqs - f0) <= rel_width * f0
    return float(power[sel].mean()) if sel.any() else 0.0


def _spectrum(profile: Profile, detrend_bins: float) -> tuple[F64, F64]:
    x = _highpass(profile.counts, detrend_bins)
    nfft = int(2 ** math.ceil(math.log2(max(len(x), 2) * FFT_PAD_FACTOR)))
    power = np.abs(np.fft.rfft(x, nfft)) ** 2
    return power, np.fft.rfftfreq(nfft, d=profile.bin_m)


def spectral_snr(profile: Profile, spacing_m: float, cfg: RowsDetectConfig) -> tuple[float, bool]:
    """(power at 1/s over the median power of periods in snr_band_m, power(1/2s) > power(1/s))."""
    power, freqs = _spectrum(profile, cfg.spacing_max_m / profile.bin_m)
    p_min, p_max = cfg.snr_band_m
    band = (freqs >= 1.0 / p_max) & (freqs <= 1.0 / p_min)
    floor = float(np.median(power[band])) if band.any() else 0.0
    peak = _band_power(power, freqs, 1.0 / spacing_m, cfg.snr_peak_rel_width)
    sub = _band_power(power, freqs, 0.5 / spacing_m, cfg.snr_peak_rel_width)
    snr = peak / floor if floor > 0 else 0.0
    return snr, sub > peak


def estimate_spacing(profile: Profile, cfg: RowsDetectConfig) -> SpacingEstimate:
    """Autocorrelation spacing in [spacing_min_m, spacing_max_m]; a period beyond the range folds to s/2."""
    bin_m = profile.bin_m
    x = _highpass(profile.smooth, cfg.spacing_max_m / bin_m)
    ac = _autocorr(x)
    lo = math.ceil(cfg.spacing_min_m / bin_m)
    hi_vine = math.floor(cfg.spacing_max_m / bin_m)
    lag, ok = _best_lag(ac, lo, math.floor(AUTOCORR_RANGE_FACTOR * cfg.spacing_max_m / bin_m))
    if ok and lag > hi_vine:
        lag, ok = lag / 2.0, lag / 2.0 >= lo
    if not ok:
        lag, _ = _best_lag(ac, lo, hi_vine)
        lag = float(np.clip(lag, lo, hi_vine))
    spacing = lag * bin_m
    snr, harmonic = spectral_snr(profile, spacing, cfg)
    return SpacingEstimate(spacing_m=float(spacing), snr=float(snr), harmonic_flag=bool(harmonic), ok=bool(ok))


def lattice_deviation(offsets_m: F64, heights: F64, spacing_m: float) -> F64:
    """|off - (o0 + k s)| / s in [0, 0.5], o0 = height-weighted circular mean of the peak phases."""
    if len(offsets_m) == 0:
        return np.zeros(0)
    phase = 2 * np.pi * offsets_m / spacing_m
    w = np.maximum(heights, 0.0) + np.finfo(np.float64).tiny
    o0 = math.atan2(float((w * np.sin(phase)).sum()), float((w * np.cos(phase)).sum()))
    frac = (phase - o0) / (2 * np.pi)
    return np.abs(frac - np.round(frac))


def find_row_offsets(profile: Profile, sp: SpacingEstimate, cfg: RowsDetectConfig) -> tuple[PeakOffset, ...]:
    """Profile peaks at least peak_min_dist_factor * s apart (no height criterion), with lattice flags."""
    distance = max(1, int(cfg.peak_min_dist_factor * sp.spacing_m / profile.bin_m))
    idx, _ = find_peaks(profile.smooth, distance=distance)
    if len(idx) == 0:
        return ()
    offsets = np.asarray(profile.centre_px(idx), dtype=np.float64)
    heights = profile.smooth[idx]
    dev = lattice_deviation(offsets * GSD_M, heights, sp.spacing_m)
    return tuple(
        PeakOffset(offset_px=float(o), height=float(h), lattice_dev_frac=float(d),
                   on_lattice=bool(d <= cfg.onlattice_tol_factor))
        for o, h, d in zip(offsets, heights, dev, strict=True)
    )
