"""Row-candidate features and reject reasons (02 §3.4 j-l, arch §4.2 vine vs orchard).

Stations are sampled along the candidate polyline in pixel space; a station's cross-section samples
the vegetation raster at 1 px steps across the row. Unknown (nodata / off-tile) stations are NaN.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import shapely
from shapely.geometry import LineString

from vineyard.geo.tiling import GSD_M
from vineyard.perception.profile import F64
from vineyard.perception.types import BoolMask

if TYPE_CHECKING:
    from vineyard.config import OrchardConfig, RowsConfig
    from vineyard.perception.profile import SpacingEstimate

# Hard rejects: never rescued downstream. Soft rejects: tile-level evidence, rescuable by rows_link.
REASON_SHORT: Final = "short"
REASON_LOW_OCCUPANCY: Final = "low_occupancy"
REASON_TOO_WIDE: Final = "too_wide"
REASON_LOW_VINE_SCORE: Final = "low_vine_score"
REASON_ANGLE_GATE: Final = "angle_gate"
REASON_MIN_BAND_AREA: Final = "min_band_area"
REASON_LOW_SNR: Final = "low_snr"
REASON_NO_PERIODICITY: Final = "no_periodicity"
HARD_REJECT_REASONS: Final = (REASON_MIN_BAND_AREA, REASON_ANGLE_GATE, REASON_SHORT, REASON_LOW_OCCUPANCY,
                              REASON_TOO_WIDE, REASON_LOW_VINE_SCORE)
SOFT_REJECT_REASONS: Final = (REASON_NO_PERIODICITY, REASON_LOW_SNR)
# Soft flags never reject; blocks/QA aggregate them (orchard rejection is block-level, 02 §3.6).
FLAG_OFFLATTICE: Final = "row_offlattice"
FLAG_HARMONIC: Final = "harmonic"
FLAG_ORCHARD_RATIO: Final = "orchard_width_ratio"
FLAG_ORCHARD_PERIOD: Final = "orchard_period"
FLAG_CURVED: Final = "curved_row"
FLAG_WIDE: Final = "wide_row"
FLAG_SEP: Final = ";"

PX_STEP: Final = 1.0
WIDTH_PCT_MED: Final = 50.0
WIDTH_PCT_HIGH: Final = 80.0
TANGENT_EPS_PX: Final = 0.5
MIN_PERIOD_CYCLES: Final = 2.0
FFT_PAD_FACTOR: Final = 4


@dataclass(frozen=True)
class RowFeatures:
    length_m: float
    support_frac: float
    width_med_m: float
    width_p80_m: float
    along_period_m: float
    along_duty: float
    gaps: tuple[tuple[float, float], ...]
    vine_score: float
    width_spacing_ratio: float
    rel_contrast: float  # (core - flank) / core of the mean cross-section vegetation density

    def gaps_json(self) -> str:
        return json.dumps([[round(a, 3), round(b, 3)] for a, b in self.gaps])


@dataclass(frozen=True, eq=False)
class Stations:
    """Points (S, 2) and unit normals (S, 2) along a polyline, `step_m` apart, starting at step/2."""

    points: F64
    normals: F64
    step_m: float


def stations(line_px: LineString, step_m: float) -> Stations:
    """Station centres every step_m along the polyline with local unit normals."""
    length = line_px.length
    step_px = step_m / GSD_M
    n = max(1, int(math.floor(length / step_px)))
    dist = (np.arange(n) + 0.5) * (length / n) if length > 0 else np.zeros(1)
    pts = shapely.get_coordinates(shapely.line_interpolate_point(line_px, dist))
    ahead = shapely.get_coordinates(shapely.line_interpolate_point(line_px, np.minimum(dist + TANGENT_EPS_PX, length)))
    behind = shapely.get_coordinates(shapely.line_interpolate_point(line_px, np.maximum(dist - TANGENT_EPS_PX, 0.0)))
    tang = ahead - behind
    norm = np.hypot(tang[:, 0], tang[:, 1])
    tang = tang / np.where(norm > 0, norm, 1.0)[:, None]
    return Stations(points=pts, normals=np.column_stack((-tang[:, 1], tang[:, 0])), step_m=length * GSD_M / n)


def _cross_samples(raster: np.ndarray, st: Stations, half_px: float) -> tuple[np.ndarray, np.ndarray]:
    """(values (S, K), inside (S, K)) of `raster` on the cross-sections; outside samples are 0."""
    offs = np.arange(-math.floor(half_px), math.floor(half_px) + PX_STEP / 2, PX_STEP)
    xy = st.points[:, None, :] + offs[None, :, None] * st.normals[:, None, :]
    cols = np.floor(xy[..., 0]).astype(np.int64)
    rows = np.floor(xy[..., 1]).astype(np.int64)
    h, w = raster.shape
    inside = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    vals = np.zeros(cols.shape, dtype=raster.dtype)
    vals[inside] = raster[rows[inside], cols[inside]]
    return vals, inside


def _known(valid: BoolMask | None, st: Stations, shape: tuple[int, int]) -> np.ndarray:
    raster = valid if valid is not None else np.ones(shape, dtype=bool)
    vals, inside = _cross_samples(raster, st, 0.0)
    return (vals[:, 0] & inside[:, 0]).astype(bool)


def along_occupancy(veg: BoolMask, line_px: LineString, half_m: float, bin_m: float,
                    valid: BoolMask | None = None) -> F64:
    """Per station (bin_m apart): 1.0 if any vegetation within ±half_m, 0.0 if none, NaN if unknown."""
    st = stations(line_px, bin_m)
    vals, _ = _cross_samples(veg, st, half_m / GSD_M)
    occ = vals.any(axis=1).astype(np.float64)
    occ[~_known(valid, st, veg.shape)] = np.nan
    return occ


def occupancy_gaps(occ: F64, bin_m: float, min_gap_m: float) -> tuple[tuple[float, float], ...]:
    """Runs of empty stations (0.0) at least min_gap_m long, as (start_m, end_m) along the line."""
    empty = np.concatenate(([False], occ == 0.0, [False]))
    edges = np.flatnonzero(np.diff(empty.astype(np.int8)))
    starts, ends = edges[0::2], edges[1::2]
    return tuple((float(s * bin_m), float(e * bin_m)) for s, e in zip(starts, ends, strict=True)
                 if (e - s) * bin_m >= min_gap_m - 1e-9)


def _run_widths(vals: np.ndarray, seed_half_px: float) -> F64:
    """Length (samples) of the vegetation run nearest the centre sample, per row; NaN if none within seed_half."""
    k = vals.shape[1]
    idx = np.arange(k)
    centre = k // 2
    near = np.abs(idx - centre)
    dist = np.where(vals & (near <= seed_half_px)[None, :], near[None, :], k + 1)
    has = dist.min(axis=1) <= k
    seed = np.where(dist == dist.min(axis=1, keepdims=True), idx[None, :], k).min(axis=1)
    left = np.where(~vals & (idx[None, :] < seed[:, None]), idx[None, :], -1).max(axis=1)
    right = np.where(~vals & (idx[None, :] > seed[:, None]), idx[None, :], k).min(axis=1)
    return np.where(has, (right - left - 1).astype(np.float64), np.nan)


def _widths_from(vals: np.ndarray, seed_half_m: float) -> F64:
    widths = _run_widths(vals.astype(bool), seed_half_m / GSD_M) * PX_STEP * GSD_M
    return widths[np.isfinite(widths)]


def perpendicular_widths(veg: BoolMask, line_px: LineString, *, station_m: float, seed_half_m: float,
                         max_half_m: float) -> F64:
    """Vegetation run width (m) across the row at stations station_m apart, runs capped at ±max_half_m.

    The run is the one containing the axis pixel or, failing that, the nearest one within ±seed_half_m.
    Stations with no vegetation within ±seed_half_m are omitted.
    """
    st = stations(line_px, station_m)
    vals, _ = _cross_samples(veg, st, max_half_m / GSD_M)
    return _widths_from(vals, seed_half_m)


def cross_contrast(vals: np.ndarray, core_half_m: float) -> float:
    """(core - flank) / core of the mean density across the row: core = |o| <= core_half_m, flank = the
    outermost core_half_m on each side (the interrow centre). NaN when the core is empty."""
    density = vals.astype(np.float64).mean(axis=0) if len(vals) else np.zeros(0)
    k = len(density)
    if k == 0:
        return float("nan")
    off = np.abs(np.arange(k) - k // 2) * PX_STEP * GSD_M
    core = float(density[off <= core_half_m].mean())
    flank = float(density[off >= off.max() - core_half_m].mean())
    return (core - flank) / core if core > 0 else float("nan")


def along_periodicity(occ: F64, bin_m: float, period_band_m: Sequence[float]) -> tuple[float, float]:
    """(dominant along-row period in period_band_m (NaN if none), duty = mean occupancy)."""
    known = np.isfinite(occ)
    if not known.any():
        return float("nan"), float("nan")
    duty = float(occ[known].mean())
    x = np.where(known, occ, duty) - duty
    length_m = len(x) * bin_m
    p_lo, p_hi = float(period_band_m[0]), min(float(period_band_m[1]), length_m / MIN_PERIOD_CYCLES)
    if not np.any(x) or p_hi <= p_lo:
        return float("nan"), duty
    nfft = int(2 ** math.ceil(math.log2(len(x) * FFT_PAD_FACTOR)))
    power = np.abs(np.fft.rfft(x, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=bin_m)
    band = (freqs >= 1.0 / p_hi) & (freqs <= 1.0 / p_lo)
    if not band.any():
        return float("nan"), duty
    f = freqs[band][int(np.argmax(power[band]))]
    return float(1.0 / f), duty


def corridor_mean(raster: np.ndarray, line_px: LineString, half_m: float, bin_m: float) -> float:
    """Mean of a float raster over the ±half_m cross-sections of the line (NaN if no sample inside)."""
    st = stations(line_px, bin_m)
    vals, inside = _cross_samples(raster, st, half_m / GSD_M)
    return float(vals[inside].mean()) if inside.any() else float("nan")


def _pct(x: F64, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def compute_row_features(veg: BoolMask, line_px: LineString, *, spacing_m: float, cfg: RowsConfig,
                         corridor_half_m: float, occ_bin_m: float, prob: np.ndarray | None = None,
                         valid: BoolMask | None = None, period_band_m: Sequence[float] = ()) -> RowFeatures:
    """Occupancy, gaps, perpendicular widths, along-row periodicity and NN vine score of one candidate."""
    occ = along_occupancy(veg, line_px, corridor_half_m, occ_bin_m, valid)
    known = occ[np.isfinite(occ)]
    st = stations(line_px, cfg.filter.width_station_m)
    cross, _ = _cross_samples(veg, st, spacing_m / 2.0 / GSD_M)
    widths = _widths_from(cross, corridor_half_m)
    period, duty = along_periodicity(occ, occ_bin_m, period_band_m or cfg.detect.snr_band_m)
    p80 = _pct(widths, WIDTH_PCT_HIGH)
    return RowFeatures(
        length_m=float(line_px.length * GSD_M),
        support_frac=float(known.mean()) if len(known) else 0.0,
        width_med_m=_pct(widths, WIDTH_PCT_MED),
        width_p80_m=p80,
        along_period_m=period,
        along_duty=duty,
        gaps=occupancy_gaps(occ, occ_bin_m, cfg.detect.gap_record_min_m),
        vine_score=float("nan") if prob is None else corridor_mean(prob, line_px, corridor_half_m, occ_bin_m),
        width_spacing_ratio=p80 / spacing_m if spacing_m > 0 else float("nan"),
        rel_contrast=cross_contrast(cross, corridor_half_m),
    )


def is_wide(f: RowFeatures, cfg: RowsConfig) -> bool:
    return math.isfinite(f.width_p80_m) and f.width_p80_m >= cfg.filter.width_p80_max_m


def orchard_period(f: RowFeatures, orchard: OrchardConfig) -> bool:
    """Discrete crowns: along-row period in orchard.along_period_m with duty below along_duty_max."""
    lo, hi = orchard.along_period_m
    periodic = math.isfinite(f.along_period_m) and lo <= f.along_period_m <= hi
    return periodic and f.along_duty < orchard.along_duty_max


def _closed_canopy(f: RowFeatures, sp: SpacingEstimate, orchard: OrchardConfig, min_rel_contrast: float) -> bool:
    # closed canopy (crown, grass field) = density does not drop towards the interrow centre
    flat = not math.isfinite(f.rel_contrast) or f.rel_contrast < min_rel_contrast
    return flat or orchard_period(f, orchard) or sp.harmonic_flag


def hard_reject_reason(f: RowFeatures, cfg: RowsConfig, *, prob_present: bool, sp: SpacingEstimate,
                       orchard: OrchardConfig) -> str | None:
    """First failed hard filter: short, low_occupancy, too_wide, low_vine_score (NN only).

    too_wide needs width p80 >= width_p80_max_m AND closed-canopy evidence (no density drop towards the
    interrow, discrete orchard crowns, or a harmonic spacing); wide grassy vine rows stay soft-flagged.
    """
    if f.length_m < cfg.detect.min_row_len_m:
        return REASON_SHORT
    if f.support_frac < cfg.filter.occupancy_min:
        return REASON_LOW_OCCUPANCY
    if is_wide(f, cfg) and _closed_canopy(f, sp, orchard, cfg.filter.wide_min_rel_contrast):
        return REASON_TOO_WIDE
    if prob_present and math.isfinite(f.vine_score) and f.vine_score < cfg.filter.min_vine_score:
        return REASON_LOW_VINE_SCORE
    return None


def soft_reject_reason(sp: SpacingEstimate, cfg: RowsConfig) -> str | None:
    """Tile-level soft reject shared by every candidate of an orientation."""
    if not sp.ok:
        return REASON_NO_PERIODICITY
    if sp.snr < cfg.detect.periodicity_min_snr:
        return REASON_LOW_SNR
    return None


def soft_flags(f: RowFeatures, sp: SpacingEstimate, cfg: RowsConfig, orchard: OrchardConfig, *,
               on_lattice: bool, is_curved: bool) -> tuple[str, ...]:
    """Informational flags (never reject): off-lattice, harmonic, orchard ratio/period, wide, curved."""
    checks = (
        (FLAG_OFFLATTICE, not on_lattice),
        (FLAG_HARMONIC, sp.harmonic_flag),
        (FLAG_ORCHARD_RATIO, math.isfinite(f.width_spacing_ratio)
         and f.width_spacing_ratio > orchard.width_spacing_ratio_max),
        (FLAG_ORCHARD_PERIOD, orchard_period(f, orchard)),
        (FLAG_WIDE, is_wide(f, cfg)),
        (FLAG_CURVED, is_curved),
    )
    return tuple(name for name, hit in checks if hit)


def final_reason(fit_reject: str | None, hard: str | None, soft: str | None) -> str | None:
    """rejected_reason: fit reject, else hard filter, else the tile-level soft reject."""
    return fit_reject or hard or soft


def is_hard_reject(reason: str | None) -> bool:
    return reason in HARD_REJECT_REASONS
