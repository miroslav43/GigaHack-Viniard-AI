"""Neighbour-guided second pass of row detection (rows_link, before linking).

A tile whose global pass accepted few rows is retried at the lattice of its 8-neighbours: rows continue
across the 51.2 m tile edge, so a neighbour's accepted rows give the angle, the spacing and the UTM phase
of the missing ones. Only a narrow angle/spacing window is searched and every guided candidate must sit on
the extended lattice with real row structure; the global gates that fail on young vines (tile-level
periodicity, the too_wide/orchard-period test) are replaced by the guided gates of `rows_guided`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from vineyard.contracts.ordering import angle_px_to_utm, canonical_normal
from vineyard.geo.tiling import GSD_M, TileRef, parse_tile_id, px_to_utm, tile_ref_from_grid
from vineyard.perception.linefit import SlabIndex, axial_diff_deg, extend_to_clip, fit_band_line, near_edge
from vineyard.perception.profile import (
    FFT_PAD_FACTOR,
    SpacingEstimate,
    find_row_offsets,
    mask_points,
    offset_profile,
    spectral_snr,
    subsample,
)
from vineyard.perception.row_features import RowFeatures, compute_row_features
from vineyard.perception.rows_detect import MIN_LINE_PX, DetectParams, OrientationResult, RowCandidate
from vineyard.perception.types import F32, BoolMask

if TYPE_CHECKING:
    from vineyard.config.sections_perception import RowsGuidedConfig

FLAG_GUIDED: Final = "guided"
METHOD_GUIDED: Final = "guided"
AXIAL_DEG: Final = 180.0


@dataclass(frozen=True)
class LatticePrior:
    """Row lattice of a neighbour tile: UTM axial angle, spacing and phase along canonical_normal(angle)."""

    angle_utm_deg: float
    spacing_m: float
    phase_m: float
    n_rows: int
    source_tile: str


@dataclass(frozen=True)
class GuidedResult:
    orientations: tuple[OrientationResult, ...]
    candidates: tuple[RowCandidate, ...]


def neighbour_tile_ids(tile_id: str) -> tuple[str, ...]:
    """The 8 grid neighbours of a tile (row-major order, off-grid indices skipped)."""
    row, col = parse_tile_id(tile_id)
    return tuple(tile_ref_from_grid(row + dr, col + dc).tile_id
                 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                 if (dr, dc) != (0, 0) and row + dr >= 0 and col + dc >= 0)


def axial_median_deg(angles: np.ndarray) -> float:
    """Median of axial angles (degrees, period 180) around their circular mean."""
    doubled = np.radians(np.asarray(angles, dtype=np.float64) * 2.0)
    mean = math.degrees(math.atan2(float(np.sin(doubled).sum()), float(np.cos(doubled).sum()))) / 2.0
    dev = (np.asarray(angles, dtype=np.float64) - mean + AXIAL_DEG / 2) % AXIAL_DEG - AXIAL_DEG / 2
    return float((mean + float(np.median(dev))) % AXIAL_DEG)


def line_offset_m(line: LineString, angle_utm_deg: float) -> float:
    """Signed offset of the line midpoint along canonical_normal(angle) (absolute UTM)."""
    mid = line.interpolate(0.5, normalized=True)
    n = canonical_normal(angle_utm_deg)
    return float(n[0] * mid.x + n[1] * mid.y)


def lattice_phase_m(offsets_m: np.ndarray, spacing_m: float) -> float:
    """Circular mean of offsets modulo spacing, in [0, spacing)."""
    ph = 2.0 * np.pi * np.asarray(offsets_m, dtype=np.float64) / spacing_m
    mean = math.atan2(float(np.sin(ph).sum()), float(np.cos(ph).sum()))
    return float((mean / (2.0 * np.pi) * spacing_m) % spacing_m)


def phase_dev_frac(offset_m: float, prior: LatticePrior) -> float:
    """|offset - nearest lattice line| / spacing, in [0, 0.5]."""
    frac = (offset_m - prior.phase_m) / prior.spacing_m
    return float(abs(frac - round(frac)))


def lattice_prior(tile_id: str, lines: list[LineString], angles_deg: np.ndarray, spacings_m: np.ndarray,
                  g: RowsGuidedConfig, spacing_range_m: tuple[float, float]) -> LatticePrior | None:
    """Prior of one neighbour tile from its accepted rows (None under prior_min_rows or out-of-range spacing)."""
    if len(lines) < g.prior_min_rows:
        return None
    angle = axial_median_deg(angles_deg)
    finite = spacings_m[np.isfinite(spacings_m)]
    if not len(finite):
        return None
    spacing = float(np.median(finite))
    if not spacing_range_m[0] <= spacing <= spacing_range_m[1]:
        return None
    offsets = np.array([line_offset_m(ln, angle) for ln in lines])
    return LatticePrior(angle_utm_deg=angle, spacing_m=spacing, phase_m=lattice_phase_m(offsets, spacing),
                        n_rows=len(lines), source_tile=tile_id)


def merge_priors(priors: list[LatticePrior], merge_deg: float) -> tuple[LatticePrior, ...]:
    """One prior per angle cluster: most rows first (ties by source tile), later priors within merge_deg dropped."""
    return tuple(cluster[0] for cluster in cluster_priors(priors, merge_deg, 1))


def cluster_priors(priors: list[LatticePrior], merge_deg: float, max_per_cluster: int
                   ) -> tuple[tuple[LatticePrior, ...], ...]:
    """Priors grouped by angle (within merge_deg of the cluster head), most rows first (ties by source tile);
    each cluster keeps its max_per_cluster strongest members (neighbours can disagree on spacing/phase)."""
    ordered = sorted(priors, key=lambda p: (-p.n_rows, p.source_tile))
    clusters: list[list[LatticePrior]] = []
    for p in ordered:
        home = next((c for c in clusters if axial_diff_deg(p.angle_utm_deg, c[0].angle_utm_deg) < merge_deg), None)
        if home is None:
            clusters.append([p])
        elif len(home) < max_per_cluster:
            home.append(p)
    return tuple(tuple(c) for c in clusters)


def _band_power(points: F32, angle_px_deg: float, bin_px: float, spacing_m: float, rel_tol: float) -> float:
    t = math.radians(angle_px_deg)
    off = points[:, 0].astype(np.float64) * -math.sin(t) + points[:, 1].astype(np.float64) * math.cos(t)
    counts = np.bincount(np.floor((off - off.min()) / bin_px).astype(np.int64)).astype(np.float64)
    nfft = int(2 ** math.ceil(math.log2(max(len(counts), 2) * FFT_PAD_FACTOR)))
    power = np.abs(np.fft.rfft(counts - counts.mean(), nfft)) ** 2
    freq = np.fft.rfftfreq(nfft, d=bin_px * GSD_M)
    band = (freq >= 1.0 / (spacing_m * (1.0 + rel_tol))) & (freq <= 1.0 / (spacing_m * (1.0 - rel_tol)))
    return float(power[band].max()) if band.any() else 0.0


def search_angle_px(points: F32, centre_px_deg: float, spacing_m: float, g: RowsGuidedConfig,
                    bin_px: float) -> float:
    """Pixel angle within centre +- angle_search_deg with the most offset-histogram power near 1/spacing."""
    steps = np.arange(-g.angle_search_deg, g.angle_search_deg + g.angle_step_deg / 2.0, g.angle_step_deg)
    best, best_score = centre_px_deg % AXIAL_DEG, -np.inf
    for a in (centre_px_deg + steps) % AXIAL_DEG:
        s = _band_power(points, float(a), bin_px, spacing_m, g.spacing_rel_tol)
        if s > best_score:
            best, best_score = float(a), s
    return best


def guided_orchard_period(f: RowFeatures, g: RowsGuidedConfig, duty_max: float) -> bool:
    lo, hi = g.orchard_along_period_m
    return math.isfinite(f.along_period_m) and lo <= f.along_period_m <= hi and f.along_duty < duty_max


def guided_reject(f: RowFeatures, g: RowsGuidedConfig, duty_max: float) -> str | None:
    """First failed guided gate (None = accepted).

    A wide row (weedy interrow, width p80 above width_p80_max_m) still counts when the density drops
    strongly towards the interrow centre (rel_contrast >= wide_min_rel_contrast); orchards and grass are flat.
    """
    wide = math.isfinite(f.width_p80_m) and f.width_p80_m > g.width_p80_max_m
    contrast = f.rel_contrast if math.isfinite(f.rel_contrast) else 0.0
    checks = (
        ("short", f.length_m < g.min_row_len_m),
        ("low_occupancy", f.support_frac < g.occupancy_min),
        ("low_contrast", contrast < g.min_rel_contrast),
        ("too_wide", wide and contrast < g.wide_min_rel_contrast),
        ("orchard_period", guided_orchard_period(f, g, duty_max)),
    )
    return next((name for name, hit in checks if hit), None)


@dataclass(frozen=True, eq=False)
class GuidedInputs:
    veg: BoolMask
    clip_px: BaseGeometry
    tile: TileRef
    valid: BoolMask | None = None
    prob: np.ndarray | None = None
    neighbour_utm: tuple[LineString, ...] = ()  # accepted rows of the 8 neighbours (continuation test)


def _to_utm(line_px: LineString, tile: TileRef) -> LineString:
    return LineString(px_to_utm(tile, np.asarray(line_px.coords, dtype=np.float64)))


def _features(line_px: LineString, prior: LatticePrior, inp: GuidedInputs, p: DetectParams) -> RowFeatures:
    return compute_row_features(inp.veg, line_px, spacing_m=prior.spacing_m, cfg=p.rows,
                                corridor_half_m=p.corridor_half_m, occ_bin_m=p.occ_bin_m, prob=inp.prob,
                                valid=inp.valid)


def _pieces_m(length_m: float, gaps_m: tuple[tuple[float, float], ...], cut_gap_m: float
              ) -> list[tuple[float, float]]:
    """(start, end) metres of the line pieces between the empty runs >= cut_gap_m."""
    pieces, start = [], 0.0
    for a, b in sorted(gaps_m):
        if b - a >= cut_gap_m:
            pieces.append((start, a))
            start = b
    return [*pieces, (start, length_m)]


def continued_segment(line_px: LineString, gaps_m: tuple[tuple[float, float], ...], inp: GuidedInputs,
                      g: RowsGuidedConfig) -> LineString | None:
    """The piece of a guided line (split at empty runs >= cut_gap_m, e.g. a road) that continues an accepted
    neighbour row (gap <= continue_max_m); None when no piece does. A row stops where its vines stop."""
    best, best_d = None, math.inf
    for a, b in _pieces_m(line_px.length * GSD_M, gaps_m, g.cut_gap_m):
        if (b - a) / GSD_M < MIN_LINE_PX:
            continue
        piece = line_px if (a <= 0.0 and b >= line_px.length * GSD_M) else substring(line_px, a / GSD_M, b / GSD_M)
        piece_utm = _to_utm(piece, inp.tile)
        d = min((piece_utm.distance(o) for o in inp.neighbour_utm), default=math.inf)
        if d < best_d:
            best, best_d = piece, d
    return best if best_d <= g.continue_max_m else None


def _far_from(line_utm: LineString, others: list[LineString], min_sep_m: float) -> bool:
    return all(not line_utm.intersects(o) and line_utm.distance(o) >= min_sep_m for o in others)


def _prior_candidates(pts: F32, prior: LatticePrior, inp: GuidedInputs, p: DetectParams, g: RowsGuidedConfig,
                      orientation: int, taken: list[LineString]
                      ) -> tuple[OrientationResult, list[RowCandidate], list[LineString]]:
    d = p.rows.detect
    bin_px = d.profile_bin_m / GSD_M
    angle = search_angle_px(subsample(pts, d.max_sample_points, p.seed), angle_px_to_utm(prior.angle_utm_deg),
                            prior.spacing_m, g, bin_px)
    prof = offset_profile(pts, angle, bin_px, d.smooth_sigma_bins)
    snr, harmonic = spectral_snr(prof, prior.spacing_m, d)
    sp = SpacingEstimate(spacing_m=prior.spacing_m, snr=snr, harmonic_flag=harmonic, ok=True)
    result = OrientationResult(angle_px_deg=angle, angle_utm_deg=angle_px_to_utm(angle), spacing=sp,
                               n_points=len(pts), n_peaks=0, n_kept=0)
    if snr < g.min_snr:
        return result, [], []
    slabs = SlabIndex.build(pts, angle)
    snap_px = d.snap_to_edge_m / GSD_M
    found: list[RowCandidate] = []
    lines: list[LineString] = []
    for peak in find_row_offsets(prof, sp, d):
        if peak.lattice_dev_frac > g.lattice_tol_frac:
            continue
        idx = slabs.slab(peak.offset_px, d.band_max_drift_m / GSD_M)
        fit = fit_band_line(pts[idx], slabs.offsets[idx], peak.offset_px, angle, d,
                            stub_len_m=p.rows.link.angle_min_len_m)
        if fit is None or fit.reject is not None:
            continue
        line = extend_to_clip(fit.line(), inp.clip_px, snap_px, ray_max_factor=d.snap_ray_max_factor)
        if line.length < MIN_LINE_PX:
            continue
        full_utm = _to_utm(line, inp.tile)
        if phase_dev_frac(line_offset_m(full_utm, prior.angle_utm_deg), prior) > g.phase_tol_frac:
            continue
        if not any(full_utm.distance(o) <= g.continue_max_m for o in inp.neighbour_utm):
            continue  # cheap pre-check: no piece of the line can continue a neighbour row
        feats = _features(line, prior, inp, p)
        seg = continued_segment(line, feats.gaps, inp, g)
        if seg is None:
            continue
        if seg is not line:
            line, feats = seg, _features(seg, prior, inp, p)
        line_utm = _to_utm(line, inp.tile)
        if not _far_from(line_utm, taken + lines, g.min_sep_m):
            continue
        if guided_reject(feats, g, p.orchard.along_duty_max) is not None:
            continue
        coords = line.coords
        found.append(RowCandidate(
            k=0, orientation=orientation, line_px=line, angle_px_deg=fit.angle_px_deg,
            peak_offset_px=peak.offset_px, features=feats, lattice_dev_frac=peak.lattice_dev_frac,
            residual_p95_m=fit.residual_p95_px * GSD_M, is_curved=False, local_spacing_m=prior.spacing_m,
            rejected_reason=None, soft_flags=(FLAG_GUIDED,),
            near_edge_start=near_edge(coords[0], inp.clip_px, snap_px),
            near_edge_end=near_edge(coords[-1], inp.clip_px, snap_px)))
        lines.append(line_utm)
    return replace(result, n_kept=len(found)), found, lines


def guided_detect(inp: GuidedInputs, priors: tuple[LatticePrior, ...], existing_utm: list[LineString],
                  p: DetectParams, g: RowsGuidedConfig) -> GuidedResult:
    """Accepted guided candidates of one tile; K left at 0.

    Only rows that continue an accepted neighbour row (inp.neighbour_utm, gap <= continue_max_m) count:
    a lattice that merely lines up with a far-away block (ploughed field, orchard) is not a vineyard.
    Priors are clustered by angle; inside a cluster every member (neighbours may disagree on spacing or
    phase) is tried and the one with the most accepted rows wins (ties: the stronger prior). A cluster
    counts only with >= min_rows rows.
    """
    pts = mask_points(inp.veg)
    if len(pts) < 2 or not priors:
        return GuidedResult(orientations=(), candidates=())
    taken = list(existing_utm)
    orientations: list[OrientationResult] = []
    cands: list[RowCandidate] = []
    for cluster in cluster_priors(list(priors), g.prior_angle_merge_deg, g.max_priors_per_cluster):
        best: tuple[OrientationResult, list[RowCandidate], list[LineString]] | None = None
        for prior in cluster:
            trial = _prior_candidates(pts, prior, inp, p, g, len(orientations), taken)
            if len(trial[1]) >= g.min_rows and (best is None or len(trial[1]) > len(best[1])):
                best = trial
        if best is None:
            continue
        orientations.append(best[0])
        cands.extend(best[1])
        taken.extend(best[2])
    return GuidedResult(orientations=tuple(orientations), candidates=tuple(cands))
