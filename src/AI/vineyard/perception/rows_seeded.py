"""Seeded row detection: rows inside a reviewed seed region (polygon + row angle [+ spacing]) of one tile.

Inside the polygon only: the angle is searched within +- angle_search_deg of the seed angle (most offset-
histogram power in the spacing band), the spacing comes from the profile autocorrelation constrained to the
seed spacing +- spacing_rel_tol (else spacing_range_m), the lattice phase from the profile peaks. Every
lattice line (refined to a nearby peak, fitted to its band when the fit agrees with the region angle) is
clipped to the polygon, split at empty runs >= cut_gap_m and trimmed to its first/last vegetation station,
then kept when its occupancy >= occupancy_min. A region needs >= min_rows parallel rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from vineyard.contracts.ordering import angle_px_to_utm
from vineyard.geo.tiling import GSD_M, TILE_PX, TileRef, utm_to_px
from vineyard.perception.linefit import SlabIndex, axial_diff_deg, fit_band_line, near_edge
from vineyard.perception.profile import (
    FFT_PAD_FACTOR,
    Profile,
    SpacingEstimate,
    estimate_spacing,
    find_row_offsets,
    mask_points,
    offset_profile,
    unit_vectors,
)
from vineyard.perception.row_features import RowFeatures, along_occupancy, compute_row_features
from vineyard.perception.row_seeds import RowSeed, seed_angle_to_px
from vineyard.perception.rows_detect import MIN_LINE_PX, DetectParams, OrientationResult, RowCandidate
from vineyard.perception.types import F32, BoolMask

if TYPE_CHECKING:
    from vineyard.config.sections_seeded import RowsSeededConfig

FLAG_SEEDED: Final = "seeded"
METHOD_SEEDED: Final = "seeded"
AXIAL_DEG: Final = 180.0
REACH_PX: Final = 2.0 * TILE_PX * math.sqrt(2.0)  # half-length of a lattice line before the polygon clip
REASON_EMPTY: Final = "empty_region"
REASON_FEW_POINTS: Final = "few_points"
REASON_FEW_ROWS: Final = "too_few_rows"


@dataclass(frozen=True, eq=False)
class SeededInputs:
    veg: BoolMask
    clip_px: BaseGeometry
    tile: TileRef
    valid: BoolMask | None = None
    existing_utm: tuple[LineString, ...] = ()  # accepted rows of the tile (seeded pieces never duplicate them)


@dataclass(frozen=True)
class SeedOutcome:
    line_no: int
    angle_px_deg: float
    spacing_m: float
    n_rows: int
    reason: str | None


@dataclass(frozen=True)
class SeededResult:
    orientations: tuple[OrientationResult, ...]
    candidates: tuple[RowCandidate, ...]
    outcomes: tuple[SeedOutcome, ...]


def region_mask(region: BaseGeometry, shape: tuple[int, int]) -> BoolMask:
    """Bool px mask of a (multi)polygon in continuous px coordinates (pixel centre = index + 0.5)."""
    mask = np.zeros(shape, dtype=np.uint8)
    polys = list(region.geoms) if isinstance(region, MultiPolygon) else [region]
    for poly in polys:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        ext = np.round(np.asarray(poly.exterior.coords) - 0.5).astype(np.int32)
        cv2.fillPoly(mask, [ext], 1)
        for hole in poly.interiors:
            cv2.fillPoly(mask, [np.round(np.asarray(hole.coords) - 0.5).astype(np.int32)], 0)
    return mask.astype(bool)


def spacing_window(seed_spacing_m: float | None, s: RowsSeededConfig) -> tuple[float, float]:
    """Autocorrelation window: seed spacing +- spacing_rel_tol, else spacing_range_m."""
    if seed_spacing_m is None:
        return float(s.spacing_range_m[0]), float(s.spacing_range_m[1])
    return seed_spacing_m * (1.0 - s.spacing_rel_tol), seed_spacing_m * (1.0 + s.spacing_rel_tol)


def band_power(points: F32, angle_px_deg: float, bin_px: float, window_m: tuple[float, float]) -> float:
    """Largest FFT power of the offset histogram at the row frequencies of window_m."""
    prof = offset_profile(points, angle_px_deg, bin_px, 0.0)
    counts = prof.counts - prof.counts.mean()
    nfft = int(2 ** math.ceil(math.log2(max(len(counts), 2) * FFT_PAD_FACTOR)))
    power = np.abs(np.fft.rfft(counts, nfft)) ** 2
    freq = np.fft.rfftfreq(nfft, d=bin_px * GSD_M)
    band = (freq >= 1.0 / window_m[1]) & (freq <= 1.0 / window_m[0])
    return float(power[band].max()) if band.any() else 0.0


def search_angle(points: F32, centre_px_deg: float, bin_px: float, window_m: tuple[float, float],
                 s: RowsSeededConfig) -> float:
    """Pixel angle within centre +- angle_search_deg with the most band power (ties: closest to centre)."""
    steps = np.arange(-s.angle_search_deg, s.angle_search_deg + s.angle_step_deg / 2.0, s.angle_step_deg)
    order = np.argsort(np.abs(steps), kind="stable")
    best, best_score = centre_px_deg % AXIAL_DEG, -np.inf
    for a in (centre_px_deg + steps[order]) % AXIAL_DEG:
        score = band_power(points, float(a), bin_px, window_m)
        if score > best_score:
            best, best_score = float(a), score
    return best


def lattice_offsets(prof: Profile, sp: SpacingEstimate, p: DetectParams, tol_frac: float
                    ) -> list[tuple[float, float]]:
    """(offset px, lattice deviation) of every lattice line over the profile's extent: phase from the
    height-weighted profile peaks; a line with a peak within tol_frac x spacing takes the peak's offset."""
    peaks = find_row_offsets(prof, sp, p.rows.detect)
    if not peaks:
        return []
    spacing_px = sp.spacing_m / GSD_M
    offs = np.array([pk.offset_px for pk in peaks])
    heights = np.array([pk.height for pk in peaks])
    ph = 2.0 * np.pi * offs / spacing_px
    w = np.maximum(heights, 0.0) + np.finfo(np.float64).tiny
    phase_px = math.atan2(float((w * np.sin(ph)).sum()), float((w * np.cos(ph)).sum())) / (2 * np.pi) * spacing_px
    lo = prof.origin_px
    hi = prof.origin_px + len(prof.counts) * prof.bin_px
    k0, k1 = math.ceil((lo - phase_px) / spacing_px), math.floor((hi - phase_px) / spacing_px)
    out = []
    for k in range(k0, k1 + 1):
        nominal = phase_px + k * spacing_px
        near = np.abs(offs - nominal)
        j = int(np.argmin(near))
        if near[j] <= tol_frac * spacing_px:
            out.append((float(offs[j]), float(near[j] / spacing_px)))
        else:
            out.append((nominal, 0.0))
    return out


def lattice_line(offset_px: float, angle_px_deg: float, reach_px: float = REACH_PX) -> LineString:
    """Long line at `offset_px` along the normal of angle_px_deg (profile.unit_vectors convention)."""
    d, n = unit_vectors(angle_px_deg)
    base = n * offset_px
    return LineString([tuple(base - d * reach_px), tuple(base + d * reach_px)])


def _fitted(offset_px: float, angle: float, pts: F32, slabs: SlabIndex, p: DetectParams,
            s: RowsSeededConfig) -> tuple[LineString, float, float]:
    """(long line, angle, residual p95 m): the band fit when it agrees with the region angle, else nominal."""
    d = p.rows.detect
    idx = slabs.slab(offset_px, d.band_max_drift_m / GSD_M)
    fit = fit_band_line(pts[idx], slabs.offsets[idx], offset_px, angle, d) if len(idx) else None
    if fit is None or fit.reject is not None or axial_diff_deg(fit.angle_px_deg, angle) > s.fit_max_angle_dev_deg:
        return lattice_line(offset_px, angle), angle, float("nan")
    dvec = np.asarray(fit.direction)
    base = np.asarray(fit.point)
    line = LineString([tuple(base - dvec * REACH_PX), tuple(base + dvec * REACH_PX)])
    return line, fit.angle_px_deg, fit.residual_p95_px * GSD_M


def _parts(geom: BaseGeometry) -> list[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, LineString) and not g.is_empty]


def trim_to_evidence(line_px: LineString, veg: BoolMask, valid: BoolMask | None, half_m: float, bin_m: float,
                     cut_gap_m: float) -> list[LineString]:
    """Pieces of the line between empty runs >= cut_gap_m, each trimmed to its first/last vegetated station."""
    occ = along_occupancy(veg, line_px, half_m, bin_m, valid)
    if not len(occ):
        return []
    step_px = line_px.length / len(occ)
    hit = np.flatnonzero(occ == 1.0)
    if not len(hit):
        return []
    cut = max(1, math.ceil(cut_gap_m / (step_px * GSD_M)))
    groups, start, prev = [], int(hit[0]), int(hit[0])
    for i in hit[1:]:
        if int(i) - prev > cut:
            groups.append((start, prev))
            start = int(i)
        prev = int(i)
    groups.append((start, prev))
    out = []
    for a, b in groups:
        piece = substring(line_px, a * step_px, (b + 1) * step_px)
        if isinstance(piece, LineString) and piece.length >= MIN_LINE_PX:
            out.append(piece)
    return out


def _existing_px(inp: SeededInputs, min_sep_m: float) -> BaseGeometry | None:
    if not inp.existing_utm or min_sep_m <= 0:
        return None
    lines = [shapely.transform(ln, lambda xy: utm_to_px(inp.tile, xy)) for ln in inp.existing_utm]
    return shapely.union_all([ln.buffer(min_sep_m / GSD_M) for ln in lines])


def _features(line: LineString, spacing_m: float, inp: SeededInputs, p: DetectParams) -> RowFeatures:
    return compute_row_features(inp.veg, line, spacing_m=spacing_m, cfg=p.rows, corridor_half_m=p.corridor_half_m,
                                occ_bin_m=p.occ_bin_m, valid=inp.valid)


@dataclass(frozen=True)
class _Row:
    line: LineString
    features: RowFeatures
    angle_px_deg: float
    offset_px: float
    lattice_dev: float
    residual_m: float


def _region_rows(region: BaseGeometry, pts: F32, angle: float, sp: SpacingEstimate, prof: Profile,
                 inp: SeededInputs, blocked: BaseGeometry | None, p: DetectParams, s: RowsSeededConfig
                 ) -> list[_Row]:
    slabs = SlabIndex.build(pts, angle)
    rows: list[_Row] = []
    for offset, dev in lattice_offsets(prof, sp, p, s.lattice_tol_frac):
        long_line, fit_angle, resid = _fitted(offset, angle, pts, slabs, p, s)
        for part in _parts(long_line.intersection(region)):
            for piece in trim_to_evidence(part, inp.veg, inp.valid, p.corridor_half_m, p.occ_bin_m, s.cut_gap_m):
                free = piece.difference(blocked) if blocked is not None else piece
                for kept in _parts(free):
                    if kept.length * GSD_M < s.min_row_len_m:
                        continue
                    feats = _features(kept, sp.spacing_m, inp, p)
                    if feats.support_frac < s.occupancy_min:
                        continue
                    rows.append(_Row(kept, feats, fit_angle, offset, dev, resid))
    return rows


def _candidate(row: _Row, spacing_m: float, orientation: int, inp: SeededInputs, snap_px: float) -> RowCandidate:
    coords = row.line.coords
    return RowCandidate(
        k=0, orientation=orientation, line_px=row.line, angle_px_deg=row.angle_px_deg, peak_offset_px=row.offset_px,
        features=row.features, lattice_dev_frac=row.lattice_dev, residual_p95_m=row.residual_m, is_curved=False,
        local_spacing_m=spacing_m, rejected_reason=None, soft_flags=(FLAG_SEEDED,),
        near_edge_start=near_edge(coords[0], inp.clip_px, snap_px),
        near_edge_end=near_edge(coords[-1], inp.clip_px, snap_px))


def seed_rows(inp: SeededInputs, seed: RowSeed, blocked: BaseGeometry | None, p: DetectParams,
              s: RowsSeededConfig) -> tuple[OrientationResult | None, list[_Row], SeedOutcome]:
    """Rows of one seed region (empty when the region fails; the outcome says why)."""
    centre = seed_angle_to_px(seed.angle_deg)
    region = seed.polygon_px.intersection(inp.clip_px) if not inp.clip_px.is_empty else Polygon()
    if region.is_empty or region.area <= 0:
        return None, [], SeedOutcome(seed.line_no, centre, math.nan, 0, REASON_EMPTY)
    pts = mask_points(inp.veg & region_mask(region, inp.veg.shape))
    if len(pts) < s.min_points:
        return None, [], SeedOutcome(seed.line_no, centre, math.nan, 0, REASON_FEW_POINTS)
    d = p.rows.detect
    bin_px = d.profile_bin_m / GSD_M
    window = spacing_window(seed.spacing_m, s)
    angle = search_angle(pts, centre, bin_px, window, s)
    prof = offset_profile(pts, angle, bin_px, d.smooth_sigma_bins)
    sp = estimate_spacing(prof, d.model_copy(update={"spacing_min_m": window[0], "spacing_max_m": window[1]}))
    rows = _region_rows(region, pts, angle, sp, prof, inp, blocked, p, s)
    orient = OrientationResult(angle_px_deg=angle, angle_utm_deg=angle_px_to_utm(angle), spacing=sp,
                               n_points=len(pts), n_peaks=len(rows), n_kept=len(rows))
    if len(rows) < s.min_rows:
        return orient, [], SeedOutcome(seed.line_no, angle, sp.spacing_m, len(rows), REASON_FEW_ROWS)
    return orient, rows, SeedOutcome(seed.line_no, angle, sp.spacing_m, len(rows), None)


def seeded_detect(inp: SeededInputs, seeds: tuple[RowSeed, ...], p: DetectParams, s: RowsSeededConfig
                  ) -> SeededResult:
    """Seeded candidates (K left at 0) of one tile's seeds, in seed order; later seeds never overlap earlier
    seeded rows nor the tile's accepted rows (within min_sep_m)."""
    blocked = _existing_px(inp, s.min_sep_m)
    snap_px = p.rows.detect.snap_to_edge_m / GSD_M
    orientations: list[OrientationResult] = []
    cands: list[RowCandidate] = []
    outcomes: list[SeedOutcome] = []
    for seed in seeds:
        orient, rows, outcome = seed_rows(inp, seed, blocked, p, s)
        outcomes.append(outcome)
        if orient is None or not rows:
            continue
        idx = len(orientations)
        orientations.append(orient)
        cands.extend(_candidate(r, orient.spacing.spacing_m, idx, inp, snap_px) for r in rows)
        taken = shapely.union_all([r.line.buffer(s.min_sep_m / GSD_M) for r in rows])
        blocked = taken if blocked is None else shapely.union_all([blocked, taken])
    return SeededResult(orientations=tuple(orientations), candidates=tuple(cands), outcomes=tuple(outcomes))

