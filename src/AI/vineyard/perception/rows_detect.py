"""Per-tile row-candidate detection (port of axes.detect_rows, 02 §3.4) and its `row_candidates` frame.

Orientation loop: dominant angle -> offset profile -> spacing/SNR -> peaks -> Huber band fits ->
features and reject reasons; explained pixels are removed and the loop repeats (max_orientations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import Source
from vineyard.contracts.ids import format_row_candidate_id
from vineyard.contracts.ordering import angle_deg_utm, canonical_normal
from vineyard.contracts.schema_defs import MIN_LINE_LENGTH_M
from vineyard.contracts.schemas import coerce_layer, empty_layer
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TileRef, px_to_utm
from vineyard.perception.linefit import (
    BandFit,
    SlabIndex,
    axial_diff_deg,
    extend_to_clip,
    fit_band_line,
    near_edge,
    track_row,
)
from vineyard.perception.profile import (
    PeakOffset,
    SpacingEstimate,
    dominant_angle_px,
    estimate_spacing,
    find_row_offsets,
    mask_points,
    offset_profile,
    periodic_angle_px,
    px_angle_to_utm,
    subsample,
)
from vineyard.perception.row_features import (
    FLAG_SEP,
    RowFeatures,
    compute_row_features,
    final_reason,
    hard_reject_reason,
    is_hard_reject,
    soft_flags,
    soft_reject_reason,
)
from vineyard.perception.types import F32, BoolMask

if TYPE_CHECKING:
    from vineyard.config import AppConfig, OrchardConfig, RowsConfig

STATUS_OK: Final = "ok"
STATUS_NO_PERIODICITY: Final = "no_periodicity"
STATUS_LOW_SNR: Final = "low_snr"
METHOD_VEG: Final = "veg"
METHOD_VEG_NN: Final = "veg_nn"
METHOD_TEXTURE: Final = "texture"
# contract §2.7: lines >= 0.05 m. A 2.0 px line can come out 0.049999999813 m in UTM (r027_c024 crashed the
# 311-tile run with a SchemaError), so the px gate keeps a relative margin above the exact 2.0 px.
MIN_LINE_PX: Final = MIN_LINE_LENGTH_M / GSD_M * (1.0 + 1e-6)


@dataclass(frozen=True)
class DetectParams:
    """Everything detect_tile_rows reads from the config (picklable, frozen)."""

    rows: RowsConfig
    orchard: OrchardConfig
    corridor_half_m: float
    occ_bin_m: float
    seed: int
    use_prob_in_profile: bool = False
    prob_threshold: float = 0.5

    @classmethod
    def from_config(cls, cfg: AppConfig) -> DetectParams:
        return cls(rows=cfg.rows, orchard=cfg.orchard, corridor_half_m=cfg.canopy.corridor_half_m,
                   occ_bin_m=cfg.row_structure.occ_bin_m, seed=cfg.runtime.seed,
                   use_prob_in_profile=cfg.nn.use_in_rows, prob_threshold=cfg.nn.prob_threshold)


@dataclass(frozen=True)
class RowCandidate:
    k: int
    orientation: int
    line_px: LineString
    angle_px_deg: float
    peak_offset_px: float
    features: RowFeatures
    lattice_dev_frac: float
    residual_p95_m: float
    is_curved: bool
    local_spacing_m: float
    rejected_reason: str | None
    soft_flags: tuple[str, ...]
    near_edge_start: bool
    near_edge_end: bool

    @property
    def hard_rejected(self) -> bool:
        return is_hard_reject(self.rejected_reason)


@dataclass(frozen=True)
class OrientationResult:
    angle_px_deg: float
    angle_utm_deg: float
    spacing: SpacingEstimate
    n_points: int
    n_peaks: int
    n_kept: int


@dataclass(frozen=True)
class TileDetection:
    tile_id: str
    status_hint: str
    method: str
    n_veg_px: int
    orientations: tuple[OrientationResult, ...]
    candidates: tuple[RowCandidate, ...]

    def summary(self) -> dict[str, Any]:
        """JSON-ready per-tile summary (angles, spacing, snr, status_hint, counts)."""
        return {
            "tile_id": self.tile_id, "status_hint": self.status_hint, "method": self.method,
            "n_veg_px": self.n_veg_px, "n_candidates": len(self.candidates),
            "n_kept": sum(1 for c in self.candidates if not c.hard_rejected),
            "n_accepted": sum(1 for c in self.candidates if c.rejected_reason is None),
            "orientations": [
                {"angle_px_deg": o.angle_px_deg, "angle_utm_deg": o.angle_utm_deg,
                 "spacing_m": o.spacing.spacing_m, "snr": o.spacing.snr, "harmonic": o.spacing.harmonic_flag,
                 "spacing_ok": o.spacing.ok, "n_points": o.n_points, "n_peaks": o.n_peaks, "n_kept": o.n_kept}
                for o in self.orientations
            ],
        }


@dataclass(frozen=True, eq=False)
class _TileInputs:
    veg: BoolMask
    valid: BoolMask | None
    prob: np.ndarray | None
    clip_px: BaseGeometry


def _profile_mask(veg: BoolMask, prob: np.ndarray | None, p: DetectParams,
                  fallback_points: BoolMask | None) -> tuple[BoolMask, str]:
    if fallback_points is not None:
        return fallback_points, METHOD_TEXTURE
    if prob is not None and p.use_prob_in_profile:
        return veg & (prob >= p.prob_threshold), METHOD_VEG_NN
    return veg, METHOD_VEG


def _line_for(fit: BandFit, slab_pts: F32, p: DetectParams) -> tuple[LineString, bool]:
    d = p.rows.detect
    is_curved = math.isfinite(fit.residual_p95_px) and fit.residual_p95_px * GSD_M > d.residual_split_m
    line = track_row(slab_pts, fit, d) if (is_curved and d.tracking_enabled) else fit.line()
    return line, is_curved


def _explained(slab_idx: np.ndarray, pts: F32, fit: BandFit, band_px: float) -> np.ndarray:
    rel = pts[slab_idx].astype(np.float64) - np.asarray(fit.point)
    nu = np.array([-fit.direction[1], fit.direction[0]])
    return slab_idx[np.abs(rel @ nu) < band_px]


@dataclass(frozen=True, eq=False)
class _Draft:
    candidate: RowCandidate
    explained: np.ndarray


def _draft(peak: PeakOffset, fit: BandFit, slab_pts: F32, slab_idx: np.ndarray, pts: F32,
           sp: SpacingEstimate, inp: _TileInputs, p: DetectParams, orientation: int) -> _Draft | None:
    d = p.rows.detect
    line, is_curved = _line_for(fit, slab_pts, p)
    snap_px = d.snap_to_edge_m / GSD_M
    line = extend_to_clip(line, inp.clip_px, snap_px, ray_max_factor=d.snap_ray_max_factor)
    if line.length < MIN_LINE_PX:
        return None
    feats = compute_row_features(inp.veg, line, spacing_m=sp.spacing_m, cfg=p.rows, corridor_half_m=p.corridor_half_m,
                                 occ_bin_m=p.occ_bin_m, prob=inp.prob, valid=inp.valid)
    hard = hard_reject_reason(feats, p.rows, prob_present=inp.prob is not None, sp=sp, orchard=p.orchard)
    reason = final_reason(fit.reject, hard, soft_reject_reason(sp, p.rows))
    coords = line.coords
    cand = RowCandidate(
        k=0, orientation=orientation, line_px=line, angle_px_deg=fit.angle_px_deg, peak_offset_px=peak.offset_px,
        features=feats, lattice_dev_frac=peak.lattice_dev_frac,
        residual_p95_m=fit.residual_p95_px * GSD_M, is_curved=is_curved, local_spacing_m=sp.spacing_m,
        rejected_reason=reason,
        soft_flags=soft_flags(feats, sp, p.rows, p.orchard, on_lattice=peak.on_lattice, is_curved=is_curved),
        near_edge_start=near_edge(coords[0], inp.clip_px, snap_px),
        near_edge_end=near_edge(coords[-1], inp.clip_px, snap_px),
    )
    explained = np.zeros(0, np.int64) if is_hard_reject(reason) else _explained(slab_idx, pts, fit,
                                                                                  d.band_m / GSD_M)
    return _Draft(candidate=cand, explained=explained)


def _line_distance_m(a: LineString, b: LineString) -> float:
    ca, cb = np.asarray(a.coords), np.asarray(b.coords)
    mid = a.interpolate(0.5, normalized=True)
    d = cb[-1] - cb[0]
    n = np.array([-d[1], d[0]]) / max(float(np.hypot(*d)), 1e-12)
    return abs(float((np.array([mid.x, mid.y]) - cb[0]) @ n)) * GSD_M if len(ca) else float("nan")


def _with_local_spacing(cands: list[RowCandidate], fallback_m: float) -> list[RowCandidate]:
    """local_spacing_m = distance to the nearest kept neighbour in profile order (fallback: tile s)."""
    kept = [i for i, c in enumerate(cands) if not c.hard_rejected]
    out = list(cands)
    for pos, i in enumerate(kept):
        nbrs = [kept[j] for j in (pos - 1, pos + 1) if 0 <= j < len(kept)]
        dists = [_line_distance_m(cands[i].line_px, cands[j].line_px) for j in nbrs]
        out[i] = replace(cands[i], local_spacing_m=min(dists) if dists else fallback_m)
    return out


def _detect_orientation(pts: F32, angle: float, inp: _TileInputs, p: DetectParams,
                        orientation: int) -> tuple[OrientationResult, list[RowCandidate], np.ndarray]:
    d = p.rows.detect
    slabs = SlabIndex.build(pts, angle)
    prof = offset_profile(pts, angle, d.profile_bin_m / GSD_M, d.smooth_sigma_bins)
    sp = estimate_spacing(prof, d)
    peaks = find_row_offsets(prof, sp, d)
    drafts: list[_Draft] = []
    for peak in peaks:
        idx = slabs.slab(peak.offset_px, d.band_max_drift_m / GSD_M)
        fit = fit_band_line(pts[idx], slabs.offsets[idx], peak.offset_px, angle, d,
                            stub_len_m=p.rows.link.angle_min_len_m)
        draft = None if fit is None else _draft(peak, fit, pts[idx], idx, pts, sp, inp, p, orientation)
        if draft is not None:
            drafts.append(draft)
    cands = _with_local_spacing([dr.candidate for dr in drafts], sp.spacing_m)
    explained = np.concatenate([dr.explained for dr in drafts]) if drafts else np.zeros(0, np.int64)
    result = OrientationResult(angle_px_deg=angle, angle_utm_deg=px_angle_to_utm(angle), spacing=sp,
                               n_points=len(pts), n_peaks=len(peaks),
                               n_kept=sum(1 for c in cands if not c.hard_rejected))
    return result, cands, explained


def _status(orientations: list[OrientationResult], p: DetectParams) -> str:
    if not orientations or not orientations[0].spacing.ok:
        return STATUS_NO_PERIODICITY
    if orientations[0].spacing.snr < p.rows.detect.periodicity_min_snr:
        return STATUS_LOW_SNR
    return STATUS_OK


def _secondary_ok(result: OrientationResult, p: DetectParams) -> bool:
    return result.spacing.ok and result.spacing.snr >= p.rows.detect.periodicity_min_snr and result.n_kept > 0


def _enough_points(n: int, n_total: int, orientation: int, p: DetectParams) -> bool:
    d = p.rows.detect
    if n * GSD_M * GSD_M < d.min_band_area_m2:
        return False
    return orientation == 0 or n >= d.multi_min_residual_frac * n_total


def _orientation_loop(all_pts: F32, inp: _TileInputs, p: DetectParams, first_angle: float | None = None
                      ) -> tuple[list[OrientationResult], list[RowCandidate]]:
    """Orientations one after the other on the points not yet explained; `first_angle` replaces the
    dominant angle of the first orientation (angle fallback)."""
    d = p.rows.detect
    alive = np.ones(len(all_pts), dtype=bool)
    results: list[OrientationResult] = []
    cands: list[RowCandidate] = []
    for o in range(d.max_orientations):
        live_idx = np.flatnonzero(alive)
        if not _enough_points(len(live_idx), len(all_pts), o, p):
            break
        pts = all_pts[live_idx]
        angle = first_angle if o == 0 and first_angle is not None else dominant_angle_px(
            subsample(pts, d.max_sample_points, p.seed + o), coarse_step_deg=d.angle_coarse_step_deg,
            fine_step_deg=d.angle_step_deg, bin_px=d.angle_bin_m / GSD_M)
        if any(axial_diff_deg(angle, r.angle_px_deg) < d.multi_min_angle_sep_deg for r in results):
            break
        result, found, explained = _detect_orientation(pts, angle, inp, p, o)
        if o > 0 and not _secondary_ok(result, p):
            break
        results.append(result)
        cands.extend(found)
        alive[live_idx[explained]] = False
        if result.n_kept == 0:
            break
    return results, cands


def _accepted(cands: list[RowCandidate]) -> tuple[int, float]:
    """(number, total px length) of the accepted candidates."""
    ok = [c for c in cands if c.rejected_reason is None]
    return len(ok), float(sum(c.line_px.length for c in ok))


def _angle_fallback(all_pts: F32, inp: _TileInputs, p: DetectParams, results: list[OrientationResult],
                    cands: list[RowCandidate]) -> tuple[list[OrientationResult], list[RowCandidate]]:
    """When nothing was accepted, retry from the angle with the most row-band periodic power.

    The variance argmax follows a tree belt or a sand edge on some tiles (r023_c014: 0 deg picked, rows at
    127 deg); no autocorrelation peak lies in the row range there, so every candidate is rejected and the
    real angle is never tried. The retry is kept only with >= angle_fallback_min_rows accepted rows and
    more accepted length, so tiles that already work are never touched.
    """
    d = p.rows.detect
    if not d.angle_fallback_enabled or not results or _accepted(cands)[1] > 0 or len(all_pts) < 2:
        return results, cands
    alt = periodic_angle_px(subsample(all_pts, d.max_sample_points, p.seed), coarse_step_deg=d.angle_coarse_step_deg,
                            fine_step_deg=d.angle_step_deg, bin_px=d.angle_bin_m / GSD_M,
                            spacing_min_m=d.spacing_min_m, spacing_max_m=d.spacing_max_m)
    if axial_diff_deg(alt, results[0].angle_px_deg) < d.multi_min_angle_sep_deg:
        return results, cands
    alt_results, alt_cands = _orientation_loop(all_pts, inp, p, first_angle=alt)
    n_ok, length = _accepted(alt_cands)
    if n_ok >= d.angle_fallback_min_rows and length > _accepted(cands)[1]:
        return alt_results, alt_cands
    return results, cands


def detect_tile_rows(veg: BoolMask, clip_px: BaseGeometry, tile_id: str, params: DetectParams, *,
                     prob: np.ndarray | None = None, valid: BoolMask | None = None,
                     fallback_points: BoolMask | None = None) -> TileDetection:
    """Row candidates of one tile (pixel space); K numbering follows profile order across orientations."""
    if veg.dtype != np.bool_ or veg.ndim != 2:
        raise ValueError(f"{tile_id}: veg must be a 2-D bool mask, got {veg.shape} {veg.dtype}")
    if prob is not None and prob.shape != veg.shape:
        raise ValueError(f"{tile_id}: prob shape {prob.shape} != veg shape {veg.shape}")
    mask, method = _profile_mask(veg, prob, params, fallback_points)
    all_pts = mask_points(mask)
    inp = _TileInputs(veg=veg, valid=valid, prob=prob, clip_px=clip_px)
    results, cands = _angle_fallback(all_pts, inp, params, *_orientation_loop(all_pts, inp, params))
    numbered = tuple(replace(c, k=i + 1) for i, c in enumerate(cands))
    return TileDetection(tile_id=tile_id, status_hint=_status(results, params), method=method,
                         n_veg_px=len(all_pts), orientations=tuple(results), candidates=numbered)


# Additive internal columns of row_candidates (02 §2.1), in on-disk order after the contract columns.
EXTRA_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("orientation", "int16"), ("method", "str"), ("snr", "float32"), ("spacing_tile_m", "float32"),
    ("width_p80_m", "float32"), ("rel_contrast", "float32"), ("lattice_dev_frac", "float32"),
    ("residual_p95_m", "float32"), ("is_curved", "bool"), ("along_period_m", "float32"),
    ("along_duty", "float32"), ("harmonic_flag", "bool"), ("soft_flags", "str"), ("gaps_json", "str"),
    ("near_edge_start", "bool"), ("near_edge_end", "bool"),
)
LAYER_NAME: Final = "row_candidates"


def _utm_line(c: RowCandidate, tile: TileRef) -> LineString:
    return LineString(px_to_utm(tile, np.asarray(c.line_px.coords, dtype=np.float64)))


def _record(c: RowCandidate, det: TileDetection, tile: TileRef, centre: np.ndarray) -> dict[str, Any]:
    line = _utm_line(c, tile)
    xy = np.asarray(line.coords)
    angle = angle_deg_utm(tuple(xy[0]), tuple(xy[-1]))
    mid = line.interpolate(0.5, normalized=True)
    offset = float(np.dot(canonical_normal(angle), np.array([mid.x, mid.y]) - centre))
    f, sp = c.features, det.orientations[c.orientation].spacing
    return {
        "cand_id": format_row_candidate_id(det.tile_id, c.k), "tile_id": det.tile_id, "angle_deg": angle,
        "offset_m": offset, "length_m": line.length, "support_frac": f.support_frac,
        "width_med_m": f.width_med_m, "local_spacing_m": c.local_spacing_m, "vine_score": f.vine_score,
        "rejected_reason": c.rejected_reason, "orientation": c.orientation, "method": det.method,
        "snr": sp.snr, "spacing_tile_m": sp.spacing_m, "width_p80_m": f.width_p80_m,
        "rel_contrast": f.rel_contrast, "lattice_dev_frac": c.lattice_dev_frac,
        "residual_p95_m": c.residual_p95_m, "is_curved": c.is_curved, "along_period_m": f.along_period_m,
        "along_duty": f.along_duty, "harmonic_flag": sp.harmonic_flag, "soft_flags": FLAG_SEP.join(c.soft_flags),
        "gaps_json": f.gaps_json(), "near_edge_start": c.near_edge_start, "near_edge_end": c.near_edge_end,
        "confidence": float(np.clip(f.support_frac, 0.0, 1.0)), "qa_flags": FLAG_SEP.join(c.soft_flags),
        "geometry": line,
    }


def detection_to_candidates(det: TileDetection, tile: TileRef, *, run_id: str, model_version: str,
                            source: Source = Source.MODEL) -> gpd.GeoDataFrame:
    """`row_candidates` frame (UTM LineStrings, every candidate incl. rejected ones), contract-coerced."""
    if tile.tile_id != det.tile_id:
        raise ValueError(f"detection of {det.tile_id} converted with the TileRef of {tile.tile_id}")
    minx, miny, maxx, maxy = tile.bounds
    centre = np.array([(minx + maxx) / 2.0, (miny + maxy) / 2.0])
    records = [_record(c, det, tile, centre) for c in det.candidates]
    base = empty_layer(LAYER_NAME)
    extras = {name: pd.Series([], dtype="string" if kind == "str" else kind) for name, kind in EXTRA_COLUMNS}
    if not records:
        return coerce_layer(base.assign(**extras), LAYER_NAME)
    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_EPSG)
    frame = frame.assign(source=source.value, run_id=run_id, model_version=model_version)
    cols = [*base.columns.drop("geometry"), *extras, "geometry"]
    typed = frame[cols].astype({name: ("string" if kind == "str" else kind) for name, kind in EXTRA_COLUMNS})
    return coerce_layer(gpd.GeoDataFrame(typed, geometry="geometry", crs=CRS_EPSG), LAYER_NAME)
