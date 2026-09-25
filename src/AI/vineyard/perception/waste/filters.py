"""Hard waste filters (A§4.9 step 2, design 03 W2). The rules list tubes, stakes, poles, wires, hoses,
stones, soil, flowering shrubs, pruning debris and vehicles as NOT WASTE: "when in doubt, leave it out".

Rules, first match wins: vehicle (> max area) -> forbidden zone -> axis (mode `all`: centroid < 0.6 m
from an axis; mode `periodic`: white and on the plant lattice) -> hose -> tube shape -> too-small white
-> bright soil (a "white" blob whose mean colour is warm and near the saturation limit: calibrated on the
2 examples, where every such survivor was a pale soil/dust patch while tubes have mean S ~ 25).
Distances use the candidate centroid and the axes of the tile (grown by the caller's margin).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, get_args

import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from vineyard.perception.waste.candidates import m2_to_px
from vineyard.perception.waste.lattice import (
    Lattice,
    LatticeParams,
    axis_projection,
    fit_lattice,
    is_periodic,
)
from vineyard.perception.waste.types import Candidate, RejectReason

if TYPE_CHECKING:
    from vineyard.config import WasteConfig

AxisMode = Literal["all", "periodic"]
KEPT: Final = "kept"
_EPS: Final = 1e-9


@dataclass(frozen=True)
class FilterParams:
    axis_exclusion_m: float
    axis_exclusion_mode: AxisMode
    plant_phase_tol: float
    lattice: LatticeParams
    white_min_area_px: float
    tube_max_aspect: float
    tube_max_area_px: float
    hose_min_len_m: float
    hose_max_width_m: float
    max_area_px: float
    bright_soil_s_min: float  # waste.colour.white_mean_s_max: a bright blob this saturated is soil
    soil_hue_range: tuple[float, float]

    def __post_init__(self) -> None:
        if self.axis_exclusion_mode not in get_args(AxisMode):
            raise ValueError(
                f"axis_exclusion_mode must be one of {get_args(AxisMode)}, got {self.axis_exclusion_mode!r}"
            )

    @classmethod
    def from_config(cls, cfg: WasteConfig, gsd_m: float) -> FilterParams:
        return cls(
            axis_exclusion_m=cfg.axis_exclusion_m,
            axis_exclusion_mode=cfg.axis_exclusion_mode,
            plant_phase_tol=cfg.plant_phase_tol,
            lattice=LatticeParams(
                pitch_range_m=tuple(cfg.lattice.pitch_range_m), min_points=cfg.lattice.min_points,
                bin_m=cfg.lattice.bin_m,
            ),
            white_min_area_px=m2_to_px(cfg.white_min_area_m2, gsd_m),
            tube_max_aspect=cfg.tube_max_aspect,
            tube_max_area_px=m2_to_px(cfg.tube_max_area_m2, gsd_m),
            hose_min_len_m=cfg.hose_min_len_m,
            hose_max_width_m=cfg.hose_max_width_m,
            max_area_px=m2_to_px(cfg.max_area_m2, gsd_m),
            bright_soil_s_min=float(cfg.colour.white_mean_s_max),
            soil_hue_range=(float(cfg.colour.soil_hue_range[0]), float(cfg.colour.soil_hue_range[1])),
        )


@dataclass(frozen=True)
class TileWasteContext:
    axes_px: tuple[np.ndarray, ...]  # row axes of the tile, CVAT continuous px
    forbidden_px: BaseGeometry | None  # forbidden zone / buildings in tile px
    gsd_m: float


@dataclass(frozen=True)
class _AxisInfo:
    axis: np.ndarray  # nearest axis index per candidate (-1 when there is no axis)
    along_m: np.ndarray
    dist_m: np.ndarray


def _axis_info(cands: Sequence[Candidate], ctx: TileWasteContext) -> _AxisInfo:
    n = len(cands)
    best = np.full(n, -1)
    along = np.full(n, np.nan)
    dist = np.full(n, np.inf)
    pts = np.array([c.centroid_px for c in cands], dtype=np.float64).reshape(-1, 2)
    for k, axis in enumerate(ctx.axes_px):
        a, d = axis_projection(axis, pts)
        closer = d * ctx.gsd_m < dist
        best = np.where(closer, k, best)
        along = np.where(closer, a * ctx.gsd_m, along)
        dist = np.where(closer, d * ctx.gsd_m, dist)
    return _AxisInfo(best, along, dist)


def _lattices(
    cands: Sequence[Candidate], info: _AxisInfo, p: FilterParams, n_axes: int
) -> dict[int, Lattice]:
    near_white = np.array([c.is_white for c in cands], dtype=bool) & (info.dist_m < p.axis_exclusion_m)
    out: dict[int, Lattice] = {}
    for k in range(n_axes):
        lat = fit_lattice(info.along_m[near_white & (info.axis == k)], p.lattice)
        if lat is not None:
            out[k] = lat
    return out


def _axis_reason(
    c: Candidate, i: int, info: _AxisInfo, lat: dict[int, Lattice], p: FilterParams
) -> RejectReason | None:
    if not info.dist_m[i] < p.axis_exclusion_m:
        return None
    if p.axis_exclusion_mode == "all":
        return RejectReason.NEAR_AXIS
    lattice = lat.get(int(info.axis[i]))
    on_lattice = lattice is not None and is_periodic(float(info.along_m[i]), lattice, p.plant_phase_tol)
    return RejectReason.PERIODIC_TUBE if c.is_white and on_lattice else None


def _shape_reason(c: Candidate, p: FilterParams) -> RejectReason | None:
    if c.length_m > p.hose_min_len_m and c.width_m <= p.hose_max_width_m + _EPS:
        return RejectReason.HOSE
    if c.aspect > p.tube_max_aspect and c.area_px < p.tube_max_area_px:
        return RejectReason.TUBE_SHAPE
    if c.is_white and c.area_px < p.white_min_area_px - _EPS:
        return RejectReason.TOO_SMALL_WHITE
    hue, sat, _ = c.mean_hsv
    lo, hi = p.soil_hue_range
    if c.is_white and sat >= p.bright_soil_s_min and lo <= hue <= hi:
        return RejectReason.BRIGHT_SOIL
    return None


def apply_filters(
    cands: Sequence[Candidate], ctx: TileWasteContext, p: FilterParams
) -> tuple[Candidate, ...]:
    """Same length and order as `cands`; rejected candidates get their first reason (inputs untouched)."""
    if not cands:
        return ()
    info = _axis_info(cands, ctx)
    lat = _lattices(cands, info, p, len(ctx.axes_px))
    zone = prep(ctx.forbidden_px) if ctx.forbidden_px is not None and not ctx.forbidden_px.is_empty else None
    out: list[Candidate] = []
    for i, c in enumerate(cands):
        if c.rejected:
            out.append(c)
            continue
        reason = (
            (RejectReason.VEHICLE if c.area_px > p.max_area_px + _EPS else None)
            or (RejectReason.FORBIDDEN if zone is not None and zone.intersects(c.box.polygon()) else None)
            or _axis_reason(c, i, info, lat, p)
            or _shape_reason(c, p)
        )
        out.append(c if reason is None else replace(c, reject_reason=reason))
    return tuple(out)


def reject_counts(cands: Sequence[Candidate]) -> dict[str, int]:
    """{"kept": n, <reason>: n, ...} sorted by key."""
    counts = Counter(KEPT if c.reject_reason is None else c.reject_reason.value for c in cands)
    return dict(sorted(counts.items()))
