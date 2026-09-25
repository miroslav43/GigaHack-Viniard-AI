"""Solve phases of one route plan (design 04 §3.8): must tour, coverage loop, optional phase, rounding.

- phase A: GTSP over the must sets;
- coverage loop (<= cover_iterations): sets whose targets are within cover_radius of the route outside
  their own two legs are dropped and the rest re-solved; the result is kept only if it is shorter and
  still covers every must target (CETSP, arch §4.11.5);
- phase B: optional sets not yet covered are added and re-solved (`optional_delta_m` reported).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.errors import RouteValidationError
from vineyard.logging_setup import get_logger, log_event
from vineyard.route.graph_types import CandidateSet, WalkGraph
from vineyard.route.tour import DistanceTable, Tour, TourParams, solve_tour
from vineyard.route.unroll import dedupe_xy
from vineyard.route.validate import covered_mask, segments

# A coverage-loop re-solve must save at least this much to replace the current tour.
MIN_GAIN_M: Final = 0.01

XY = Mapping[str, tuple[float, float]]
_log = get_logger("route.plan_steps")


@dataclass(frozen=True)
class PhaseParams:
    tour: TourParams
    cover_iterations: int
    cover_radius_m: float
    include_optional: bool


@dataclass(frozen=True, eq=False)
class PhaseResult:
    tour: Tour
    used: tuple[CandidateSet, ...]
    iterations: int
    must_length_m: float | None
    optional_delta_m: float | None


def points(sets: Sequence[CandidateSet], xy_of: XY) -> np.ndarray:
    return np.array([xy_of[t] for s in sets for t in s.target_ids], dtype=np.float64).reshape(-1, 2)


def free_sets(tour: Tour, sets: Sequence[CandidateSet], xy_of: XY, radius_m: float) -> frozenset[int]:
    """Sets whose targets are all within `radius_m` of the route outside the legs into and out of them."""
    rows = [(k, tid) for k, owner in enumerate(tour.stop_sets) for tid in sets[owner].target_ids]
    if not rows:
        return frozenset()
    stop = np.array([k for k, _ in rows])
    lo, hi = np.asarray(tour.anchors)[stop], np.asarray(tour.anchors)[stop + 2]
    pts = shapely.points(np.array([xy_of[tid] for _, tid in rows]))
    pi, si = shapely.STRtree(segments(tour.line)).query(pts, predicate="dwithin", distance=radius_m)
    elsewhere = (si < lo[pi]) | (si >= hi[pi])
    free_pt = np.bincount(pi[elsewhere], minlength=len(rows)) > 0
    free = set(tour.stop_sets)
    for (k, _), ok in zip(rows, free_pt, strict=True):
        if not ok:
            free.discard(tour.stop_sets[k])
    return frozenset(free)


def cover_loop(g: WalkGraph, must: tuple[CandidateSet, ...], xy_of: XY, eroded: BaseGeometry, p: PhaseParams,
               table: DistanceTable) -> tuple[Tour, int, tuple[CandidateSet, ...]]:
    active, best = must, solve_tour(g, must, eroded, p.tour, table)
    all_pts, iterations = points(must, xy_of), 0
    for _ in range(p.cover_iterations):
        free = free_sets(best, active, xy_of, p.cover_radius_m)
        trial_sets = tuple(s for k, s in enumerate(active) if k not in free)
        if not free or not trial_sets:
            break
        trial = solve_tour(g, trial_sets, eroded, p.tour, table)
        kept = trial.length_m < best.length_m - MIN_GAIN_M and bool(
            covered_mask(trial.line, all_pts, p.cover_radius_m).all())
        log_event(_log, "route.cover.iteration", n_free=len(free), length_m=round(trial.length_m, 2), kept=kept)
        if not kept:
            break
        best, active, iterations = trial, trial_sets, iterations + 1
    return best, iterations, active


def _uncovered(best: Tour, optional: tuple[CandidateSet, ...], xy_of: XY, radius_m: float) -> tuple[CandidateSet, ...]:
    done = covered_mask(best.line, points(optional, xy_of), radius_m)
    flags = np.split(done, np.cumsum([len(s.target_ids) for s in optional])[:-1])
    return tuple(s for s, f in zip(optional, flags, strict=True) if not f.all())


def solve_phases(g: WalkGraph, must: tuple[CandidateSet, ...], optional: tuple[CandidateSet, ...], xy_of: XY,
                 eroded: BaseGeometry, p: PhaseParams, table: DistanceTable) -> PhaseResult:
    """Phase A + coverage loop on the must sets, then phase B on the optional ones still uncovered."""
    optional = optional if p.include_optional else ()
    if not must and not optional:
        raise RouteValidationError("no reachable targets to route")
    best, iterations, active = cover_loop(g, must, xy_of, eroded, p, table) if must else (None, 0, ())
    todo = _uncovered(best, optional, xy_of, p.cover_radius_m) if best is not None and optional else optional
    if best is not None and not todo:
        return PhaseResult(best, active, iterations, best.length_m, 0.0 if optional else None)
    used = (*active, *todo)
    final = solve_tour(g, used, eroded, p.tour, table)
    must_len = best.length_m if best is not None else None
    return PhaseResult(final, used, iterations, must_len, final.length_m - (must_len or 0.0))


def finalize(tour: Tour, start_xy: tuple[float, float], decimals: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Round like the GeoJSON writer (START exact at both ends), dedupe, remap anchors."""
    xy = np.round(np.asarray(tour.xy, dtype=np.float64), decimals) + 0.0
    xy[0] = xy[-1] = np.round(np.asarray(start_xy, dtype=np.float64), decimals) + 0.0
    return dedupe_xy(xy, tour.anchors)


__all__ = ["MIN_GAIN_M", "PhaseParams", "PhaseResult", "cover_loop", "finalize", "free_sets", "points",
           "solve_phases"]
