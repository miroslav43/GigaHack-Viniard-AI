"""Solve phases of one route plan (design 04 §3.8): must tour, coverage loop, optional phase, rounding.

- phase A: GTSP over the must sets;
- coverage loop (<= cover_iterations): sets whose targets are within cover_radius of the route outside
  their own two legs are dropped and the rest re-solved; the result is kept only if it is shorter and
  still covers every must target (CETSP, arch §4.11.5);
- phase B: optional sets not yet covered are added and re-solved (`optional_delta_m` reported), but only
  those whose cheapest insertion into the must tour (between two consecutive stops, graph cost: metres +
  outside penalty) is <= optional_max_detour_m; the others are `skipped` (optional = visited when cheap).
  Without must sets there is no tour to insert into and every optional set is routed.
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
    optional_max_detour_m: float


@dataclass(frozen=True, eq=False)
class PhaseResult:
    tour: Tour
    used: tuple[CandidateSet, ...]
    iterations: int
    must_length_m: float | None
    optional_delta_m: float | None
    skipped: tuple[CandidateSet, ...] = ()


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


def insertion_detours(g: WalkGraph, tour: Tour, sets: Sequence[CandidateSet], table: DistanceTable) -> np.ndarray:
    """Per set: min over its nodes and the tour's legs of d(a, n) + d(n, b) - d(a, b) (graph cost, m)."""
    if not sets:
        return np.zeros(0, dtype=np.float64)
    seq = np.array([g.start_node, *tour.stop_nodes, g.start_node], dtype=np.int64)
    nodes = np.array([n for s in sets for n in s.nodes], dtype=np.int64)
    owner = np.repeat(np.arange(len(sets)), [len(s.nodes) for s in sets])
    pool = np.unique(np.concatenate([seq, nodes]))
    d = table.matrix(pool)
    a, b, c = (np.searchsorted(pool, x) for x in (seq[:-1], seq[1:], nodes))
    via = d[np.ix_(a, c)] + d[np.ix_(c, b)].T - d[a, b][:, None]
    out = np.full(len(sets), np.inf)
    np.minimum.at(out, owner, via.min(axis=0))
    return out


def cheap_sets(g: WalkGraph, tour: Tour, sets: tuple[CandidateSet, ...], table: DistanceTable,
               max_detour_m: float) -> tuple[tuple[CandidateSet, ...], tuple[CandidateSet, ...]]:
    """(sets worth inserting into `tour`, sets whose cheapest insertion costs more than `max_detour_m`)."""
    ok = insertion_detours(g, tour, sets, table) <= max_detour_m
    return (tuple(s for s, keep in zip(sets, ok, strict=True) if keep),
            tuple(s for s, keep in zip(sets, ok, strict=True) if not keep))


def solve_phases(g: WalkGraph, must: tuple[CandidateSet, ...], optional: tuple[CandidateSet, ...], xy_of: XY,
                 eroded: BaseGeometry, p: PhaseParams, table: DistanceTable) -> PhaseResult:
    """Phase A + coverage loop on the must sets, then phase B on the cheap optional ones still uncovered."""
    optional = optional if p.include_optional else ()
    if not must and not optional:
        raise RouteValidationError("no reachable targets to route")
    if not must:
        final = solve_tour(g, optional, eroded, p.tour, table)
        return PhaseResult(final, optional, 0, None, final.length_m)
    best, iterations, active = cover_loop(g, must, xy_of, eroded, p, table)
    todo, skipped = cheap_sets(g, best, _uncovered(best, optional, xy_of, p.cover_radius_m) if optional else (),
                               table, p.optional_max_detour_m)
    delta = 0.0 if optional else None
    if not todo:
        return PhaseResult(best, active, iterations, best.length_m, delta, skipped)
    used = (*active, *todo)
    final = solve_tour(g, used, eroded, p.tour, table)
    return PhaseResult(final, used, iterations, best.length_m, final.length_m - best.length_m, skipped)


def finalize(tour: Tour, start_xy: tuple[float, float], decimals: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Round like the GeoJSON writer (START exact at both ends), dedupe, remap anchors."""
    xy = np.round(np.asarray(tour.xy, dtype=np.float64), decimals) + 0.0
    xy[0] = xy[-1] = np.round(np.asarray(start_xy, dtype=np.float64), decimals) + 0.0
    return dedupe_xy(xy, tour.anchors)


__all__ = ["MIN_GAIN_M", "PhaseParams", "PhaseResult", "cheap_sets", "cover_loop", "finalize", "free_sets",
           "insertion_detours", "points", "solve_phases"]
