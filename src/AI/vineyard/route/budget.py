"""Outside-budget trimming (headland risk, critic S5): which stops to drop so the route fits the budget.

Stops are grouped into runs joined by legs without outside length (one run = one entry into an
interrow). Dropping a run replaces its entry and exit legs by the shortest path from the stop before it
to the stop after it, so the saving is in + out - bypass (bypass measured on the graph). Runs are
dropped by (contains a must stop last, largest saving per target, tour order), never two neighbours in
one round, until the savings cover the excess; the caller re-solves and repeats.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from vineyard.route.candidates import ROLE_MUST
from vineyard.route.distances import leg_paths
from vineyard.route.graph_types import WalkGraph
from vineyard.route.policies import INSIDE_TOL_M, DropMode
from vineyard.route.tour import DistanceTable
from vineyard.route.unroll import edge_index

NOTE_OUTSIDE_BUDGET: Final = "outside_budget"
# The number of rounds is route.solver.budget_rounds; they stop as soon as the route fits.


@dataclass(frozen=True)
class StopRun:
    first: int
    last: int
    saving_m: float
    has_must: bool
    n_targets: int

    @property
    def positions(self) -> range:
        return range(self.first, self.last + 1)


def leg_outside(legs_outside: Sequence[tuple[int, float]], n_stops: int) -> np.ndarray:
    """Outside length per leg; leg k joins anchor k and anchor k + 1 (anchor 0 = START)."""
    per_leg = np.zeros(n_stops + 1, dtype=np.float64)
    for leg, length in legs_outside:
        if not 0 <= leg <= n_stops:
            raise ValueError(f"leg index {leg} outside [0, {n_stops}]")
        per_leg[leg] += length
    return per_leg


def run_bounds(per_leg: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of stops whose internal legs stay inside (first, last stop position)."""
    n_stops = len(per_leg) - 1
    bounds, first = [], 0
    for k in range(1, n_stops):
        if per_leg[k] > INSIDE_TOL_M:
            bounds.append((first, k - 1))
            first = k
    return [*bounds, (first, n_stops - 1)] if n_stops else []


def bypass_outside(g: WalkGraph, stop_nodes: Sequence[int], bounds: Sequence[tuple[int, int]],
                   table: DistanceTable, chunk: int) -> np.ndarray:
    """Outside length of the shortest path from the stop before each run to the stop after it."""
    seq = [g.start_node, *(int(n) for n in stop_nodes), g.start_node]
    if not bounds:
        return np.zeros(0, dtype=np.float64)
    pairs = [(seq[a], seq[b + 2]) for a, b in bounds]
    d = table.matrix(np.array(seq, dtype=np.int64))
    limits = [float(d[a, b + 2]) for a, b in bounds]
    paths = leg_paths(g.adjacency("cost"), pairs, chunk, limits)
    outside, lookup = g.outside_len_m(), edge_index(g, "cost")
    return np.array([sum(float(outside[lookup[(u, v)]]) for u, v in zip(p[:-1], p[1:], strict=True))
                     for p in paths], dtype=np.float64)


def stop_runs(per_leg: np.ndarray, bounds: Sequence[tuple[int, int]], bypass: np.ndarray, roles: Sequence[str],
              n_targets: Sequence[int]) -> tuple[StopRun, ...]:
    return tuple(StopRun(a, b, max(float(per_leg[a] + per_leg[b + 1] - bypass[i]), 0.0),
                         any(roles[k] == ROLE_MUST for k in range(a, b + 1)), int(sum(n_targets[a:b + 1])))
                 for i, (a, b) in enumerate(bounds))


def pick_runs(runs: Sequence[StopRun], excess_m: float, mode: DropMode) -> tuple[int, ...]:
    """Stop positions of the runs to drop so that their savings cover `excess_m`."""
    if mode == "none" or excess_m <= 0.0:
        return ()
    eligible = [i for i, r in enumerate(runs) if r.saving_m > 0.0 and (mode == "any" or not r.has_must)]
    order = sorted(eligible, key=lambda i: (runs[i].has_must, -runs[i].saving_m / max(runs[i].n_targets, 1), i))
    picked, blocked, total = [], set(), 0.0
    for i in order:
        if total >= excess_m:
            break
        if i in blocked:
            continue
        picked.append(i)
        blocked |= {i - 1, i + 1}
        total += runs[i].saving_m
    return tuple(k for i in sorted(picked) for k in runs[i].positions)


def plan_drops(g: WalkGraph, stop_nodes: Sequence[int], legs_outside: Sequence[tuple[int, float]],
               roles: Sequence[str], n_targets: Sequence[int], table: DistanceTable, chunk: int, excess_m: float,
               mode: DropMode) -> tuple[int, ...]:
    """Stop positions to drop this round (empty when nothing helps or nothing is allowed)."""
    if mode == "none" or excess_m <= 0.0 or not stop_nodes:
        return ()
    per_leg = leg_outside(legs_outside, len(stop_nodes))
    bounds = run_bounds(per_leg)
    runs = stop_runs(per_leg, bounds, bypass_outside(g, stop_nodes, bounds, table, chunk), roles, n_targets)
    return pick_runs(runs, excess_m, mode)


__all__ = [
    "NOTE_OUTSIDE_BUDGET", "StopRun", "bypass_outside", "leg_outside", "pick_runs", "plan_drops", "run_bounds",
    "stop_runs",
]
