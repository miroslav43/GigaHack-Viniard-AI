"""Solvers used without OR-Tools (arch §4.11.5, design 04 §3.8).

- `solve_gtsp_exact`: Held-Karp over sets with member choice; optimal, for tiny instances.
- `solve_gtsp_fallback`: python-tsp local search (2-opt) then Lin-Kernighan on one representative per
  set (the member nearest START), then an exact DP that picks the best member of every set for that order.

python-tsp draws neighbours with the global `random` module; it is seeded and restored around the call so
the fallback is deterministic and leaves no global side effect.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Final

import numpy as np

from vineyard.route.solver import DEPOT, SolverResult, tour_cost_cm

FALLBACK_SEED: Final = 20260926
# Order/member refinement rounds: re-optimise the order on the members the DP chose, then re-pick members.
# CONFIG-REQUEST: route.solver.fallback_rounds = 2
FALLBACK_ROUNDS: Final = 2
# CONFIG-REQUEST: route.solver.lk_max_nodes = 400
LK_MAX_NODES: Final = 400
_INF: Final = np.iinfo(np.int64).max // 4


@contextmanager
def _seeded_random(seed: int) -> Iterator[None]:
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def _flatten(sets: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    members = np.array([int(i) for s in sets for i in s], dtype=np.int64)
    owner = np.array([k for k, s in enumerate(sets) for _ in s], dtype=np.int64)
    return members, owner


def _relax_mask(dp: np.ndarray, parent: np.ndarray, mask: int, d: np.ndarray, members: np.ndarray,
                positions: Sequence[np.ndarray]) -> None:
    row = dp[mask]
    active = np.flatnonzero(row < _INF)
    for s, target in enumerate(positions):
        if mask >> s & 1:
            continue
        cand = row[active][:, None] + d[np.ix_(members[active], members[target])]
        arg = cand.argmin(axis=0)
        best = cand[arg, np.arange(len(target))]
        nxt = mask | (1 << s)
        better = best < dp[nxt, target]
        dp[nxt, target[better]] = best[better]
        parent[nxt, target[better]] = active[arg[better]]


def _backtrack(parent: np.ndarray, owner: np.ndarray, full: int, end: int) -> list[int]:
    path, mask, pos = [], full, end
    while pos >= 0:
        path.append(pos)
        prev = int(parent[mask, pos])
        mask &= ~(1 << int(owner[pos]))
        pos = prev
    return path[::-1]


def solve_gtsp_exact(d_cm: np.ndarray, sets: Sequence[Sequence[int]]) -> SolverResult:
    """Optimal GTSP tour by dynamic programming over (visited sets, last member); O(2^m m K^2)."""
    t0 = time.perf_counter()
    d = np.asarray(d_cm, dtype=np.int64)
    members, owner = _flatten(sets)
    m = len(sets)
    positions = [np.flatnonzero(owner == s) for s in range(m)]
    dp = np.full((1 << m, len(members)), _INF, dtype=np.int64)
    parent = np.full((1 << m, len(members)), -1, dtype=np.int64)
    for pos, node in enumerate(members):
        dp[1 << int(owner[pos]), pos] = d[DEPOT, node]
    for mask in range(1, 1 << m):
        _relax_mask(dp, parent, mask, d, members, positions)
    full = (1 << m) - 1
    end = int(np.argmin(dp[full] + d[members, DEPOT]))
    order = (DEPOT, *(int(members[p]) for p in _backtrack(parent, owner, full, end)))
    return SolverResult(order=order, cost_cm=tour_cost_cm(d, order), solver="exact_dp",
                        solve_time_s=round(time.perf_counter() - t0, 3))


def best_members_for_order(d_cm: np.ndarray, set_order: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], int]:
    """Best member of each set for a fixed cyclic set order starting and ending at the depot."""
    d = np.asarray(d_cm, dtype=np.int64)
    layers = [np.asarray(s, dtype=np.int64) for s in set_order]
    cost = d[DEPOT, layers[0]].copy()
    parents: list[np.ndarray] = []
    for prev, nxt in zip(layers[:-1], layers[1:], strict=True):
        cand = cost[:, None] + d[np.ix_(prev, nxt)]
        arg = cand.argmin(axis=0)
        parents.append(arg)
        cost = cand[arg, np.arange(len(nxt))]
    pos = int(np.argmin(cost + d[layers[-1], DEPOT]))
    chosen = [pos]
    for arg in reversed(parents):
        chosen.append(int(arg[chosen[-1]]))
    chosen.reverse()
    members = tuple(int(layer[p]) for layer, p in zip(layers, chosen, strict=True))
    return members, tour_cost_cm(d, (DEPOT, *members))


def nearest_neighbour_tour(d: np.ndarray) -> list[int]:
    """Greedy tour from the depot; ties go to the lowest index."""
    n = len(d)
    tour, left = [DEPOT], set(range(1, n))
    while left:
        here = tour[-1]
        nxt = min(left, key=lambda j: (d[here, j], j))
        tour.append(nxt)
        left.remove(nxt)
    return tour


def _rotate_to_depot(perm: Sequence[int]) -> list[int]:
    perm = [int(p) for p in perm]
    k = perm.index(DEPOT)
    return perm[k:] + perm[:k]


def _improve_order(reduced: np.ndarray, time_s: float) -> list[int]:
    from python_tsp.heuristics import solve_tsp_lin_kernighan, solve_tsp_local_search

    x0 = nearest_neighbour_tour(reduced)
    if len(reduced) < 3:
        return x0
    with _seeded_random(FALLBACK_SEED):
        perm, _ = solve_tsp_local_search(reduced, x0=x0, perturbation_scheme="two_opt",
                                         max_processing_time=time_s)
        if len(reduced) <= LK_MAX_NODES:
            perm, _ = solve_tsp_lin_kernighan(reduced, x0=_rotate_to_depot(perm))
    return _rotate_to_depot(perm)


def _order_then_members(d: np.ndarray, sets: Sequence[Sequence[int]], reps: Sequence[int],
                        time_s: float) -> tuple[tuple[int, ...], int]:
    nodes = np.array([DEPOT, *reps], dtype=np.int64)
    perm = _improve_order(d[np.ix_(nodes, nodes)].astype(np.float64), time_s)
    return best_members_for_order(d, [sets[p - 1] for p in perm[1:]])


def solve_gtsp_fallback(d_cm: np.ndarray, sets: Sequence[Sequence[int]], *, time_s: float) -> SolverResult:
    """python-tsp on set representatives, then the exact member choice for that set order (refined)."""
    t0 = time.perf_counter()
    d = np.asarray(d_cm, dtype=np.int64)
    owner = {int(i): k for k, s in enumerate(sets) for i in s}
    reps = [min((int(i) for i in s), key=lambda i: (d[DEPOT, i], i)) for s in sets]
    best_members, best_cost = _order_then_members(d, sets, reps, time_s)
    for _ in range(FALLBACK_ROUNDS - 1):
        reps = sorted(best_members, key=lambda i: owner[i])
        members, cost = _order_then_members(d, sets, reps, time_s)
        if cost >= best_cost:
            break
        best_members, best_cost = members, cost
    return SolverResult(order=(DEPOT, *best_members), cost_cm=best_cost, solver="python_tsp",
                        solve_time_s=round(time.perf_counter() - t0, 3))


__all__ = [
    "FALLBACK_ROUNDS", "FALLBACK_SEED", "LK_MAX_NODES", "best_members_for_order", "nearest_neighbour_tour",
    "solve_gtsp_exact", "solve_gtsp_fallback",
]
