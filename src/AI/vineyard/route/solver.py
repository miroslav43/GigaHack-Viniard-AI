"""GTSP solver (arch §4.11.5): OR-Tools routing with one mandatory disjunction per candidate set.

Index space: 0 = START (depot), every other index belongs to exactly one set. Costs are integer cm.
`method=auto` uses the exact DP for tiny instances, OR-Tools otherwise, and python-tsp when OR-Tools
is missing or fails; the reason is logged and recorded in `SolverResult.fallback_reason`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from vineyard.config import RouteSolverConfig
from vineyard.errors import ConfigError
from vineyard.logging_setup import get_logger, log_event

DEPOT: Final = 0
N_VEHICLES: Final = 1
# CONFIG-REQUEST: route.solver.exact_max_sets = 8
EXACT_MAX_SETS: Final = 8
# CONFIG-REQUEST: route.solver.exact_max_candidates = 64
EXACT_MAX_CANDIDATES: Final = 64

_log = get_logger("route.solver")


@dataclass(frozen=True)
class SolverResult:
    """`order` starts at the depot and lists one index per set (the return to START is implicit)."""

    order: tuple[int, ...]
    cost_cm: int
    solver: str
    solve_time_s: float
    fallback_reason: str = ""


class SolverFailed(RuntimeError):
    """OR-Tools returned no solution."""


def validate_gtsp_input(d_cm: np.ndarray, sets: Sequence[Sequence[int]]) -> None:
    """Square matrix; sets non-empty, disjoint, never contain the depot, and cover every other index."""
    d = np.asarray(d_cm)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 1:
        raise ValueError(f"cost matrix must be square and non-empty, got shape {d.shape}")
    n = d.shape[0]
    owner = np.full(n, -1, dtype=np.int64)
    for k, members in enumerate(sets):
        if len(members) == 0:
            raise ValueError(f"candidate set {k} is empty")
        for i in members:
            if i == DEPOT:
                raise ValueError(f"candidate set {k} contains the depot index {DEPOT}")
            if not 0 < i < n:
                raise ValueError(f"candidate set {k}: index {i} out of range [1, {n})")
            if owner[i] >= 0:
                raise ValueError(f"index {i} is in more than one set ({owner[i]} and {k})")
            owner[i] = k
    missing = np.flatnonzero(owner[1:] < 0) + 1
    if len(missing):
        raise ValueError(f"indices not covered by any set: {missing[:10].tolist()}")


def tour_cost_cm(d_cm: np.ndarray, order: Sequence[int]) -> int:
    """Closed-tour cost of `order` (starting at the depot) including the return leg."""
    idx = np.asarray(order, dtype=np.int64)
    if len(idx) <= 1:
        return 0
    nxt = np.roll(idx, -1)
    return int(np.asarray(d_cm, dtype=np.int64)[idx, nxt].sum())


def time_limit_for(cfg: RouteSolverConfig) -> int:
    return cfg.time_limit_final_s if cfg.final else cfg.time_limit_s


def _enum_value(enum_type: object, name: str, what: str) -> int:
    value = getattr(enum_type, name, None)
    if not isinstance(value, int):
        raise ConfigError(f"unknown OR-Tools {what}", key=f"route.solver.{what}", value=name)
    return value


def _read_order(routing: object, manager: object, solution: object) -> tuple[int, ...]:
    index = routing.Start(0)
    order: list[int] = []
    while not routing.IsEnd(index):
        order.append(int(manager.IndexToNode(index)))
        index = solution.Value(routing.NextVar(index))
    return tuple(order)


def solve_ortools(d_cm: np.ndarray, sets: Sequence[Sequence[int]], cfg: RouteSolverConfig,
                  time_limit_s: int) -> tuple[int, ...]:
    """OR-Tools GTSP; raises ImportError when OR-Tools is absent and SolverFailed without a solution."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    manager = pywrapcp.RoutingIndexManager(len(d_cm), N_VEHICLES, DEPOT)
    routing = pywrapcp.RoutingModel(manager)
    transit = routing.RegisterTransitMatrix(np.asarray(d_cm, dtype=np.int64).tolist())
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    for members in sets:
        # No penalty argument: the disjunction is mandatory, exactly one member is visited.
        routing.AddDisjunction([manager.NodeToIndex(int(i)) for i in members])
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = _enum_value(routing_enums_pb2.FirstSolutionStrategy, cfg.first_solution,
                                                 "first_solution")
    params.local_search_metaheuristic = _enum_value(routing_enums_pb2.LocalSearchMetaheuristic, cfg.metaheuristic,
                                                    "metaheuristic")
    params.time_limit.FromSeconds(int(time_limit_s))
    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise SolverFailed(f"OR-Tools found no solution (status {routing.status()}, n={len(d_cm)})")
    return _read_order(routing, manager, solution)


def _result(d_cm: np.ndarray, order: tuple[int, ...], solver: str, t0: float, reason: str = "") -> SolverResult:
    return SolverResult(order=order, cost_cm=tour_cost_cm(d_cm, order), solver=solver,
                        solve_time_s=round(time.perf_counter() - t0, 3), fallback_reason=reason)


def _is_tiny(sets: Sequence[Sequence[int]]) -> bool:
    return len(sets) <= EXACT_MAX_SETS and sum(len(s) for s in sets) <= EXACT_MAX_CANDIDATES


def _auto(d_cm: np.ndarray, sets: Sequence[Sequence[int]], cfg: RouteSolverConfig, limit: int) -> SolverResult:
    from vineyard.route.solver_fallback import solve_gtsp_exact, solve_gtsp_fallback

    if _is_tiny(sets):
        return solve_gtsp_exact(d_cm, sets)
    t0 = time.perf_counter()
    try:
        return _result(d_cm, solve_ortools(d_cm, sets, cfg, limit), "ortools", t0)
    except ConfigError:
        raise
    except Exception as exc:  # design 04 §3.8: any OR-Tools failure falls back, logged with its reason
        reason = f"{type(exc).__name__}: {exc}"
        log_event(_log, "route.solver.fallback", level=logging.WARNING, reason=reason, n_sets=len(sets))
    fallback = solve_gtsp_fallback(d_cm, sets, time_s=cfg.fallback_time_s)
    return SolverResult(fallback.order, fallback.cost_cm, fallback.solver, fallback.solve_time_s, reason)


def solve_gtsp(d_cm: np.ndarray, sets: Sequence[Sequence[int]], cfg: RouteSolverConfig, *,
               time_limit_s: int | None = None) -> SolverResult:
    """Best closed tour from the depot visiting exactly one index of every set."""
    validate_gtsp_input(d_cm, sets)
    if not sets:
        return SolverResult(order=(DEPOT,), cost_cm=0, solver="trivial", solve_time_s=0.0)
    limit = time_limit_s if time_limit_s is not None else time_limit_for(cfg)
    if cfg.method == "ortools":
        t0 = time.perf_counter()
        return _result(d_cm, solve_ortools(d_cm, sets, cfg, limit), "ortools", t0)
    if cfg.method == "fallback":
        from vineyard.route.solver_fallback import solve_gtsp_fallback

        return solve_gtsp_fallback(d_cm, sets, time_s=cfg.fallback_time_s)
    return _auto(d_cm, sets, cfg, limit)


__all__ = [
    "DEPOT", "EXACT_MAX_CANDIDATES", "EXACT_MAX_SETS", "SolverFailed", "SolverResult", "solve_gtsp",
    "solve_ortools", "time_limit_for", "tour_cost_cm", "validate_gtsp_input",
]
