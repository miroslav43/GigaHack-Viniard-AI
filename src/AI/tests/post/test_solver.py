"""GTSP solvers (arch §4.11.5): OR-Tools, python-tsp fallback + set DP, exact DP, all against brute force."""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pytest

from vineyard.config import RouteSolverConfig, load_config
from vineyard.route.solver import SolverResult, solve_gtsp, tour_cost_cm, validate_gtsp_input
from vineyard.route.solver_fallback import best_members_for_order, solve_gtsp_exact, solve_gtsp_fallback


@pytest.fixture(scope="module")
def solver_cfg() -> RouteSolverConfig:
    return load_config().route.solver.model_copy(update={"time_limit_s": 1, "fallback_time_s": 1})


def _instance(n_sets: int, per_set: int, seed: int) -> tuple[np.ndarray, list[list[int]]]:
    rng = np.random.default_rng(seed)
    pts = rng.random((1 + n_sets * per_set, 2)) * 100.0
    d = np.rint(np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)) * 100).astype(np.int64)
    sets = [list(range(1 + s * per_set, 1 + (s + 1) * per_set)) for s in range(n_sets)]
    return d, sets


def _clustered_instance(n_sets: int, per_set: int, seed: int) -> tuple[np.ndarray, list[list[int]]]:
    """Realistic GTSP: each set's candidates lie within 1.9 m of its target (targets 10-100 m apart)."""
    rng = np.random.default_rng(seed)
    centres = rng.random((n_sets, 2)) * 100.0
    angles = rng.random((n_sets, per_set)) * 2 * np.pi
    cands = centres[:, None, :] + 1.9 * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    pts = np.vstack([[50.0, -5.0], cands.reshape(-1, 2)])
    d = np.rint(np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)) * 100).astype(np.int64)
    sets = [list(range(1 + s * per_set, 1 + (s + 1) * per_set)) for s in range(n_sets)]
    return d, sets


def _brute_force(d: np.ndarray, sets: list[list[int]]) -> int:
    """Every set order x every member combination (members enumerated exhaustively per order)."""
    best = None
    for perm in itertools.permutations(range(len(sets))):
        if perm[0] > perm[-1]:
            continue  # symmetric costs: a tour and its reverse cost the same
        combos = np.array(list(itertools.product(*(sets[s] for s in perm))), dtype=np.int64)
        path = np.hstack([np.zeros((len(combos), 1), np.int64), combos, np.zeros((len(combos), 1), np.int64)])
        cost = int(d[path[:, :-1], path[:, 1:]].sum(axis=1).min())
        best = cost if best is None or cost < best else best
    assert best is not None
    return best


def _far_side_toy() -> tuple[np.ndarray, list[list[int]]]:
    """START at x=0; each target has a near candidate (x=10+k, y=0) and a far-side one (y=2.6).

    The far-side candidates lie on a straight corridor, so choosing them all is optimal only jointly.
    """
    near = [(10.0 + 3 * k, 0.0) for k in range(3)]
    far = [(10.0 + 3 * k, 2.6) for k in range(3)]
    pts = np.array([(0.0, 2.6), *[p for pair in zip(near, far, strict=True) for p in pair]])
    d = np.rint(np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)) * 100).astype(np.int64)
    sets = [[1 + 2 * k, 2 + 2 * k] for k in range(3)]
    return d, sets


@pytest.mark.parametrize(("n_sets", "per_set", "seed"), [(3, 2, 0), (5, 2, 1), (6, 3, 2), (7, 2, 3)])
def test_ortools_matches_brute_force(solver_cfg: RouteSolverConfig, n_sets: int, per_set: int, seed: int) -> None:
    d, sets = _instance(n_sets, per_set, seed)
    res = solve_gtsp(d, sets, solver_cfg.model_copy(update={"method": "ortools"}))
    assert res.solver == "ortools"
    assert res.cost_cm == _brute_force(d, sets)
    assert res.cost_cm == tour_cost_cm(d, res.order)
    assert res.order[0] == 0 and len(res.order) == n_sets + 1


@pytest.mark.parametrize(("n_sets", "per_set", "seed"), [(3, 2, 0), (5, 2, 1), (6, 3, 2), (7, 2, 3)])
def test_exact_matches_brute_force(n_sets: int, per_set: int, seed: int) -> None:
    d, sets = _instance(n_sets, per_set, seed)
    res = solve_gtsp_exact(d, sets)
    assert res.cost_cm == _brute_force(d, sets)
    assert res.cost_cm == tour_cost_cm(d, res.order)


@pytest.mark.parametrize(("n_sets", "per_set", "seed"), [(3, 2, 0), (5, 2, 1), (6, 3, 2), (7, 2, 3)])
def test_fallback_matches_brute_force_on_clustered_sets(n_sets: int, per_set: int, seed: int) -> None:
    d, sets = _clustered_instance(n_sets, per_set, seed)
    optimum = _brute_force(d, sets)
    fallback = solve_gtsp_fallback(d, sets, time_s=1)
    assert fallback.solver == "python_tsp"
    assert fallback.cost_cm == optimum


def test_far_side_candidate_is_chosen(solver_cfg: RouteSolverConfig) -> None:
    d, sets = _far_side_toy()
    optimum = _brute_force(d, sets)
    for method in ("ortools", "fallback", "auto"):
        res = solve_gtsp(d, sets, solver_cfg.model_copy(update={"method": method}))
        assert res.cost_cm == optimum, method
    assert set(solve_gtsp_exact(d, sets).order[1:]) == {2, 4, 6}


def test_each_set_visited_exactly_once(solver_cfg: RouteSolverConfig) -> None:
    d, sets = _instance(6, 3, 7)
    res = solve_gtsp(d, sets, solver_cfg.model_copy(update={"method": "ortools"}))
    visited = res.order[1:]
    assert sorted(next(i for i, s in enumerate(sets) if n in s) for n in visited) == list(range(6))


def test_auto_uses_exact_for_small_and_is_deterministic(solver_cfg: RouteSolverConfig) -> None:
    d, sets = _instance(5, 2, 11)
    a = solve_gtsp(d, sets, solver_cfg)
    b = solve_gtsp(d, sets, solver_cfg)
    assert a.solver == "exact_dp"
    assert (a.order, a.cost_cm) == (b.order, b.cost_cm)


def test_auto_falls_back_without_ortools(solver_cfg: RouteSolverConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ortools.constraint_solver", None)
    d, sets = _instance(9, 2, 5)
    res = solve_gtsp(d, sets, solver_cfg)
    assert res.solver == "python_tsp"
    assert res.cost_cm == tour_cost_cm(d, res.order)


def test_ortools_method_without_ortools_raises(solver_cfg: RouteSolverConfig,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ortools.constraint_solver", None)
    d, sets = _instance(3, 2, 5)
    with pytest.raises(ImportError):
        solve_gtsp(d, sets, solver_cfg.model_copy(update={"method": "ortools"}))


def test_fallback_is_deterministic() -> None:
    d, sets = _instance(12, 2, 4)
    a = solve_gtsp_fallback(d, sets, time_s=1)
    b = solve_gtsp_fallback(d, sets, time_s=1)
    assert (a.order, a.cost_cm) == (b.order, b.cost_cm)


def test_no_sets_is_trivial(solver_cfg: RouteSolverConfig) -> None:
    res = solve_gtsp(np.zeros((1, 1), np.int64), [], solver_cfg)
    assert res == SolverResult(order=(0,), cost_cm=0, solver="trivial", solve_time_s=0.0)


def test_best_members_for_fixed_order() -> None:
    d, sets = _far_side_toy()
    members, cost = best_members_for_order(d, sets)
    assert members == (2, 4, 6)
    assert cost == tour_cost_cm(d, (0, *members))


@pytest.mark.parametrize(
    ("sets", "match"),
    [([[1], [1]], "more than one set"), ([[0]], "depot"), ([[5]], "out of range"), ([[]], "empty"),
     ([[1]], "not covered")],
)
def test_validate_gtsp_input_errors(sets: list[list[int]], match: str) -> None:
    d = np.zeros((3, 3), np.int64)
    with pytest.raises(ValueError, match=match):
        validate_gtsp_input(d, sets)


def test_validate_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        validate_gtsp_input(np.zeros((2, 3), np.int64), [[1]])
