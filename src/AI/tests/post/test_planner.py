"""plan_route end to end on synthetic mini-vineyards (arch §4.11, design 04 §3.8-3.9)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, box

from vineyard.config import load_config
from vineyard.errors import RouteValidationError
from vineyard.route.baseline import compute_baseline
from vineyard.route.budget import NOTE_OUTSIDE_BUDGET, StopRun, leg_outside, pick_runs, run_bounds, stop_runs
from vineyard.route.candidates import NOTE_OUTSIDE_ONLY, ROLE_OPTIONAL, TargetPoint
from vineyard.route.planner import PlanParams, candidates_with_ladder, plan_route
from vineyard.route.policies import DEFAULT_POLICIES
from vineyard.route.stops import NOTE_NOT_ROUTED

from .route_factories import CO, IC, PC, graph_from_lines, mini_route


@pytest.fixture(scope="module")
def cfg():
    route = load_config().route
    return route.model_copy(update={"solver": route.solver.model_copy(update={"time_limit_s": 1,
                                                                              "fallback_time_s": 1})})


def _params(cfg, start_xy, **changes) -> PlanParams:
    return dataclasses.replace(PlanParams.from_route_cfg(cfg, start_xy), **changes)


def _tp(tid: str, x: float, y: float, role: str = "must", **kw) -> TargetPoint:
    return TargetPoint(target_id=tid, x=x, y=y, role=role, **kw)


def _vineyard_targets(m) -> list[TargetPoint]:
    rows = m.row_ys
    gaps = [(1012.0, rows[1]), (1030.0, rows[1]), (1045.0, rows[4]), (1020.0, rows[4]), (1050.0, rows[7]),
            (1008.0, rows[7])]
    out = [_tp(f"T-GAP-{k:04d}", x, y) for k, (x, y) in enumerate(gaps, start=1)]
    return out + [_tp("T-WST-0001", m.x0 + m.length_m + 2.5, m.centre_ys[4] + 0.7, priority=1)]


def test_mini_vineyard_route_validates_and_beats_baselines(cfg) -> None:
    m = mini_route(n_rows=10)
    targets = _vineyard_targets(m)
    plan = plan_route(m.graph, targets, m.inner, m.eroded, _params(cfg, m.start_xy))
    v = plan.validation
    assert v.passed, v.failures
    assert v.coverage_required == 1.0 and v.outside_frac <= 0.005
    assert plan.policy == "penalty"
    assert tuple(plan.xy[0]) == m.start_xy and tuple(plan.xy[-1]) == m.start_xy
    assert all(vis.covered for vis in plan.visits)
    base = compute_baseline(plan.graph, plan.sets, v.length_m, m.start_xy, chunk=64, walking_speed_kmh=4.0)
    assert v.length_m < base.all_interrows_m < base.serpentine_est_m
    assert v.length_m <= base.id_order_m
    assert [s.target_id for s in plan.stops if s.target_id] and plan.stops[0].seq == 0
    assert plan.stops[-1].cum_dist_m == pytest.approx(v.length_m)


def test_route_is_deterministic(cfg) -> None:
    m = mini_route(n_rows=10)
    a = plan_route(m.graph, _vineyard_targets(m), m.inner, m.eroded, _params(cfg, m.start_xy))
    b = plan_route(m.graph, _vineyard_targets(m), m.inner, m.eroded, _params(cfg, m.start_xy))
    assert np.array_equal(a.xy, b.xy) and a.stops == b.stops


def test_dead_end_out_and_back_is_accepted(cfg) -> None:
    m = mini_route(n_rows=3, east_gap_m=1.0)
    t = _tp("T-GAP-0001", m.x0 + 55.0, m.row_ys[1])
    plan = plan_route(m.graph, [t], m.inner, m.eroded, _params(cfg, m.start_xy))
    assert plan.validation.passed, plan.validation.failures
    assert not plan.line.is_simple
    assert plan.validation.outside_len_m == 0.0


def test_headland_gap_switches_to_a_stricter_policy(cfg) -> None:
    m = mini_route(n_rows=5, east_gap_m=1.0)
    targets = [_tp("T-GAP-0001", m.x0 + 50.0, m.centre_ys[0] - 1.3), _tp("T-GAP-0002", m.x0 + 50.0,
                                                                          m.centre_ys[3] + 1.3)]
    plan = plan_route(m.graph, targets, m.inner, m.eroded, _params(cfg, m.start_xy))
    first = plan.reports[0]
    assert first.policy == "penalty" and not first.passed and first.outside_frac > 0.005
    assert plan.policy != "penalty" and plan.validation.passed
    assert plan.validation.outside_frac <= 0.005


def test_target_only_reachable_outside_becomes_outside_only(cfg) -> None:
    probe = mini_route(n_rows=5)
    y_top, xw = probe.centre_ys[-1], probe.start_xy[0]
    island = [(LineString([(xw, y_top), (xw, y_top + 8.0)]), CO, "N"),
              (LineString([(xw, y_top + 8.0), (xw + 30.0, y_top + 8.0)]), IC, "island")]
    m = mini_route(n_rows=5, extra_lines=island)
    targets = [_tp("T-GAP-0001", m.x0 + 30.0, m.row_ys[2]), _tp("T-GAP-0002", xw + 20.0, y_top + 9.0)]
    plan = plan_route(m.graph, targets, m.inner, m.eroded, _params(cfg, m.start_xy))
    assert plan.validation.passed and plan.policy == "drop_outside"
    visit = {v.target_id: v for v in plan.visits}
    assert visit["T-GAP-0002"].reach_note == NOTE_OUTSIDE_BUDGET and not visit["T-GAP-0002"].reachable_final
    assert visit["T-GAP-0001"].covered and plan.dropped == ("T-GAP-0002",)
    assert [r.policy for r in plan.reports] == ["penalty", "penalty_x10", "drop_optional", "must_only",
                                                "drop_outside"]
    assert plan.reports[-1].n_outside_budget == 1
    capped = _params(cfg, m.start_xy, policies=tuple(p for p in DEFAULT_POLICIES if p.name in ("penalty", "cap_outside")))
    plan = plan_route(m.graph, targets, m.inner, m.eroded, capped)
    assert plan.validation.passed and plan.policy == "cap_outside"
    visit = {v.target_id: v for v in plan.visits}
    assert visit["T-GAP-0002"].reach_note == NOTE_OUTSIDE_ONLY and plan.reports[-1].n_outside_only == 1


def _free_cover_case():
    lines = [(LineString([(0.0, 0.0), (40.0, 0.0)]), PC, "m1"), (LineString([(0.0, -1.2), (40.0, -1.2)]), PC, "m2"),
             (LineString([(0.0, 0.0), (0.0, -1.2)]), CO, "c0"), (LineString([(40.0, 0.0), (40.0, -1.2)]), CO, "c1")]
    dom = box(-1.0, -1.8, 41.0, 0.6)
    inner, eroded = dom.buffer(-0.05, join_style="mitre"), dom.buffer(-0.3, join_style="mitre")
    shapely.prepare(inner)
    shapely.prepare(eroded)
    g = graph_from_lines(lines, (0.0, -1.2), node_step_m=4.0, inner=inner)
    targets = [_tp("T-GAP-0001", 20.0, 0.6), _tp("T-WST-0001", 40.0, -2.2)]
    return g, targets, inner, eroded


def test_target_covered_for_free_is_dropped_and_coverage_stays_full(cfg) -> None:
    g, targets, inner, eroded = _free_cover_case()
    params = _params(cfg, (0.0, -1.2))
    params = dataclasses.replace(params, candidates=params.candidates.with_changes(max_per_side=1))
    plan = plan_route(g, targets, inner, eroded, params)
    assert plan.iterations == 1
    assert plan.validation.coverage_required == 1.0 and plan.validation.passed
    assert [s.target_id for s in plan.stops if s.target_id] == ["T-WST-0001"]
    assert plan.must_length_m == pytest.approx(plan.validation.length_m, abs=0.02)


def test_no_coverage_loop_keeps_every_anchor(cfg) -> None:
    g, targets, inner, eroded = _free_cover_case()
    params = _params(cfg, (0.0, -1.2), cover_iterations=0)
    params = dataclasses.replace(params, candidates=params.candidates.with_changes(max_per_side=1))
    plan = plan_route(g, targets, inner, eroded, params)
    assert plan.iterations == 0
    assert sorted(s.target_id for s in plan.stops if s.target_id) == ["T-GAP-0001", "T-WST-0001"]


def test_optional_phase_adds_uncovered_optional_targets(cfg) -> None:
    m = mini_route(n_rows=6)
    must = _tp("T-GAP-0001", m.x0 + 10.0, m.row_ys[1])
    opt = _tp("T-MSP-0001", m.x0 + 40.0, m.row_ys[4], role=ROLE_OPTIONAL, priority=3)
    with_opt = plan_route(m.graph, [must, opt], m.inner, m.eroded, _params(cfg, m.start_xy))
    without = plan_route(m.graph, [must, opt], m.inner, m.eroded, _params(cfg, m.start_xy, include_optional=False))
    assert with_opt.optional_delta_m is not None and with_opt.optional_delta_m > 0.0
    assert {v.target_id: v.covered for v in with_opt.visits}["T-MSP-0001"]
    assert without.optional_delta_m is None
    assert without.validation.length_m < with_opt.validation.length_m
    skipped = {v.target_id: v for v in without.visits}["T-MSP-0001"]
    assert not skipped.covered and not skipped.reachable_final and skipped.reach_note == NOTE_NOT_ROUTED


def test_optional_targets_left_out_by_must_only_are_reported_unreachable(cfg) -> None:
    m = mini_route(n_rows=6)
    must = _tp("T-GAP-0001", m.x0 + 10.0, m.row_ys[1])
    opt = _tp("T-MSP-0001", m.x0 + 40.0, m.row_ys[4], role=ROLE_OPTIONAL, priority=3)
    must_only = tuple(p for p in DEFAULT_POLICIES if p.name == "must_only")
    plan = plan_route(m.graph, [must, opt], m.inner, m.eroded, _params(cfg, m.start_xy, policies=must_only))
    visit = {v.target_id: v for v in plan.visits}
    assert plan.policy == "must_only" and plan.validation.passed and visit["T-GAP-0001"].covered
    assert not visit["T-MSP-0001"].reachable_final and visit["T-MSP-0001"].reach_note == NOTE_OUTSIDE_BUDGET
    assert all(v.covered for v in plan.visits if v.reachable_final)


def test_only_optional_targets_are_routed(cfg) -> None:
    m = mini_route(n_rows=4)
    opt = _tp("T-MSP-0001", m.x0 + 40.0, m.row_ys[1], role=ROLE_OPTIONAL, priority=3)
    plan = plan_route(m.graph, [opt], m.inner, m.eroded, _params(cfg, m.start_xy))
    assert plan.validation.passed and plan.must_length_m is None
    assert plan.visits[0].covered


def test_unreachable_target_is_reported_not_required(cfg) -> None:
    m = mini_route(n_rows=4)
    targets = [_tp("T-GAP-0001", m.x0 + 20.0, m.row_ys[1]), _tp("T-WST-0001", m.x0 + 20.0, m.row_ys[-1] + 30.0)]
    plan = plan_route(m.graph, targets, m.inner, m.eroded, _params(cfg, m.start_xy))
    assert plan.validation.passed
    visit = {v.target_id: v for v in plan.visits}
    assert not visit["T-WST-0001"].reachable_final and visit["T-WST-0001"].reach_note == "too_far"
    assert plan.validation.coverage_est == pytest.approx(0.5)


def test_no_reachable_target_raises(cfg) -> None:
    m = mini_route(n_rows=4)
    with pytest.raises(RouteValidationError, match="no outside policy"):
        plan_route(m.graph, [_tp("T-WST-0001", 0.0, 0.0)], m.inner, m.eroded, _params(cfg, m.start_xy))


def test_node_cap_ladder_steps_down(cfg) -> None:
    m = mini_route(n_rows=6)
    targets = [_tp(f"T-GAP-{k:04d}", m.x0 + 5.3 * k, m.row_ys[2]) for k in range(1, 8)]
    aug0, level0 = candidates_with_ladder(m.graph, targets, m.inner, _params(cfg, m.start_xy))
    n0 = 1 + sum(len(s.nodes) for s in aug0.sets)
    aug1, level1 = candidates_with_ladder(m.graph, targets, m.inner, _params(cfg, m.start_xy, max_tsp_nodes=n0 - 1))
    assert (level0, level1) == (0, 1)
    assert all(len(s.nodes) <= 2 for s in aug1.sets)


def test_ladder_truncates_optional_then_raises_on_must(cfg) -> None:
    m = mini_route(n_rows=6)
    must = [_tp(f"T-GAP-{k:04d}", m.x0 + 8.0 * k, m.row_ys[2]) for k in range(1, 3)]
    opt = [_tp(f"T-MSP-{k:04d}", m.x0 + 8.0 * k, m.row_ys[4], role=ROLE_OPTIONAL, priority=3, gap_length_m=2.0 + k)
           for k in range(1, 4)]
    aug, level = candidates_with_ladder(m.graph, [*must, *opt], m.inner, _params(cfg, m.start_xy, max_tsp_nodes=7))
    assert level == 3
    kept = {t for s in aug.sets for t in s.target_ids}
    assert set(t.target_id for t in must) <= kept and "T-MSP-0003" in kept
    assert {r.target_id for r in aug.reach if r.note == "truncated"} == {"T-MSP-0001", "T-MSP-0002"}
    with pytest.raises(RouteValidationError, match="max_tsp_nodes"):
        candidates_with_ladder(m.graph, must, m.inner, _params(cfg, m.start_xy, max_tsp_nodes=2))


def test_policy_report_json_has_no_nan(cfg) -> None:
    m = mini_route(n_rows=4)
    plan = plan_route(m.graph, [_tp("T-GAP-0001", m.x0 + 20.0, m.row_ys[1])], m.inner, m.eroded,
                      _params(cfg, m.start_xy))
    doc = plan.reports[0].to_json()
    assert doc["policy"] == "penalty" and doc["passed"] is True and isinstance(doc["failures"], list)


# ------------------------------------------------------------------ outside budget (budget.py)


def test_leg_outside_and_runs() -> None:
    per_leg = leg_outside([(0, 2.0), (3, 3.0), (5, 1.0)], 5)
    assert per_leg.tolist() == [2.0, 0.0, 0.0, 3.0, 0.0, 1.0]
    assert run_bounds(per_leg) == [(0, 2), (3, 4)]
    assert run_bounds(np.zeros(1)) == []
    with pytest.raises(ValueError, match="leg index"):
        leg_outside([(7, 1.0)], 5)


def test_stop_runs_saving_uses_bypass() -> None:
    per_leg = np.array([2.0, 0.0, 0.0, 3.0, 0.0, 1.0])
    runs = stop_runs(per_leg, [(0, 2), (3, 4)], np.array([1.0, 5.0]), ["optional", "must", "optional",
                                                                          "optional", "optional"], [1, 1, 2, 1, 1])
    assert (runs[0].saving_m, runs[0].has_must, runs[0].n_targets) == (4.0, True, 4)
    assert (runs[1].saving_m, runs[1].has_must) == (0.0, False)


def test_pick_runs_modes_order_and_neighbours() -> None:
    runs = (StopRun(0, 0, 4.0, False, 1), StopRun(1, 2, 6.0, False, 2), StopRun(3, 3, 5.0, True, 1),
            StopRun(4, 4, 1.0, False, 1))
    assert pick_runs(runs, 3.0, "none") == () and pick_runs(runs, 0.0, "any") == ()
    assert pick_runs(runs, 3.5, "optional") == (0,)            # 4.0/1 beats 6.0/2; covers the excess
    assert pick_runs(runs, 100.0, "optional") == (0, 4)        # run 1 is a neighbour of run 0 this round
    assert pick_runs(runs, 100.0, "any") == (0, 4)             # run 2 (must) is last and next to run 3
    lone_must = (StopRun(0, 1, 1.0, True, 2),)
    assert pick_runs(lone_must, 0.5, "any") == (0, 1) and pick_runs(lone_must, 0.5, "optional") == ()
