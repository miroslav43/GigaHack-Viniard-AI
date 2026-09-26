"""policy_choice: plan acceptance against the planning limit, ranking by reach, early stop, reports."""

from __future__ import annotations

import dataclasses

import pytest
from shapely.geometry import LineString

from tests.post.route_factories import CO, IC, PC, graph_from_lines
from vineyard.route.budget import NOTE_OUTSIDE_BUDGET
from vineyard.route.candidates import NOTE_OUTSIDE_ONLY, ROLE_MUST, ROLE_OPTIONAL
from vineyard.route.policy_choice import PlanScore, policy_report, reachable_interrows, score_plan
from vineyard.route.stops import TargetVisit
from vineyard.route.validate import RouteValidation

LIMIT = 0.014


def _validation(outside_frac: float, passed: bool = True, length_m: float = 100.0) -> RouteValidation:
    return RouteValidation(length_m=length_m, closure_m=0.0, outside_len_m=outside_frac * length_m,
                           outside_frac=outside_frac, n_targets=3, n_reachable=1, n_visited_est=2, coverage_est=0.6,
                           coverage_required=1.0, legs_outside=(), fast_covered=False, single_linestring=True,
                           is_valid=True, zero_length_segments=0, passed=passed,
                           failures=() if passed else ("outside_frac",))


def _visit(tid: str, role: str, covered: bool, note: str = "") -> TargetVisit:
    return TargetVisit(target_id=tid, x=0.0, y=0.0, reachable_final=covered, reach_note=note, n_candidates=1,
                       covered=covered, visit_dist_m=0.0 if covered else 9.0, route_role=role)


VISITS = (_visit("T-GAP-0001", ROLE_MUST, True), _visit("T-GAP-0002", ROLE_MUST, False, NOTE_OUTSIDE_ONLY),
          _visit("T-MSP-0001", ROLE_OPTIONAL, True), _visit("T-MSP-0002", ROLE_OPTIONAL, False, NOTE_OUTSIDE_BUDGET))


def test_score_counts_visits_and_lost_must_targets() -> None:
    s = score_plan(_validation(0.01), VISITS, LIMIT, n_interrows=4)
    assert (s.acceptable, s.n_must_visited, s.n_optional_visited, s.n_must_lost, s.n_interrows) == (True, 1, 1, 1, 4)
    assert not s.final


def test_share_between_plan_limit_and_validator_limit_is_not_acceptable() -> None:
    s = score_plan(_validation(0.0145, passed=True), VISITS, LIMIT, n_interrows=4)
    assert not s.acceptable
    assert not score_plan(_validation(0.001, passed=False), VISITS, LIMIT, n_interrows=4).acceptable


def test_acceptable_plan_without_lost_must_targets_is_final() -> None:
    kept = (VISITS[0], VISITS[2])
    assert score_plan(_validation(0.01), kept, LIMIT, n_interrows=1).final


def _score(**kw) -> PlanScore:
    base = PlanScore(acceptable=True, outside_frac=0.01, length_m=100.0, n_must_visited=5, n_optional_visited=3,
                     n_must_lost=0, n_interrows=10)
    return dataclasses.replace(base, **kw)


def test_ranking_prefers_acceptable_then_reach_then_length() -> None:
    order = [_score(acceptable=False, outside_frac=0.001), _score(n_must_visited=4, length_m=10.0),
             _score(n_optional_visited=2), _score(n_interrows=9), _score(length_m=120.0), _score()]
    ranked = sorted(order, key=lambda s: s.key)
    assert ranked == [order[5], order[4], order[3], order[2], order[1], order[0]]
    worse = [_score(acceptable=False, outside_frac=0.03), _score(acceptable=False, outside_frac=0.02)]
    assert min(worse, key=lambda s: s.key).outside_frac == pytest.approx(0.02)


def test_reachable_interrows_counts_distinct_refs_in_start_component() -> None:
    lines = [(LineString([(0, 0), (0, 10)]), PC, "west"), (LineString([(0, 0), (20, 0)]), IC, "V01-I001"),
             (LineString([(0, 10), (20, 10)]), IC, "V01-I002"), (LineString([(50, 0), (70, 0)]), IC, "V01-I003"),
             (LineString([(20, 0), (20, 2)]), CO, "c")]
    g = graph_from_lines(lines, (0.0, 0.0), node_step_m=5.0)
    assert reachable_interrows(g) == 2


def test_policy_report_json() -> None:
    s = score_plan(_validation(0.01), VISITS, LIMIT, n_interrows=4)
    doc = policy_report("penalty", _validation(0.01), s, n_sets=3, n_outside_only=1, n_outside_budget=1).to_json()
    assert doc["policy"] == "penalty" and doc["acceptable"] is True and doc["n_must_lost"] == 1
    assert doc["n_interrows_reachable"] == 4 and isinstance(doc["failures"], list)
    err = policy_report("inside_only", None, None, error="boom").to_json()
    assert err["passed"] is False and err["acceptable"] is False and err["error"] == "boom"
