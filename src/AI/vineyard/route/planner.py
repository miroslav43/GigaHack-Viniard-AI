"""plan_route (arch §4.11.3-6, design 04 §3.8-3.9).

1. Candidates with the node-cap ladder: configured -> 1 per side -> + merge within merge_node_m ->
   truncate optional sets by (priority, -gap length, id); must sets alone over the cap is an error.
2. Outside policies (policies.py), in order: each is probed with a short solver limit and no coverage
   loop; `drop_*` policies re-solve up to `budget_rounds` times, dropping the stops that carry the most
   outside length above the planning limit (route.plan_outside_frac = the 1.5 % publish/validator limit
   minus route.plan_outside_margin). Probing stops at the first plan that is acceptable (validates and
   fits the planning limit) and loses no must target; otherwise every policy is probed and the one
   reaching most (policy_choice.py: must visited, optional visited, interrows reachable, length) wins.
3. The chosen policy is solved again with the full time limit and the coverage loop (plan_steps.py);
   the better of probe and final is returned. One distance table per policy graph serves every solve.
Optional targets are routed only when their insertion into the must tour is cheap (plan_steps.py).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from vineyard.config import RouteConfig
from vineyard.errors import RouteValidationError
from vineyard.logging_setup import get_logger, log_event
from vineyard.route.budget import NOTE_OUTSIDE_BUDGET, plan_drops
from vineyard.route.candidates import (
    NOTE_OUTSIDE_ONLY,
    ROLE_MUST,
    Augmented,
    CandidateParams,
    TargetPoint,
    augment_with_targets,
)
from vineyard.route.graph_types import CandidateSet, Reach, WalkGraph
from vineyard.route.plan_steps import PhaseParams, PhaseResult, finalize, points, solve_phases
from vineyard.route.policies import DEFAULT_POLICIES, OutsidePolicy, apply_policy, restrict_sets
from vineyard.route.policy_choice import (
    PlanScore,
    PolicyReport,
    policy_report,
    reachable_interrows,
    score_plan,
)
from vineyard.route.stops import (
    NOTE_NOT_ROUTED,
    NOTE_OPTIONAL_DETOUR,
    NOTE_TRUNCATED,
    Stop,
    TargetVisit,
    build_stops,
    build_visits,
)
from vineyard.route.tour import DistanceTable, TourParams
from vineyard.route.validate import RouteValidation, ValidateParams, validate_route

_log = get_logger("route.planner")
_NO_LIMIT: Final = math.nan


@dataclass(frozen=True)
class PlanParams:
    candidates: CandidateParams
    tour: TourParams
    validate: ValidateParams
    cover_iterations: int
    cover_radius_m: float
    max_tsp_nodes: int
    merge_node_m: float
    base_penalty: float
    decimals: int
    plan_outside_frac: float
    include_optional: bool
    optional_max_detour_m: float
    probe_time_limit_s: int
    budget_rounds: int
    policies: tuple[OutsidePolicy, ...] = DEFAULT_POLICIES

    @classmethod
    def from_route_cfg(cls, cfg: RouteConfig, start_xy: tuple[float, float], *,
                       time_limit_s: int | None = None) -> PlanParams:
        s = cfg.solver
        return cls(candidates=CandidateParams.from_route_cfg(cfg),
                   tour=TourParams.from_route_cfg(cfg, start_xy, time_limit_s),
                   validate=ValidateParams.from_route_cfg(cfg), cover_iterations=s.cover_iterations,
                   cover_radius_m=cfg.candidate_radius_m, max_tsp_nodes=s.max_tsp_nodes, merge_node_m=s.merge_node_m,
                   base_penalty=cfg.graph.connector_outside_penalty, decimals=cfg.validate_.coord_decimals,
                   plan_outside_frac=cfg.plan_outside_frac, include_optional=s.include_optional,
                   optional_max_detour_m=s.optional_max_detour_m, probe_time_limit_s=s.probe_time_limit_s,
                   budget_rounds=s.budget_rounds)

    def phases(self, *, probe: bool, optional: bool = True) -> PhaseParams:
        limit = min(self.probe_time_limit_s, self.tour.time_limit_s) if probe else self.tour.time_limit_s
        return PhaseParams(tour=dataclasses.replace(self.tour, time_limit_s=limit),
                           cover_iterations=0 if probe else self.cover_iterations,
                           cover_radius_m=self.cover_radius_m, include_optional=self.include_optional and optional,
                           optional_max_detour_m=self.optional_max_detour_m)


@dataclass(frozen=True, eq=False)
class RoutePlan:
    xy: np.ndarray
    anchors: tuple[int, ...]
    stops: tuple[Stop, ...]
    visits: tuple[TargetVisit, ...]
    validation: RouteValidation
    policy: str
    solver: str
    solve_time_s: float
    fallback_reason: str
    iterations: int
    must_length_m: float | None
    optional_delta_m: float | None
    ladder_level: int
    graph: WalkGraph
    sets: tuple[CandidateSet, ...]
    stop_sets: tuple[int, ...]
    stop_nodes: tuple[int, ...]
    dropped: tuple[str, ...] = ()
    reports: tuple[PolicyReport, ...] = ()
    accepted: bool = False
    plan_outside_limit: float = _NO_LIMIT

    @property
    def line(self) -> LineString:
        return LineString(self.xy)


@dataclass(frozen=True, eq=False)
class _PolicyGraph:
    policy: OutsidePolicy
    graph: WalkGraph
    sets: tuple[CandidateSet, ...]
    lost: tuple[str, ...]
    table: DistanceTable
    n_interrows: int


def _n_nodes(sets: Sequence[CandidateSet], include_optional: bool) -> int:
    return 1 + sum(len(s.nodes) for s in sets if include_optional or s.role == ROLE_MUST)


def _truncate_optional(aug: Augmented, targets: Sequence[TargetPoint], params: PlanParams) -> Augmented:
    must = [s for s in aug.sets if s.role == ROLE_MUST]
    budget = params.max_tsp_nodes - _n_nodes(must, False)
    if budget < 0:
        raise RouteValidationError("must targets alone exceed route.solver.max_tsp_nodes",
                                   n_nodes=_n_nodes(must, False), cap=params.max_tsp_nodes, n_sets=len(must))
    info = {t.target_id: t for t in targets}

    def rank(s: CandidateSet) -> tuple[int, float, str]:
        gap = max((info[t].gap_length_m for t in s.target_ids if math.isfinite(info[t].gap_length_m)), default=0.0)
        return (min(info[t].priority for t in s.target_ids), -gap, s.target_ids[0])

    kept, cut = [], []
    for s in sorted((s for s in aug.sets if s.role != ROLE_MUST), key=rank):
        if len(s.nodes) <= budget:
            kept.append(s)
            budget -= len(s.nodes)
        else:
            cut.append(s)
    cut_ids = {t for s in cut for t in s.target_ids}
    reach = tuple(dataclasses.replace(r, note=NOTE_TRUNCATED) if r.target_id in cut_ids else r for r in aug.reach)
    log_event(_log, "route.ladder.truncated", n_cut=len(cut_ids), n_kept_optional=len(kept))
    return Augmented(aug.graph, tuple(sorted([*must, *kept], key=lambda s: s.target_ids[0])), reach,
                     aug.target_nodes)


def candidates_with_ladder(g: WalkGraph, targets: Sequence[TargetPoint], inner: BaseGeometry | None,
                           params: PlanParams) -> tuple[Augmented, int]:
    base = params.candidates
    ladder = (base, base.with_changes(max_per_side=1),
              base.with_changes(max_per_side=1, merge_m=params.merge_node_m))
    augs = []
    for level, cand in enumerate(ladder):
        augs.append(augment_with_targets(g, targets, inner, cand))
        n_nodes = _n_nodes(augs[-1].sets, params.include_optional)
        if n_nodes <= params.max_tsp_nodes:
            return augs[-1], level
        log_event(_log, "route.ladder.step", level=level, n_nodes=n_nodes, cap=params.max_tsp_nodes)
    return _truncate_optional(augs[-1], targets, params), len(ladder)


def _policy_graph(aug: Augmented, policy: OutsidePolicy, params: PlanParams) -> _PolicyGraph:
    g = apply_policy(aug.graph, policy, params.base_penalty)
    sets, lost = restrict_sets(g, aug.sets)
    return _PolicyGraph(policy, g, sets, lost, DistanceTable.for_sets(g, sets, params.tour.chunk),
                        reachable_interrows(g))


def _reach(reach: Sequence[Reach], lost: Sequence[str], dropped: frozenset[str]) -> dict[str, Reach]:
    gone, out = frozenset(lost), {}
    for r in reach:
        if r.target_id in gone:
            r = dataclasses.replace(r, reachable=False, note=NOTE_OUTSIDE_ONLY, n_candidates=0)
        elif r.target_id in dropped:
            r = dataclasses.replace(r, reachable=False, note=NOTE_OUTSIDE_BUDGET)
        out[r.target_id] = r
    return out


def _settle_unrouted(visits: Sequence[TargetVisit], routed: frozenset[str], note: str,
                     detour: frozenset[str] = frozenset()) -> tuple[TargetVisit, ...]:
    """A reachable target that is neither routed nor passed within the visit radius is reported unreachable
    (the route must cover 100 % of the targets it calls reachable): `optional_detour` when phase B found it
    too expensive, else `note`; an earlier note (truncated) is kept."""
    def settle(v: TargetVisit) -> TargetVisit:
        if not v.reachable_final or v.covered or v.target_id in routed:
            return v
        why = NOTE_OPTIONAL_DETOUR if v.target_id in detour else note
        return dataclasses.replace(v, reachable_final=False, reach_note=v.reach_note or why)
    return tuple(settle(v) for v in visits)


def _visits(pg: _PolicyGraph, aug: Augmented, targets: Sequence[TargetPoint], line: LineString,
            sets: Sequence[CandidateSet], res: PhaseResult, phases: PhaseParams, dropped: frozenset[str],
            radius_m: float) -> tuple[TargetVisit, ...]:
    detour = frozenset(t for s in res.skipped for t in s.target_ids)
    routed = frozenset(t for s in sets if phases.include_optional or s.role == ROLE_MUST
                       for t in s.target_ids) - detour
    return _settle_unrouted(build_visits(line, targets, _reach(aug.reach, pg.lost, dropped), radius_m), routed,
                            NOTE_NOT_ROUTED if pg.policy.optional else NOTE_OUTSIDE_BUDGET, detour)


def _plan_once(pg: _PolicyGraph, aug: Augmented, targets: Sequence[TargetPoint], inner: BaseGeometry,
               eroded: BaseGeometry, params: PlanParams, level: int, dropped: frozenset[str],
               probe: bool) -> RoutePlan:
    sets = tuple(s for s in pg.sets if not dropped.intersection(s.target_ids))
    xy_of = {t.target_id: (t.x, t.y) for t in targets}
    must = tuple(s for s in sets if s.role == ROLE_MUST)
    phases = params.phases(probe=probe, optional=pg.policy.optional)
    res = solve_phases(pg.graph, must, tuple(s for s in sets if s.role != ROLE_MUST), xy_of, eroded, phases, pg.table)
    start = params.tour.start_xy
    xy, anchors = finalize(res.tour, start, params.decimals)
    all_xy = np.array([[t.x, t.y] for t in targets], dtype=np.float64).reshape(-1, 2)
    validation = validate_route(LineString(xy), inner, start_xy=start, required_xy=points(must, xy_of),
                                all_xy=all_xy, params=params.validate, anchor_idx=anchors)
    stops = build_stops(xy, anchors, [res.used[k].target_ids for k in res.tour.stop_sets])
    visits = _visits(pg, aug, targets, LineString(xy), sets, res, phases, dropped, params.validate.visit_radius_m)
    return RoutePlan(xy=xy, anchors=anchors, stops=stops, visits=visits, validation=validation,
                     policy=pg.policy.name, solver=res.tour.solver, solve_time_s=res.tour.solve_time_s,
                     fallback_reason=res.tour.fallback_reason, iterations=res.iterations,
                     must_length_m=res.must_length_m, optional_delta_m=res.optional_delta_m, ladder_level=level,
                     graph=pg.graph, sets=res.used, stop_sets=res.tour.stop_sets, stop_nodes=res.tour.stop_nodes,
                     dropped=tuple(sorted(dropped)))


def _plan_budget(pg: _PolicyGraph, aug: Augmented, targets: Sequence[TargetPoint], inner: BaseGeometry,
                 eroded: BaseGeometry, params: PlanParams, level: int) -> RoutePlan:
    """Probe plan; `drop_*` policies re-solve without the stops that carry the most outside length."""
    dropped: frozenset[str] = frozenset()
    plan = _plan_once(pg, aug, targets, inner, eroded, params, level, dropped, probe=True)
    for _ in range(params.budget_rounds if pg.policy.drop != "none" else 0):
        v = plan.validation
        excess = v.outside_len_m - params.plan_outside_frac * v.length_m
        owners = [plan.sets[k] for k in plan.stop_sets]
        picks = plan_drops(pg.graph, plan.stop_nodes, v.legs_outside, [s.role for s in owners],
                           [len(s.target_ids) for s in owners], pg.table, params.tour.chunk, excess, pg.policy.drop)
        if not picks:
            break
        dropped = dropped | {t for p in picks for t in plan.sets[plan.stop_sets[p]].target_ids}
        log_event(_log, "route.budget.drop", policy=pg.policy.name, n_dropped=len(dropped), excess_m=round(excess, 2))
        plan = _plan_once(pg, aug, targets, inner, eroded, params, level, dropped, probe=True)
    return plan


def _score(plan: RoutePlan, pg: _PolicyGraph, params: PlanParams) -> PlanScore:
    return score_plan(plan.validation, plan.visits, params.plan_outside_frac, pg.n_interrows)


def _report(policy: OutsidePolicy, plan: RoutePlan, score: PlanScore) -> PolicyReport:
    n_lost = sum(1 for vis in plan.visits if vis.reach_note == NOTE_OUTSIDE_ONLY)
    return policy_report(policy.name, plan.validation, score, n_sets=len(plan.sets), n_outside_only=n_lost,
                         n_outside_budget=len(plan.dropped))


_Probe = tuple[_PolicyGraph, RoutePlan, PlanScore]


def _probe_policies(aug: Augmented, targets: Sequence[TargetPoint], inner: BaseGeometry, eroded: BaseGeometry,
                    params: PlanParams, level: int) -> tuple[list[_Probe], list[PolicyReport]]:
    graphs: dict[tuple[float, float | None], _PolicyGraph] = {}
    probes: list[_Probe] = []
    reports: list[PolicyReport] = []
    for policy in params.policies:
        key = (policy.penalty_mult, policy.max_edge_outside_m)
        try:
            if key not in graphs:
                graphs[key] = _policy_graph(aug, policy, params)
            pg = dataclasses.replace(graphs[key], policy=policy)
            plan = _plan_budget(pg, aug, targets, inner, eroded, params, level)
        except RouteValidationError as exc:
            log_event(_log, "route.policy.failed", policy=policy.name, error=str(exc))
            reports.append(policy_report(policy.name, None, None, error=str(exc)))
            continue
        score = _score(plan, pg, params)
        probes.append((pg, plan, score))
        reports.append(_report(policy, plan, score))
        log_event(_log, "route.policy", policy=policy.name, passed=plan.validation.passed, acceptable=score.acceptable,
                  length_m=round(plan.validation.length_m, 2), outside_frac=round(plan.validation.outside_frac, 5),
                  must_visited=score.n_must_visited, optional_visited=score.n_optional_visited,
                  must_lost=score.n_must_lost, interrows=score.n_interrows)
        if score.final:
            break
    return probes, reports


def plan_route(g: WalkGraph, targets: Sequence[TargetPoint], inner: BaseGeometry, eroded: BaseGeometry,
               params: PlanParams) -> RoutePlan:
    """The best-reaching acceptable policy (else the smallest outside share), solved again in full."""
    aug, level = candidates_with_ladder(g, targets, inner, params)
    probes, reports = _probe_policies(aug, targets, inner, eroded, params, level)
    if not probes:
        raise RouteValidationError("no outside policy produced a route", reports=[r.error for r in reports])
    pg, probe, _ = min(probes, key=lambda item: item[2].key)
    final = _plan_once(pg, aug, targets, inner, eroded, params, level, frozenset(probe.dropped), probe=False)
    best, score = min(((p, _score(p, pg, params)) for p in (final, probe)), key=lambda item: item[1].key)
    log_event(_log, "route.policy.chosen", policy=best.policy, acceptable=score.acceptable,
              outside_frac=round(best.validation.outside_frac, 5), plan_limit=params.plan_outside_frac)
    return dataclasses.replace(best, reports=tuple(reports), accepted=score.acceptable,
                               plan_outside_limit=params.plan_outside_frac)


__all__ = ["PlanParams", "PolicyReport", "RoutePlan", "candidates_with_ladder", "plan_route"]
