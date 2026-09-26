"""Which outside policy's plan the route keeps (critic S5 headland risk; user decision: 1.5 % limit).

A plan is *acceptable* when the validator passes (outside share <= route.max_outside_frac_publish) and its
outside share is also <= route.plan_outside_frac (that limit minus route.plan_outside_margin). Plans are
ranked: acceptable first (else the smallest outside share), then most must targets visited, most
optional targets visited, most interrows reachable in the policy's graph, shortest. A plan that is
acceptable and loses no must target to its policy (`outside_only` / `outside_budget`) is `final`: the
policies after it only penalise or cut outside edges further, so none of them reaches more.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from vineyard.contracts.enums import EdgeKind
from vineyard.route.budget import NOTE_OUTSIDE_BUDGET
from vineyard.route.candidates import NOTE_OUTSIDE_ONLY, ROLE_MUST
from vineyard.route.graph_types import WalkGraph
from vineyard.route.stops import TargetVisit
from vineyard.route.validate import RouteValidation

LOST_NOTES: Final = frozenset({NOTE_OUTSIDE_ONLY, NOTE_OUTSIDE_BUDGET})
ERROR_FAILURE: Final = "error"


@dataclass(frozen=True)
class PlanScore:
    acceptable: bool
    outside_frac: float
    length_m: float
    n_must_visited: int
    n_optional_visited: int
    n_must_lost: int
    n_interrows: int

    @property
    def key(self) -> tuple[bool, float, int, int, int, float]:
        """Sort key, best first."""
        return (not self.acceptable, 0.0 if self.acceptable else self.outside_frac, -self.n_must_visited,
                -self.n_optional_visited, -self.n_interrows, self.length_m)

    @property
    def final(self) -> bool:
        return self.acceptable and self.n_must_lost == 0


def score_plan(v: RouteValidation, visits: Sequence[TargetVisit], plan_limit: float, n_interrows: int) -> PlanScore:
    must = [t for t in visits if t.route_role == ROLE_MUST]
    return PlanScore(
        acceptable=bool(v.passed and v.outside_frac <= plan_limit), outside_frac=v.outside_frac, length_m=v.length_m,
        n_must_visited=sum(t.covered for t in must),
        n_optional_visited=sum(t.covered for t in visits if t.route_role != ROLE_MUST),
        n_must_lost=sum(t.reach_note in LOST_NOTES for t in must), n_interrows=n_interrows)


def reachable_interrows(g: WalkGraph) -> int:
    """Distinct interrows with a centerline edge in START's component."""
    if g.n_edges == 0:
        return 0
    in_start = g.component[g.edge_uv[:, 0]] == g.start_component
    refs = {g.edge_ref[i] for i in np.flatnonzero(in_start) if g.edge_kind[i] == EdgeKind.INTERROW_CENTERLINE}
    return len(refs)


@dataclass(frozen=True)
class PolicyReport:
    policy: str
    passed: bool
    acceptable: bool
    failures: tuple[str, ...]
    length_m: float | None
    outside_len_m: float | None
    outside_frac: float | None
    coverage_required: float | None
    n_sets: int
    n_outside_only: int
    n_outside_budget: int
    n_must_visited: int = 0
    n_optional_visited: int = 0
    n_must_lost: int = 0
    n_interrows_reachable: int = 0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"failures": list(self.failures)}


def policy_report(policy: str, v: RouteValidation | None, score: PlanScore | None, *, n_sets: int = 0,
                  n_outside_only: int = 0, n_outside_budget: int = 0, error: str = "") -> PolicyReport:
    """One line of `route_validation.json` `policies` (a policy that raised has no validation)."""
    if v is None or score is None:
        return PolicyReport(policy, False, False, (ERROR_FAILURE,), None, None, None, None, 0, 0, 0, error=error)
    return PolicyReport(policy, v.passed, score.acceptable, v.failures, v.length_m, v.outside_len_m, v.outside_frac,
                        v.coverage_required, n_sets, n_outside_only, n_outside_budget, score.n_must_visited,
                        score.n_optional_visited, score.n_must_lost, score.n_interrows, error)


__all__ = ["LOST_NOTES", "PlanScore", "PolicyReport", "policy_report", "reachable_interrows", "score_plan"]
