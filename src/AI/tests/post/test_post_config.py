"""Post-Marcaj config keys: must/optional target kinds, edge margin, the single outside-share limit."""

from __future__ import annotations

from typing import get_args

import pytest

from vineyard.config import load_config
from vineyard.config.sections_post import TargetKindName
from vineyard.contracts.enums import TargetKind
from vineyard.errors import ConfigError

NO_ENV: dict[str, str] = {}


def _cfg(*overrides: str):
    return load_config(overrides=overrides, environ=NO_ENV)


def test_target_kind_names_match_the_contract_enum() -> None:
    assert set(get_args(TargetKindName)) == {k.value for k in TargetKind}


def test_default_must_kinds_and_edge_keys() -> None:
    t = _cfg().targets
    assert t.must_kinds == ("row_gap", "missing_row", "waste")
    assert t.edge_margin_m == pytest.approx(3.0)
    assert t.end_skip_outer_rows is True
    assert t.end_neighbour_min_offset_m == pytest.approx(1.5)


def test_must_kinds_reject_unknown_and_duplicate_kinds() -> None:
    with pytest.raises((ConfigError, ValueError)):
        _cfg("targets.must_kinds=[row_gap, not_a_kind]")
    with pytest.raises((ConfigError, ValueError), match="duplicate"):
        _cfg("targets.must_kinds=[row_gap, row_gap]")


def test_edge_margin_must_be_positive() -> None:
    with pytest.raises((ConfigError, ValueError)):
        _cfg("targets.edge_margin_m=0.0")


def test_outside_limits_default_to_1_5_percent_with_2_percent_official() -> None:
    route = _cfg().route
    assert route.max_outside_frac_publish == pytest.approx(0.015)
    assert route.max_outside_frac_official == pytest.approx(0.02)
    assert route.plan_outside_margin == pytest.approx(0.001)
    assert route.plan_outside_frac == pytest.approx(0.014)


def test_outside_limit_cannot_exceed_the_official_one() -> None:
    with pytest.raises((ConfigError, ValueError), match="max_outside_frac_publish"):
        _cfg("route.max_outside_frac_publish=0.025")


def test_plan_margin_must_leave_a_positive_planning_limit() -> None:
    with pytest.raises((ConfigError, ValueError), match="plan_outside_margin"):
        _cfg("route.plan_outside_margin=0.015")


def test_solver_planning_keys_moved_from_module_constants() -> None:
    solver = _cfg().route.solver
    assert solver.include_optional is True
    assert solver.probe_time_limit_s == 5
    assert solver.budget_rounds == 10
    assert solver.optional_max_detour_m == pytest.approx(25.0)
