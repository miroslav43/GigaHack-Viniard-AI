"""Route validator (arch §4.11.6, contract §5.1/§10): 2 levels, closure, coverage, no is_simple check."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point, box

from vineyard.config import load_config
from vineyard.route.validate import ValidateParams, outside_lengths, validate_route

START = (1.0, 2.0)
CORRIDOR = box(0.0, 0.0, 100.0, 4.0)
INNER = CORRIDOR.buffer(-0.05, join_style="mitre")
shapely.prepare(INNER)


@pytest.fixture(scope="module")
def params() -> ValidateParams:
    return ValidateParams.from_route_cfg(load_config().route)


def _out_and_back(x_end: float = 50.0, excursion_y: float | None = None) -> LineString:
    pts = [START, (x_end, 2.0)]
    if excursion_y is not None:
        pts += [(x_end, excursion_y), (x_end, 2.0)]
    return LineString([*pts, START])


def _validate(line: object, params: ValidateParams, required: list[tuple[float, float]] | None = None,
              **kw: object) -> object:
    req = np.array(required or [(25.0, 3.0)], float)
    return validate_route(line, INNER, start_xy=START, required_xy=req, all_xy=req, params=params, **kw)


def test_params_from_config(params: ValidateParams) -> None:
    assert params.max_outside_frac == pytest.approx(0.005)
    assert params.closure_max_m == pytest.approx(0.01)
    assert params.visit_radius_m == pytest.approx(2.0)
    assert params.grid_size_m == pytest.approx(0.001)


def test_valid_out_and_back_passes_although_not_simple(params: ValidateParams) -> None:
    line = _out_and_back()
    assert not line.is_simple
    v = _validate(line, params)
    assert v.passed, v.failures
    assert v.outside_len_m == 0.0 and v.fast_covered
    assert v.closure_m == 0.0
    assert v.length_m == pytest.approx(98.0)


def test_three_percent_outside_is_rejected(params: ValidateParams) -> None:
    line = _out_and_back(excursion_y=5.5)  # 1.55 m outside, twice, on a 105 m route
    v = _validate(line, params)
    assert not v.passed
    assert "outside_frac" in v.failures
    assert v.outside_frac == pytest.approx(0.03, abs=0.005)
    assert not v.fast_covered


def test_outside_traversed_twice_counts_twice(params: ValidateParams) -> None:
    line = _out_and_back(excursion_y=4.95)
    v = _validate(line, params)
    assert v.outside_len_m == pytest.approx(2.0, abs=2e-3)


def test_outside_lengths_per_segment() -> None:
    per_seg = outside_lengths(_out_and_back(excursion_y=4.95), INNER, 0.001)
    assert per_seg.shape == (4,)
    assert per_seg[[0, 3]].tolist() == [0.0, 0.0]
    assert per_seg[1] == pytest.approx(1.0, abs=2e-3) and per_seg[2] == pytest.approx(1.0, abs=2e-3)


def test_legs_outside_reported_per_anchor_leg(params: ValidateParams) -> None:
    line = _out_and_back(excursion_y=4.95)
    v = _validate(line, params, anchor_idx=(0, 2, 4))
    assert [leg for leg, _ in v.legs_outside] == [0, 1]
    assert sum(length for _, length in v.legs_outside) == pytest.approx(2.0, abs=2e-3)


def test_start_off_by_two_cm_fails_closure(params: ValidateParams) -> None:
    line = LineString([(1.02, 2.0), (50.0, 2.0), START])
    v = _validate(line, params)
    assert not v.passed and "closure" in v.failures
    assert v.closure_m == pytest.approx(0.02)


def test_multilinestring_fails(params: ValidateParams) -> None:
    multi = MultiLineString([[START, (50.0, 2.0)], [(50.0, 2.0), START]])
    v = _validate(multi, params)
    assert not v.passed and "single_linestring" in v.failures


def test_non_line_geometry_fails(params: ValidateParams) -> None:
    v = _validate(Point(START), params)
    assert not v.passed and "single_linestring" in v.failures


def test_zero_length_segment_fails(params: ValidateParams) -> None:
    line = LineString([START, (20.0, 2.0), (20.0, 2.0), (50.0, 2.0), START])
    v = _validate(line, params)
    assert v.zero_length_segments == 1
    assert not v.passed and "zero_length_segments" in v.failures


@pytest.mark.parametrize(("offset", "ok"), [(1.95, True), (2.05, False)])
def test_required_target_coverage_is_blocking(params: ValidateParams, offset: float, ok: bool) -> None:
    line = LineString([START, (50.0, 2.0), START])
    v = _validate(line, params, required=[(30.0, 2.0 + offset)])
    assert v.passed is ok
    assert v.coverage_required == (1.0 if ok else 0.0)
    assert ("coverage" in v.failures) is (not ok)


def test_all_targets_coverage_is_reported_not_blocking(params: ValidateParams) -> None:
    line = LineString([START, (50.0, 2.0), START])
    required = np.array([(30.0, 3.0)])
    everything = np.array([(30.0, 3.0), (90.0, 3.0)])
    v = validate_route(line, INNER, start_xy=START, required_xy=required, all_xy=everything, params=params)
    assert v.passed
    assert (v.n_targets, v.n_reachable, v.n_visited_est) == (2, 1, 1)
    assert v.coverage_est == pytest.approx(0.5)


def test_no_required_targets_is_full_coverage(params: ValidateParams) -> None:
    empty = np.zeros((0, 2))
    v = validate_route(_out_and_back(), INNER, start_xy=START, required_xy=empty, all_xy=empty, params=params)
    assert v.coverage_required == 1.0 and v.coverage_est == 1.0 and v.passed


def test_to_json_has_contract_keys(params: ValidateParams) -> None:
    doc = _validate(_out_and_back(), params).to_json()
    for key in ("length_m", "closure_m", "outside_len_m", "outside_frac", "n_targets", "n_reachable",
                "n_visited_est", "coverage_est", "legs_outside", "passed", "failures"):
        assert key in doc
    assert doc["legs_outside"] == []
