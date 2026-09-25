from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import box

from tests.waste.synth_waste import make_candidate
from vineyard.perception.waste.filters import (
    FilterParams,
    TileWasteContext,
    apply_filters,
    reject_counts,
)
from vineyard.perception.waste.types import RejectReason

GSD = 0.025
AXIS = np.array([[0.0, 500.0], [2048.0, 500.0]])


@pytest.fixture
def params(cfg) -> FilterParams:
    return FilterParams.from_config(cfg.waste, GSD)


def ctx(axes: tuple[np.ndarray, ...] = (AXIS,), forbidden=None) -> TileWasteContext:
    return TileWasteContext(axes_px=axes, forbidden_px=forbidden, gsd_m=GSD)


def at_m(u: float, d_m: float, **kw) -> object:
    return make_candidate(centroid=(u, 500.0 + d_m / GSD), **kw)


def reasons(cands) -> list[RejectReason | None]:
    return [c.reject_reason for c in cands]


def test_params_from_config(params: FilterParams) -> None:
    assert params.white_min_area_px == pytest.approx(48.0)
    assert params.tube_max_area_px == pytest.approx(192.0)
    assert params.max_area_px == pytest.approx(9600.0)
    assert params.axis_exclusion_mode == "all"


def test_axis_exclusion_threshold(params: FilterParams) -> None:
    out = apply_filters([at_m(300, 0.59), at_m(300, 0.61), at_m(300, -0.59)], ctx(), params)
    assert reasons(out) == [RejectReason.NEAR_AXIS, None, RejectReason.NEAR_AXIS]


def test_periodic_mode_keeps_vivid_and_rejects_white_on_lattice(params: FilterParams) -> None:
    p = replace(params, axis_exclusion_mode="periodic")
    pitch_px, phase_px = 1.2 / GSD, 0.3 / GSD
    tubes = [at_m(phase_px + k * pitch_px, 0.05, is_white=True, area_px=60) for k in range(8)]
    red = at_m(phase_px + 3.5 * pitch_px, 0.3, area_px=200)
    white_on = at_m(phase_px + 9 * pitch_px, 0.2, is_white=True, area_px=300)
    white_off = at_m(phase_px + 9.5 * pitch_px, 0.2, is_white=True, area_px=300)
    out = apply_filters([*tubes, red, white_on, white_off], ctx(), p)
    assert reasons(out)[:8] == [RejectReason.PERIODIC_TUBE] * 8
    assert reasons(out)[8:] == [None, RejectReason.PERIODIC_TUBE, None]


def test_periodic_mode_without_lattice_keeps_near_axis_white(params: FilterParams) -> None:
    p = replace(params, axis_exclusion_mode="periodic")
    out = apply_filters([at_m(300, 0.2, is_white=True, area_px=300)], ctx(), p)
    assert reasons(out) == [None]


def test_too_small_white(params: FilterParams) -> None:
    out = apply_filters(
        [
            at_m(300, 5, is_white=True, area_px=47),
            at_m(900, 5, is_white=True, area_px=48),
            at_m(1500, 5, area_px=30),
        ],
        ctx(),
        params,
    )
    assert reasons(out) == [RejectReason.TOO_SMALL_WHITE, None, None]


def test_tube_shape(params: FilterParams) -> None:
    out = apply_filters(
        [
            at_m(300, 5, aspect=3.0, area_px=150),
            at_m(900, 5, aspect=3.0, area_px=200),
            at_m(1500, 5, aspect=2.4, area_px=150),
        ],
        ctx(),
        params,
    )
    assert reasons(out) == [RejectReason.TUBE_SHAPE, None, None]


def test_hose(params: FilterParams) -> None:
    hose = at_m(300, 5, aspect=65.0, area_px=260, length_px=130.0)
    short = at_m(900, 5, aspect=50.0, area_px=200, length_px=100.0)
    thick = at_m(1500, 5, aspect=40.0, area_px=520, length_px=130.0)
    assert reasons(apply_filters([hose, short, thick], ctx(), params)) == [RejectReason.HOSE, None, None]


def test_forbidden_and_vehicle(params: FilterParams) -> None:
    zone = box(0, 0, 200, 200)
    out = apply_filters(
        [
            make_candidate(centroid=(195.0, 100.0)),
            make_candidate(centroid=(300.0, 100.0)),
            make_candidate(centroid=(300.0, 1500.0), area_px=9601),
            make_candidate(centroid=(900.0, 1500.0), area_px=9600),
        ],
        ctx((), zone),
        params,
    )
    assert reasons(out) == [RejectReason.FORBIDDEN, None, RejectReason.VEHICLE, None]


def test_same_order_frozen_inputs_and_already_rejected_untouched(params: FilterParams) -> None:
    first = replace(make_candidate(centroid=(300.0, 1500.0)), reject_reason=RejectReason.NMS)
    inputs = (first, at_m(300, 0.1), make_candidate(centroid=(900.0, 1500.0)))
    out = apply_filters(inputs, ctx(), params)
    assert [c.cand_key for c in out] == [c.cand_key for c in inputs]
    assert reasons(out) == [RejectReason.NMS, RejectReason.NEAR_AXIS, None]
    assert inputs[1].reject_reason is None


def test_nearest_of_several_axes(params: FilterParams) -> None:
    other = np.array([[0.0, 1000.0], [2048.0, 1000.0]])
    out = apply_filters(
        [make_candidate(centroid=(300.0, 990.0)), make_candidate(centroid=(300.0, 750.0))],
        ctx((AXIS, other)),
        params,
    )
    assert reasons(out) == [RejectReason.NEAR_AXIS, None]


def test_reject_counts(params: FilterParams) -> None:
    out = apply_filters([at_m(300, 0.1), at_m(900, 0.1), at_m(1500, 5)], ctx(), params)
    assert reject_counts(out) == {"kept": 1, "near_axis": 2}


def test_empty_input(params: FilterParams) -> None:
    assert apply_filters([], ctx(), params) == ()


def test_invalid_mode_rejected(params: FilterParams) -> None:
    with pytest.raises(ValueError, match="axis_exclusion_mode"):
        replace(params, axis_exclusion_mode="bogus")


def test_bright_soil_rule(params: FilterParams) -> None:
    soil_like = replace(at_m(300, 5, is_white=True, area_px=300), mean_hsv=(18.0, 38.0, 215.0))
    white = replace(at_m(900, 5, is_white=True, area_px=300), mean_hsv=(18.0, 12.0, 240.0))
    pale_blue = replace(at_m(1500, 5, is_white=True, area_px=300), mean_hsv=(105.0, 38.0, 230.0))
    vivid = replace(at_m(1800, 5, area_px=300), mean_hsv=(18.0, 200.0, 180.0))
    out = apply_filters([soil_like, white, pale_blue, vivid], ctx(), params)
    assert reasons(out) == [RejectReason.BRIGHT_SOIL, None, None, None]
