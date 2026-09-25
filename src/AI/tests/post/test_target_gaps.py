"""The gap adapter (route/target_gaps.py) and the reference test engine that stands in for perception.attrs."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
from shapely.geometry import LineString, box

from tests.post.factories import reference_annset
from tests.post.tgt_helpers import reference_gap_fn
from vineyard.errors import StageError
from vineyard.pipeline.registry import module_available
from vineyard.route.target_gaps import (
    ENGINE_MODULE,
    RowGap,
    engine_gap_fn,
    merge_intervals,
    pixel_frame,
    row_gaps,
    split_by_unknown,
    unknown_intervals,
)

AXIS = LineString([(0.0, 0.0), (50.0, 0.0)])


def _canopy(lo: float, hi: float):
    return box(lo, -0.2, hi, 0.2)


# ------------------------------------------------------------------ RowGap


def test_row_gap_properties():
    gap = RowGap(10.0, 16.0, "interior")
    assert gap.length_m == 6.0
    assert gap.centre_m == 13.0
    assert gap.censored is False


@pytest.mark.parametrize("args", [(5.0, 4.0, "interior"), (0.0, 1.0, "middle"), (float("nan"), 1.0, "head")])
def test_row_gap_rejects_bad_values(args):
    with pytest.raises(ValueError):
        RowGap(*args)


# ------------------------------------------------------------------ interval helpers


def test_merge_intervals_joins_touching_and_overlapping():
    assert merge_intervals([(5, 6), (0, 1), (1, 2), (5.5, 7)]) == ((0, 2), (5, 7))


def test_unknown_intervals_projects_the_axis_part_inside_unknown():
    assert unknown_intervals(AXIS, box(25, -5, 30, 5)) == ((25.0, 30.0),)
    assert unknown_intervals(AXIS, None) == ()
    assert unknown_intervals(AXIS, box(100, 100, 101, 101)) == ()


# ------------------------------------------------------------------ reference engine (test oracle)


def test_reference_engine_interior_gaps_on_a_50m_line():
    canopies = [_canopy(0, 10), _canopy(15, 20), _canopy(40, 50)]
    gaps = row_gaps(reference_gap_fn, AXIS, canopies, None)
    interior = [(round(g.start_m, 6), round(g.end_m, 6)) for g in gaps if g.kind == "interior"]
    assert interior == [(10.0, 15.0), (20.0, 40.0)]
    assert all(not g.censored for g in gaps)


def test_reference_engine_unknown_splits_and_censors():
    canopies = [_canopy(0, 10), _canopy(15, 20), _canopy(40, 50)]
    gaps = row_gaps(reference_gap_fn, AXIS, canopies, box(25, -1, 30, 1))
    long = [g for g in gaps if g.start_m >= 20.0 - 1e-9]
    assert [(round(g.start_m, 6), round(g.end_m, 6), g.censored) for g in long] == [
        (20.0, 25.0, True), (30.0, 40.0, True)]


def test_reference_engine_head_and_tail_gaps():
    gaps = row_gaps(reference_gap_fn, AXIS, [_canopy(3, 10), _canopy(15, 45)], None)
    kinds = {(g.kind, round(g.length_m, 6)) for g in gaps}
    assert kinds == {("head", 3.0), ("interior", 5.0), ("tail", 5.0)}


@pytest.mark.examples
def test_reference_engine_reproduces_documented_r006_gaps(examples_xml):
    ann = reference_annset(examples_xml)
    expected = {"V02-R06": [5.054], "V02-R08": [7.09, 9.25, 7.745], "V02-R09": [12.683, 7.812, 12.903],
                "V02-R15": [5.009]}
    found = {}
    for piece in ann.row_pieces.itertuples():
        near = [p for p in ann.canopies.geometry if p.distance(piece.geometry) < 1.0]
        gaps = row_gaps(reference_gap_fn, piece.geometry, near, None, min_length_m=5.0)
        interior = [round(g.length_m, 3) for g in gaps if g.kind == "interior"]
        if interior:
            found[piece.row_id] = interior
    assert found == expected


# ------------------------------------------------------------------ row_gaps validation


def test_row_gaps_sorts_and_validates_output():
    def fn(axis, canopies, unknown, min_length_m):
        return [RowGap(30.0, 40.0, "interior"), RowGap(0.0, 2.0, "head")]

    assert [g.start_m for g in row_gaps(fn, AXIS, [], None)] == [0.0, 30.0]


def test_row_gaps_rejects_gap_beyond_axis():
    def fn(axis, canopies, unknown, min_length_m):
        return [RowGap(45.0, 55.0, "tail")]

    with pytest.raises(StageError, match="outside its row axis"):
        row_gaps(fn, AXIS, [], None, row_id="V01-R001")


def test_row_gaps_rejects_foreign_types():
    def fn(axis, canopies, unknown, min_length_m):
        return [(1.0, 2.0)]

    with pytest.raises(StageError, match="RowGap"):
        row_gaps(fn, AXIS, [], None)


def test_row_gaps_passes_min_length_to_the_engine():
    seen = {}

    def fn(axis, canopies, unknown, min_length_m):
        seen["min"] = min_length_m
        return []

    row_gaps(fn, AXIS, [], None, min_length_m=2.0)
    assert seen == {"min": 2.0}


# ------------------------------------------------------------------ unknown spans cut out of engine gaps


@pytest.mark.parametrize(("gap", "unknown", "expected"), [
    (RowGap(10, 40, "interior"), ((20.0, 25.0),), [(10, 20, "interior", True), (25, 40, "interior", True)]),
    (RowGap(0, 15, "head"), ((0.0, 5.0),), [(5, 15, "interior", True)]),
    (RowGap(0, 50, "full"), ((20.0, 30.0),), [(0, 20, "head", True), (30, 50, "tail", True)]),
    (RowGap(30, 50, "tail"), ((45.0, 60.0),), [(30, 45, "interior", True)]),
    (RowGap(10, 20, "interior", True), ((30.0, 35.0),), [(10, 20, "interior", True)]),
    (RowGap(10, 20, "interior"), ((5.0, 25.0),), []),
])
def test_split_by_unknown(gap, unknown, expected):
    parts = split_by_unknown([gap], unknown, 50.0)
    assert [(p.start_m, p.end_m, p.kind, p.censored) for p in parts] == expected


# ------------------------------------------------------------------ engine adapter (perception.attrs API)

X0, Y0 = 629560.0, 5220290.0  # tile r018_c011; +50 m east crosses into r018_c012 at x = 629606.4
UTM_AXIS = LineString([(X0, Y0), (X0 + 50.0, Y0)])


@dataclass(frozen=True)
class _EngineGap:
    start_m: float
    end_m: float
    kind: str
    censored: bool


def _fake_engine(gaps, calls):
    module = types.ModuleType(ENGINE_MODULE)

    def axis_to_px(axis, tile):
        calls.append(("axis", tile.tile_id))
        return np.asarray(axis.coords)

    def canopy_pixels(canopies, tile):
        calls.append(("pixels", tile.tile_id, len(canopies)))
        return np.array([[1.0, 2.0]])

    def occupancy_profile(axis_px, canopy_ij, *, half_px):
        calls.append(("profile", canopy_ij.tolist(), half_px))
        return "profile"

    def smooth_profile(profile, cfg):
        calls.append(("smooth",))
        return profile

    def find_gaps(profile, *, include_ends, min_length_m):
        calls.append(("gaps", profile, include_ends, min_length_m))
        return gaps

    for fn in (axis_to_px, canopy_pixels, occupancy_profile, smooth_profile, find_gaps):
        setattr(module, fn.__name__, fn)
    return module


def test_pixel_frame_spans_every_tile_of_the_corridor():
    frame = pixel_frame(UTM_AXIS, 0.3)
    assert frame.ref.tile_id == "siret3_r018_c011"
    assert [t.tile_id for t in frame.tiles] == ["siret3_r018_c011", "siret3_r018_c012"]
    assert frame.offset(frame.tiles[1]).tolist() == [2048.0, 0.0]


def test_engine_gap_fn_shifts_pixels_filters_and_converts(monkeypatch):
    calls: list = []
    engine = _fake_engine([_EngineGap(1.0, 7.0, "interior", True), _EngineGap(10.0, 11.0, "interior", False)],
                          calls)
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, engine)
    canopies = [box(X0 + 1, Y0 - 0.2, X0 + 2, Y0 + 0.2), box(X0 + 48, Y0 - 0.2, X0 + 49, Y0 + 0.2)]
    gaps = row_gaps(engine_gap_fn(types.SimpleNamespace(occ_smoothing_enabled=False), 0.3), UTM_AXIS, canopies,
                    None, min_length_m=2.0)
    assert gaps == (RowGap(1.0, 7.0, "interior", True),)
    assert calls == [("axis", "siret3_r018_c011"), ("pixels", "siret3_r018_c011", 1),
                     ("pixels", "siret3_r018_c012", 1), ("profile", [[1.0, 2.0], [2049.0, 2.0]], pytest.approx(12.0)),
                     ("gaps", "profile", True, 0.0)]


def test_engine_gap_fn_smooths_and_cuts_unknown(monkeypatch):
    calls: list = []
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, _fake_engine([_EngineGap(10.0, 40.0, "interior", False)],
                                                                 calls))
    fn = engine_gap_fn(types.SimpleNamespace(occ_smoothing_enabled=True), 0.3)
    hole = box(X0 + 20.0, Y0 - 5.0, X0 + 25.0, Y0 + 5.0)
    gaps = fn(UTM_AXIS, [], hole, 0.0)
    assert [(round(g.start_m, 6), round(g.end_m, 6), g.censored) for g in gaps] == [(10.0, 20.0, True),
                                                                                     (25.0, 40.0, True)]
    assert ("smooth",) in calls and not any(c[0] == "pixels" for c in calls)


def test_engine_gap_fn_accepts_position_attribute(monkeypatch):
    gap = types.SimpleNamespace(start_m=0.0, end_m=4.0, position="head")
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, _fake_engine([gap], []))
    assert engine_gap_fn(None, 0.3)(UTM_AXIS, [], None, 0.0) == (RowGap(0.0, 4.0, "head", False),)


def test_engine_gap_fn_rejects_unusable_engine_gaps(monkeypatch):
    no_kind = types.SimpleNamespace(start_m=0.0, end_m=1.0)
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, _fake_engine([no_kind], []))
    with pytest.raises(StageError, match="kind/position"):
        engine_gap_fn(None, 0.3)(UTM_AXIS, [], None, 0.0)
    bad = types.SimpleNamespace(start_m=5.0, end_m=1.0, kind="interior")
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, _fake_engine([bad], []))
    with pytest.raises(StageError, match="unusable gap"):
        engine_gap_fn(None, 0.3)(UTM_AXIS, [], None, 0.0)


def test_engine_gap_fn_fails_loudly_when_engine_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, None)
    fn = engine_gap_fn(None, 0.3)  # creating the adapter never imports the engine
    with pytest.raises(StageError, match="not available"):
        fn(UTM_AXIS, [], None, 0.0)


def test_engine_gap_fn_fails_loudly_when_engine_incomplete(monkeypatch):
    monkeypatch.setitem(sys.modules, ENGINE_MODULE, types.ModuleType(ENGINE_MODULE))
    with pytest.raises(StageError, match="lacks required functions"):
        engine_gap_fn(None, 0.3)(UTM_AXIS, [], None, 0.0)


@pytest.mark.examples
@pytest.mark.skipif(not module_available(ENGINE_MODULE), reason="perception.attrs not merged yet")
def test_real_engine_on_the_reference_rows(examples_xml):
    from vineyard.config import load_config

    cfg = load_config(environ={})
    fn = engine_gap_fn(cfg.row_structure, cfg.canopy.corridor_half_m)
    ann = reference_annset(examples_xml)
    found = {}
    for piece in ann.row_pieces.itertuples():
        near = [p for p in ann.canopies.geometry if p.distance(piece.geometry) < 1.0]
        interior = [round(g.length_m, 3) for g in row_gaps(fn, piece.geometry, near, None, min_length_m=4.9)
                    if g.kind == "interior"]
        if interior:
            found[piece.row_id] = interior
    documented = {"V02-R06": [5.054], "V02-R08": [7.09, 9.25, 7.745], "V02-R09": [12.683, 7.812, 12.903],
                  "V02-R15": [5.009]}
    assert set(found) == set(documented)
    for row_id, lengths in documented.items():
        assert sorted(found[row_id]) == pytest.approx(sorted(lengths), abs=0.06)
