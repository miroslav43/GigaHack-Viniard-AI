"""Target rules on synthetic rows: sampling, GAP, MSP, SPR, END (a/b), MRW, WST."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, Point, box

from tests.post.factories import Prov, layer_frame, waste_record
from vineyard.config import load_config
from vineyard.contracts.enums import TargetKind
from vineyard.route.target_gaps import RowGap
from vineyard.route.target_rules import (
    END_SKIP_BOUNDARY,
    END_SKIP_NOT_LATERAL,
    END_SKIP_UNOBSERVED,
    RowContext,
    TargetDraft,
    end_extension_drafts,
    end_extensions,
    end_gap_drafts,
    end_samples,
    gap_drafts,
    missing_plant_drafts,
    missing_row_drafts,
    priority_for,
    segment_samples,
    sparse_drafts,
)
from vineyard.route.target_waste import waste_drafts

TILE = "siret3_r018_c011"
X0, Y0 = 629560.0, 5220290.0


@pytest.fixture(scope="module")
def cfg():
    return load_config(environ={}).targets


def _axis(length: float, y: float = 0.0, x0: float = 0.0) -> LineString:
    return LineString([(X0 + x0, Y0 + y), (X0 + x0 + length, Y0 + y)])


def _ctx(gaps=(), length=60.0, *, y=0.0, x0=0.0, row_index=1, unknown=(), head_b=False, tail_b=False,
         row_id="V01-R001") -> RowContext:
    return RowContext(row_id=row_id, vineyard_id="V01", row_index=row_index, axis=_axis(length, y, x0),
                      gaps=tuple(gaps), unknown=tuple(unknown), head_on_boundary=head_b, tail_on_boundary=tail_b)


def _along(draft: TargetDraft) -> float:
    return draft.x - X0


# ------------------------------------------------------------------ sampling and priority


@pytest.mark.parametrize(("length", "centres"), [(40.0, [6.667, 20.0, 33.333]), (20.0, [10.0]),
                                                 (20.5, [5.125, 15.375]), (6.0, [3.0])])
def test_segment_samples_split_long_segments(length, centres):
    parts = segment_samples(0.0, length, 20.0, 15.0)
    assert [round((a + b) / 2, 3) for a, b in parts] == centres
    assert math.isclose(sum(b - a for a, b in parts), length)


def test_segment_samples_rejects_bad_steps():
    with pytest.raises(ValueError):
        segment_samples(0.0, 10.0, 20.0, 0.0)


def test_priority_rules(cfg):
    assert priority_for(TargetKind.WASTE, math.nan, cfg) == 1
    assert priority_for(TargetKind.ROW_GAP, 12.7, cfg) == 1
    assert priority_for(TargetKind.ROW_GAP, 9.25, cfg) == 2
    assert priority_for(TargetKind.ROW_END_SHORT, 30.0, cfg) == 2
    assert priority_for(TargetKind.MISSING_ROW, math.nan, cfg) == 2
    assert priority_for(TargetKind.MISSING_PLANT, 3.0, cfg) == 3
    assert priority_for(TargetKind.SPARSE, math.nan, cfg) == 3


# ------------------------------------------------------------------ GAP / MSP


def test_gap_drafts_interior_only_at_centres(cfg):
    ctx = _ctx([RowGap(0, 6, "head"), RowGap(10, 16, "interior"), RowGap(20, 32.9, "interior"),
                RowGap(40, 44, "interior"), RowGap(52, 60, "tail")])
    drafts = gap_drafts(ctx, cfg)
    assert [round(_along(d), 3) for d in drafts] == [13.0, 26.45]
    assert [d.priority for d in drafts] == [2, 1]
    assert all(d.kind is TargetKind.ROW_GAP and d.row_id == "V01-R001" for d in drafts)
    assert [round(d.gap_length_m, 3) for d in drafts] == [6.0, 12.9]
    assert all(ctx.axis.distance(Point(d.x, d.y)) < 1e-9 for d in drafts)
    assert [round(d.extent.length, 6) for d in drafts] == [6.0, 12.9]


def test_gap_drafts_exact_threshold_counts(cfg):
    assert len(gap_drafts(_ctx([RowGap(10, 15, "interior")]), cfg)) == 1
    assert gap_drafts(_ctx([RowGap(10, 14.99, "interior")]), cfg) == ()


def test_long_gap_gets_one_target_per_step(cfg):
    drafts = gap_drafts(_ctx([RowGap(10, 50, "interior")], length=70.0), cfg)
    assert [round(_along(d), 2) for d in drafts] == [16.67, 30.0, 43.33]
    assert math.isclose(sum(d.extent.length for d in drafts), 40.0)
    assert {d.gap_length_m for d in drafts} == {40.0}
    assert {d.priority for d in drafts} == {1}


def test_censored_gap_is_kept_when_its_known_part_is_long_enough(cfg):
    ctx = _ctx([RowGap(20, 25, "interior", censored=True), RowGap(30, 34, "interior", censored=True)],
               unknown=((25.0, 30.0),))
    drafts = gap_drafts(ctx, cfg)
    assert [round(_along(d), 3) for d in drafts] == [22.5]
    assert "censored" in drafts[0].reason


def test_missing_plant_drafts_between_min_and_gap(cfg):
    ctx = _ctx([RowGap(0, 3, "head"), RowGap(5, 7, "interior"), RowGap(10, 11.9, "interior"),
                RowGap(20, 24.9, "interior"), RowGap(30, 36, "interior")])
    drafts = missing_plant_drafts(ctx, cfg)
    assert [round(_along(d), 2) for d in drafts] == [6.0, 22.45]
    assert {d.kind for d in drafts} == {TargetKind.MISSING_PLANT}
    assert {d.priority for d in drafts} == {3}


# ------------------------------------------------------------------ SPR


def _sparse_gaps(length: float, canopy_m: float, pitch: float) -> list[RowGap]:
    gaps, start = [], 0.0
    while start + canopy_m < length:
        nxt = start + pitch
        gaps.append(RowGap(start + canopy_m, min(nxt, length), "interior" if nxt < length else "tail"))
        start = nxt
    return gaps


def test_sparse_row_gets_sparse_targets(cfg):
    ctx = _ctx(_sparse_gaps(60.0, 0.3, 4.0), length=60.0)
    drafts = sparse_drafts(ctx, cfg)
    assert drafts, "0.3 m of canopy every 4 m (7.5 %) is sparse"
    assert {d.kind for d in drafts} == {TargetKind.SPARSE}
    assert all(0.0 <= _along(d) <= 60.0 for d in drafts)
    assert len(drafts) == 4  # one 60 m run sampled every 15 m


def test_dense_row_and_rows_with_big_gaps_are_not_sparse(cfg):
    assert sparse_drafts(_ctx(_sparse_gaps(60.0, 0.8, 1.0)), cfg) == ()
    with_gap = _sparse_gaps(60.0, 0.3, 4.0) + [RowGap(40.0, 46.0, "interior")]
    drafts = sparse_drafts(_ctx(sorted(with_gap, key=lambda g: g.start_m)), cfg)
    assert len(drafts) == 3  # windows overlapping the 6 m gap are excluded: run [0, 40]
    assert all(_along(d) < 40.0 for d in drafts)


def test_sparse_needs_known_windows(cfg):
    ctx = _ctx(_sparse_gaps(60.0, 0.3, 4.0), unknown=((0.0, 60.0),))
    assert sparse_drafts(ctx, cfg) == ()


# ------------------------------------------------------------------ END (a) and (b)


def test_end_gap_rule_needs_an_end_inside_coverage(cfg):
    ctx = _ctx([RowGap(0, 2, "head"), RowGap(48, 60, "tail")])
    drafts = end_gap_drafts(ctx, cfg)
    assert [round(_along(d), 2) for d in drafts] == [54.0]
    assert drafts[0].kind is TargetKind.ROW_END_SHORT and drafts[0].priority == 2
    on_boundary = _ctx([RowGap(48, 60, "tail")], tail_b=True)
    assert end_gap_drafts(on_boundary, cfg) == ()
    assert end_gap_drafts(_ctx([RowGap(56, 60, "tail")]), cfg) == ()


def test_end_extension_rule_uses_both_neighbours(cfg):
    short = _ctx(length=40.0, y=0.0, row_index=2, row_id="V01-R002")
    up = _ctx(length=47.0, y=2.5, row_index=1, row_id="V01-R001")
    down = _ctx(length=48.0, y=-2.5, row_index=3, row_id="V01-R003")
    drafts = end_extension_drafts(short, (up, down), cfg)
    assert [round(_along(d), 2) for d in drafts] == [43.5]
    assert abs(drafts[0].y - Y0) < 1e-9
    assert drafts[0].gap_length_m == pytest.approx(7.0)
    assert drafts[0].row_id == "V01-R002"


def test_end_extension_rule_rejects_one_neighbour_or_boundary(cfg):
    short = _ctx(length=40.0, row_index=2, row_id="V01-R002")
    up = _ctx(length=47.0, y=2.5, row_index=1, row_id="V01-R001")
    down_short = _ctx(length=43.0, y=-2.5, row_index=3, row_id="V01-R003")
    assert end_extension_drafts(short, (up,), cfg) == ()
    assert end_extension_drafts(short, (up, down_short), cfg) == ()
    boundary = _ctx(length=40.0, row_index=2, row_id="V01-R002", tail_b=True)
    down = _ctx(length=48.0, y=-2.5, row_index=3, row_id="V01-R003")
    assert end_extension_drafts(boundary, (up, down), cfg) == ()


def test_end_extension_rule_at_the_head(cfg):
    short = _ctx(length=40.0, x0=8.0, row_index=2, row_id="V01-R002")
    up = _ctx(length=48.0, y=2.5, row_index=1, row_id="V01-R001")
    down = _ctx(length=48.0, y=-2.5, row_index=3, row_id="V01-R003")
    drafts = end_extension_drafts(short, (up, down), cfg)
    assert [round(_along(d), 2) for d in drafts] == [4.0]


def _short_between(up_y: float = 2.5, down_y: float = -2.5, up_len: float = 48.0) -> tuple[RowContext, ...]:
    short = _ctx(length=40.0, row_index=2, row_id="V01-R002")
    up = _ctx(length=up_len, y=up_y, row_index=1, row_id="V01-R001")
    down = _ctx(length=48.0, y=down_y, row_index=3, row_id="V01-R003")
    return short, up, down


def test_end_extensions_report_a_well_defined_end(cfg):
    short, up, down = _short_between()
    (ext,) = end_extensions(short, (up, down), cfg)
    assert (ext.at_head, ext.skip) == (False, "")
    assert ext.length_m == pytest.approx(8.0)
    assert ext.line.length == pytest.approx(8.0)
    assert end_samples(ext.length_m, cfg) == 1 and end_samples(40.0, cfg) == 3


def test_collinear_fragment_is_not_a_neighbour(cfg):
    short = _ctx(length=40.0, row_index=2, row_id="V01-R002")
    fragment = _ctx(length=15.0, x0=45.0, y=0.1, row_index=1, row_id="V01-R001")  # same row, other tile
    down = _ctx(length=60.0, y=-2.5, row_index=3, row_id="V01-R003")
    (ext,) = end_extensions(short, (fragment, down), cfg)
    assert ext.skip == END_SKIP_NOT_LATERAL and ext.length_m == pytest.approx(20.0)
    assert end_extension_drafts(short, (fragment, down), cfg) == ()


def test_neighbours_on_one_side_are_ill_defined(cfg):
    short, up, down = _short_between(up_y=2.5, down_y=5.0)
    (ext,) = end_extensions(short, (up, down), cfg)
    assert ext.skip == END_SKIP_NOT_LATERAL
    assert end_extension_drafts(short, (up, down), cfg) == ()


def test_spurious_row_inside_an_interrow_is_ill_defined(cfg):
    # neighbours one spacing apart (2.6 m) with the short "row" between them: it cannot be a real row
    short, up, down = _short_between(up_y=1.2, down_y=-1.4)
    assert end_extensions(short, (up, down), cfg)[0].skip == END_SKIP_NOT_LATERAL
    assert end_extension_drafts(short, (up, down), cfg) == ()


def test_min_offset_is_configurable(cfg):
    short = _ctx(length=40.0, row_index=2, row_id="V01-R002")
    near = _ctx(length=48.0, y=0.8, row_index=1, row_id="V01-R001")
    down = _ctx(length=48.0, y=-2.5, row_index=3, row_id="V01-R003")
    assert end_extensions(short, (near, down), cfg)[0].skip == END_SKIP_NOT_LATERAL
    loose = cfg.model_copy(update={"end_neighbour_min_offset_m": 0.5})
    assert end_extensions(short, (near, down), loose)[0].skip == ""


def test_extension_leaving_the_coverage_is_ill_defined(cfg):
    short, up, down = _short_between()
    seen = box(X0 - 5.0, Y0 - 5.0, X0 + 100.0, Y0 + 5.0)
    cut = box(X0 - 5.0, Y0 - 5.0, X0 + 44.0, Y0 + 5.0).union(box(X0 + 46.0, Y0 - 5.0, X0 + 100.0, Y0 + 5.0))
    assert end_extensions(short, (up, down), cfg, seen)[0].skip == ""
    assert len(end_extension_drafts(short, (up, down), cfg, seen)) == 1
    assert end_extensions(short, (up, down), cfg, cut)[0].skip == END_SKIP_UNOBSERVED
    assert end_extension_drafts(short, (up, down), cfg, cut) == ()


def test_end_on_the_boundary_is_reported(cfg):
    short = _ctx(length=40.0, row_index=2, row_id="V01-R002", tail_b=True)
    _, up, down = _short_between()
    (ext,) = end_extensions(short, (up, down), cfg)
    assert ext.skip == END_SKIP_BOUNDARY


# ------------------------------------------------------------------ MRW


def _block(spacings: list[float], length: float = 50.0) -> list[RowContext]:
    ys, rows = [0.0], []
    for s in spacings:
        ys.append(ys[-1] - s)
    for k, y in enumerate(ys, start=1):
        rows.append(_ctx(length=length, y=y, row_index=k, row_id=f"V01-R{k:03d}"))
    return rows


def test_missing_row_between_widely_spaced_rows(cfg):
    drafts = missing_row_drafts(_block([2.5, 2.5, 5.2, 2.5]), cfg)
    assert drafts, "a 5.2 m pair in a 2.5 m block is a missing row"
    assert {d.kind for d in drafts} == {TargetKind.MISSING_ROW}
    assert all(abs(d.y - (Y0 - 5.0 - 2.6)) < 1e-6 for d in drafts)
    assert all(d.row_id is None for d in drafts)
    assert "V01-R003" in drafts[0].reason and "V01-R004" in drafts[0].reason
    assert math.isclose(sum(d.extent.length for d in drafts), 50.0, abs_tol=1e-6)


def test_no_missing_row_below_the_spacing_factor(cfg):
    assert missing_row_drafts(_block([2.5, 2.5, 4.0, 2.5]), cfg) == ()
    assert missing_row_drafts(_block([2.5]), cfg) == ()


def test_two_missing_rows_get_two_virtual_axes(cfg):
    drafts = missing_row_drafts(_block([2.5, 2.5, 7.5, 2.5, 2.5], length=10.0), cfg)
    assert sorted(round(Y0 - d.y, 3) for d in drafts) == [7.5, 10.0]


# ------------------------------------------------------------------ WST


def _waste(records):
    return layer_frame("waste", records, Prov())


def test_split_waste_boxes_give_one_target_at_the_union_centre():
    parts = [waste_record("W0001a", TILE, (X0 + 1.0, Y0), 2.0, vineyard_id="V01"),
             waste_record("W0001b", TILE, (X0 + 3.0, Y0), 2.0, vineyard_id="V01"),
             waste_record("W0002", TILE, (X0 + 20.0, Y0 + 5.0), 1.0)]
    drafts = waste_drafts(_waste(parts))
    assert [(d.waste_id, round(d.x - X0, 6), round(d.y - Y0, 6)) for d in drafts] == [
        ("W0001", 2.0, 0.0), ("W0002", 20.0, 5.0)]
    assert [d.vineyard_id for d in drafts] == ["V01", ""]
    assert {d.priority for d in drafts} == {1}
    assert all(d.extent is None for d in drafts)


def test_waste_drafts_empty_layer():
    assert waste_drafts(_waste([])) == ()


def test_waste_with_non_standard_id_is_kept_as_is():
    rec = waste_record("W0003", TILE, (X0, Y0), 1.0)
    frame = _waste([rec]).assign(waste_id=["custom-7"])
    drafts = waste_drafts(frame)
    assert [d.waste_id for d in drafts] == ["custom-7"]
    assert drafts[0].extent is None and box(X0 - 1, Y0 - 1, X0 + 1, Y0 + 1).contains(Point(drafts[0].x, drafts[0].y))
