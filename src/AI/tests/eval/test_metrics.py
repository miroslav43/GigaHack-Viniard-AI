"""eval.metrics: the official metric primitives on synthetic geometries (design 01 §5 test list)."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, MultiLineString, box

from vineyard.contracts.enums import RowStructure
from vineyard.eval.metrics import (
    AreaOverlap,
    area_overlap,
    attribute_scores,
    canopy_metrics,
    class_iou,
    f1_from_counts,
    fp_canopy_penalty,
    grouping_consistency,
    mutual_cover,
    relative_error,
    row_axis_f1,
    row_matches,
    value_score,
    waste_f1,
    weighted_canopy_score,
)

W = {"match_iou": 0.5, "w_iou": 0.6, "w_f1": 0.4}
ROW = {"tol_m": 0.4, "min_cover": 0.8}
LABELS = tuple(v.value for v in RowStructure)
REF_ROW = LineString([(0, 0), (10, 0)])


# ------------------------------------------------------------------ canopy


def test_canopy_shifted_20_percent_matches() -> None:
    s = canopy_metrics([box(0.2, 0, 1.2, 1)], [box(0, 0, 1, 1)], **W)
    assert s.iou == pytest.approx(2 / 3, abs=1e-4)
    assert (s.tp, s.n_pred, s.n_ref, s.f1) == (1, 1, 1, 1.0)
    assert s.score == pytest.approx(0.8)


def test_canopy_shifted_half_does_not_match() -> None:
    s = canopy_metrics([box(0.5, 0, 1.5, 1)], [box(0, 0, 1, 1)], **W)
    assert s.iou == pytest.approx(1 / 3)
    assert s.tp == 0 and s.f1 == 0.0
    assert s.score == pytest.approx(0.2)


def test_canopy_ten_percent_dropped() -> None:
    ref = [box(2 * k, 0, 2 * k + 1, 1) for k in range(10)]
    s = canopy_metrics(ref[:9], ref, **W)
    assert s.f1 == pytest.approx(0.947, abs=1e-3)
    assert s.iou == pytest.approx(0.9)
    assert s.inter_m2 == pytest.approx(9.0) and s.union_m2 == pytest.approx(10.0)


def test_canopy_identical_is_exactly_one() -> None:
    ref = [box(2 * k, 0, 2 * k + 1.3, 0.7) for k in range(20)]
    s = canopy_metrics(list(ref), ref, **W)
    assert (s.iou, s.f1, s.score) == (1.0, 1.0, 1.0)


def test_canopy_both_empty_is_perfect_and_one_side_empty_is_zero() -> None:
    assert canopy_metrics([], [], **W).score == 1.0
    assert canopy_metrics([box(0, 0, 1, 1)], [], **W).score == 0.0
    assert canopy_metrics([], [box(0, 0, 1, 1)], **W).score == 0.0


def test_class_iou_uses_union_of_overlapping_polygons() -> None:
    pred = [box(0, 0, 1, 1), box(0.5, 0, 1.5, 1)]  # union area 1.5, overlapping parts not double-counted
    assert class_iou(pred, [box(0, 0, 1.5, 1)]) == pytest.approx(1.0)
    ov = area_overlap(pred, [box(0, 0, 1, 1)])
    assert (ov.inter_m2, ov.union_m2) == pytest.approx((1.0, 1.5))
    assert AreaOverlap(0.0, 0.0).iou == 1.0


def test_weighted_score_and_f1_counts() -> None:
    assert weighted_canopy_score(0.5, 1.0, w_iou=0.6, w_f1=0.4) == pytest.approx(0.7)
    assert f1_from_counts(0, 0, 0) == 1.0
    assert f1_from_counts(3, 4, 5) == pytest.approx(6 / 9)


# ------------------------------------------------------------------ rows


def test_row_offset_03_matches() -> None:
    s = row_axis_f1([LineString([(0, 0.3), (10, 0.3)])], [REF_ROW], **ROW)
    assert (s.tp, s.f1, s.matches) == (1, 1.0, ((0, 0),))


def test_row_offset_05_does_not_match() -> None:
    s = row_axis_f1([LineString([(0, 0.5), (10, 0.5)])], [REF_ROW], **ROW)
    assert s.tp == 0 and s.f1 == 0.0


def test_row_half_length_fails_mutual_cover() -> None:
    short = LineString([(0, 0.3), (5, 0.3)])
    cov_short, cov_ref = mutual_cover(short, REF_ROW, 0.4)
    assert cov_short == pytest.approx(1.0)
    assert cov_ref == pytest.approx((5 + (0.4**2 - 0.3**2) ** 0.5) / 10, abs=1e-3)
    assert row_axis_f1([short], [REF_ROW], **ROW).tp == 0


def test_mutual_cover_is_exact_on_long_lines() -> None:
    a = LineString([(0, 0), (100, 0)])
    b = LineString([(0, 0.39), (100, 0.39)])
    assert mutual_cover(a, b, 0.4) == pytest.approx((1.0, 1.0))
    assert mutual_cover(LineString(), b, 0.4) == (0.0, 0.0)


def test_row_multilinestring_prediction() -> None:
    pred = MultiLineString([[(0, 0.1), (4.5, 0.1)], [(5, 0.1), (10, 0.1)]])
    assert row_axis_f1([pred], [REF_ROW], **ROW).tp == 1


def test_row_optimal_assignment_beats_first_fit() -> None:
    ref = [LineString([(0, 0), (10, 0)]), LineString([(0, 0.7), (10, 0.7)])]
    pred = [LineString([(0, 0.35), (10, 0.35)]), LineString([(0, 0.0), (10, 0.0)])]
    # pred 0 fits both refs; first-fit would give it ref 0 and strand pred 1.
    s = row_axis_f1(pred, ref, **ROW)
    assert s.tp == 2 and s.matches == ((0, 1), (1, 0))
    assert [(i, j) for i, j, _ in row_matches(pred, ref, **ROW)] == [(0, 1), (1, 0)]


def test_rows_empty() -> None:
    assert row_axis_f1([], [], **ROW).f1 == 1.0
    assert row_axis_f1([REF_ROW], [], **ROW).f1 == 0.0


# ------------------------------------------------------------------ attributes, grouping, counts


def test_attribute_scores_brief_example() -> None:
    s = attribute_scores(["regular", "disrupted", "regular"], ["regular", "regular", None], LABELS)
    assert s.accuracy == pytest.approx(1 / 3, abs=1e-4)
    assert s.macro_f1 == pytest.approx(0.25)
    assert s.score == pytest.approx(0.2917, abs=1e-4)
    assert s.n_ref == 3


def test_attribute_scores_perfect_empty_and_unknown_value() -> None:
    assert attribute_scores(["regular", "disrupted"], ["regular", "disrupted"], LABELS).score == 1.0
    assert attribute_scores([], [], LABELS).score == 1.0
    s = attribute_scores(["regular"], ["garbage"], LABELS)
    assert (s.accuracy, s.macro_f1) == (0.0, 0.0)
    with pytest.raises(ValueError, match="length"):
        attribute_scores(["regular"], [], LABELS)


def test_grouping_consistency() -> None:
    assert grouping_consistency(["A", "A", "B"], ["X", "X", "Y"]) == 1.0
    assert grouping_consistency(["A", "A", "B"], ["Y", "Y", "X"]) == 1.0  # swapped labels
    assert grouping_consistency(["A", "A", "B", "B"], ["X"] * 4) == pytest.approx(0.5)
    assert grouping_consistency(["A", "A", "A"], ["X", "Y", "Z"]) == 0.0
    assert grouping_consistency(["A", "B"], ["X", "Y"]) == 1.0
    assert grouping_consistency(["A", "A"], [None, None]) == 0.0  # missing ids never group
    assert grouping_consistency([], []) == 1.0
    with pytest.raises(ValueError, match="length"):
        grouping_consistency(["A"], [])


def test_value_score_and_relative_error() -> None:
    assert value_score(11, 10, 0.15) == pytest.approx(0.3333, abs=1e-4)
    assert value_score(12, 10, 0.15) == 0.0
    assert value_score(10, 10, 0.15) == 1.0
    assert value_score(0, 0, 0.15) == 1.0
    assert value_score(1, 0, 0.15) == 0.0
    assert relative_error(9, 10) == pytest.approx(0.1)
    assert relative_error(1, 0) == float("inf")


# ------------------------------------------------------------------ waste, fp penalty


def test_waste_boxes_iou_035_match_025_do_not() -> None:
    ref = [box(0, 0, 1, 1)]
    # IoU(box(0,0,1,1), box(a,0,1+a,1)) = (1-a)/(1+a): a=0.48148 -> 0.35, a=0.6 -> 0.25
    assert waste_f1([box(0.48148, 0, 1.48148, 1)], ref, match_iou=0.3).tp == 1
    assert waste_f1([box(0.6, 0, 1.6, 1)], ref, match_iou=0.3).tp == 0


def test_waste_duplicates_are_false_positives() -> None:
    ref = [box(0, 0, 1, 1)]
    s = waste_f1([box(0, 0, 1, 1), box(0.05, 0, 1.05, 1)], ref, match_iou=0.3)
    assert (s.tp, s.n_pred, s.n_ref) == (1, 2, 1)
    assert s.f1 == pytest.approx(2 / 3)
    assert waste_f1([], [], match_iou=0.3).f1 == 1.0


def test_fp_canopy_penalty() -> None:
    side = 262.144**0.5
    assert fp_canopy_penalty([box(0, 0, side, side)], 2621.44, factor=0.5) == pytest.approx(0.05)
    assert fp_canopy_penalty([], 2621.44, factor=0.5) == 0.0
    overlapping = [box(0, 0, 10, 10), box(5, 0, 15, 10)]
    assert fp_canopy_penalty(overlapping, 1000.0, factor=0.5) == pytest.approx(0.5 * 150 / 1000)
    with pytest.raises(ValueError, match="tile_area"):
        fp_canopy_penalty([], 0.0, factor=0.5)
