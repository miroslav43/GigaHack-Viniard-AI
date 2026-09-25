"""eval.matching: IoU pairs, greedy and optimal one-to-one matching, line candidates."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from vineyard.eval.matching import (
    LayerObjects,
    clean_polygonal,
    greedy_one_to_one,
    iou_pairs,
    line_candidates,
    match_polygons,
    optimal_one_to_one,
    pair_iou,
)


def test_pair_iou_basic_and_disjoint() -> None:
    assert pair_iou(box(0, 0, 1, 1), box(0.2, 0, 1.2, 1)) == pytest.approx(0.8 / 1.2)
    assert pair_iou(box(0, 0, 1, 1), box(2, 2, 3, 3)) == 0.0
    assert pair_iou(Polygon(), box(0, 0, 1, 1)) == 0.0


def test_clean_polygonal_repairs_bowtie_and_drops_lines() -> None:
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    fixed = clean_polygonal(bowtie)
    assert fixed.is_valid and fixed.area == pytest.approx(2.0)
    assert clean_polygonal(LineString([(0, 0), (1, 1)])).is_empty
    multi = MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)])
    assert clean_polygonal(multi).area == pytest.approx(2.0)


def test_iou_pairs_only_intersecting_candidates() -> None:
    pred = [box(0, 0, 1, 1), box(5, 5, 6, 6)]
    ref = [box(0.5, 0, 1.5, 1), box(10, 10, 11, 11)]
    i, j, iou = iou_pairs(pred, ref)
    assert list(i) == [0] and list(j) == [0]
    assert iou[0] == pytest.approx(1 / 3)


def test_iou_pairs_empty_inputs() -> None:
    i, j, iou = iou_pairs([], [box(0, 0, 1, 1)])
    assert len(i) == len(j) == len(iou) == 0


def test_greedy_is_deterministic_and_one_to_one() -> None:
    pairs = [(0, 0, 0.6), (1, 0, 0.9), (1, 1, 0.7), (0, 1, 0.55)]
    assert greedy_one_to_one(pairs, min_score=0.5) == ((0, 1, 0.55), (1, 0, 0.9))
    assert greedy_one_to_one(pairs, min_score=0.8) == ((1, 0, 0.9),)
    assert greedy_one_to_one([], min_score=0.5) == ()


def test_greedy_ties_broken_by_index() -> None:
    pairs = [(1, 0, 0.7), (0, 0, 0.7)]
    assert greedy_one_to_one(pairs, min_score=0.5) == ((0, 0, 0.7),)


def test_optimal_maximizes_cardinality_where_greedy_does_not() -> None:
    # greedy takes (0,0) first and strands pred 1; optimal matches both.
    pairs = [(0, 0, 0.9), (0, 1, 0.5), (1, 0, 0.6)]
    assert len(greedy_one_to_one(pairs, min_score=0.3)) == 1
    assert optimal_one_to_one(pairs, min_score=0.3) == ((0, 1, 0.5), (1, 0, 0.6))


def test_optimal_prefers_higher_score_at_equal_cardinality() -> None:
    pairs = [(0, 0, 0.4), (0, 1, 0.8)]
    assert optimal_one_to_one(pairs, min_score=0.3) == ((0, 1, 0.8),)
    assert optimal_one_to_one(pairs, min_score=0.9) == ()


def test_match_polygons_methods() -> None:
    ref = [box(0, 0, 1, 1), box(2, 0, 3, 1)]
    pred = [box(0.1, 0, 1.1, 1), box(2.6, 0, 3.6, 1)]
    greedy = match_polygons(pred, ref, min_iou=0.5)
    assert [(i, j) for i, j, _ in greedy] == [(0, 0)]
    optimal = match_polygons(pred, ref, min_iou=0.2, method="optimal")
    assert [(i, j) for i, j, _ in optimal] == [(0, 0), (1, 1)]
    with pytest.raises(ValueError, match="method"):
        match_polygons(pred, ref, min_iou=0.5, method="nope")


def test_line_candidates_within_tolerance() -> None:
    ref = [LineString([(0, 0), (10, 0)])]
    pred = [LineString([(0, 0.3), (10, 0.3)]), LineString([(0, 5), (10, 5)])]
    i, j = line_candidates(pred, ref, tol_m=0.4)
    assert list(i) == [0] and list(j) == [0]
    i, j = line_candidates([], ref, tol_m=0.4)
    assert len(i) == len(j) == 0


def test_layer_objects_defaults_and_length_check() -> None:
    lay = LayerObjects((box(0, 0, 1, 1), box(2, 0, 3, 1)), vineyard_ids=("V01", "V01"))
    assert lay.object_ids == (None, None) and lay.attrs == (None, None)
    with pytest.raises(ValueError, match="attrs has 1 values for 2 geometries"):
        LayerObjects((box(0, 0, 1, 1), box(2, 0, 3, 1)), attrs=("regular",))
