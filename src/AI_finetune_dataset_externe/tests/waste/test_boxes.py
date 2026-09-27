from __future__ import annotations

import numpy as np
import pytest

from fte.waste.boxes import (
    BoxParams,
    box_iou,
    cap_per_image,
    centre_inside,
    drop_in_regions,
    heatmap_to_boxes,
    match_boxes,
    nms,
)


def _heat(shape=(100, 100)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def test_component_gives_tight_box_with_max_score() -> None:
    heat = _heat()
    heat[10:20, 30:45] = 0.6
    heat[15, 40] = 0.9
    boxes = heatmap_to_boxes(heat, BoxParams(threshold=0.5))
    assert boxes.shape == (1, 5)
    np.testing.assert_allclose(boxes[0, :4], [30, 10, 45, 20])
    assert boxes[0, 4] == pytest.approx(0.9)


def test_uint8_heatmap_equivalent_to_float() -> None:
    heat = _heat()
    heat[10:20, 30:45] = 0.8
    u8 = np.round(heat * 255).astype(np.uint8)
    a = heatmap_to_boxes(heat, BoxParams(threshold=0.5))
    b = heatmap_to_boxes(u8, BoxParams(threshold=0.5))
    np.testing.assert_allclose(a[:, :4], b[:, :4])
    assert b[0, 4] == pytest.approx(0.8, abs=0.01)


def test_area_gate_drops_tiny_and_huge_components() -> None:
    heat = _heat((300, 300))
    heat[0:3, 0:3] = 1.0  # 9 px = 0.0056 m2 < 0.015
    heat[50:60, 50:60] = 1.0  # 100 px = 0.0625 m2 kept
    boxes = heatmap_to_boxes(heat, BoxParams(threshold=0.5))
    assert len(boxes) == 1
    huge = np.ones((300, 300), dtype=np.float32)  # 90000 px = 56 m2 > 6
    assert len(heatmap_to_boxes(huge, BoxParams(threshold=0.5))) == 0


def test_threshold_splits_components() -> None:
    heat = _heat()
    heat[10:20, 10:20] = 0.9
    heat[10:20, 20:22] = 0.4  # bridge
    heat[10:20, 22:32] = 0.9
    assert len(heatmap_to_boxes(heat, BoxParams(threshold=0.3))) == 1
    assert len(heatmap_to_boxes(heat, BoxParams(threshold=0.5))) == 2


def test_empty_heatmap() -> None:
    assert heatmap_to_boxes(_heat(), BoxParams(threshold=0.5)).shape == (0, 5)


def test_box_params_validation() -> None:
    with pytest.raises(ValueError):
        BoxParams(threshold=0.0)
    with pytest.raises(ValueError):
        BoxParams(min_area_m2=7.0)


def test_iou_and_nms() -> None:
    a = np.array([[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [50, 50, 60, 60, 0.7]], dtype=float)
    iou = box_iou(a, a)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(81 / 119)
    assert iou[0, 2] == 0.0
    kept = nms(a, 0.3)
    assert len(kept) == 2
    assert kept[0, 4] == pytest.approx(0.9)
    assert a.shape == (3, 5)  # input untouched


def test_match_is_one_to_one() -> None:
    ref = np.array([[0, 0, 10, 10]], dtype=float)
    pred = np.array([[0, 0, 10, 10, 0.9], [0, 0, 9, 10, 0.8]], dtype=float)
    s = match_boxes(pred, ref, 0.3)
    assert (s.tp, s.n_pred, s.n_ref) == (1, 2, 1)
    assert s.f1 == pytest.approx(2 / 3)


def test_match_optimal_assignment() -> None:
    ref = np.array([[0, 0, 10, 10], [6, 0, 16, 10]], dtype=float)
    pred = np.array([[3, 0, 13, 10, 0.9], [8, 0, 18, 10, 0.8]], dtype=float)
    assert match_boxes(pred, ref, 0.3).tp == 2


def test_empty_match_scores() -> None:
    s = match_boxes(np.zeros((0, 5)), np.zeros((0, 4)), 0.3)
    assert s.f1 == 1.0
    s2 = match_boxes(np.zeros((0, 5)), np.array([[0, 0, 1, 1]], float), 0.3)
    assert s2.f1 == 0.0 and s2.recall == 0.0


def test_centre_inside_and_drop() -> None:
    pred = np.array([[0, 0, 4, 4, 0.9], [20, 20, 24, 24, 0.5]], dtype=float)
    regions = np.array([[0, 0, 5, 5]], dtype=float)
    assert centre_inside(pred, regions).tolist() == [True, False]
    assert len(drop_in_regions(pred, regions)) == 1


def test_cap_keeps_highest() -> None:
    b = np.array([[0, 0, 1, 1, 0.2], [0, 0, 1, 1, 0.9], [0, 0, 1, 1, 0.5]], dtype=float)
    out = cap_per_image(b, 2)
    assert out[:, 4].tolist() == [0.9, 0.5]
