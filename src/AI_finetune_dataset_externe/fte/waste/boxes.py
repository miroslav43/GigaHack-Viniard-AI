"""Heatmap -> waste boxes (torch-free): connected components of p >= t, tight px boxes, area gate, NMS.

Boxes are float arrays (N, 5) = xtl, ytl, xbr, ybr, score in continuous tile px (a component covering
pixel columns 3..5 spans x 3.0 .. 6.0), the convention of `configs/waste_confirmed.csv`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

GSD_M: Final = 0.025
MIN_AREA_M2: Final = 0.015
MAX_AREA_M2: Final = 6.0
NMS_IOU: Final = 0.3
MATCH_IOU: Final = 0.3
BOX_COLS: Final = 5


@dataclass(frozen=True)
class BoxParams:
    threshold: float = 0.5
    min_area_m2: float = MIN_AREA_M2
    max_area_m2: float = MAX_AREA_M2
    gsd_m: float = GSD_M
    nms_iou: float = NMS_IOU

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if not 0.0 <= self.min_area_m2 < self.max_area_m2:
            raise ValueError("need 0 <= min_area_m2 < max_area_m2")

    def area_px(self) -> tuple[float, float]:
        px = self.gsd_m * self.gsd_m
        return self.min_area_m2 / px, self.max_area_m2 / px


def empty_boxes() -> np.ndarray:
    return np.zeros((0, BOX_COLS), dtype=np.float64)


def as_prob(heat: np.ndarray) -> np.ndarray:
    """uint8 0..255 or float 0..1 heatmap -> float32 probabilities."""
    if heat.ndim != 2:
        raise ValueError(f"heatmap must be 2-D, got shape {heat.shape}")
    if heat.dtype == np.uint8:
        return heat.astype(np.float32) / 255.0
    return heat.astype(np.float32, copy=False)


def heatmap_to_boxes(heat: np.ndarray, p: BoxParams) -> np.ndarray:
    """Tight boxes of the components of ``heat >= p.threshold`` with score = max p inside, then NMS."""
    prob = as_prob(heat)
    binary = (prob >= p.threshold - 1e-6).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return empty_boxes()
    lo, hi = p.area_px()
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = np.flatnonzero((area >= lo) & (area <= hi)) + 1
    if keep.size == 0:
        return empty_boxes()
    max_p = np.zeros(n, dtype=np.float64)
    np.maximum.at(max_p, labels.ravel(), prob.ravel().astype(np.float64))
    x, y = stats[keep, cv2.CC_STAT_LEFT], stats[keep, cv2.CC_STAT_TOP]
    w, h = stats[keep, cv2.CC_STAT_WIDTH], stats[keep, cv2.CC_STAT_HEIGHT]
    boxes = np.column_stack([x, y, x + w, y + h, max_p[keep]]).astype(np.float64)
    return nms(boxes, p.nms_iou)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) IoU of xyxy boxes (extra columns ignored)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax0, ay0, ax1, ay1 = (a[:, i][:, None] for i in range(4))
    bx0, by0, bx1, by1 = (b[:, i][None, :] for i in range(4))
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def nms(boxes: np.ndarray, iou: float = NMS_IOU) -> np.ndarray:
    """Greedy NMS by descending score (column 4); returns a new array sorted by score."""
    if len(boxes) == 0:
        return empty_boxes()
    order = np.argsort(-boxes[:, 4], kind="stable")
    ordered = boxes[order]
    ious = box_iou(ordered, ordered)
    alive = np.ones(len(ordered), dtype=bool)
    for i in range(len(ordered)):
        if alive[i]:
            later = np.arange(len(ordered)) > i
            alive &= ~(later & (ious[i] >= iou))
    return ordered[alive].copy()


def centres(boxes: np.ndarray) -> np.ndarray:
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0])


def centre_inside(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Bool (len(boxes),): the box centre lies inside at least one region box."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes), dtype=bool)
    c = centres(boxes)
    inside = ((c[:, 0:1] >= regions[None, :, 0]) & (c[:, 0:1] <= regions[None, :, 2])
              & (c[:, 1:2] >= regions[None, :, 1]) & (c[:, 1:2] <= regions[None, :, 3]))
    return inside.any(axis=1)


@dataclass(frozen=True)
class MatchScore:
    tp: int
    n_pred: int
    n_ref: int

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 1.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_ref if self.n_ref else 1.0

    @property
    def f1(self) -> float:
        denom = self.n_pred + self.n_ref
        return 2.0 * self.tp / denom if denom else 1.0

    def __add__(self, other: MatchScore) -> MatchScore:
        return MatchScore(self.tp + other.tp, self.n_pred + other.n_pred, self.n_ref + other.n_ref)


def match_boxes(pred: np.ndarray, ref: np.ndarray, iou: float = MATCH_IOU) -> MatchScore:
    """Optimal one-to-one matching at IoU >= ``iou`` (duplicates are false positives)."""
    if len(pred) == 0 or len(ref) == 0:
        return MatchScore(0, len(pred), len(ref))
    m = box_iou(pred, ref)
    cost = np.where(m >= iou, -m, 1.0)
    rows, cols = linear_sum_assignment(cost)
    return MatchScore(int((m[rows, cols] >= iou).sum()), len(pred), len(ref))


def drop_in_regions(pred: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Predictions whose centre falls in a don't-care region are removed (new array)."""
    return pred[~centre_inside(pred, regions)].copy()


def cap_per_image(boxes: np.ndarray, cap: int) -> np.ndarray:
    """At most ``cap`` highest-score boxes."""
    if cap < 0:
        raise ValueError("cap must be >= 0")
    order = np.argsort(-boxes[:, 4], kind="stable")[:cap]
    return boxes[order].copy()


def stack_boxes(parts: Sequence[np.ndarray]) -> np.ndarray:
    non_empty = [p for p in parts if len(p)]
    return np.concatenate(non_empty, axis=0) if non_empty else empty_boxes()
