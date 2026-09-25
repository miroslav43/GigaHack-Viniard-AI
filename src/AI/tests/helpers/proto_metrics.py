"""Test-only port of the prototype raster metrics (analiza_exemple/scripts/fit.py, axes.py).

Kept verbatim in behaviour (including the prototype's pixel quirks: np.round(p) references and
round(p*16) corridors without the -0.5 shift) so vector metrics can be cross-checked against the
numbers in the analysis report. Do not use in production code.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from vineyard.geo.tiling import GSD_M, TILE_PX

TILE_SHAPE = (TILE_PX, TILE_PX)
PROTO_CORRIDOR_HALF_M = 0.30
PROTO_MIN_AREA_M2 = 0.19
PROTO_MATCH_IOU = 0.5
PROTO_W_IOU, PROTO_W_F1 = 0.6, 0.4
PROTO_ROW_TOL_M, PROTO_ROW_NEED = 0.4, 0.8
_COVER_SAMPLES = 200
_SHIFT = 4
_SCALE = 1 << _SHIFT


def reference_labels(canopies_px: Sequence[np.ndarray], shape: tuple[int, int] = TILE_SHAPE) -> np.ndarray:
    """int32 label raster, canopy i -> i+1, fillPoly(np.round(p)) as in fit.py."""
    ref = np.zeros(shape, np.int32)
    for i, p in enumerate(canopies_px):
        cv2.fillPoly(ref, [np.round(p).astype(np.int32)], i + 1)
    return ref


def corridor(rows_px: Sequence[np.ndarray], half_m: float = PROTO_CORRIDOR_HALF_M,
             shape: tuple[int, int] = TILE_SHAPE) -> np.ndarray:
    """uint8 corridor of flat-capped rectangles around first->last row vertices (fit.corridor)."""
    m = np.zeros(shape, np.uint8)
    for r in rows_px:
        a, b = r[0], r[-1]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]]) * half_m / GSD_M
        poly = np.array([a + n, b + n, b - n, a - n])
        cv2.fillPoly(m, [np.round(poly * _SCALE).astype(np.int32)], 1, shift=_SHIFT)
    return m


def canopy_labels(veg: np.ndarray, rows_px: Sequence[np.ndarray], *, half_m: float = PROTO_CORRIDOR_HALF_M,
                  min_area_m2: float = PROTO_MIN_AREA_M2) -> np.ndarray:
    """veg ∩ corridor -> 8-connected components >= min area -> int32 labels 1..n (axes.canopies)."""
    mm = (veg.astype(np.uint8) > 0).astype(np.uint8) & corridor(rows_px, half_m, veg.shape)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(mm, connectivity=8)
    keep = np.zeros(nl, np.int32)
    j = 0
    for i in range(1, nl):
        if st[i, cv2.CC_STAT_AREA] * GSD_M * GSD_M >= min_area_m2:
            j += 1
            keep[i] = j
    return keep[lab]


def score(pred_lab: np.ndarray, ref_lab: np.ndarray, nref: int) -> tuple[float, float, int]:
    """(class IoU, instance F1 at IoU >= 0.5, n_pred) on label rasters (fit.score)."""
    pm, rm = pred_lab > 0, ref_lab > 0
    iou = float((pm & rm).sum()) / max(int((pm | rm).sum()), 1)
    npred = int(pred_lab.max())
    both = pm & rm
    pairs = np.stack([pred_lab[both], ref_lab[both]], 1)
    ua, cnt = np.unique(pairs, axis=0, return_counts=True) if len(pairs) else (np.zeros((0, 2), int), [])
    pa = np.bincount(pred_lab.ravel(), minlength=npred + 1)
    ra = np.bincount(ref_lab.ravel(), minlength=nref + 1)
    tp = sum(1 for (p, r), c in zip(ua, cnt, strict=True) if c / (pa[p] + ra[r] - c) >= PROTO_MATCH_IOU)
    f1 = 2 * tp / max(npred + nref, 1)
    return iou, f1, npred


def canopy_score(iou: float, f1: float) -> float:
    return PROTO_W_IOU * iou + PROTO_W_F1 * f1


def cover(a: np.ndarray, b: np.ndarray, tol_px: float) -> float:
    """Share of segment a (200 samples) within tol of segment b (axes.cover)."""
    s = np.linspace(0, 1, _COVER_SAMPLES)[:, None]
    pts = a[0] + s * (a[-1] - a[0])
    ab = b[-1] - b[0]
    t = np.clip(((pts - b[0]) @ ab) / (ab @ ab), 0, 1)
    dist = np.linalg.norm(pts - (b[0] + t[:, None] * ab), axis=1)
    return float(np.mean(dist <= tol_px))


def row_f1(pred: Sequence[np.ndarray], ref: Sequence[np.ndarray], tol_m: float = PROTO_ROW_TOL_M,
           need: float = PROTO_ROW_NEED) -> tuple[float, int]:
    """Greedy first-fit mutual-cover row F1 in px (axes.row_f1)."""
    used: set[int] = set()
    tp = 0
    for r in ref:
        for j, p in enumerate(pred):
            if j in used:
                continue
            if cover(p, r, tol_m / GSD_M) >= need and cover(r, p, tol_m / GSD_M) >= need:
                used.add(j)
                tp += 1
                break
    return 2 * tp / max(len(pred) + len(ref), 1), tp
