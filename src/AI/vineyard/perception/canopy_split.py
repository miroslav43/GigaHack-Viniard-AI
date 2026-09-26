"""Split touching vine canopies at visible narrowings (annotation rules §2.2).

Where neighbouring plants touch, the reference splits the foliage "at the point where it visibly narrows".
A connected canopy component long enough to hold two plants is profiled along its main direction (PCA of
its pixels): the perpendicular width per `bin_px` slice, smoothed over 3 slices. A slice is a neck when it is
the narrowest within ±`min_piece_m` and at most `neck_ratio` × the smaller of the widest slices within
`window_m` on either side; both resulting parts must be at least `min_piece_m` long and `min_part_px` large.
Necks are cut by clearing the pixels of that slice (2 px thick, so the parts stay 8-disconnected). The
input mask is not modified; components without a qualifying neck are returned unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from vineyard.geo.tiling import GSD_M
from vineyard.perception.types import BoolMask

MIN_BIN_PX: Final = 2  # a cut thinner than 2 px leaves the parts 8-connected
_SMOOTH: Final = np.array([1.0, 1.0, 1.0]) / 3.0


@dataclass(frozen=True)
class NeckSplitOptions:
    min_along_m: float  # components shorter than this along their main direction are not examined
    neck_ratio: float  # neck width <= ratio * min(widest left, widest right)
    min_piece_m: float  # each part keeps at least this length (and necks are this far apart)
    window_m: float  # the widest slice is searched this far on each side of the neck
    min_part_px: int  # each part keeps at least this many pixels
    bin_px: int = MIN_BIN_PX

    def __post_init__(self) -> None:
        if not 0 < self.neck_ratio < 1:
            raise ValueError(f"neck_ratio {self.neck_ratio} must be in (0, 1)")
        if self.bin_px < MIN_BIN_PX:
            raise ValueError(f"bin_px {self.bin_px} must be >= {MIN_BIN_PX}")
        if not 0 < 2 * self.min_piece_m <= self.min_along_m:
            raise ValueError(f"need 0 < 2 * min_piece_m ({self.min_piece_m}) <= min_along_m ({self.min_along_m})")
        if self.window_m <= 0 or self.min_part_px < 1:
            raise ValueError("window_m must be > 0 and min_part_px >= 1")


def neck_bins(width: np.ndarray, *, ratio: float, min_piece_bins: int, window_bins: int) -> list[int]:
    """Indices of the neck slices of a width profile (deepest first, at least min_piece_bins apart)."""
    n = len(width)
    w = np.convolve(width, _SMOOTH, mode="same") if n >= len(_SMOOTH) else width.astype(np.float64)
    scored: list[tuple[float, int]] = []
    for i in range(min_piece_bins, n - min_piece_bins):
        lo, hi = max(0, i - min_piece_bins), min(n, i + min_piece_bins + 1)
        if w[i] > w[lo:hi].min():
            continue
        left = w[max(0, i - window_bins) : i].max()
        right = w[i + 1 : i + 1 + window_bins].max()
        depth = w[i] / max(min(left, right), 1e-9)
        if depth <= ratio:
            scored.append((depth, i))
    chosen: list[int] = []
    for _, i in sorted(scored):
        if all(abs(i - j) >= min_piece_bins for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def _profile(ys: np.ndarray, xs: np.ndarray, bin_px: int) -> tuple[np.ndarray, np.ndarray]:
    """(slice index per pixel, width per slice in px) along the main direction of the pixels."""
    pts = np.column_stack([xs, ys]).astype(np.float64)
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    t = centred @ vt[0]
    idx = np.floor((t - t.min()) / bin_px).astype(np.int64)
    counts = np.bincount(idx).astype(np.float64)
    return idx, counts / bin_px


def _cut_component(ys: np.ndarray, xs: np.ndarray, opts: NeckSplitOptions) -> np.ndarray:
    """Bool per pixel: True where the pixel is cleared (neck slices), all False when nothing qualifies."""
    idx, width = _profile(ys, xs, opts.bin_px)
    bin_m = opts.bin_px * GSD_M
    if len(width) * bin_m < opts.min_along_m:
        return np.zeros(len(ys), dtype=bool)
    necks = neck_bins(width, ratio=opts.neck_ratio, min_piece_bins=max(1, round(opts.min_piece_m / bin_m)),
                      window_bins=max(1, round(opts.window_m / bin_m)))
    cut = np.zeros(len(ys), dtype=bool)
    edges = [-1, *necks, len(width)]
    counts = np.bincount(idx, minlength=len(width))
    parts = [int(counts[a + 1 : b].sum()) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    if not necks or min(parts) < opts.min_part_px:
        return cut
    return np.isin(idx, necks)


def split_at_necks(mask: BoolMask, connectivity: int, opts: NeckSplitOptions) -> BoolMask:
    """Copy of `mask` with the neck slices of its long components cleared."""
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    out = np.array(mask, dtype=bool, copy=True)
    min_px = opts.min_along_m / GSD_M
    for label in range(1, n):
        x, y, w, h = (int(v) for v in stats[label, :4])
        if max(w, h) * 1.0 < min_px / np.sqrt(2) or stats[label, cv2.CC_STAT_AREA] < 2 * opts.min_part_px:
            continue
        ys, xs = np.nonzero(comp[y : y + h, x : x + w] == label)
        cut = _cut_component(ys, xs, opts)
        if cut.any():
            out[ys[cut] + y, xs[cut] + x] = False
    return out
