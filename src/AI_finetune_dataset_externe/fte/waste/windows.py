"""Sliding-window math (torch-free): window starts covering an axis and the blending weight ramp."""

from __future__ import annotations

from typing import Final

import numpy as np

STRIDE: Final = 32  # U-Net input side must be a multiple of the encoder stride


def ceil_to(n: int, k: int = STRIDE) -> int:
    return ((n + k - 1) // k) * k


def window_starts(length: int, window: int, overlap: int) -> list[int]:
    """Starts of ``window``-long windows with >= ``overlap`` overlap covering [0, length); last flush."""
    if window <= 0 or not 0 <= overlap < window:
        raise ValueError(f"need window > 0 and 0 <= overlap < window, got {window}, {overlap}")
    if length <= window:
        return [0]
    step = window - overlap
    starts = list(range(0, length - window, step))
    starts.append(length - window)
    return sorted(set(starts))


def ramp_weight(window: int, overlap: int, floor: float = 1e-3) -> np.ndarray:
    """(window, window) float32 weight: 1 inside, linear ramp to ``floor`` over ``overlap`` px at the edges."""
    if overlap <= 0:
        return np.ones((window, window), dtype=np.float32)
    idx = np.arange(window, dtype=np.float32)
    edge = np.minimum(idx + 0.5, window - idx - 0.5) / float(overlap)
    one_d = np.clip(edge, floor, 1.0)
    return np.outer(one_d, one_d).astype(np.float32)


def pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Reflect-pad ``img`` (H, W[, C]) at the bottom/right to at least (h, w) (new array)."""
    ph, pw = max(0, h - img.shape[0]), max(0, w - img.shape[1])
    if ph == 0 and pw == 0:
        return img.copy()
    pads = ((0, ph), (0, pw)) + (((0, 0),) if img.ndim == 3 else ())
    mode = "reflect" if img.shape[0] > ph and img.shape[1] > pw else "edge"
    return np.pad(img, pads, mode=mode)
