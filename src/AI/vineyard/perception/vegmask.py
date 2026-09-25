"""Vegetation mask (port of analiza_exemple/scripts/axes.py::veg_mask) and visibility codes.

na = -(a* - 128) from 8-bit Lab, Gaussian blur, `na > threshold`, AND valid. No closing by default.
NN probability upsampling and fusion live in the torch-free vineyard/nn/{probs,fusion}.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from vineyard.perception.types import (
    F32,
    U8,
    VIS_NODATA,
    VIS_OK,
    VIS_OVEREXPOSED,
    VIS_SHADOW,
    BoolMask,
    TileMasks,
)

if TYPE_CHECKING:
    from vineyard.config import VegConfig

METHOD_OTSU: Final = "otsu"
_LAB_A_OFFSET: Final = 128.0  # OpenCV 8-bit Lab stores a* + 128
_OTSU_BINS: Final = 256


def _require_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8 RGB, got {rgb.shape} {rgb.dtype}")


def _require_mask(mask: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if mask.shape != shape:
        raise ValueError(f"{name} shape {mask.shape} != image shape {shape}")


def neg_a(rgb: U8, blur_sigma_px: float) -> F32:
    """-(a* - 128) as float32, Gaussian-blurred with sigma `blur_sigma_px` (0 = no blur)."""
    _require_rgb(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    na = _LAB_A_OFFSET - lab[..., 1].astype(np.float32)
    if blur_sigma_px > 0:
        na = cv2.GaussianBlur(na, (0, 0), blur_sigma_px)
    return na


def veg_mask(rgb: U8, valid: BoolMask, *, blur_sigma_px: float, threshold: float) -> BoolMask:
    """Vegetation = neg_a > threshold, restricted to `valid`."""
    _require_rgb(rgb)
    _require_mask(valid, rgb.shape[:2], "valid")
    return (neg_a(rgb, blur_sigma_px) > threshold) & valid


def vis_codes(rgb: U8, valid: BoolMask, *, shadow_v_max: int, overexp_v_min: int) -> U8:
    """Per-pixel visibility: VIS_SHADOW if V < shadow_v_max, VIS_OVEREXPOSED if V > overexp_v_min,
    VIS_NODATA outside `valid` (wins over the others), else VIS_OK. V = max(R, G, B) (HSV value)."""
    _require_rgb(rgb)
    _require_mask(valid, rgb.shape[:2], "valid")
    v = rgb.max(axis=2)
    codes = np.full(v.shape, VIS_OK, dtype=np.uint8)
    codes[v < shadow_v_max] = VIS_SHADOW
    codes[v > overexp_v_min] = VIS_OVEREXPOSED
    codes[~valid] = VIS_NODATA
    return codes


def otsu_threshold(values: F32) -> float:
    """Otsu threshold of a 1-D sample (256-bin histogram); constant input returns that constant."""
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("otsu_threshold: empty sample")
    if not np.isfinite(x).all():
        raise ValueError("otsu_threshold: sample contains NaN or inf")
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        return lo
    hist, edges = np.histogram(x, bins=_OTSU_BINS, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)[:-1].astype(np.float64)
    w1 = x.size - w0
    s0 = np.cumsum(hist * centers)[:-1]
    m0 = s0 / np.maximum(w0, 1.0)
    m1 = (float((hist * centers).sum()) - s0) / np.maximum(w1, 1.0)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(edges[1:-1][int(np.argmax(between))])


def _valid_fraction(veg: BoolMask, valid: BoolMask) -> float:
    n_valid = int(valid.sum())
    return float((veg & valid).sum()) / n_valid if n_valid else 0.0


def _close(veg: BoolMask, kernel_px: int) -> BoolMask:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    return cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def _needs_otsu(veg_frac: float, valid: BoolMask, cfg: VegConfig) -> bool:
    lo, hi = cfg.otsu_fallback_veg_frac
    return cfg.otsu_fallback_enabled and bool(valid.any()) and not (lo <= veg_frac <= hi)


def compute_tile_masks(rgb: U8, valid_eroded: BoolMask, cfg: VegConfig) -> TileMasks:
    """tile_prep masks: veg (fixed threshold, Otsu fallback when enabled and the tile-wide
    veg fraction is outside cfg.otsu_fallback_veg_frac), vis codes, method, threshold, veg_frac."""
    _require_rgb(rgb)
    _require_mask(valid_eroded, rgb.shape[:2], "valid_eroded")
    na = neg_a(rgb, cfg.blur_sigma_px)
    method, threshold = cfg.method, float(cfg.threshold)
    veg = (na > threshold) & valid_eroded
    if _needs_otsu(_valid_fraction(veg, valid_eroded), valid_eroded, cfg):
        method, threshold = METHOD_OTSU, otsu_threshold(na[valid_eroded])
        veg = (na > threshold) & valid_eroded
    if cfg.morph_close_px > 0:
        veg = _close(veg, cfg.morph_close_px) & valid_eroded
    vis = vis_codes(rgb, valid_eroded, shadow_v_max=cfg.shadow_v_max, overexp_v_min=cfg.overexposed_v_min)
    return TileMasks(veg=veg, vis=vis, method=method, threshold=threshold, veg_frac=_valid_fraction(veg, valid_eroded))
