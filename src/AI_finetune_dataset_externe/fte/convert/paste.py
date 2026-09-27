"""Copy-paste of external waste instances (polygon cut-outs) onto Sireț3 windows.

Each instance is rotated (any angle), rescaled (+-20 %), luminance-matched part of the way to the local
background (the sources have other exposures), pasted with a feathered (~1 px) alpha and an optional soft
shadow. The label becomes 1 where alpha >= 0.5; everything else of the background label is kept.
All functions return new arrays (inputs are never modified).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from fte.convert.coco import WASTE, StoreImage, WasteStore, object_mask, read_rgb

MIN_AREA_M2: Final = 0.015
MAX_AREA_M2: Final = 3.0
MAX_SIDE_PX: Final = 200
GSD_M: Final = 0.025
CROP_MARGIN_PX: Final = 3


@dataclass(frozen=True)
class Instance:
    rgb: np.ndarray  # (h, w, 3) uint8 crop around the object
    mask: np.ndarray  # (h, w) bool object mask
    source: str


@dataclass(frozen=True)
class PasteSpec:
    scale_jitter: float = 0.2
    feather_sigma_px: float = 0.7
    lum_strength: tuple[float, float] = (0.0, 0.5)
    shadow_p: float = 0.5
    shadow_offset_px: tuple[int, int] = (1, 4)
    shadow_strength: tuple[float, float] = (0.1, 0.35)
    n_objects: tuple[int, int] = (1, 4)
    max_side_px: int = MAX_SIDE_PX


def _area_ok(area_px: float) -> bool:
    m2 = area_px * GSD_M * GSD_M
    return MIN_AREA_M2 <= m2 <= MAX_AREA_M2


def image_instances(rgb: np.ndarray, img: StoreImage, source: str, max_side: int = MAX_SIDE_PX) -> list[Instance]:
    """Cut-outs of the waste objects (with polygons) of one scaled store image."""
    h, w = rgb.shape[:2]
    out = []
    for obj in img.objects:
        if obj.role != WASTE or not obj.polygons:
            continue
        mask = object_mask(obj, (h, w))
        ys, xs = np.nonzero(mask)
        if len(ys) == 0 or not _area_ok(len(ys)):
            continue
        y0, y1 = max(0, ys.min() - CROP_MARGIN_PX), min(h, ys.max() + 1 + CROP_MARGIN_PX)
        x0, x1 = max(0, xs.min() - CROP_MARGIN_PX), min(w, xs.max() + 1 + CROP_MARGIN_PX)
        if max(y1 - y0, x1 - x0) > max_side:
            continue
        out.append(Instance(rgb[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), source))
    return out


def instance_bank(store: WasteStore, images: Sequence[StoreImage]) -> tuple[Instance, ...]:
    bank: list[Instance] = []
    for img in images:
        if img.n_waste:
            bank.extend(image_instances(read_rgb(store.image_path(img)), img, store.name))
    return tuple(bank)


def transform_instance(inst: Instance, angle_deg: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate + scale on an expanded canvas; returns (rgb uint8, alpha float32 in [0, 1])."""
    h, w = inst.mask.shape
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    rgb = cv2.resize(inst.rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    alpha = cv2.resize(inst.mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)
    side = int(np.ceil(np.hypot(nh, nw))) + 2
    m = cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), angle_deg, 1.0)
    m[0, 2] += (side - nw) / 2.0
    m[1, 2] += (side - nh) / 2.0
    rgb_r = cv2.warpAffine(rgb, m, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    alpha_r = cv2.warpAffine(alpha, m, (side, side), flags=cv2.INTER_LINEAR, borderValue=0.0)
    ys, xs = np.nonzero(alpha_r > 0.01)
    if len(ys) == 0:
        return rgb_r[:1, :1], np.zeros((1, 1), np.float32)
    sl = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgb_r[sl].copy(), np.clip(alpha_r[sl], 0.0, 1.0)


def feather(alpha: np.ndarray, sigma: float) -> np.ndarray:
    return alpha if sigma <= 0 else cv2.GaussianBlur(alpha, (0, 0), sigma)


def luminance_match(obj: np.ndarray, alpha: np.ndarray, bg: np.ndarray, strength: float) -> np.ndarray:
    """Scale the object's brightness part of the way (``strength``) toward the background mean."""
    wsum = float(alpha.sum())
    if wsum <= 0 or strength <= 0:
        return obj.copy()
    obj_l = float((obj.astype(np.float32).mean(axis=2) * alpha).sum() / wsum)
    bg_l = float(bg.astype(np.float32).mean())
    ratio = (bg_l + 1.0) / (obj_l + 1.0)
    gain = 1.0 + strength * (ratio - 1.0)
    return np.clip(obj.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def shadow(bg: np.ndarray, alpha: np.ndarray, dx: int, dy: int, strength: float) -> np.ndarray:
    """Darken ``bg`` under ``alpha`` shifted by (dx, dy) and blurred (same shape as ``alpha``)."""
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    sh = cv2.warpAffine(alpha, m, (alpha.shape[1], alpha.shape[0]), borderValue=0.0)
    sh = cv2.GaussianBlur(sh, (0, 0), 1.5) * strength
    return np.clip(bg.astype(np.float32) * (1.0 - sh[:, :, None]), 0, 255).astype(np.uint8)


def composite(img: np.ndarray, label: np.ndarray, obj: np.ndarray, alpha: np.ndarray, x: int, y: int,
              ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float] | None]:
    """Alpha-blend ``obj`` at top-left (x, y) (clipped); label 1 where alpha >= 0.5; returns the box."""
    h, w = label.shape
    oh, ow = alpha.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + ow), min(h, y + oh)
    new_img, new_lbl = img.copy(), label.copy()
    if x1 <= x0 or y1 <= y0:
        return new_img, new_lbl, None
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    o = obj[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    region = new_img[y0:y1, x0:x1].astype(np.float32)
    new_img[y0:y1, x0:x1] = np.clip(region * (1 - a[:, :, None]) + o * a[:, :, None], 0, 255).astype(np.uint8)
    on = a >= 0.5
    if not on.any():
        return new_img, new_lbl, None
    new_lbl[y0:y1, x0:x1][on] = 1
    ys, xs = np.nonzero(on)
    return new_img, new_lbl, (float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max() + 1),
                              float(y0 + ys.max() + 1))


def paste_one(img: np.ndarray, label: np.ndarray, inst: Instance, spec: PasteSpec, rng: np.random.Generator,
              ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float] | None]:
    scale = 1.0 + rng.uniform(-spec.scale_jitter, spec.scale_jitter)
    obj, alpha = transform_instance(inst, float(rng.uniform(0, 360)), scale)
    alpha = feather(alpha, spec.feather_sigma_px)
    h, w = label.shape
    oh, ow = alpha.shape
    x = int(rng.integers(-ow // 4, max(1, w - 3 * ow // 4)))
    y = int(rng.integers(-oh // 4, max(1, h - 3 * oh // 4)))
    bg = img[max(0, y):max(0, y) + oh, max(0, x):max(0, x) + ow]
    obj = luminance_match(obj, alpha, bg if bg.size else img, float(rng.uniform(*spec.lum_strength)))
    base = img
    if rng.random() < spec.shadow_p:
        base = _with_shadow(img, alpha, x, y, spec, rng)
    return composite(base, label, obj, alpha, x, y)


def _with_shadow(img: np.ndarray, alpha: np.ndarray, x: int, y: int, spec: PasteSpec,
                 rng: np.random.Generator) -> np.ndarray:
    lo, hi = spec.shadow_offset_px
    dx, dy = (int(v) for v in rng.integers(lo, hi + 1, size=2))
    pad = hi + 3
    canvas = np.pad(alpha, pad)
    x0, y0 = x - pad, y - pad
    h, w = img.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x0 + canvas.shape[1]), min(h, y0 + canvas.shape[0])
    if cx1 <= cx0 or cy1 <= cy0:
        return img.copy()
    out = img.copy()
    sub_alpha = canvas[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    out[cy0:cy1, cx0:cx1] = shadow(img[cy0:cy1, cx0:cx1], sub_alpha, dx, dy, float(rng.uniform(*spec.shadow_strength)))
    return out


def paste_many(img: np.ndarray, label: np.ndarray, bank: Sequence[Instance], spec: PasteSpec,
               rng: np.random.Generator, n: int | None = None) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Paste ``n`` (default: random in spec.n_objects) random instances; returns img, label, boxes."""
    if not bank:
        raise ValueError("empty instance bank")
    k = n if n is not None else int(rng.integers(spec.n_objects[0], spec.n_objects[1] + 1))
    boxes: list[tuple] = []
    for _ in range(k):
        img, label, box = paste_one(img, label, bank[int(rng.integers(len(bank)))], spec, rng)
        if box is not None:
            boxes.append(box)
    return img, label, boxes
