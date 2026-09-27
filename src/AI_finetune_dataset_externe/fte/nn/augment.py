"""Numpy augmentation for (image uint8 HxWx3, labels uint8 HxWxK) pairs; returns new arrays.

Geometric ops (dihedral, scale jitter) move image and labels together (labels with nearest
neighbour, so 255 = ignore survives). Photometric ops touch the image only. Hue is kept almost
fixed: vine vs grass colour is signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentSpec:
    scale_jitter: float = 0.15
    brightness: float = 0.25
    contrast: float = 0.2
    saturation: float = 0.3
    hue: float = 0.02
    gamma: tuple[float, float] = (0.8, 1.25)
    channel_gain: float = 0.05
    blur_p: float = 0.3
    jpeg_p: float = 0.3
    noise_sigma: float = 3.0


def dihedral(img: np.ndarray, lbl: np.ndarray, k: int, flip: bool) -> tuple[np.ndarray, np.ndarray]:
    img2, lbl2 = np.rot90(img, k, axes=(0, 1)), np.rot90(lbl, k, axes=(0, 1))
    if flip:
        img2, lbl2 = img2[:, ::-1], lbl2[:, ::-1]
    return np.ascontiguousarray(img2), np.ascontiguousarray(lbl2)


def scale_crop(img: np.ndarray, lbl: np.ndarray, factor: float, out: int,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Resize by ``factor`` then take a random ``out`` x ``out`` crop (pads labels with 255)."""
    h, w = img.shape[:2]
    nh, nw = max(out, int(round(h * factor))), max(out, int(round(w * factor)))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lbl2 = cv2.resize(lbl, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if lbl2.ndim == 2:
        lbl2 = lbl2[:, :, None]
    y0 = int(rng.integers(0, nh - out + 1))
    x0 = int(rng.integers(0, nw - out + 1))
    return img2[y0:y0 + out, x0:x0 + out].copy(), lbl2[y0:y0 + out, x0:x0 + out].copy()


def photometric(img: np.ndarray, spec: AugmentSpec, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = x * (1.0 + rng.uniform(-spec.brightness, spec.brightness))
    mean = x.mean()
    x = (x - mean) * (1.0 + rng.uniform(-spec.contrast, spec.contrast)) + mean
    x = x * (1.0 + rng.uniform(-spec.channel_gain, spec.channel_gain, size=3)).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + 360.0 * rng.uniform(-spec.hue, spec.hue)) % 360.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + rng.uniform(-spec.saturation, spec.saturation)), 0.0, 1.0)
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    x = np.power(np.clip(x, 0.0, 1.0), rng.uniform(*spec.gamma))
    out = np.clip(x * 255.0 + rng.normal(0.0, spec.noise_sigma, x.shape), 0, 255).astype(np.uint8)
    if rng.random() < spec.blur_p:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 1.0)))
    if rng.random() < spec.jpeg_p:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return out


def augment(img: np.ndarray, lbl: np.ndarray, out: int, spec: AugmentSpec,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Full chain: scale jitter + random crop -> dihedral -> photometric. ``lbl`` is HxWxK."""
    factor = 1.0 + rng.uniform(-spec.scale_jitter, spec.scale_jitter)
    img2, lbl2 = scale_crop(img, lbl, factor, out, rng)
    img2, lbl2 = dihedral(img2, lbl2, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
    return photometric(img2, spec, rng), lbl2
