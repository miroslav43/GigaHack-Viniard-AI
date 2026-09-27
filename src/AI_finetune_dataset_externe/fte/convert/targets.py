"""Training targets shared by the ICAERUS and Sireț3 canopy samples.

Channels (uint8, 255 = ignore): c0 vine canopy foreground, c1 contact band between two different
plants (instance separation), c2 ground vegetation (green that is not vine).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IGNORE = 255
N_CHANNELS = 3
C_FG, C_CONTACT, C_GROUND = 0, 1, 2
_NO_ID = np.iinfo(np.uint16).max


@dataclass(frozen=True)
class GroundVegSpec:
    blur_sigma_px: float = 2.0
    hi: float = 35.0  # ExG >= hi outside vine regions -> ground vegetation
    lo: float = 15.0  # ExG <= lo -> not vegetation


def disk(radius_px: int) -> np.ndarray:
    r = max(0, int(radius_px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary dilation of a bool/0-1 mask with a disk; returns bool."""
    if radius_px <= 0:
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), disk(radius_px)) > 0


def erode(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)
    return cv2.erode(mask.astype(np.uint8), disk(radius_px)) > 0


def fg_from_instances(inst: np.ndarray) -> np.ndarray:
    return (inst > 0).astype(np.uint8)


def contact_band(inst: np.ndarray, radius_px: int = 2) -> np.ndarray:
    """bool mask of pixels within ``radius_px`` of at least two different instances.

    Equivalent to "covered by >= 2 distinct instances dilated by radius_px": the max and the min of
    the non-zero ids in the disk neighbourhood differ exactly there.
    """
    if inst.dtype != np.uint16:
        raise ValueError(f"instance map must be uint16, got {inst.dtype}")
    k = disk(radius_px)
    hi = cv2.dilate(inst, k)
    lo_src = np.where(inst > 0, inst, _NO_ID).astype(np.uint16)
    lo = cv2.erode(lo_src, k)
    return (hi > 0) & (lo != _NO_ID) & (hi != lo)


def excess_green(rgb: np.ndarray, blur_sigma_px: float = 2.0) -> np.ndarray:
    """2G - R - B on uint8 values (range -510..510) as float32, Gaussian-blurred."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {rgb.shape}")
    f = rgb.astype(np.float32)
    exg = 2.0 * f[..., 1] - f[..., 0] - f[..., 2]
    if blur_sigma_px > 0:
        exg = cv2.GaussianBlur(exg, (0, 0), blur_sigma_px)
    return exg


def ground_veg_labels(rgb: np.ndarray, vine_region: np.ndarray, spec: GroundVegSpec | None = None) -> np.ndarray:
    """c2: 1 = green outside the vine region, 0 = not green (anywhere), 255 otherwise."""
    spec = spec or GroundVegSpec()
    exg = excess_green(rgb, spec.blur_sigma_px)
    out = np.full(exg.shape, IGNORE, dtype=np.uint8)
    out[exg <= spec.lo] = 0
    out[(exg >= spec.hi) & ~vine_region.astype(bool)] = 1
    return out


def stack_labels(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, ignore: np.ndarray | None = None) -> np.ndarray:
    """HxWx3 uint8; ``ignore`` (bool) sets every channel to 255."""
    lbl = np.stack([c0, c1, c2], axis=-1).astype(np.uint8)
    if ignore is not None:
        lbl = np.where(ignore[..., None], np.uint8(IGNORE), lbl)
    return lbl
