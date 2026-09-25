"""Crops around waste candidates (A§4.9 step 3, design 03 W3/W6): probe/review crops (box + context,
square, min side, shifted inside the tile, resized) and 512 px SAM windows; box <-> crop transforms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from vineyard.perception.waste.types import BoxPx

XYXY = tuple[float, float, float, float]
_EPS: Final = 1e-9


@dataclass(frozen=True)
class CropWindow:
    """Square window [x0, x0 + side) x [y0, y0 + side) in tile px."""

    x0: int
    y0: int
    side: int

    def as_xyxy(self) -> XYXY:
        return (float(self.x0), float(self.y0), float(self.x0 + self.side), float(self.y0 + self.side))


def _placed(centre: float, side: int, extent: int) -> int:
    return int(min(max(round(centre - side / 2.0), 0), extent - side))


def crop_window(box: BoxPx, context: float, min_px: int, extent_px: int) -> CropWindow:
    """Square of side max(min_px, (1 + context) * max(w, h)) centred on the box, shifted inside [0, extent]."""
    if context < 0 or min_px <= 0 or extent_px <= 0:
        raise ValueError(
            f"invalid crop parameters: context={context}, min_px={min_px}, extent_px={extent_px}"
        )
    side = min(max(min_px, math.ceil((1.0 + context) * max(box.width, box.height) - _EPS)), extent_px)
    cx, cy = box.centre
    return CropWindow(_placed(cx, side, extent_px), _placed(cy, side, extent_px), side)


def sam_window(centroid: tuple[float, float], crop_px: int, extent_px: int) -> CropWindow:
    """crop_px window centred on the centroid, clamped to [0, extent - crop_px]."""
    side = min(crop_px, extent_px)
    return CropWindow(_placed(centroid[0], side, extent_px), _placed(centroid[1], side, extent_px), side)


def extract_crop(rgb: np.ndarray, window: CropWindow, out_px: int) -> np.ndarray:
    """uint8 (out_px, out_px, 3) crop of `window`, resized with INTER_CUBIC (INTER_AREA when shrinking)."""
    h, w = rgb.shape[:2]
    if window.x0 < 0 or window.y0 < 0 or window.x0 + window.side > w or window.y0 + window.side > h:
        raise ValueError(f"crop window {window} outside the image {w}x{h}")
    patch = rgb[window.y0 : window.y0 + window.side, window.x0 : window.x0 + window.side]
    interp = cv2.INTER_AREA if window.side > out_px else cv2.INTER_CUBIC
    return np.ascontiguousarray(cv2.resize(patch, (out_px, out_px), interpolation=interp))


def _scale(window: CropWindow, out_px: int | None) -> float:
    return 1.0 if out_px is None else out_px / window.side


def box_to_crop(xyxy: XYXY, window: CropWindow, out_px: int | None = None) -> XYXY:
    """Tile px box -> crop px (optionally of the resized crop)."""
    s = _scale(window, out_px)
    x0, y0, x1, y1 = xyxy
    return ((x0 - window.x0) * s, (y0 - window.y0) * s, (x1 - window.x0) * s, (y1 - window.y0) * s)


def box_to_tile(xyxy: XYXY, window: CropWindow, out_px: int | None = None) -> XYXY:
    """Crop px box -> tile px (inverse of box_to_crop)."""
    s = _scale(window, out_px)
    x0, y0, x1, y1 = xyxy
    return (x0 / s + window.x0, y0 / s + window.y0, x1 / s + window.x0, y1 / s + window.y0)
