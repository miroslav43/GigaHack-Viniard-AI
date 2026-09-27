"""Ground-sampling-distance matching to the Sireț3 tiles.

Sireț3 was captured at 3.52 cm/px and delivered resampled to 2.5 cm/px, so it carries the blur of
a 3.52 cm capture. External imagery is degraded through the same chain
(native -> capture GSD with INTER_AREA -> target GSD with INTER_LINEAR) so textures match.
"""

from __future__ import annotations

import cv2
import numpy as np

TARGET_GSD_M = 0.025
CAPTURE_GSD_M = 0.0352


def scaled_size(size: int, src_gsd: float, dst_gsd: float) -> int:
    """Pixel length of ``size`` px at ``src_gsd`` when resampled to ``dst_gsd`` (at least 1)."""
    if src_gsd <= 0 or dst_gsd <= 0:
        raise ValueError("GSD must be positive")
    return max(1, int(round(size * src_gsd / dst_gsd)))


def degrade_to_target(
    img: np.ndarray,
    src_gsd: float,
    dst_gsd: float = TARGET_GSD_M,
    capture_gsd: float = CAPTURE_GSD_M,
) -> np.ndarray:
    """Resample ``img`` (H, W[, C]) from ``src_gsd`` to ``dst_gsd`` through the capture GSD.

    When the source is already coarser than the capture GSD, the intermediate step is skipped.
    """
    h, w = img.shape[:2]
    out_w, out_h = scaled_size(w, src_gsd, dst_gsd), scaled_size(h, src_gsd, dst_gsd)
    if src_gsd < capture_gsd:
        mid_w, mid_h = scaled_size(w, src_gsd, capture_gsd), scaled_size(h, src_gsd, capture_gsd)
        mid = cv2.resize(img, (mid_w, mid_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(mid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    interp = cv2.INTER_AREA if src_gsd < dst_gsd else cv2.INTER_LINEAR
    return cv2.resize(img, (out_w, out_h), interpolation=interp)


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a label/instance mask to ``shape_hw``."""
    h, w = shape_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
