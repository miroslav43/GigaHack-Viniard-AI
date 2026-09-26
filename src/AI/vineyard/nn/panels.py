"""Qualitative NN panels (A§4.6): RGB | a* | NN probability | difference, as JPEG, torch-free.

The difference panel compares the NN mask (p >= nn.prob_threshold) with the classic a* vegetation mask:
white = both, green = NN only, magenta = a* only. Hard tiles are picked automatically as the tiles
where the two masks disagree most (grass between rows, shade, trees in the row).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from vineyard.pipeline.atomic import atomic_write_bytes

PANEL_PX: Final = 768
JPEG_QUALITY: Final = 90
NA_PERCENTILES: Final = (1.0, 99.0)
U8_MAX: Final = 255
LABEL_BAR_PX: Final = 28
LABEL_SCALE: Final = 0.7
LABEL_THICKNESS: Final = 2
LABEL_ORIGIN: Final = (8, 20)
LABELS: Final = ("RGB", "a* (-a)", "NN prob", "diff: white=both green=NN magenta=a*")
COLOUR_BOTH: Final = (255, 255, 255)  # RGB
COLOUR_NN_ONLY: Final = (0, 200, 0)
COLOUR_VEG_ONLY: Final = (220, 0, 220)
LABEL_COLOUR: Final = (255, 255, 255)

__all__ = ["COLOUR_BOTH", "COLOUR_NN_ONLY", "COLOUR_VEG_ONLY", "diff_image", "disagreement", "pick_hard_tiles",
           "render_panel", "write_jpeg"]


def disagreement(prob: np.ndarray, veg: np.ndarray, valid: np.ndarray, threshold: float) -> float:
    """Share of valid pixels where (prob >= threshold) and the a* mask differ."""
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0
    return float((((prob >= threshold) ^ veg) & valid).sum()) / n_valid


def pick_hard_tiles(scores: Mapping[str, float], n: int, exclude: Sequence[str] = ()) -> tuple[str, ...]:
    """The n tiles with the largest disagreement (ties by tile id), excluding `exclude`."""
    skip = set(exclude)
    ranked = sorted(((-v, t) for t, v in scores.items() if t not in skip))
    return tuple(t for _, t in ranked[: max(n, 0)])


def diff_image(nn: np.ndarray, veg: np.ndarray) -> np.ndarray:
    """RGB uint8 colour-coded comparison of two bool masks."""
    out = np.zeros((*nn.shape, 3), dtype=np.uint8)
    out[nn & veg] = COLOUR_BOTH
    out[nn & ~veg] = COLOUR_NN_ONLY
    out[~nn & veg] = COLOUR_VEG_ONLY
    return out


def _grey(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scale = (np.clip(values, lo, hi) - lo) / max(hi - lo, np.finfo(np.float32).eps)
    grey = (scale * U8_MAX).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2)


def _labelled(rgb: np.ndarray, label: str, panel_px: int) -> np.ndarray:
    img = cv2.resize(rgb, (panel_px, panel_px), interpolation=cv2.INTER_AREA)
    bar = np.zeros((LABEL_BAR_PX, panel_px, 3), dtype=np.uint8)
    cv2.putText(bar, label, LABEL_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_COLOUR, LABEL_THICKNESS,
                cv2.LINE_AA)
    return np.vstack([bar, img])


def render_panel(rgb: np.ndarray, na: np.ndarray, prob: np.ndarray, veg: np.ndarray, threshold: float, *,
                 panel_px: int = PANEL_PX) -> np.ndarray:
    """RGB uint8 image of the four labelled panels side by side; all inputs share the tile shape."""
    shape = rgb.shape[:2]
    for name, arr in (("na", na), ("prob", prob), ("veg", veg)):
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} != RGB shape {shape}")
    lo, hi = (float(np.percentile(na, p)) for p in NA_PERCENTILES)
    tiles = (rgb, _grey(na, lo, hi), _grey(prob, 0.0, 1.0), diff_image(prob >= threshold, veg))
    return np.hstack([_labelled(img, label, panel_px) for img, label in zip(tiles, LABELS, strict=True)])


def write_jpeg(path: Path, image_rgb: np.ndarray, quality: int = JPEG_QUALITY) -> Path:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"JPEG encoding failed for {path}")
    return atomic_write_bytes(Path(path), encoded.tobytes())
