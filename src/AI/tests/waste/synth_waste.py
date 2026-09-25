"""Synthetic tiles and candidate factories for the waste tests (GSD 0.025 m: 1 px = 0.000625 m²)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vineyard.perception.waste.types import BoxPx, Candidate, ColourClass

SOIL_RGB = (120, 100, 80)
TILE_ID = "siret3_r021_c012"
PX_M2 = 0.025 * 0.025

MakeCandidate = Callable[..., Candidate]


def soil(size: int = 256) -> np.ndarray:
    """Uniform brown soil image (no candidates)."""
    img = np.empty((size, size, 3), np.uint8)
    img[...] = SOIL_RGB
    return img


def paint(img: np.ndarray, rgb: tuple[int, int, int], rows: slice, cols: slice) -> np.ndarray:
    out = img.copy()
    out[rows, cols] = rgb
    return out


def make_candidate(
    *,
    tile_id: str = TILE_ID,
    centroid: tuple[float, float] = (500.0, 500.0),
    area_px: int = 200,
    is_white: bool = False,
    aspect: float = 1.0,
    length_px: float = 16.0,
    half: float = 8.0,
    key: str | None = None,
) -> Candidate:
    cx, cy = centroid
    width_px = area_px / length_px
    return Candidate(
        tile_id=tile_id,
        cand_key=key or f"{tile_id}@{round(cx):04d}_{round(cy):04d}",
        box=BoxPx(cx - half, cy - half, cx + half, cy + half),
        centroid_px=(cx, cy),
        area_px=area_px,
        area_m2=area_px * PX_M2,
        colour_class=ColourClass.BRIGHT if is_white else ColourClass.VIVID,
        is_white=is_white,
        aspect=aspect,
        length_m=length_px * 0.025,
        width_m=width_px * 0.025,
        mean_hsv=(0.0, 20.0 if is_white else 200.0, 230.0 if is_white else 180.0),
    )
