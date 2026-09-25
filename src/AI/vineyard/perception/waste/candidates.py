"""Waste candidates (A§4.9 step 1, design 03 W1): vivid and bright HSV blobs on valid pixels.

- vivid  = S >= vivid_s_min and V >= vivid_v_min and H outside veg_hue_range (OpenCV H in 0..179);
- bright = V > bright_v_min and S < bright_s_max (A§4.9 strict inequalities);
- both masks are ANDed with `valid`, opened by `open_px`, then split in 8-connected components;
- components below min area are dropped; large ones are kept so the vehicle filter can name them.
Boxes use the corner convention (pixels i0..i1 -> [i0, i1 + 1]) plus box_pad of each size in total.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from vineyard.perception.waste.types import BoxPx, Candidate, ColourClass

if TYPE_CHECKING:
    from vineyard.config import WasteConfig

CONNECTIVITY: Final = 8
PIXEL_CENTRE: Final = 0.5
_BACKGROUND: Final = 0
_RGB_BANDS: Final = 3
_AREA_DECIMALS: Final = 6  # absorbs float noise when converting m² thresholds to px


def m2_to_px(area_m2: float, gsd_m: float) -> float:
    """Area in m² -> px at `gsd_m` (rounded to 1e-6 px so 0.03 m² is exactly 48 px)."""
    return round(area_m2 / (gsd_m * gsd_m), _AREA_DECIMALS)


@dataclass(frozen=True)
class CandidateParams:
    vivid_s_min: int
    vivid_v_min: int
    veg_hue_range: tuple[int, int]
    bright_v_min: int
    bright_s_max: int
    open_px: int
    min_area_px: int
    box_pad: float
    gsd_m: float

    @classmethod
    def from_config(cls, cfg: WasteConfig, gsd_m: float) -> CandidateParams:
        c = cfg.colour
        return cls(
            vivid_s_min=c.vivid_s_min,
            vivid_v_min=c.vivid_v_min,
            veg_hue_range=tuple(c.veg_hue_range),
            bright_v_min=c.bright_v_min,
            bright_s_max=c.bright_s_max,
            open_px=c.open_px,
            min_area_px=math.ceil(m2_to_px(cfg.min_area_m2, gsd_m)),
            box_pad=cfg.box_pad,
            gsd_m=gsd_m,
        )


def colour_masks(hsv: np.ndarray, valid: np.ndarray, p: CandidateParams) -> dict[ColourClass, np.ndarray]:
    """uint8 {0,1} masks per colour class, restricted to valid pixels and opened."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = p.veg_hue_range
    vivid = (s >= p.vivid_s_min) & (v >= p.vivid_v_min) & ~((h >= lo) & (h <= hi))
    bright = (v > p.bright_v_min) & (s < p.bright_s_max)
    masks = {ColourClass.VIVID: vivid & valid, ColourClass.BRIGHT: bright & valid}
    return {k: _open(m.astype(np.uint8), p.open_px) for k, m in masks.items()}


def _open(mask: np.ndarray, open_px: int) -> np.ndarray:
    if open_px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * open_px + 1, 2 * open_px + 1))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _padded_box(x: int, y: int, w: int, h: int, pad: float, shape: tuple[int, int]) -> BoxPx:
    px, py = pad / 2.0 * w, pad / 2.0 * h
    height, width = shape
    return BoxPx(
        max(0.0, x - px), max(0.0, y - py), min(float(width), x + w + px), min(float(height), y + h + py)
    )


def _shape_stats(labels: np.ndarray, k: int, x: int, y: int, w: int, h: int) -> tuple[float, float]:
    """(long side, short side) in px of the component's minimum-area rectangle (pixel extents)."""
    jj, ii = np.nonzero(labels[y : y + h, x : x + w] == k)
    pts = np.column_stack((ii + x, jj + y)).astype(np.float32)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    return max(rw, rh) + 1.0, min(rw, rh) + 1.0


@dataclass(frozen=True)
class _Components:
    labels: np.ndarray
    stats: np.ndarray
    centroids: np.ndarray
    mean_hsv: np.ndarray


def _components(mask: np.ndarray, hsv: np.ndarray) -> _Components:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=CONNECTIVITY)
    flat = labels.ravel()
    area = np.maximum(stats[:, cv2.CC_STAT_AREA].astype(np.float64), 1.0)
    sums = [np.bincount(flat, weights=hsv[..., c].ravel().astype(np.float64), minlength=n) for c in range(3)]
    return _Components(labels, stats, centroids, np.column_stack(sums) / area[:, None])


def _candidate(comp: _Components, k: int, cls: ColourClass, tile_id: str, p: CandidateParams) -> Candidate:
    x, y, w, h, area = (int(v) for v in comp.stats[k, :5])
    long_px, short_px = _shape_stats(comp.labels, k, x, y, w, h)
    cx, cy = (float(v) + PIXEL_CENTRE for v in comp.centroids[k])
    return Candidate(
        tile_id=tile_id,
        cand_key=f"{tile_id}@{round(cx):04d}_{round(cy):04d}",
        box=_padded_box(x, y, w, h, p.box_pad, comp.labels.shape),
        centroid_px=(cx, cy),
        area_px=area,
        area_m2=area * p.gsd_m * p.gsd_m,
        colour_class=cls,
        is_white=cls == ColourClass.BRIGHT,
        aspect=long_px / short_px,
        length_m=long_px * p.gsd_m,
        width_m=area / long_px * p.gsd_m,
        mean_hsv=tuple(float(v) for v in comp.mean_hsv[k]),
    )


def _unique_keys(cands: list[Candidate]) -> list[Candidate]:
    seen: dict[str, int] = {}
    out = []
    for c in cands:
        seen[c.cand_key] = seen.get(c.cand_key, 0) + 1
        dup = seen[c.cand_key]
        out.append(c if dup == 1 else Candidate(**{**vars(c), "cand_key": f"{c.cand_key}-{dup}"}))
    return out


def _check_inputs(rgb: np.ndarray, valid: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != _RGB_BANDS or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8 RGB, got {rgb.shape} {rgb.dtype}")
    if valid.shape != rgb.shape[:2]:
        raise ValueError(f"valid mask shape {valid.shape} != image shape {rgb.shape[:2]}")


def find_candidates(
    rgb: np.ndarray, valid: np.ndarray, tile_id: str, p: CandidateParams
) -> tuple[Candidate, ...]:
    """Every vivid / bright component >= min area, sorted by (centroid v, centroid u, colour class)."""
    _check_inputs(rgb, valid)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    found: list[Candidate] = []
    for cls, mask in colour_masks(hsv, valid.astype(bool), p).items():
        comp = _components(mask, hsv)
        big = np.flatnonzero(comp.stats[:, cv2.CC_STAT_AREA] >= p.min_area_px)
        found.extend(_candidate(comp, int(k), cls, tile_id, p) for k in big if k != _BACKGROUND)
    ordered = sorted(
        found, key=lambda c: (round(c.centroid_px[1]), round(c.centroid_px[0]), c.colour_class.value)
    )
    return tuple(_unique_keys(ordered))
