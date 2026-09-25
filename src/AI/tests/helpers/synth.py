"""Synthetic rasters for tests: striped vine rows, arcs, orchard blobs, RGB rendering, nodata.

Pixel conventions follow contract §1.4: pixel (row j, col i) has its centre at (u, v) = (i+0.5, j+0.5);
angles are in pixel space, measured from +u towards +v (so UTM angle = -angle_px mod 180).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from shapely.geometry import LineString, box

# neg_a: vegetation +18, soil -9; the midpoint sits near veg.threshold (4) so blur keeps stripe widths.
VEG_RGB: tuple[int, int, int] = (95, 120, 72)
SOIL_RGB: tuple[int, int, int] = (150, 118, 92)
Corner = str  # "tl" | "tr" | "bl" | "br"


def _pixel_centres(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    return u + 0.5, v + 0.5


def _normal(angle_deg: float) -> tuple[float, float]:
    t = np.radians(angle_deg)
    return float(-np.sin(t)), float(np.cos(t))


def striped_mask(
    shape: tuple[int, int], *, angle_deg: float, spacing_px: float, width_px: float, offset_px: float = 0.0
) -> np.ndarray:
    """Bool mask of parallel stripes (rows): axis direction `angle_deg`, centres at offset + k*spacing."""
    u, v = _pixel_centres(shape)
    nx, ny = _normal(angle_deg)
    off = u * nx + v * ny - offset_px
    phase = np.mod(off, spacing_px)
    return (phase < width_px / 2) | (phase > spacing_px - width_px / 2)


def row_axes(
    shape: tuple[int, int], *, angle_deg: float, spacing_px: float, offset_px: float = 0.0
) -> list[np.ndarray]:
    """Stripe centre lines of striped_mask clipped to the image, as (2,2) px endpoint arrays."""
    h, w = shape
    frame = box(0.0, 0.0, float(w), float(h))
    t = np.radians(angle_deg)
    d = np.array([np.cos(t), np.sin(t)])
    n = np.array(_normal(angle_deg))
    reach = float(np.hypot(h, w))
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float) @ n
    k0 = int(np.floor((corners.min() - offset_px) / spacing_px))
    k1 = int(np.ceil((corners.max() - offset_px) / spacing_px))
    axes: list[np.ndarray] = []
    for k in range(k0, k1 + 1):
        c = n * (offset_px + k * spacing_px)
        seg = LineString([c - d * reach, c + d * reach]).intersection(frame)
        if isinstance(seg, LineString) and seg.length > 0:
            axes.append(np.asarray(seg.coords, dtype=np.float64))
    return axes


def arc_mask(
    shape: tuple[int, int], *, centre: tuple[float, float], r0_px: float, spacing_px: float, width_px: float,
    n_rows: int,
) -> np.ndarray:
    """Concentric curved rows: rings of radius r0 + k*spacing (k < n_rows) and the given width."""
    u, v = _pixel_centres(shape)
    r = np.hypot(u - centre[0], v - centre[1])
    k = np.round((r - r0_px) / spacing_px)
    near = np.abs(r - (r0_px + k * spacing_px)) < width_px / 2
    return near & (k >= 0) & (k < n_rows)


def blobs_mask(shape: tuple[int, int], centres: Sequence[tuple[float, float]], radius_px: float) -> np.ndarray:
    """Round blobs (orchard crowns / trees) at px centres."""
    u, v = _pixel_centres(shape)
    out = np.zeros(shape, dtype=bool)
    for cu, cv in centres:
        out |= np.hypot(u - cu, v - cv) <= radius_px
    return out


def rgb_from_mask(
    mask: np.ndarray,
    *,
    veg_rgb: tuple[int, int, int] = VEG_RGB,
    soil_rgb: tuple[int, int, int] = SOIL_RGB,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Render a bool mask as uint8 RGB (vegetation green on brown soil, optional Gaussian noise)."""
    img = np.where(mask[..., None], np.array(veg_rgb, np.float64), np.array(soil_rgb, np.float64))
    if noise_sigma > 0:
        img = img + np.random.default_rng(seed).normal(0.0, noise_sigma, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def with_nodata_corner(rgb: np.ndarray, size_px: int, corner: Corner = "tl", *, ring_px: int = 0) -> np.ndarray:
    """Copy of `rgb` with a black square in a corner (plus an optional 1-10 DN JPEG-like ring)."""
    out = rgb.copy()
    h, w = out.shape[:2]
    rows = slice(0, size_px + ring_px) if corner[0] == "t" else slice(h - size_px - ring_px, h)
    cols = slice(0, size_px + ring_px) if corner[1] == "l" else slice(w - size_px - ring_px, w)
    out[rows, cols] = 5 if ring_px else 0
    inner_r = slice(0, size_px) if corner[0] == "t" else slice(h - size_px, h)
    inner_c = slice(0, size_px) if corner[1] == "l" else slice(w - size_px, w)
    out[inner_r, inner_c] = 0
    return out
