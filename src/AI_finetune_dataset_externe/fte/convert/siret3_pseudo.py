"""Sireț3 pseudo-labels from the base run (complete-v4): canopy polygons + row axes -> c0 / c1 / c2.

c0 (canopy): 1 inside the base canopy polygons, 0 in the row gaps (< 0.30 m from an axis) and away
from the rows (> 0.50 m), 255 on a ±2 px band around polygon edges and between 0.30 and 0.50 m from
the axes. On tiles known to contain unmapped vines ("partial"), the far field is 255 instead of 0.
c1 (contact): 255 everywhere (contact is learned from ICAERUS only).
c2 (ground vegetation): ExG labels with the vine region = dilated canopy ∪ ±0.50 m corridor.
Nodata pixels are 255 in every channel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from fte.convert.targets import (
    IGNORE,
    GroundVegSpec,
    dilate,
    erode,
    ground_veg_labels,
    stack_labels,
)

GSD_M = 0.025


class TileKind(StrEnum):
    VINEYARD = "vineyard"
    PARTIAL = "partial"  # mapped vines are trusted, the far field is not
    NEGATIVE = "negative"


@dataclass(frozen=True)
class PseudoSpec:
    edge_band_px: int = 2
    gap_max_m: float = 0.30
    corridor_half_m: float = 0.50
    vine_dilate_px: int = 4
    ground: GroundVegSpec = GroundVegSpec()


def axis_distance_px(lines_px: Sequence[np.ndarray], shape_hw: tuple[int, int]) -> np.ndarray:
    """float32 distance (px) of every pixel to the nearest polyline (continuous px coords); inf if none."""
    if not lines_px:
        return np.full(shape_hw, np.inf, dtype=np.float32)
    canvas = np.full(shape_hw, 255, dtype=np.uint8)
    rings = [np.round(np.asarray(ln, dtype=np.float64) - 0.5).astype(np.int32).reshape(-1, 1, 2) for ln in lines_px]
    cv2.polylines(canvas, rings, isClosed=False, color=0, thickness=1)
    return cv2.distanceTransform(canvas, cv2.DIST_L2, 5)


def lines_to_px(geoms: Sequence[BaseGeometry], utm_to_px) -> list[np.ndarray]:  # noqa: ANN001
    """UTM (Multi)LineStrings -> list of (N, 2) pixel polylines via ``utm_to_px(xy) -> uv``."""
    out: list[np.ndarray] = []
    for g in geoms:
        parts = g.geoms if isinstance(g, MultiLineString) else [g]
        for part in parts:
            if isinstance(part, LineString) and not part.is_empty and len(part.coords) >= 2:
                out.append(utm_to_px(np.asarray(part.coords)[:, :2]))
    return out


def canopy_c0(canopy: np.ndarray, axis_dist_m: np.ndarray, kind: TileKind, spec: PseudoSpec) -> np.ndarray:
    can = canopy.astype(bool)
    near_gap = axis_dist_m < spec.gap_max_m
    band = (axis_dist_m >= spec.gap_max_m) & (axis_dist_m <= spec.corridor_half_m)
    far = axis_dist_m > spec.corridor_half_m
    c0 = np.full(can.shape, IGNORE, dtype=np.uint8)
    c0[~can & near_gap] = 0
    c0[~can & far] = IGNORE if kind is TileKind.PARTIAL else 0
    c0[~can & band] = IGNORE
    c0[can] = 1
    edge = dilate(can, spec.edge_band_px) & ~erode(can, spec.edge_band_px)
    c0[edge] = IGNORE
    return c0


def pseudo_labels(rgb: np.ndarray, valid: np.ndarray, canopy: np.ndarray, axis_dist_px: np.ndarray,
                  kind: TileKind, spec: PseudoSpec | None = None) -> np.ndarray:
    """HxWx3 uint8 labels of one tile (see module doc)."""
    spec = spec or PseudoSpec()
    shape = rgb.shape[:2]
    for name, arr in (("valid", valid), ("canopy", canopy), ("axis_dist_px", axis_dist_px)):
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} != rgb shape {shape}")
    nodata = ~valid.astype(bool) | (rgb.max(axis=2) == 0)
    c1 = np.full(shape, IGNORE, dtype=np.uint8)
    if kind is TileKind.NEGATIVE:
        c0 = np.zeros(shape, dtype=np.uint8)
        c2 = ground_veg_labels(rgb, np.zeros(shape, bool), spec.ground)
        return stack_labels(c0, c1, c2, ignore=nodata)
    dist_m = axis_dist_px * GSD_M
    c0 = canopy_c0(canopy, dist_m, kind, spec)
    if kind is TileKind.PARTIAL:
        vine_region = np.ones(shape, bool)
    else:
        vine_region = dilate(canopy.astype(bool), spec.vine_dilate_px) | (dist_m <= spec.corridor_half_m)
    c2 = ground_veg_labels(rgb, vine_region, spec.ground)
    return stack_labels(c0, c1, c2, ignore=nodata)


def sampling_weights(lbl: np.ndarray, axis_dist_px: np.ndarray, kind: TileKind, cell_px: int = 64,
                     corridor_half_m: float = 0.5) -> np.ndarray:
    """Coarse (H/cell, W/cell) float32 map used to draw patch centres (corridor or green fraction)."""
    if kind is TileKind.NEGATIVE:
        src = (lbl[..., 2] == 1).astype(np.float32) + 0.2 * (lbl[..., 0] != IGNORE)
    else:
        src = ((axis_dist_px * GSD_M) <= corridor_half_m).astype(np.float32)
    h, w = src.shape
    return cv2.resize(src, (w // cell_px, h // cell_px), interpolation=cv2.INTER_AREA).astype(np.float32)
