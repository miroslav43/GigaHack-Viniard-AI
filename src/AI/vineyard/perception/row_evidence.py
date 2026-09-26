"""Vegetation evidence along a line from the tile_prep veg masks (used by row_regularize via `blocks`).

`VegEvidence(line, offset_m)` is the veg fraction of the valid pixels in a band of half-width band_m around
`line` shifted sideways by offset_m, sampled every step_m; None when fewer than half of the samples fall on
valid pixels of a tile whose masks exist. Masks are loaded lazily, once per tile (memo only).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString

from vineyard.contracts.ids import format_tile_id
from vineyard.errors import StageError
from vineyard.geo.tiling import GRID_ORIGIN_X, GRID_ORIGIN_Y, GSD_M, TILE_M, TILE_PX

MIN_KNOWN_FRAC: Final = 0.5
_EPS: Final = 1e-9

MaskLoader = Callable[[Path, str], np.ndarray]


class VegEvidence:
    """Callable evidence source over the per-tile veg / valid masks of `cache_dir`."""

    def __init__(self, cache_dir: Path, band_m: float, step_m: float, load_veg: MaskLoader,
                 load_valid: MaskLoader) -> None:
        self._cache_dir = Path(cache_dir)
        self._band_m = band_m
        self._step_m = step_m
        self._load = (load_veg, load_valid)
        self._masks: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def _tile(self, tile_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        if tile_id not in self._masks:
            try:
                self._masks[tile_id] = (self._load[0](self._cache_dir, tile_id), self._load[1](self._cache_dir, tile_id))
            except (StageError, OSError, ValueError):  # tile outside the run / not prepared: unknown evidence
                self._masks[tile_id] = None
        return self._masks[tile_id]

    def samples(self, line: LineString, offset_m: float) -> np.ndarray:
        """(N, 2) UTM sample points of the band around `line` shifted by offset_m."""
        ln = line if abs(offset_m) < _EPS else line.offset_curve(offset_m)
        if ln.is_empty or ln.length < _EPS or ln.geom_type != "LineString":
            return np.zeros((0, 2))
        d = np.linspace(0.0, ln.length, max(2, math.ceil(ln.length / self._step_m) + 1))
        pts = shapely.get_coordinates(shapely.line_interpolate_point(ln, d))
        c = np.asarray(ln.coords, dtype=np.float64)[:, :2]
        chord = c[-1] - c[0]
        nrm = np.array([-chord[1], chord[0]]) / max(float(np.hypot(*chord)), _EPS)
        offs = np.arange(-self._band_m, self._band_m + _EPS, self._step_m)
        return (pts[:, None, :] + offs[None, :, None] * nrm[None, None, :]).reshape(-1, 2)

    def __call__(self, line: LineString, offset_m: float) -> float | None:
        pts = self.samples(line, offset_m)
        if not len(pts):
            return None
        col = np.floor((pts[:, 0] - GRID_ORIGIN_X) / TILE_M).astype(int)
        row = np.floor((GRID_ORIGIN_Y - pts[:, 1]) / TILE_M).astype(int)
        hits, known = 0, 0
        for r, c in sorted(set(zip(row.tolist(), col.tolist(), strict=True))):
            masks = self._tile(format_tile_id(r, c)) if r >= 0 and c >= 0 else None
            if masks is None:
                continue
            sel = (row == r) & (col == c)
            px = np.clip(((pts[sel, 0] - GRID_ORIGIN_X - c * TILE_M) / GSD_M).astype(int), 0, TILE_PX - 1)
            py = np.clip(((GRID_ORIGIN_Y - r * TILE_M - pts[sel, 1]) / GSD_M).astype(int), 0, TILE_PX - 1)
            veg, valid = masks
            ok = valid[py, px]
            known += int(ok.sum())
            hits += int((veg[py, px] & ok).sum())
        return hits / known if known >= MIN_KNOWN_FRAC * len(pts) else None
