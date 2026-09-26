"""Lateral refinement of row pieces on the canopy mask (canopy stage only).

Row axes are fitted on the Lab a* mask, which includes the cast shadow on the anti-sun side of the plants,
so they sit a few cm towards the shadow (r021: 4.3 cm on average, p90 11.7 cm) and the ±0.30 m corridor
cuts the sunlit side of the canopy. Before the corridor labels are rasterized, each local piece is moved
along its normal onto the mean offset of the canopy-mask pixels inside a ±band_m strip (a few iterations,
capped at max_shift_m, skipped without min_px supporting pixels). Direction and extent stay unchanged,
and the published rows layer is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString

from vineyard.geo.tiling import GSD_M, TileRef, utm_to_px
from vineyard.perception.types import BoolMask

_PIXEL_CENTRE: float = 0.5


@dataclass(frozen=True)
class AxisRefineOptions:
    band_m: float
    iterations: int
    max_shift_m: float
    min_px: int

    def __post_init__(self) -> None:
        if self.band_m <= 0 or self.iterations < 1 or self.max_shift_m < 0 or self.min_px < 1:
            raise ValueError(f"invalid axis refinement options: {self}")


def _pixel_points(mask: BoolMask) -> np.ndarray:
    v, u = np.nonzero(mask)
    return np.column_stack((u + _PIXEL_CENTRE, v + _PIXEL_CENTRE)).astype(np.float64)


def _piece_shift_px(axis_px: np.ndarray, pts: np.ndarray, opts: AxisRefineOptions) -> tuple[float, np.ndarray]:
    """(signed shift in px along the normal n, n) for one piece; 0 when the support is too small."""
    a, b = axis_px[0], axis_px[-1]
    length = float(np.hypot(*(b - a)))
    n = np.zeros(2)
    if length <= 0 or len(pts) == 0:
        return 0.0, n
    t = (b - a) / length
    n = np.array([-t[1], t[0]])
    rel = pts - a
    along, off = rel @ t, rel @ n
    band_px, cap_px = opts.band_m / GSD_M, opts.max_shift_m / GSD_M
    off = off[(along > 0) & (along < length) & (np.abs(off) < band_px * (opts.iterations + 1))]
    total = 0.0
    for _ in range(opts.iterations):
        sel = np.abs(off - total) < band_px
        if int(sel.sum()) < opts.min_px:
            break
        total += float((off[sel] - total).mean())
    return float(np.clip(total, -cap_px, cap_px)), n


def refine_offsets_m(pieces: gpd.GeoDataFrame, mask: BoolMask, tile: TileRef, opts: AxisRefineOptions) -> np.ndarray:
    """(n_pieces,) lateral shift in metres of each piece (sign: along the px normal (-t_v, t_u))."""
    pts = _pixel_points(mask)
    out = np.zeros(len(pieces), dtype=np.float64)
    for k, geom in enumerate(pieces.geometry):
        shift, _ = _piece_shift_px(utm_to_px(tile, np.asarray(geom.coords)), pts, opts)
        out[k] = shift * GSD_M
    return out


def refine_pieces(pieces: gpd.GeoDataFrame, mask: BoolMask, tile: TileRef,
                  opts: AxisRefineOptions | None) -> gpd.GeoDataFrame:
    """New frame whose piece axes are moved onto the canopy mask (the input is not modified)."""
    if opts is None or pieces.empty:
        return pieces
    pts = _pixel_points(mask)
    geoms: list[LineString] = []
    for geom in pieces.geometry:
        shift, n = _piece_shift_px(utm_to_px(tile, np.asarray(geom.coords)), pts, opts)
        # px (u, v) -> UTM (x, y): x grows with u, y shrinks with v
        dx, dy = n[0] * shift * GSD_M, -n[1] * shift * GSD_M
        geoms.append(shapely.transform(geom, lambda xy, d=(dx, dy): xy + np.asarray(d)))
    return pieces.set_geometry(gpd.GeoSeries(geoms, index=pieces.index, crs=pieces.crs))
