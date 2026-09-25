"""CONTRACT (00_contracte §1.2-§1.4, §8): the Sireț3 tile grid and pixel <-> UTM conversions.

CVAT continuous pixel coordinates (u, v) span [0, 2048]; (0, 0) is the upper-left corner of the
first pixel (RasterPixelIsArea). X = x0 + GSD*u, Y = y0 - GSD*v.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Polygon, box

from vineyard.contracts.ids import format_tile_id, parse_tile_id, tile_grid_ids, tile_id_from_file_name
from vineyard.errors import IngestError, SchemaError

GSD_M: Final = 0.025
TILE_PX: Final = 2048
TILE_M: Final = 51.2
GRID_ORIGIN_X: Final = 628992.0
GRID_ORIGIN_Y: Final = 5221222.4
CRS_EPSG: Final = 32635
TIEPOINT_TOL_M: Final = 1e-6
GRID_DECIMALS: Final = 6  # x0/y0 rounded so they equal the tag doubles exactly
_GRID_EPS: Final = 1e-9  # in tile units: absorbs float noise on shared edges
_SCALE_TOL: Final = 1e-12


@dataclass(frozen=True)
class TileRef:
    tile_id: str
    grid_row: int
    grid_col: int
    x0: float  # UL corner = ModelTiepoint
    y0: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) in EPSG:32635."""
        return (self.x0, self.y0 - TILE_M, self.x0 + TILE_M, self.y0)


def tile_ref_from_grid(row: int, col: int) -> TileRef:
    """TileRef for grid cell (row, col); row grows southwards, col eastwards."""
    if row < 0 or col < 0:
        raise ValueError(f"grid indices must be >= 0, got row={row} col={col}")
    x0 = round(GRID_ORIGIN_X + TILE_M * col, GRID_DECIMALS)
    y0 = round(GRID_ORIGIN_Y - TILE_M * row, GRID_DECIMALS)
    return TileRef(format_tile_id(row, col), row, col, x0, y0)


def tile_ref(tile_id: str) -> TileRef:
    """TileRef for a tile id such as 'siret3_r021_c012' (grid formula, no I/O)."""
    row, col = parse_tile_id(tile_id)
    return tile_ref_from_grid(row, col)


def tile_of_point(x: float, y: float) -> TileRef:
    """Grid cell containing (x, y); a point on a shared edge goes to the east/south cell."""
    col = math.floor((x - GRID_ORIGIN_X) / TILE_M + _GRID_EPS)
    row = math.floor((GRID_ORIGIN_Y - y) / TILE_M + _GRID_EPS)
    return tile_ref_from_grid(row, col)


def tile_box(t: TileRef) -> Polygon:
    """The 51.2 m tile square (CCW) in EPSG:32635."""
    return box(*t.bounds)


def _tile_id_for_file(path: Path) -> str:
    try:
        return tile_id_from_file_name(path.name)
    except SchemaError as exc:
        raise IngestError(f"{path.name}: not a tile file name", path=str(path)) from exc


def _check_transform(tile_id: str, transform: Affine, expected: TileRef, tol_m: float) -> None:
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    if abs(a - GSD_M) > _SCALE_TOL or abs(e + GSD_M) > _SCALE_TOL or b != 0.0 or d != 0.0:
        raise IngestError(
            f"{tile_id}: transform scale/rotation != ({GSD_M}, 0, 0, {-GSD_M})", tile_id=tile_id, abde=(a, b, d, e)
        )
    dx, dy = c - expected.x0, f - expected.y0
    if abs(dx) >= tol_m or abs(dy) >= tol_m:
        raise IngestError(
            f"{tile_id}: transform origin off grid by ({dx:.6g}, {dy:.6g}) m",
            tile_id=tile_id, origin=(c, f), grid=(expected.x0, expected.y0),
        )


def tile_ref_from_tags(path: str | Path, *, tol_m: float = TIEPOINT_TOL_M) -> TileRef:
    """Read CRS and transform with rasterio and check them against the grid cell named by the file."""
    source = Path(path)
    tile_id = _tile_id_for_file(source)
    expected = tile_ref(tile_id)
    with rasterio.open(source) as src:
        crs, transform = src.crs, src.transform
    epsg = None if crs is None else crs.to_epsg()
    if epsg != CRS_EPSG:
        raise IngestError(f"{tile_id}: CRS EPSG:{epsg} != EPSG:{CRS_EPSG}", tile_id=tile_id, path=str(source))
    _check_transform(tile_id, transform, expected, tol_m)
    return expected


def _as_points(a: np.ndarray, what: str) -> np.ndarray:
    pts = np.asarray(a, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{what}: expected an (N, 2) array, got shape {pts.shape}")
    return pts


def px_to_utm(t: TileRef, uv: np.ndarray) -> np.ndarray:
    """(N,2) CVAT continuous (u, v) -> (N,2) UTM (x, y)."""
    p = _as_points(uv, "uv")
    return np.column_stack((t.x0 + GSD_M * p[:, 0], t.y0 - GSD_M * p[:, 1]))


def utm_to_px(t: TileRef, xy: np.ndarray) -> np.ndarray:
    """(N,2) UTM (x, y) -> (N,2) CVAT continuous (u, v); no clipping."""
    p = _as_points(xy, "xy")
    return np.column_stack(((p[:, 0] - t.x0) / GSD_M, (t.y0 - p[:, 1]) / GSD_M))


def index_to_px(ij: np.ndarray) -> np.ndarray:
    """(N,2) index/contour coords (i=col, j=row) -> pixel-centre CVAT coords (+0.5)."""
    return _as_points(ij, "ij") + 0.5


def mosaic_px(t: TileRef, uv: np.ndarray) -> np.ndarray:
    """(N,2) tile (u, v) -> mosaic (U, V) = (2048*col + u, 2048*row + v)."""
    p = _as_points(uv, "uv")
    return np.column_stack((TILE_PX * t.grid_col + p[:, 0], TILE_PX * t.grid_row + p[:, 1]))


@lru_cache(maxsize=1)
def existing_tile_ids() -> frozenset[str]:
    """The 311 tile ids of the challenge (package data contracts/tile_grid.txt)."""
    return frozenset(tile_grid_ids())


@lru_cache(maxsize=1)
def _existing_cells() -> frozenset[tuple[int, int]]:
    return frozenset(parse_tile_id(tile_id) for tile_id in existing_tile_ids())


def _cell_range(lo: float, hi: float) -> range:
    return range(math.floor(lo + _GRID_EPS), math.ceil(hi - _GRID_EPS))


def tiles_for_bounds(minx: float, miny: float, maxx: float, maxy: float) -> list[str]:
    """Existing tiles whose interior overlaps the box (touching an edge does not count); sorted."""
    if minx > maxx or miny > maxy:
        raise ValueError(f"invalid bounds ({minx}, {miny}, {maxx}, {maxy})")
    cells = _existing_cells()
    max_row = max(r for r, _ in cells)
    max_col = max(c for _, c in cells)
    rows = _cell_range((GRID_ORIGIN_Y - maxy) / TILE_M, (GRID_ORIGIN_Y - miny) / TILE_M)
    cols = _cell_range((minx - GRID_ORIGIN_X) / TILE_M, (maxx - GRID_ORIGIN_X) / TILE_M)
    rows = range(max(rows.start, 0), min(rows.stop, max_row + 1))
    cols = range(max(cols.start, 0), min(cols.stop, max_col + 1))
    return [format_tile_id(r, c) for r in rows for c in cols if (r, c) in cells]
