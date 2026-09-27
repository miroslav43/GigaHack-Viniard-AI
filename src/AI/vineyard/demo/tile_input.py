"""Uploaded tile check for `vineyard demo tile`: a GeoTIFF is accepted only when it is one of the 311 Sireț3 tiles.

The tile_id comes from the raster origin (the grid cell of its north-west corner), not from the file name, so a
renamed upload still maps to its tile. Every refusal is a DemoInputError whose `code` the web page translates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import rasterio
from rasterio.errors import RasterioIOError

from vineyard.errors import VineyardError
from vineyard.geo.raster import RGB_BANDS
from vineyard.geo.tiling import (
    CRS_EPSG,
    GSD_M,
    TIEPOINT_TOL_M,
    TILE_PX,
    existing_tile_ids,
    tile_of_point,
)

U8: Final = "uint8"
_SCALE_TOL: Final = 1e-9
_INSIDE_M: Final = GSD_M / 2  # probe point half a pixel inside the NW corner, so it falls in the tile's own cell


class DemoInputError(VineyardError):
    """The upload is not a Sireț3 tile; `code` is one of the codes in the web contract."""

    def __init__(self, code: str, message: str, **context: object) -> None:
        super().__init__(message, code=code, **context)
        self.code = code


@dataclass(frozen=True)
class TileInput:
    tile_id: str
    path: Path


def _open_profile(path: Path) -> tuple[int | None, int, tuple[str, ...], int, int, rasterio.Affine]:
    try:
        with rasterio.open(path) as src:
            epsg = None if src.crs is None else src.crs.to_epsg()
            return epsg, src.count, tuple(src.dtypes), src.width, src.height, src.transform
    except RasterioIOError as exc:
        raise DemoInputError("not_geotiff", f"{path.name}: not a readable GeoTIFF") from exc


def _grid_tile(transform: rasterio.Affine, name: str) -> str:
    x0, y0 = transform.c, transform.f
    ref = tile_of_point(x0 + _INSIDE_M, y0 - _INSIDE_M)
    if abs(x0 - ref.x0) >= TIEPOINT_TOL_M or abs(y0 - ref.y0) >= TIEPOINT_TOL_M:
        raise DemoInputError("off_grid", f"{name}: origin ({x0}, {y0}) is not on the Sireț3 tile grid")
    if ref.tile_id not in existing_tile_ids():
        raise DemoInputError("off_grid", f"{name}: grid cell {ref.tile_id} is not one of the 311 challenge tiles")
    return ref.tile_id


def check_tile(path: str | Path) -> TileInput:
    """TileInput for a GeoTIFF on the Sireț3 grid (EPSG:32635, RGB uint8, 2048 px, 0.025 m/px); else DemoInputError."""
    source = Path(path)
    epsg, count, dtypes, width, height, transform = _open_profile(source)
    if epsg != CRS_EPSG:
        raise DemoInputError("wrong_crs", f"{source.name}: CRS EPSG:{epsg}, expected EPSG:{CRS_EPSG}")
    if count < RGB_BANDS or any(dt != U8 for dt in dtypes[:RGB_BANDS]):
        raise DemoInputError("wrong_bands", f"{source.name}: expected 3 uint8 bands, found {count} {dtypes}")
    if (width, height) != (TILE_PX, TILE_PX):
        raise DemoInputError("wrong_size", f"{source.name}: {width} x {height} px, expected {TILE_PX} x {TILE_PX}")
    a, b, d, e = transform.a, transform.b, transform.d, transform.e
    if abs(a - GSD_M) > _SCALE_TOL or abs(e + GSD_M) > _SCALE_TOL or b != 0.0 or d != 0.0:
        raise DemoInputError("wrong_gsd", f"{source.name}: pixel size ({a}, {e}), expected ({GSD_M}, {-GSD_M})")
    return TileInput(_grid_tile(transform, source.name), source)
