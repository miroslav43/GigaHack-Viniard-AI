"""vineyard.demo.tile_input: an uploaded GeoTIFF is accepted only on the Sireț3 grid; tile_id comes from its origin."""

from __future__ import annotations

from pathlib import Path

import pytest

from vineyard.demo.tile_input import DemoInputError, check_tile
from vineyard.geo.tiling import TILE_PX

TILE = "siret3_r021_c012"


def _code(path: Path) -> str:
    with pytest.raises(DemoInputError) as info:
        check_tile(path)
    return info.value.code


def test_grid_tile_gives_its_id_whatever_the_file_name(tmp_path: Path, make_geotiff) -> None:
    tif = make_geotiff(tmp_path / "upload (1).tif", tile_id=TILE, size=TILE_PX)
    got = check_tile(tif)
    assert got.tile_id == TILE
    assert got.path == tif


def test_not_a_geotiff(tmp_path: Path) -> None:
    junk = tmp_path / "x.tif"
    junk.write_bytes(b"not a tiff")
    assert _code(junk) == "not_geotiff"


def test_missing_file(tmp_path: Path) -> None:
    assert _code(tmp_path / "missing.tif") == "not_geotiff"


def test_wrong_crs(tmp_path: Path, make_geotiff) -> None:
    assert _code(make_geotiff(tmp_path / "a.tif", tile_id=TILE, size=TILE_PX, epsg=4326)) == "wrong_crs"


def test_wrong_bands(tmp_path: Path, make_geotiff) -> None:
    tif = make_geotiff(tmp_path / "a.tif", tile_id=TILE, size=TILE_PX, bands=1, compress="DEFLATE")
    assert _code(tif) == "wrong_bands"


def test_wrong_size(tmp_path: Path, make_geotiff) -> None:
    assert _code(make_geotiff(tmp_path / "a.tif", tile_id=TILE, size=64)) == "wrong_size"


def test_wrong_gsd(tmp_path: Path, make_geotiff) -> None:
    assert _code(make_geotiff(tmp_path / "a.tif", tile_id=TILE, size=TILE_PX, gsd=0.05)) == "wrong_gsd"


def test_origin_between_grid_cells(tmp_path: Path, make_geotiff) -> None:
    assert _code(make_geotiff(tmp_path / "a.tif", tile_id=TILE, size=TILE_PX, dx=3.0)) == "off_grid"


def test_grid_cell_outside_the_study_area(tmp_path: Path, make_geotiff) -> None:
    tif = make_geotiff(tmp_path / "a.tif", tile_id="siret3_r000_c000", size=TILE_PX)
    assert _code(tif) == "off_grid"
