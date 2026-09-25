"""ingest.tiles: stream tiles out of ZIPs with sha256, verify names / count / tags, build tile_index."""

from __future__ import annotations

import hashlib
import os
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
import shapely

from vineyard.contracts.schemas import validate_layer
from vineyard.errors import IngestError
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.ingest.tiles import ingest_tiles, scan_zips, verify_tile_raster

TILE_PX = 2048
TOL_M = 1e-6
MakeGeotiff = Callable[..., Path]
MakeZip = Callable[..., Path]
IDS = ("siret3_r006_c004", "siret3_r018_c010", "siret3_r021_c012")


@pytest.fixture
def tifs(tmp_path: Path, make_geotiff: MakeGeotiff) -> dict[str, Path]:
    src = tmp_path / "src"
    return {t: make_geotiff(src / f"{t}.tif", size=TILE_PX) for t in IDS}


@pytest.fixture
def zips(tmp_path: Path, tifs: dict[str, Path], make_zip: MakeZip) -> list[Path]:
    a = make_zip(tmp_path / "zips" / "part1of2.zip", {f"{IDS[0]}.tif": tifs[IDS[0]], f"{IDS[1]}.tif": tifs[IDS[1]]})
    b = make_zip(tmp_path / "zips" / "part2of2.zip", {f"{IDS[2]}.tif": tifs[IDS[2]]})
    return [a, b]


def _ingest(zips: list[Path], tiles_dir: Path, expected: int = 3) -> tuple:
    return ingest_tiles(zips, tiles_dir, expected_tiles=expected, tile_px=TILE_PX, tol_m=TOL_M)


def test_ingest_builds_a_valid_tile_index(tmp_path: Path, zips: list[Path], tifs: dict[str, Path]) -> None:
    index, summary = _ingest(zips, tmp_path / "tiles")
    validate_layer(index, "tile_index")
    assert list(index["tile_id"]) == sorted(IDS)
    assert (summary.n_tiles, summary.n_extracted, summary.n_skipped) == (3, 3, 0)
    row = index.set_index("tile_id").loc["siret3_r021_c012"]
    assert row["file_name"] == "siret3_r021_c012.tif"
    assert (row["grid_row"], row["grid_col"]) == (21, 12)
    assert row["src_zip"] == "part2of2.zip"
    data = tifs["siret3_r021_c012"].read_bytes()
    assert row["sha256"] == hashlib.sha256(data).hexdigest() and row["file_size"] == len(data)
    assert Path(row["path"]) == (tmp_path / "tiles" / "siret3_r021_c012.tif").resolve()
    t = tile_ref("siret3_r021_c012")
    assert (row["x0"], row["y0"], row["x1"], row["y1"]) == (t.x0, t.y0, t.x0 + 51.2, t.y0 - 51.2)
    assert (row["width_px"], row["height_px"], row["gsd_m"]) == (TILE_PX, TILE_PX, 0.025)
    assert index["nodata_frac"].isna().all() and index["valid_area_m2"].isna().all()
    geom = index.set_index("tile_id").geometry.loc["siret3_r021_c012"]
    assert shapely.equals_exact(geom.normalize(), tile_box(t).normalize(), 1e-9)


def test_second_run_is_fully_cached(tmp_path: Path, zips: list[Path]) -> None:
    tiles = tmp_path / "tiles"
    first, _ = _ingest(zips, tiles)
    mtimes = {p.name: os.stat(p).st_mtime_ns for p in tiles.glob("*.tif")}
    second, summary = _ingest(zips, tiles)
    assert (summary.n_extracted, summary.n_skipped) == (0, 3)
    assert {p.name: os.stat(p).st_mtime_ns for p in tiles.glob("*.tif")} == mtimes
    assert list(second["sha256"]) == list(first["sha256"])


def test_corrupted_copy_is_re_extracted(tmp_path: Path, zips: list[Path]) -> None:
    tiles = tmp_path / "tiles"
    _ingest(zips, tiles)
    (tiles / f"{IDS[0]}.tif").write_bytes(b"garbage")
    _, summary = _ingest(zips, tiles)
    assert summary.n_extracted == 1


def test_no_leftover_tmp_files(tmp_path: Path, zips: list[Path]) -> None:
    _ingest(zips, tmp_path / "tiles")
    assert not [p for p in (tmp_path / "tiles").iterdir() if ".tmp-" in p.name]


def test_wrong_count_is_an_error(tmp_path: Path, zips: list[Path]) -> None:
    with pytest.raises(IngestError, match="expected 4"):
        _ingest(zips, tmp_path / "tiles", expected=4)


def test_duplicate_across_zips_is_an_error(tmp_path: Path, zips: list[Path], tifs: dict[str, Path],
                                           make_zip: MakeZip) -> None:
    dup = make_zip(tmp_path / "zips" / "dup.zip", {f"{IDS[0]}.tif": tifs[IDS[0]]})
    with pytest.raises(IngestError, match="duplicate") as info:
        scan_zips([*zips, dup])
    assert IDS[0] in str(info.value)


@pytest.mark.parametrize("name", ["foo.tif", "images/siret3_r006_c004.tif", "siret3_r000_c000.tif", "notes.txt"])
def test_bad_member_names_are_errors(tmp_path: Path, tifs: dict[str, Path], make_zip: MakeZip, name: str) -> None:
    bad = make_zip(tmp_path / "zips" / "bad.zip", {name: tifs[IDS[0]]})
    with pytest.raises(IngestError) as info:
        scan_zips([bad])
    assert name in str(info.value)


def test_directory_entries_are_ignored(tmp_path: Path, tifs: dict[str, Path]) -> None:
    path = tmp_path / "dirs.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("somedir/", b"")
        zf.write(tifs[IDS[0]], f"{IDS[0]}.tif")
    assert [m.tile_id for m in scan_zips([path])] == [IDS[0]]


def test_no_zips_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no tile ZIPs"):
        _ingest([], tmp_path / "tiles")


def test_tiepoint_off_grid_names_the_tile(tmp_path: Path, make_geotiff: MakeGeotiff, make_zip: MakeZip) -> None:
    tif = make_geotiff(tmp_path / "src" / f"{IDS[0]}.tif", size=TILE_PX, dx=0.1)
    z = make_zip(tmp_path / "zips" / "p.zip", {f"{IDS[0]}.tif": tif})
    with pytest.raises(IngestError) as info:
        _ingest([z], tmp_path / "tiles", expected=1)
    assert IDS[0] in str(info.value)


def test_wrong_size_is_an_error(tmp_path: Path, make_geotiff: MakeGeotiff) -> None:
    tif = make_geotiff(tmp_path / f"{IDS[0]}.tif", size=64)
    with pytest.raises(IngestError, match="2048"):
        verify_tile_raster(tif, tile_px=TILE_PX, tol_m=TOL_M)


def test_wrong_band_count_is_an_error(tmp_path: Path, make_geotiff: MakeGeotiff) -> None:
    tif = make_geotiff(tmp_path / f"{IDS[0]}.tif", size=TILE_PX, bands=1, compress="DEFLATE")
    with pytest.raises(IngestError, match="bands"):
        verify_tile_raster(tif, tile_px=TILE_PX, tol_m=TOL_M)


@pytest.mark.examples
def test_example_tile_passes_verification(example_tif) -> None:
    ref = verify_tile_raster(example_tif("siret3_r006_c004"), tile_px=TILE_PX, tol_m=TOL_M)
    assert ref == tile_ref("siret3_r006_c004")
