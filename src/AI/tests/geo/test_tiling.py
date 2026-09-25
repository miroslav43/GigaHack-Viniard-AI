"""Contract §8 grid and pixel <-> UTM functions (oracles measured on the tile tags)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from vineyard.errors import IngestError
from vineyard.geo import tiling
from vineyard.geo.tiling import (
    GRID_ORIGIN_X,
    GRID_ORIGIN_Y,
    GSD_M,
    TILE_M,
    TILE_PX,
    TileRef,
    existing_tile_ids,
    index_to_px,
    mosaic_px,
    px_to_utm,
    tile_box,
    tile_of_point,
    tile_ref,
    tile_ref_from_grid,
    tile_ref_from_tags,
    tiles_for_bounds,
    utm_to_px,
)

START_XY = (629504.70, 5220250.75)
EXAMPLES = ("siret3_r006_c004", "siret3_r021_c012")


def test_constants() -> None:
    assert (GSD_M, TILE_PX, TILE_M) == (0.025, 2048, 51.2)
    assert (GRID_ORIGIN_X, GRID_ORIGIN_Y) == (628992.0, 5221222.4)
    assert abs(TILE_PX * GSD_M - TILE_M) < 1e-12


def test_tile_ref_from_grid_r006_c004() -> None:
    t = tile_ref_from_grid(6, 4)
    assert t == TileRef("siret3_r006_c004", 6, 4, 629196.8, 5220915.2)
    assert t.bounds == (629196.8, 5220864.0, 629248.0, 5220915.2)


def test_tile_ref_from_grid_r021_c012() -> None:
    t = tile_ref_from_grid(21, 12)
    assert t.x0 == round(628992.0 + 51.2 * 12, 6) == 629606.4
    assert t.y0 == round(5221222.4 - 51.2 * 21, 6) == 5220147.2


def test_tile_ref_from_grid_rejects_negative() -> None:
    with pytest.raises(ValueError):
        tile_ref_from_grid(-1, 3)


def test_tile_ref_parses_id() -> None:
    assert tile_ref("siret3_r021_c012") == tile_ref_from_grid(21, 12)


def test_start_maps_to_r018_c010() -> None:
    t = tile_ref("siret3_r018_c010")
    uv = utm_to_px(t, np.array([START_XY]))
    np.testing.assert_allclose(uv, [[28.0, 2002.0]], atol=1e-6)
    np.testing.assert_allclose(px_to_utm(t, np.array([[28.0, 2002.0]])), [START_XY], atol=1e-6)
    assert tile_of_point(*START_XY) == t


def test_px_utm_roundtrip_random() -> None:
    rng = np.random.default_rng(0)
    t = tile_ref("siret3_r021_c012")
    xy = np.column_stack(
        (rng.uniform(t.bounds[0] - 100, t.bounds[2] + 100, 1000), rng.uniform(t.bounds[1] - 100, t.bounds[3] + 100, 1000))
    )
    back = px_to_utm(t, utm_to_px(t, xy))
    assert np.max(np.abs(back - xy)) <= 1e-9


def test_px_to_utm_corners_and_input_untouched() -> None:
    t = tile_ref("siret3_r006_c004")
    uv = np.array([[0.0, 0.0], [2048.0, 2048.0]])
    before = uv.copy()
    xy = px_to_utm(t, uv)
    np.testing.assert_allclose(xy, [[629196.8, 5220915.2], [629248.0, 5220864.0]], atol=1e-9)
    np.testing.assert_array_equal(uv, before)


def test_point_shape_validation() -> None:
    t = tile_ref("siret3_r006_c004")
    with pytest.raises(ValueError):
        px_to_utm(t, np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        utm_to_px(t, np.zeros((3, 3)))
    assert px_to_utm(t, np.zeros((0, 2))).shape == (0, 2)


def test_index_to_px_adds_half() -> None:
    ij = np.array([[0, 0], [10, 3]], dtype=np.int32)
    np.testing.assert_array_equal(index_to_px(ij), [[0.5, 0.5], [10.5, 3.5]])
    assert index_to_px(ij).dtype == np.float64


def test_mosaic_px() -> None:
    t = tile_ref("siret3_r006_c004")
    np.testing.assert_array_equal(mosaic_px(t, np.array([[0.0, 0.0]])), [[8192.0, 12288.0]])
    np.testing.assert_array_equal(mosaic_px(t, np.array([[1.5, 2.0]])), [[8193.5, 12290.0]])


def test_tile_box() -> None:
    t = tile_ref("siret3_r006_c004")
    b = tile_box(t)
    assert b.bounds == t.bounds
    assert b.area == pytest.approx(TILE_M * TILE_M)
    assert b.exterior.is_ccw


def test_tile_of_point_boundaries() -> None:
    t = tile_ref("siret3_r006_c004")
    assert tile_of_point(t.x0, t.y0) == t  # UL corner belongs to the tile
    assert tile_of_point(t.x0 + TILE_M, t.y0).grid_col == 5


def test_existing_tile_ids_has_311_and_examples() -> None:
    ids = existing_tile_ids()
    assert len(ids) == 311
    assert set(EXAMPLES) <= ids


def test_tiles_for_bounds_strict_interior() -> None:
    t = tile_ref("siret3_r006_c004")
    assert tiles_for_bounds(*tile_box(t).bounds) == ["siret3_r006_c004"]
    cx, cy = t.x0 + TILE_M, t.y0 - TILE_M  # SE corner shared by 4 tiles
    four = tiles_for_bounds(cx - 1, cy - 1, cx + 1, cy + 1)
    expected = [f"siret3_r{r:03d}_c{c:03d}" for r in (6, 7) for c in (4, 5)]
    assert four == [tid for tid in expected if tid in existing_tile_ids()]
    assert tiles_for_bounds(0.0, 0.0, 1.0, 1.0) == []
    with pytest.raises(ValueError):
        tiles_for_bounds(10.0, 0.0, 0.0, 1.0)


def test_tiles_for_bounds_filters_nonexistent() -> None:
    ids = existing_tile_ids()
    everything = tiles_for_bounds(GRID_ORIGIN_X - 1e3, GRID_ORIGIN_Y - 40 * TILE_M, GRID_ORIGIN_X + 40 * TILE_M, GRID_ORIGIN_Y + 1e3)
    assert everything == sorted(ids)


@pytest.mark.examples
@pytest.mark.parametrize("tile_id", EXAMPLES)
def test_tile_ref_from_tags_examples(example_tif, tile_id: str) -> None:
    assert tile_ref_from_tags(example_tif(tile_id)) == tile_ref(tile_id)


def test_tile_ref_from_tags_synthetic(make_geotiff, tmp_path: Path) -> None:
    path = make_geotiff(tmp_path / "siret3_r021_c012.tif", size=64)
    assert tile_ref_from_tags(path) == tile_ref("siret3_r021_c012")


def test_tile_ref_from_tags_offset_raises(make_geotiff, tmp_path: Path) -> None:
    path = make_geotiff(tmp_path / "siret3_r021_c012.tif", size=64, dx=0.1)
    with pytest.raises(IngestError, match="siret3_r021_c012"):
        tile_ref_from_tags(path)


def test_tile_ref_from_tags_wrong_crs_raises(make_geotiff, tmp_path: Path) -> None:
    path = make_geotiff(tmp_path / "siret3_r021_c012.tif", size=64, epsg=32634)
    with pytest.raises(IngestError, match="32634"):
        tile_ref_from_tags(path)


def test_tile_ref_from_tags_wrong_scale_raises(make_geotiff, tmp_path: Path) -> None:
    path = make_geotiff(tmp_path / "siret3_r021_c012.tif", size=64, gsd=0.05)
    with pytest.raises(IngestError, match="transform"):
        tile_ref_from_tags(path)


def test_tile_ref_from_tags_bad_name_raises(make_geotiff, tmp_path: Path) -> None:
    path = make_geotiff(tmp_path / "foo.tif", tile_id="siret3_r021_c012", size=64)
    with pytest.raises(IngestError, match="foo.tif"):
        tile_ref_from_tags(path)


@pytest.mark.needs_tiles
def test_all_zip_tiles_match_grid(data_root: Path, tmp_path: Path) -> None:
    names: list[str] = []
    for zpath in sorted((data_root / "01_tiles").glob("siret3_challenge_tiles_part*of5.zip")):
        with zipfile.ZipFile(zpath) as zf:
            names.extend(zf.namelist())
    ids = sorted(n.removesuffix(".tif") for n in names)
    assert len(ids) == len(set(ids)) == 311
    assert set(ids) == existing_tile_ids()
    zpath = data_root / "01_tiles" / "siret3_challenge_tiles_part1of5.zip"
    with zipfile.ZipFile(zpath) as zf:
        for name in sorted(zf.namelist())[:10]:
            out = tmp_path / name
            out.write_bytes(zf.read(name))
            assert tile_ref_from_tags(out) == tile_ref(name.removesuffix(".tif"))


def test_module_exports_contract_names() -> None:
    for name in ("GSD_M", "TILE_PX", "TILE_M", "GRID_ORIGIN_X", "GRID_ORIGIN_Y", "TileRef", "tile_ref_from_grid",
                 "tile_ref_from_tags", "px_to_utm", "utm_to_px", "index_to_px", "tiles_for_bounds", "mosaic_px"):
        assert hasattr(tiling, name)
