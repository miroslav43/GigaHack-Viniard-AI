"""Raster helpers: tile reading, nodata, mask <-> polygon conventions, rasterization, PNG I/O."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from vineyard.errors import IngestError
from vineyard.geo.raster import (
    mask_to_polygons,
    rasterize_px,
    rasterize_utm,
    read_mask_png,
    read_tile,
    read_u8_png,
    resize_aligned,
    valid_mask,
    valid_polygon,
    write_mask_png,
    write_u8_png,
)
from vineyard.geo.tiling import GSD_M, TILE_M, px_to_utm, tile_box, tile_ref

T = tile_ref("siret3_r021_c012")
PX2 = GSD_M * GSD_M
NODATA_KW = {"max_rgb": 10, "min_area_px": 800, "close_px": 5, "dilate_px": 4}


def _utm_area_px(polys: list[Polygon]) -> float:
    return sum(p.area for p in polys) / PX2


def _px_coords(poly: Polygon) -> np.ndarray:
    xy = np.asarray(poly.exterior.coords)
    return np.column_stack(((xy[:, 0] - T.x0) / GSD_M, (T.y0 - xy[:, 1]) / GSD_M))


# ------------------------------------------------------------------ read_tile


@pytest.mark.examples
def test_read_tile_example(example_tif) -> None:
    rgb = read_tile(example_tif("siret3_r021_c012"))
    assert rgb.shape == (2048, 2048, 3)
    assert rgb.dtype == np.uint8
    assert rgb.flags["C_CONTIGUOUS"]
    assert 20 < rgb.mean() < 235


def test_read_tile_synthetic_matches_written(make_geotiff, tmp_path: Path) -> None:
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 200, 100, 50
    path = make_geotiff(tmp_path / "siret3_r021_c012.tif", rgb=img)
    rgb = read_tile(path)
    assert rgb.shape == (64, 64, 3)
    assert np.abs(rgb.astype(int) - img.astype(int)).max() <= 3  # JPEG


def test_read_tile_rejects_non_rgb(make_geotiff, tmp_path: Path) -> None:
    path = make_geotiff(tmp_path / "siret3_r021_c012.tif", size=32, bands=1)
    with pytest.raises(IngestError, match="3 bands"):
        read_tile(path)


# ------------------------------------------------------------------ valid_mask


def _tile_with_black(size: int = 512) -> np.ndarray:
    rgb = np.full((size, size, 3), 120, np.uint8)
    rgb[:60, :60] = 0  # corner nodata touching the border
    rgb[60:62, :62] = 5  # JPEG ringing 1-10 DN at its edge
    rgb[:62, 60:62] = 5
    rgb[200:230, 200:230] = 0  # interior deep shadow stays valid
    rgb[:20, 300:320] = 0  # border blob below min_area (400 px < 800) stays valid
    return rgb


def test_valid_mask_border_nodata_and_interior_shadow() -> None:
    rgb = _tile_with_black()
    valid = valid_mask(rgb, **NODATA_KW)
    assert valid.dtype == np.bool_ and valid.shape == rgb.shape[:2]
    assert not valid[:62, :62].any()
    assert valid[200:230, 200:230].all()
    assert valid[:20, 300:320].all()
    # dilated by 4 px beyond the ringing edge
    assert not valid[30, 65] and valid[30, 67]
    assert not valid[65, 30] and valid[67, 30]
    assert valid[300:, :].all()


def test_valid_mask_all_valid_and_all_black() -> None:
    assert valid_mask(np.full((64, 64, 3), 90, np.uint8), **NODATA_KW).all()
    assert not valid_mask(np.zeros((64, 64, 3), np.uint8), **NODATA_KW).any()


def test_valid_mask_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        valid_mask(np.zeros((8, 8), np.uint8), **NODATA_KW)


def test_valid_polygon_full_tile_equals_tile_box() -> None:
    mp = valid_polygon(np.ones((2048, 2048), bool), T, approx_eps_px=2.0)
    assert isinstance(mp, MultiPolygon)
    assert mp.symmetric_difference(tile_box(T)).area < 1e-6


def test_valid_polygon_with_nodata_corner() -> None:
    valid = np.ones((2048, 2048), bool)
    valid[:1024, :1024] = False
    mp = valid_polygon(valid, T, approx_eps_px=2.0)
    assert mp.area == pytest.approx(0.75 * TILE_M * TILE_M, rel=1e-3)
    assert mp.within(tile_box(T).buffer(1e-6))
    assert valid_polygon(np.zeros((64, 64), bool), T, approx_eps_px=2.0).is_empty


# ------------------------------------------------------------------ mask_to_polygons


def test_mask_to_polygons_raw_convention_integer_vertices() -> None:
    mask = np.zeros((64, 64), bool)
    mask[10:13, 20:23] = True  # 3x3 block
    polys = mask_to_polygons(mask, T, approx_eps_px=0.0, min_area_px=0.0)
    assert len(polys) == 1
    uv = _px_coords(polys[0])
    np.testing.assert_allclose(uv, np.round(uv), atol=1e-6)
    assert _utm_area_px(polys) == pytest.approx(4.0, abs=1e-6)
    assert polys[0].exterior.is_ccw


def test_mask_to_polygons_pixel_square_convention() -> None:
    mask = np.zeros((64, 64), bool)
    mask[10:13, 20:23] = True
    polys = mask_to_polygons(mask, T, approx_eps_px=0.0, min_area_px=0.0, pixel_offset=0.5, outset_px=0.5)
    square = Polygon(px_to_utm(T, np.array([[20, 10], [23, 10], [23, 13], [20, 13]], float)))
    assert len(polys) == 1
    assert polys[0].symmetric_difference(square).area < 1e-9


def test_mask_to_polygons_square_areas_both_conventions() -> None:
    mask = rasterize_px([np.array([[10, 10], [20, 10], [20, 20], [10, 20]], float)], (64, 64)).astype(bool)
    raw = mask_to_polygons(mask, T, approx_eps_px=0.0, min_area_px=0.0)
    full = mask_to_polygons(mask, T, approx_eps_px=0.0, min_area_px=0.0, pixel_offset=0.5, outset_px=0.5)
    assert _utm_area_px(raw) == pytest.approx(81.0, abs=1e-6)
    assert _utm_area_px(full) == pytest.approx(100.0, abs=2.0)


def test_mask_to_polygons_single_pixel_and_min_area() -> None:
    mask = np.zeros((32, 32), bool)
    mask[5, 5] = True
    assert mask_to_polygons(mask, T, approx_eps_px=0.0, min_area_px=0.0) == []
    sq = mask_to_polygons(mask, T, approx_eps_px=0.0, min_area_px=0.0, pixel_offset=0.5, outset_px=0.5)
    assert _utm_area_px(sq) == pytest.approx(1.0, abs=1e-6)
    mask[20:30, 20:30] = True
    kept = mask_to_polygons(mask, T, approx_eps_px=0.0, min_area_px=50.0)
    assert len(kept) == 1 and _utm_area_px(kept) == pytest.approx(81.0, abs=1e-6)


def test_mask_to_polygons_holes_kept_and_oriented() -> None:
    mask = np.zeros((64, 64), bool)
    mask[10:40, 10:40] = True
    mask[20:30, 20:30] = False
    polys = mask_to_polygons(mask, T, approx_eps_px=0.0, min_area_px=0.0)
    assert len(polys) == 1
    p = polys[0]
    assert len(p.interiors) == 1
    assert p.exterior.is_ccw and not p.interiors[0].is_ccw
    assert p.is_valid


def test_mask_to_polygons_diagonal_pinch_is_valid() -> None:
    mask = np.zeros((32, 32), bool)
    mask[5:10, 5:10] = True
    mask[10:15, 10:15] = True  # touches the first block at one corner pixel
    polys = mask_to_polygons(mask, T, approx_eps_px=0.0, min_area_px=0.0)
    assert polys and all(p.is_valid and p.exterior.is_ccw for p in polys)
    assert _utm_area_px(polys) == pytest.approx(32.0, abs=1.0)


def test_mask_to_polygons_approx_reduces_vertices() -> None:
    mask = np.zeros((128, 128), np.uint8)
    cv2.circle(mask, (64, 64), 40, 1, -1)
    dense = mask_to_polygons(mask.astype(bool), T, approx_eps_px=0.0, min_area_px=0.0)
    coarse = mask_to_polygons(mask.astype(bool), T, approx_eps_px=2.5, min_area_px=0.0)
    assert len(coarse[0].exterior.coords) < len(dense[0].exterior.coords) / 3
    assert abs(coarse[0].area - dense[0].area) / dense[0].area < 0.03


def test_mask_to_polygons_input_untouched() -> None:
    mask = np.zeros((16, 16), bool)
    mask[2:6, 2:6] = True
    before = mask.copy()
    mask_to_polygons(mask, T, approx_eps_px=1.0, min_area_px=0.0)
    np.testing.assert_array_equal(mask, before)


# ------------------------------------------------------------------ rasterize


def test_rasterize_px_continuous_square_is_100px() -> None:
    m = rasterize_px([np.array([[10, 10], [20, 10], [20, 20], [10, 20]], float)], (64, 64))
    assert m.dtype == np.uint8 and m.sum() == 100
    rows, cols = np.nonzero(m)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (10, 19, 10, 19)


def test_rasterize_px_value_and_dtype() -> None:
    ring = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float)
    assert rasterize_px([ring], (8, 8), value=7).max() == 7
    big = rasterize_px([ring], (8, 8), value=1000)
    assert big.dtype == np.int32 and big.max() == 1000
    assert rasterize_px([], (8, 8)).sum() == 0
    with pytest.raises(ValueError):
        rasterize_px([ring], (8, 8), convention="bogus")  # type: ignore[arg-type]


def test_rasterize_px_adjacent_polygons_partition() -> None:
    a = np.array([[0, 0], [30.3, 0], [20.7, 64], [0, 64]])
    b = np.array([[30.3, 0], [64, 0], [64, 64], [20.7, 64]])
    ma, mb = rasterize_px([a], (64, 64)), rasterize_px([b], (64, 64))
    assert int((ma & mb).sum()) == 0
    assert int(((ma | mb) == 0).sum()) == 0


def test_rasterize_px_index_inverts_raw_contour() -> None:
    mask = np.zeros((96, 96), np.uint8)
    cv2.circle(mask, (40, 40), 18, 1, -1)
    mask[38:43, 5:90] = 1
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    rings = [c.reshape(-1, 2).astype(float) for c in contours]
    back = rasterize_px(rings, mask.shape, convention="index")
    np.testing.assert_array_equal(back, mask)


def test_rasterize_utm_index_convention_inverts_mask_to_polygons() -> None:
    mask = np.zeros((96, 96), bool)
    mask[10:50, 10:50] = True
    mask[20:30, 20:30] = False
    mask[60:70, 60:90] = True
    polys = mask_to_polygons(mask, T, approx_eps_px=0.0, min_area_px=0.0)
    back = rasterize_utm(polys, T, (96, 96), convention="index")
    np.testing.assert_array_equal(back.astype(bool), mask)


def test_rasterize_utm_continuous_tile_box_and_holes() -> None:
    full = rasterize_utm([tile_box(T)], T)
    assert full.shape == (2048, 2048) and full.all()
    uv = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], float)
    hole = np.array([[150, 150], [250, 150], [250, 250], [150, 250]], float)
    poly = Polygon(px_to_utm(T, uv), [px_to_utm(T, hole)])
    m = rasterize_utm([MultiPolygon([poly])], T, (512, 512))
    assert m.sum() == 200 * 200 - 100 * 100
    with pytest.raises(TypeError):
        rasterize_utm([poly.exterior], T)


# ------------------------------------------------------------------ resize


def test_resize_aligned_block_average() -> None:
    arr = np.zeros((2048, 2048), np.float32)
    arr[0:2, 0:2] = 1.0
    arr[2, 2] = 1.0
    down = resize_aligned(arr, 1024, mode="down")
    assert down.shape == (1024, 1024)
    assert down[0, 0] == pytest.approx(1.0) and down[1, 1] == pytest.approx(0.25)
    up = resize_aligned(down, 2048, mode="up")
    assert up.shape == (2048, 2048)
    boolean = resize_aligned(np.ones((8, 8), bool), 4, mode="down")
    assert boolean.dtype == np.float32 and boolean.min() == 1.0


def test_resize_aligned_mode_checks() -> None:
    with pytest.raises(ValueError):
        resize_aligned(np.zeros((8, 8), np.uint8), 16, mode="down")
    with pytest.raises(ValueError):
        resize_aligned(np.zeros((8, 8), np.uint8), 4, mode="up")
    with pytest.raises(ValueError):
        resize_aligned(np.zeros((8, 6), np.uint8), 4, mode="down")


# ------------------------------------------------------------------ PNG


def test_mask_png_roundtrip_is_1bit_and_atomic(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    mask = rng.random((300, 200)) > 0.5
    path = write_mask_png(tmp_path / "veg" / "t.png", mask)
    assert not [p for p in path.parent.iterdir() if ".tmp-" in p.name]
    back = read_mask_png(path)
    assert back.dtype == np.bool_
    np.testing.assert_array_equal(back, mask)
    assert path.stat().st_size < mask.size / 8 + 2048  # 1 bit per pixel plus overhead


def test_u8_png_roundtrip(tmp_path: Path) -> None:
    codes = np.random.default_rng(1).integers(0, 4, (128, 64), dtype=np.uint8)
    back = read_u8_png(write_u8_png(tmp_path / "vis.png", codes))
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, codes)


def test_png_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_mask_png(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises(ValueError, match="bad.png"):
        read_u8_png(bad)
    with pytest.raises(ValueError):
        write_mask_png(tmp_path / "x.png", np.zeros((4, 4, 3), bool))
    with pytest.raises(ValueError):
        write_u8_png(tmp_path / "y.png", np.zeros((4, 4), np.int32))
