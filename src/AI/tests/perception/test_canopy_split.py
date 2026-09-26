"""perception.canopy_split: touching plants split at the narrowing (annotation rules §2.2)."""

from __future__ import annotations

import dataclasses

import cv2
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from vineyard.geo.tiling import CRS_EPSG, TILE_PX, px_to_utm, tile_box, tile_ref
from vineyard.perception.canopy import CanopyOptions, extract_canopies
from vineyard.perception.canopy_split import NeckSplitOptions, neck_bins, split_at_necks
from vineyard.perception.corridor import clip_rows_to_tile

TILE = tile_ref("siret3_r021_c012")
NECK = NeckSplitOptions(min_along_m=1.6, neck_ratio=0.3, min_piece_m=0.5, window_m=0.6, min_part_px=304)
V = 500


def _canvas() -> np.ndarray:
    return np.zeros((TILE_PX, TILE_PX), dtype=bool)


def _two_plants(neck_half_px: int, *, half_px: int = 10, angle_deg: float = 0.0) -> np.ndarray:
    """Two 1.25 m plants (50 px) along a row joined by a 12 px long neck of half width neck_half_px."""
    m = _canvas()
    m[V - half_px : V + half_px, 100:150] = True
    m[V - neck_half_px : V + neck_half_px, 150:162] = True
    m[V - half_px : V + half_px, 162:212] = True
    if angle_deg:
        rot = cv2.getRotationMatrix2D((150.0, float(V)), angle_deg, 1.0)
        m = cv2.warpAffine(m.astype(np.uint8), rot, (TILE_PX, TILE_PX), flags=cv2.INTER_NEAREST) > 0
    return m


def _n_components(mask: np.ndarray) -> int:
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1


def test_neck_bins_finds_the_deep_minimum_only() -> None:
    width = np.array([10.0] * 12 + [2.0] * 3 + [10.0] * 12)
    assert neck_bins(width, ratio=0.3, min_piece_bins=10, window_bins=12) == [13]
    assert neck_bins(width, ratio=0.1, min_piece_bins=10, window_bins=12) == []  # 2/10 is not <= 0.1
    single = np.array([10.0] * 12 + [2.0] + [10.0] * 12)  # one 5 cm slice: smoothed away (noise)
    assert neck_bins(single, ratio=0.3, min_piece_bins=10, window_bins=12) == []
    shallow = np.array([10.0] * 12 + [7.0] * 3 + [10.0] * 12)
    assert neck_bins(shallow, ratio=0.3, min_piece_bins=10, window_bins=12) == []
    near_end = np.array([10.0] * 4 + [1.0] * 3 + [10.0] * 20)
    assert neck_bins(near_end, ratio=0.3, min_piece_bins=10, window_bins=12) == []


@pytest.mark.parametrize("angle", [0.0, 35.0, -60.0])
def test_touching_plants_split_at_the_narrowing(angle: float) -> None:
    mask = _two_plants(2, angle_deg=angle)
    out = split_at_necks(mask, 8, NECK)
    assert _n_components(mask) == 1 and _n_components(out) == 2
    assert (mask.sum() - out.sum()) < 0.05 * mask.sum()  # only the thin neck slice is cleared
    assert not (out & ~mask).any()


def test_continuous_or_short_canopies_stay_whole() -> None:
    bar = _canvas()
    bar[V - 10 : V + 10, 100:300] = True  # 5 m, no narrowing
    assert np.array_equal(split_at_necks(bar, 8, NECK), bar)
    wide_neck = _two_plants(8)  # 16 px neck of 20 px plants: not a visible narrowing
    assert np.array_equal(split_at_necks(wide_neck, 8, NECK), wide_neck)
    short = _canvas()
    short[V - 10 : V + 10, 100:125] = True
    short[V - 2 : V + 2, 125:137] = True
    short[V - 10 : V + 10, 137:160] = True  # 1.5 m < min_along_m
    assert np.array_equal(split_at_necks(short, 8, NECK), short)


def test_small_parts_are_not_cut_off() -> None:
    m = _canvas()
    m[V - 10 : V + 10, 100:180] = True
    m[V - 2 : V + 2, 180:192] = True
    m[V - 4 : V + 4, 192:222] = True  # 30 x 8 = 240 px < min_part_px
    assert np.array_equal(split_at_necks(m, 8, NECK), m)


def test_input_not_modified_and_option_validation() -> None:
    mask = _two_plants(2)
    before = mask.copy()
    split_at_necks(mask, 8, NECK)
    assert np.array_equal(mask, before)
    for bad in ({"neck_ratio": 1.2}, {"bin_px": 1}, {"min_piece_m": 1.0}, {"window_m": 0.0}):
        with pytest.raises(ValueError):
            dataclasses.replace(NECK, **bad)


def test_extract_canopies_splits_touching_plants_when_enabled() -> None:
    line = LineString(px_to_utm(TILE, np.array([[-100.0, V], [2200.0, V]])))
    rows = gpd.GeoDataFrame({"row_id": ["V01-R001"], "vineyard_id": "V01", "row_index": [1], "qa_flags": [""]},
                            geometry=[line], crs=f"EPSG:{CRS_EPSG}")
    pieces = clip_rows_to_tile(rows, TILE, tile_box(TILE), margin_m=1.0)
    base = CanopyOptions(corridor_half_m=0.30, min_area_m2=0.19, connectivity=8, approx_eps_px=2.0, offset_px=0.0,
                         outset_px=0.0, clump_area_m2=2.0, clip_to_corridor=False, interpolated_min_veg_frac=0.15,
                         label_convention="index")
    mask = _two_plants(2)
    assert len(extract_canopies(mask, pieces, TILE, tile_box(TILE), base).canopies) == 1
    split = dataclasses.replace(base, neck_split=NECK)
    assert len(extract_canopies(mask, pieces, TILE, tile_box(TILE), split).canopies) == 2


def test_neck_split_options_from_config() -> None:
    from vineyard.config import load_config

    cfg = load_config()
    opts = CanopyOptions.from_config(cfg.canopy)
    assert opts.neck_split is not None and opts.neck_split.neck_ratio == cfg.canopy.neck_split_ratio
    assert opts.neck_split.min_part_px == opts.min_component_px
    off = CanopyOptions.from_config(cfg.canopy.model_copy(update={"neck_split_enabled": False}))
    assert off.neck_split is None
