"""perception.trees: compact wide blobs on a row are trees; elongated grass is not."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString

from tests.helpers.examples import load_examples, row_lines_utm
from vineyard.config import load_config
from vineyard.geo.raster import read_tile
from vineyard.geo.tiling import CRS_EPSG, TILE_PX, px_to_utm, tile_box, tile_ref
from vineyard.perception.canopy import CanopyOptions, extract_canopies
from vineyard.perception.corridor import clip_rows_to_tile, corridor_labels
from vineyard.perception.trees import TreeParams, blob_is_tree, tree_mask, tree_mask_from_labels
from vineyard.perception.vegmask import compute_tile_masks

TILE = tile_ref("siret3_r006_c004")
PARAMS = TreeParams(min_area_m2=2.0, max_axis_ratio=1.5, min_width_m=1.5)


def _disk(mask: np.ndarray, centre: tuple[int, int], r: int) -> np.ndarray:
    out = mask.astype(np.uint8).copy()
    cv2.circle(out, centre, r, 1, -1)
    return out > 0


def _pieces(v: float) -> gpd.GeoDataFrame:
    line = LineString(px_to_utm(TILE, np.array([[0.0, v], [2048.0, v]])))
    rows = gpd.GeoDataFrame({"row_id": ["V01-R001"], "vineyard_id": ["V01"]}, geometry=[line],
                            crs=f"EPSG:{CRS_EPSG}")
    return clip_rows_to_tile(rows, TILE, tile_box(TILE))


def test_blob_is_tree_geometry() -> None:
    ys, xs = np.nonzero(_disk(np.zeros((200, 200), bool), (100, 100), 40))
    pts = np.column_stack((xs, ys))
    assert blob_is_tree(pts, len(pts), PARAMS)
    strip = np.argwhere(np.ones((24, 400), bool))[:, ::-1]
    assert not blob_is_tree(strip, len(strip), PARAMS)  # elongated grass
    assert not blob_is_tree(pts[:10], 10, PARAMS)  # too small
    assert not blob_is_tree(np.array([[0, 0], [1, 1], [2, 2]]), 5000, PARAMS)  # degenerate


def test_tree_mask_keeps_trees_on_rows_only() -> None:
    veg = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    veg = _disk(veg, (500, 1000), 40)  # tree on the row
    veg = _disk(veg, (1500, 300), 40)  # tree far from the row
    veg[990:1010, 1200:1800] = True  # grass along the row
    labels = np.zeros(veg.shape, dtype=np.int32)
    labels[988:1012, :] = 1
    mask = tree_mask_from_labels(veg, labels, PARAMS)
    assert mask[1000, 500] and not mask[300, 1500] and not mask[1000, 1500]
    prob = np.ones(veg.shape, dtype=np.float32)
    vine_tree = tree_mask_from_labels(veg, labels, TreeParams(2.0, 1.5, 1.5, max_prob=0.5), prob)
    assert not vine_tree.any()


def test_tree_mask_from_config() -> None:
    cfg = load_config()
    veg = _disk(np.zeros((TILE_PX, TILE_PX), dtype=bool), (500, 1000), 40)
    mask = tree_mask(veg, _pieces(1000.0), TILE, cfg.canopy, cfg.orchard)
    assert mask.dtype == np.bool_ and mask[1000, 500]


def test_tree_crown_on_the_axis_removes_its_canopies() -> None:
    cfg = load_config()
    pieces = _pieces(1000.0)
    veg = _disk(np.zeros((TILE_PX, TILE_PX), dtype=bool), (500, 1000), 60)  # 3 m crown on the row
    veg[990:1010, 1200:1300] = True  # a vine plant further along
    tree = tree_mask(veg, pieces, TILE, cfg.canopy, cfg.orchard)
    assert tree[1000, 500] and not tree[1000, 1250]
    opts = CanopyOptions.from_config(cfg.canopy)
    kept = extract_canopies(veg, pieces, TILE, tile_box(TILE), opts, tree=tree).canopies
    assert len(kept) == 1 and kept.geometry.iloc[0].centroid.distance(
        shapely.Point(px_to_utm(TILE, np.array([[1250.0, 1000.0]]))[0])) < 1.0
    assert len(extract_canopies(veg, pieces, TILE, tile_box(TILE), opts).canopies) == 2


def test_grass_strip_along_the_row_is_not_a_tree() -> None:
    cfg = load_config()
    veg = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    veg[964:1036, 200:1000] = True  # 20 m x 1.8 m
    assert not tree_mask(veg, _pieces(1000.0), TILE, cfg.canopy, cfg.orchard).any()


@pytest.mark.examples
@pytest.mark.parametrize("tile_id", ["siret3_r021_c012", "siret3_r006_c004"])
def test_examples_have_no_trees_in_rows(tile_id: str, examples_xml: bytes, example_tif: Callable[[str], Path]) -> None:
    cfg = load_config()
    img = load_examples(examples_xml)[tile_id]
    tile = tile_ref(tile_id)
    rows = gpd.GeoDataFrame({"row_id": [s.attributes["row_id"] for s in img.by_label("row")], "vineyard_id": "V01"},
                            geometry=row_lines_utm(img), crs=f"EPSG:{CRS_EPSG}")
    pieces = clip_rows_to_tile(rows, tile, tile_box(tile), margin_m=cfg.canopy.rows_margin_m)
    rgb = read_tile(example_tif(tile_id))
    veg = compute_tile_masks(rgb, np.ones(rgb.shape[:2], dtype=bool), cfg.veg).veg
    labels = corridor_labels(pieces, tile, cfg.canopy.corridor_half_m)
    tree = tree_mask(veg, pieces, tile, cfg.canopy, cfg.orchard)
    assert int((tree & (labels > 0)).sum()) == 0
