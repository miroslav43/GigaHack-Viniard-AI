"""Pseudo-labels (design 03 §3 N2, §5): label raster rules and RGB downsampling."""

from __future__ import annotations

import subprocess
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from vineyard.config import load_config
from vineyard.geo.tiling import CRS_EPSG, TILE_PX, px_to_utm, tile_ref
from vineyard.nn.pseudolabels import (
    IGNORE,
    NEGATIVE,
    POSITIVE,
    LabelParams,
    PseudoLabelError,
    build_pseudolabel,
    classify_corridor_veg,
    corridor_inputs,
    downsample_rgb,
    edge_band,
    empty_tile_label,
    label_counts,
)
from vineyard.pipeline.stages.canopy import local_pieces

HOLDOUT = ("siret3_r006_c004", "siret3_r021_c012")
TILE_ID = "siret3_r020_c012"
N = 64  # small native rasters for the pure functions (factor 2 -> 32)


def _params(**kw: object) -> LabelParams:
    base = dict(factor=2, ignore_band_px=3, edge_band_px=1, ignore_small_in_corridor=True, ignore_clumps=True,
                tree_filter=True, min_component_px=4, clump_min_px=400, connectivity=8)
    return LabelParams(**{**base, **kw})  # type: ignore[arg-type]


def _all_eligible(labels: np.ndarray) -> np.ndarray:
    return np.ones(int(labels.max()) + 1, dtype=bool)


# ------------------------------------------------------------------ import hygiene


def test_pseudolabels_is_torch_free() -> None:
    code = "import sys, vineyard.nn.pseudolabels; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------------ LabelParams


def test_label_params_from_default_config() -> None:
    p = LabelParams.from_config(load_config())
    assert p.factor == 2
    assert p.ignore_band_px == 3 and p.edge_band_px == 1
    assert p.min_component_px == 304  # ceil(0.19 m² / 0.025²)
    assert p.clump_min_px == 3200  # 2.0 m² / 0.025²
    assert p.connectivity == 8


def test_label_params_reject_non_integer_factor() -> None:
    cfg = load_config(overrides=["nn.in_gsd_m=0.06"])
    with pytest.raises(PseudoLabelError, match="integer"):
        LabelParams.from_config(cfg)


def test_label_params_validate_values() -> None:
    with pytest.raises(ValueError, match="factor"):
        _params(factor=0)
    with pytest.raises(ValueError, match="connectivity"):
        _params(connectivity=6)
    with pytest.raises(ValueError, match="band"):
        _params(edge_band_px=-1)


# ------------------------------------------------------------------ build_pseudolabel


def _square_label(edge_band_px: int) -> np.ndarray:
    vine = np.zeros((2048, 2048), dtype=bool)
    vine[400:440, 600:640] = True  # 20x20 at the label resolution
    return build_pseudolabel(vine, np.zeros_like(vine), np.ones_like(vine), _params(edge_band_px=edge_band_px))


def test_isolated_square_gets_480_px_band_and_196_px_core_with_a_3px_band() -> None:
    label = _square_label(3)  # design 03 §5 case (7x7 kernel)
    assert label.shape == (1024, 1024) and label.dtype == np.uint8
    assert int((label == IGNORE).sum()) == 26 * 26 - 14 * 14 == 480
    assert int((label == POSITIVE).sum()) == 14 * 14
    assert int((label == NEGATIVE).sum()) == 1024 * 1024 - 26 * 26


def test_isolated_square_with_the_default_1px_band() -> None:
    label = _square_label(1)
    assert int((label == IGNORE).sum()) == 22 * 22 - 18 * 18
    assert int((label == POSITIVE).sum()) == 18 * 18


def test_nodata_becomes_ignore() -> None:
    valid = np.ones((N, N), dtype=bool)
    valid[:, :10] = False  # native cols 0..9 -> label cols 0..4 touch nodata
    zero = np.zeros_like(valid)
    label = build_pseudolabel(zero, zero, valid, _params(edge_band_px=0))
    assert (label[:, :5] == IGNORE).all()
    assert (label[:, 5:] == NEGATIVE).all()


def test_ignore_mask_wins_over_positive() -> None:
    vine = np.zeros((N, N), dtype=bool)
    vine[20:40, 20:40] = True
    ignore = np.zeros_like(vine)
    ignore[20:22, 20:22] = True
    label = build_pseudolabel(vine, ignore, np.ones_like(vine), _params(edge_band_px=0))
    assert label[10, 10] == IGNORE
    assert label[15, 15] == POSITIVE


def test_zero_band_keeps_edges() -> None:
    vine = np.zeros((N, N), dtype=bool)
    vine[20:40, 20:40] = True
    label = build_pseudolabel(vine, np.zeros_like(vine), np.ones_like(vine), _params(edge_band_px=0))
    assert int((label == POSITIVE).sum()) == 100 and not (label == IGNORE).any()


def test_build_pseudolabel_rejects_bad_shapes() -> None:
    a = np.zeros((N, N), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        build_pseudolabel(a, a, np.zeros((N, N + 2), dtype=bool), _params())
    odd = np.zeros((N + 1, N + 1), dtype=bool)
    with pytest.raises(ValueError, match="divisible"):
        build_pseudolabel(odd, odd, odd, _params())


def test_inputs_are_not_modified() -> None:
    vine = np.zeros((N, N), dtype=bool)
    vine[20:40, 20:40] = True
    before = vine.copy()
    build_pseudolabel(vine, np.zeros_like(vine), np.ones_like(vine), _params())
    assert np.array_equal(vine, before)


def test_edge_band_of_empty_and_full_masks_is_empty() -> None:
    assert not edge_band(np.zeros((10, 10), dtype=bool), 3).any()
    assert not edge_band(np.ones((10, 10), dtype=bool), 3).any()
    with pytest.raises(ValueError, match="band"):
        edge_band(np.zeros((4, 4), dtype=bool), -1)


def test_empty_tile_label_is_zero_except_nodata() -> None:
    valid = np.ones((N, N), dtype=bool)
    valid[:4, :4] = False
    label = empty_tile_label(valid, _params())
    assert label.shape == (N // 2, N // 2)
    assert (label[:2, :2] == IGNORE).all()
    assert int((label == IGNORE).sum()) == 4
    assert int((label == NEGATIVE).sum()) == (N // 2) ** 2 - 4


def test_label_counts() -> None:
    label = np.array([[0, 1], [255, 1]], dtype=np.uint8)
    counts = label_counts(label)
    assert (counts.n_pos, counts.n_neg, counts.n_ignore) == (2, 1, 1)
    with pytest.raises(ValueError, match="values"):
        label_counts(np.array([[3]], dtype=np.uint8))


# ------------------------------------------------------------------ classify_corridor_veg


def _corridor_labels() -> np.ndarray:
    labels = np.zeros((N, N), dtype=np.int32)
    labels[10:30, :] = 1
    labels[40:60, :] = 2
    return labels


def test_vegetation_outside_the_corridor_is_negative_beyond_the_overhang_band() -> None:
    veg = np.zeros((N, N), dtype=bool)
    veg[0:10, 0:40] = True  # above corridor 1 (rows 10..29): rows 4..9 are within 3 label px = 6 native px
    veg[12:28, 5:25] = True  # inside corridor 1, 320 px
    labels = _corridor_labels()
    masks = classify_corridor_veg(veg, labels, _all_eligible(labels), None, _params())
    assert not masks.vine[0:10].any()
    assert not masks.ignore[0:4].any() and masks.ignore[4:10, 0:40].all()
    assert masks.n_overhang_px == 6 * 40
    assert masks.vine[12:28, 5:25].all()
    assert np.array_equal(masks.corridor, labels > 0)
    no_band = classify_corridor_veg(veg, labels, _all_eligible(labels), None, _params(ignore_band_px=0))
    assert not no_band.ignore[0:10].any() and no_band.n_overhang_px == 0


def test_soil_next_to_the_corridor_stays_negative() -> None:
    labels = _corridor_labels()
    masks = classify_corridor_veg(np.zeros((N, N), dtype=bool), labels, _all_eligible(labels), None, _params())
    assert not masks.ignore.any() and not masks.vine.any()


def test_clump_component_is_ignored_or_positive_by_flag() -> None:
    veg = np.zeros((N, N), dtype=bool)
    veg[10:30, 0:30] = True  # 600 px > clump_min_px 400
    labels = _corridor_labels()
    on = classify_corridor_veg(veg, labels, _all_eligible(labels), None, _params())
    assert on.ignore[10:30, 0:30].all() and not on.vine.any()
    assert on.n_clump_px == 600
    off = classify_corridor_veg(veg, labels, _all_eligible(labels), None, _params(ignore_clumps=False))
    assert off.vine[10:30, 0:30].all() and not off.ignore.any()


def test_small_in_corridor_component_by_flag() -> None:
    veg = np.zeros((N, N), dtype=bool)
    veg[15, 15] = True  # 1 px < min_component_px 4
    labels = _corridor_labels()
    on = classify_corridor_veg(veg, labels, _all_eligible(labels), None, _params())
    assert on.ignore[15, 15] and not on.vine[15, 15] and on.n_small_px == 1
    off = classify_corridor_veg(veg, labels, _all_eligible(labels), None, _params(ignore_small_in_corridor=False))
    assert not off.ignore[15, 15] and not off.vine[15, 15]


def test_tree_pixels_are_ignored_when_the_filter_is_on() -> None:
    veg = np.zeros((N, N), dtype=bool)
    veg[12:28, 5:25] = True
    tree = np.zeros_like(veg)
    tree[12:28, 5:15] = True
    labels = _corridor_labels()
    on = classify_corridor_veg(veg, labels, _all_eligible(labels), tree, _params())
    assert on.ignore[12:28, 5:15].all() and on.vine[12:28, 15:25].all() and on.n_tree_px == 160
    off = classify_corridor_veg(veg, labels, _all_eligible(labels), tree, _params(tree_filter=False))
    assert off.vine[12:28, 5:25].all() and off.n_tree_px == 0


def test_ineligible_piece_vegetation_is_ignored() -> None:
    veg = np.zeros((N, N), dtype=bool)
    veg[42:58, 5:25] = True  # corridor 2
    labels = _corridor_labels()
    eligible = np.array([False, True, False])
    masks = classify_corridor_veg(veg, labels, eligible, None, _params())
    assert masks.ignore[42:58, 5:25].all() and not masks.vine.any() and masks.n_ineligible_px == 320


def test_classify_rejects_short_eligible_lookup() -> None:
    labels = _corridor_labels()
    with pytest.raises(ValueError, match="eligible"):
        classify_corridor_veg(np.zeros((N, N), dtype=bool), labels, np.ones(2, dtype=bool), None, _params())


# ------------------------------------------------------------------ corridor_inputs (perception functions)


def _rows(tile_id: str, vs: tuple[float, ...]) -> gpd.GeoDataFrame:
    tile = tile_ref(tile_id)
    lines = [LineString(px_to_utm(tile, np.array([[-400.0, v], [1900.0, v]]))) for v in vs]
    return gpd.GeoDataFrame(
        {"row_id": [f"V01-R{k + 1:03d}" for k in range(len(vs))], "vineyard_id": "V01",
         "row_index": range(1, len(vs) + 1), "qa_flags": "", "interp_tile_ids": ""},
        geometry=lines, crs=f"EPSG:{CRS_EPSG}",
    )


def test_corridor_inputs_from_row_pieces() -> None:
    cfg = load_config()
    tile = tile_ref(TILE_ID)
    pieces = local_pieces(_rows(TILE_ID, (500.0, 620.0)), tile, cfg.canopy.rows_margin_m)
    veg = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    veg[495:505, 100:1800] = True
    inputs = corridor_inputs(pieces, tile, veg, np.ones_like(veg), cfg, _params(tree_filter=True))
    assert inputs.labels.shape == (TILE_PX, TILE_PX)
    half_px = cfg.canopy.corridor_half_m / 0.025
    assert inputs.labels[500, 1000] > 0 and inputs.labels[int(500 + half_px + 2), 1000] == 0
    assert inputs.eligible.shape == (3,) and inputs.eligible[1:].all() and not inputs.eligible[0]
    assert inputs.tree is not None and not inputs.tree.any()
    no_tree = corridor_inputs(pieces, tile, veg, np.ones_like(veg), cfg, _params(tree_filter=False))
    assert no_tree.tree is None


# ------------------------------------------------------------------ downsample_rgb


def test_downsample_rgb_zeroes_nodata_before_area_average() -> None:
    rgb = np.full((4, 4, 3), 200, dtype=np.uint8)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    out = downsample_rgb(rgb, valid, 2)
    assert out.shape == (2, 2, 3) and out.dtype == np.uint8
    assert (out[0, 0] == 150).all() and (out[1, 1] == 200).all()
    assert (rgb == 200).all()  # input untouched


def test_downsample_rgb_alignment() -> None:
    rgb = np.zeros((2048, 2048, 3), dtype=np.uint8)
    rgb[20:22, 10:12] = 255
    out = downsample_rgb(rgb, np.ones((2048, 2048), dtype=bool), 2)
    ys, xs = np.nonzero(out[..., 0])
    assert list(ys) == [10] and list(xs) == [5]


def test_downsample_rgb_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="RGB"):
        downsample_rgb(np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=bool), 2)
    with pytest.raises(ValueError, match="valid"):
        downsample_rgb(np.zeros((4, 4, 3), dtype=np.uint8), np.ones((2, 2), dtype=bool), 2)
    assert downsample_rgb(np.ones((4, 4, 3), dtype=np.uint8), np.ones((4, 4), dtype=bool), 1).shape == (4, 4, 3)
