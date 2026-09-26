"""nn.validate: official vector canopy score of a probability raster through fusion + canopy extraction."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import shapely

from tests.nn.n2_synth import (
    OTHER_TILE,
    TILE_ID,
    reference_canopies,
    synth_tile,
    write_tile_prep_cache,
    write_tile_valid,
)
from vineyard.config import load_config
from vineyard.contracts.enums import FusionVariant
from vineyard.errors import StageError
from vineyard.geo.tiling import TILE_PX, tile_box, tile_ref
from vineyard.nn.probs import PROB_PX
from vineyard.nn.validate import (
    TileScore,
    ValResult,
    ValTile,
    input_factor,
    load_val_tiles,
    make_val_tile,
    network_input,
    predict_canopies,
    prob_to_tile,
    score_tile,
    val_variant,
    validate_probs,
    variant_mask,
)

CFG = load_config()


def _half(mask: np.ndarray) -> np.ndarray:
    """2048² bool -> 1024² float probability (exact for masks made of 2x2-aligned blocks)."""
    return mask.reshape(PROB_PX, 2, PROB_PX, 2).mean(axis=(1, 3)).astype(np.float32)


@pytest.fixture(scope="module")
def val_tile() -> ValTile:
    s = synth_tile()
    clip = tile_box(tile_ref(TILE_ID))
    base = make_val_tile(TILE_ID, s.rgb, s.valid, s.veg, s.pieces, clip, (), input_factor(CFG))
    ref = tuple(predict_canopies(base, None, FusionVariant.A, CFG))
    return replace(base, ref=ref)


def test_import_is_torch_free() -> None:
    code = "import sys, vineyard.nn.validate; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_val_variant_comes_from_config() -> None:
    assert val_variant(CFG) is FusionVariant.B
    cfg_c = CFG.model_copy(update={"nn": CFG.nn.model_copy(update={
        "train": CFG.nn.train.model_copy(update={"val_variant": "C"})})})
    assert val_variant(cfg_c) is FusionVariant.C


def test_input_factor_and_network_input() -> None:
    assert input_factor(CFG) == 2
    rgb = np.full((4, 4, 3), 200, np.uint8)
    valid = np.ones((4, 4), bool)
    valid[:2, :2] = False
    out = network_input(rgb, valid, 2)
    assert out.shape == (2, 2, 3) and out.dtype == np.uint8
    assert out[0, 0].tolist() == [0, 0, 0]  # a fully invalid block becomes 0
    assert out[1, 1].tolist() == [200, 200, 200]
    with pytest.raises(ValueError, match="RGB"):
        network_input(rgb[..., 0], valid, 2)


def test_prob_to_tile_quantises_like_the_cache() -> None:
    prob = np.zeros((PROB_PX, PROB_PX), np.float32)
    prob[0, 0] = 0.5  # stored as 128 -> 0.502 >= 0.5
    prob[0, 1] = 0.498  # stored as 127 -> 0.498 < 0.5
    up = prob_to_tile(prob)
    assert up.shape == (TILE_PX, TILE_PX) and up.dtype == np.float32
    assert up[0, 0] >= 0.5 and up[0, 3] < 0.5
    as_u8 = prob_to_tile(np.rint(prob * 255).astype(np.uint8))
    np.testing.assert_allclose(as_u8, up)
    with pytest.raises(ValueError, match="shape"):
        prob_to_tile(np.zeros((10, 10), np.float32))


def test_variant_mask_truth() -> None:
    veg = np.array([[True, True, False, False]])
    prob = np.array([[0.9, 0.1, 0.9, 0.1]], np.float32)
    assert variant_mask(veg, prob, FusionVariant.B, 0.5).tolist() == [[True, False, True, False]]
    assert variant_mask(veg, prob, FusionVariant.C, 0.5).tolist() == [[True, False, False, False]]
    assert variant_mask(veg, prob, FusionVariant.D, 0.5).tolist() == [[True, True, True, False]]
    assert variant_mask(veg, None, FusionVariant.B, 0.5).tolist() == veg.tolist()


def test_reference_vs_itself_is_perfect(val_tile: ValTile) -> None:
    assert len(val_tile.ref) == 5
    s = score_tile(val_tile, _half(synth_tile().canopy), FusionVariant.B, CFG)
    assert isinstance(s, TileScore)
    assert (s.iou, s.f1, s.score, s.n_pred, s.n_ref) == pytest.approx((1.0, 1.0, 1.0, 5, 5))
    assert s.variant == "B" and s.tile_id == TILE_ID


def test_empty_prediction_scores_zero(val_tile: ValTile) -> None:
    s = score_tile(val_tile, np.zeros((PROB_PX, PROB_PX), np.float32), FusionVariant.B, CFG)
    assert (s.iou, s.f1, s.score, s.n_pred) == (0.0, 0.0, 0.0, 0)


def test_shifted_canopies_are_not_matched(val_tile: ValTile) -> None:
    shifted = np.roll(synth_tile().canopy, 36, axis=1)  # 60 px wide canopies moved by 36 px: IoU 0.25
    s = score_tile(val_tile, _half(shifted), FusionVariant.B, CFG)
    assert s.f1 == 0.0 and 0.0 < s.iou < 0.5


def test_grass_outside_the_corridor_only_counts_for_e(val_tile: ValTile) -> None:
    prob = _half(synth_tile().veg)  # canopies + grass far from the row
    b = score_tile(val_tile, prob, FusionVariant.B, CFG)
    e = score_tile(val_tile, prob, FusionVariant.E, CFG)
    assert b.score == pytest.approx(1.0)
    assert e.n_pred == 6 and e.score < 1.0


def test_validate_probs_means(val_tile: ValTile) -> None:
    other = replace(val_tile, tile_id=OTHER_TILE)
    probs = {TILE_ID: _half(synth_tile().canopy), OTHER_TILE: np.zeros((PROB_PX, PROB_PX), np.float32)}
    res = validate_probs((val_tile, other), probs, CFG)
    assert isinstance(res, ValResult)
    assert [s.tile_id for s in res.scores] == [TILE_ID, OTHER_TILE]
    assert res.mean_score == pytest.approx(0.5)
    doc = res.to_dict()
    assert doc["mean"]["score"] == pytest.approx(0.5) and doc["variant"] == "B"
    assert doc["tiles"][TILE_ID]["score"] == pytest.approx(1.0)
    with pytest.raises(StageError, match="probability"):
        validate_probs((val_tile,), {}, CFG)


def test_load_val_tiles(tmp_path: Path) -> None:
    s = synth_tile()
    cache = tmp_path / "cache"
    write_tile_prep_cache(cache, s)
    tile_valid = write_tile_valid(tmp_path / "layers" / "tile_valid.parquet", (TILE_ID,))
    c = tile_box(tile_ref(TILE_ID)).centroid
    outside = shapely.box(0, 0, 1, 1)  # far from the tile: dropped by the clip to the tile
    ref = reference_canopies(TILE_ID, [shapely.box(c.x, c.y, c.x + 1, c.y + 1), outside])
    tiles = load_val_tiles(CFG, cache, tile_valid, (TILE_ID,), s.pieces, ref, lambda _t: s.rgb)
    (vt,) = tiles
    assert vt.tile_id == TILE_ID and len(vt.ref) == 1 and len(vt.pieces) == 1
    assert vt.net_input.shape == (PROB_PX, PROB_PX, 3)
    assert vt.veg.sum() == s.veg.sum() and vt.valid.all()
    with pytest.raises(StageError, match="row pieces"):
        load_val_tiles(CFG, cache, tile_valid, (TILE_ID,), s.pieces.iloc[0:0], ref, lambda _t: s.rgb)
