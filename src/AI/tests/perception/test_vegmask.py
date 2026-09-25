"""Vegetation mask (port of analiza_exemple/scripts/axes.py veg_mask), vis codes, Otsu fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.helpers.synth import rgb_from_mask, striped_mask
from vineyard.config import VegConfig, load_config
from vineyard.geo.raster import read_tile
from vineyard.perception.types import VIS_NODATA, VIS_OK, VIS_OVEREXPOSED, VIS_SHADOW
from vineyard.perception.vegmask import (
    compute_tile_masks,
    neg_a,
    otsu_threshold,
    veg_mask,
    vis_codes,
)

EXAMPLES = ("siret3_r006_c004", "siret3_r021_c012")
PARITY_MIN = 0.995


@pytest.fixture(scope="module")
def veg_cfg() -> VegConfig:
    return load_config().veg


def _prototype_veg_mask(rgb: np.ndarray, thr: float = 4.0, blur: float = 2.5) -> np.ndarray:
    """analiza_exemple/scripts/axes.py::veg_mask, verbatim maths."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    na = -(lab[..., 1] - 128)
    na = cv2.GaussianBlur(na, (0, 0), blur)
    return na > thr


# ------------------------------------------------------------------ neg_a / veg_mask


def test_neg_a_green_positive_soil_negative() -> None:
    rgb = np.zeros((4, 8, 3), np.uint8)
    rgb[:, :4] = (60, 130, 40)  # vine green
    rgb[:, 4:] = (150, 115, 90)  # brown soil
    na = neg_a(rgb, 0.0)
    assert na.dtype == np.float32 and na.shape == (4, 8)
    assert (na[:, :4] > 10).all()
    assert (na[:, 4:] < 0).all()


def test_neg_a_blur_changes_edges_only() -> None:
    rgb = np.full((32, 32, 3), (150, 115, 90), np.uint8)
    rgb[:, 16:] = (60, 130, 40)
    sharp, blurred = neg_a(rgb, 0.0), neg_a(rgb, 2.5)
    assert blurred[:, 0] == pytest.approx(sharp[:, 0], abs=1e-3)
    assert blurred[0, 15] > sharp[0, 15]


def test_veg_mask_stripe_fraction_and_valid() -> None:
    stripes = striped_mask((512, 512), angle_deg=30.0, spacing_px=100.0, width_px=30.0)
    rgb = rgb_from_mask(stripes, noise_sigma=3.0, seed=1)
    valid = np.ones(stripes.shape, bool)
    veg = veg_mask(rgb, valid, blur_sigma_px=2.5, threshold=4.0)
    assert veg.dtype == np.bool_
    assert veg.mean() == pytest.approx(stripes.mean(), abs=0.02)
    valid[:, :256] = False
    assert not veg_mask(rgb, valid, blur_sigma_px=2.5, threshold=4.0)[:, :256].any()


def test_veg_mask_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        veg_mask(np.zeros((8, 8), np.uint8), np.ones((8, 8), bool), blur_sigma_px=0.0, threshold=4.0)
    with pytest.raises(ValueError):
        veg_mask(np.zeros((8, 8, 3), np.uint8), np.ones((4, 4), bool), blur_sigma_px=0.0, threshold=4.0)


# ------------------------------------------------------------------ vis codes


def test_vis_codes_priorities() -> None:
    rgb = np.full((2, 4, 3), 120, np.uint8)
    rgb[0, 1] = (10, 20, 30)  # V=30 -> shadow
    rgb[0, 2] = (250, 250, 250)  # V=250 -> overexposed
    rgb[0, 3] = (0, 0, 0)  # would be shadow, but nodata wins
    valid = np.ones((2, 4), bool)
    valid[0, 3] = False
    codes = vis_codes(rgb, valid, shadow_v_max=50, overexp_v_min=245)
    assert codes.dtype == np.uint8
    assert codes[0].tolist() == [VIS_OK, VIS_SHADOW, VIS_OVEREXPOSED, VIS_NODATA]
    assert (codes[1] == VIS_OK).all()


def test_vis_codes_threshold_edges() -> None:
    rgb = np.zeros((1, 4, 3), np.uint8)
    rgb[0, :, 1] = [49, 50, 245, 246]
    codes = vis_codes(rgb, np.ones((1, 4), bool), shadow_v_max=50, overexp_v_min=245)
    assert codes[0].tolist() == [VIS_SHADOW, VIS_OK, VIS_OK, VIS_OVEREXPOSED]


# ------------------------------------------------------------------ otsu


def test_otsu_threshold_bimodal() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(-5, 1, 5000), rng.normal(12, 1, 3000)]).astype(np.float32)
    thr = otsu_threshold(values)
    assert -2.0 < thr < 9.0


def test_otsu_threshold_degenerate() -> None:
    assert otsu_threshold(np.full(10, 3.0, np.float32)) == 3.0
    with pytest.raises(ValueError):
        otsu_threshold(np.array([], np.float32))
    with pytest.raises(ValueError):
        otsu_threshold(np.array([1.0, np.nan], np.float32))


# ------------------------------------------------------------------ compute_tile_masks


def test_compute_tile_masks_classic(veg_cfg: VegConfig) -> None:
    stripes = striped_mask((256, 256), angle_deg=0.0, spacing_px=64.0, width_px=16.0)
    rgb = rgb_from_mask(stripes)
    valid = np.ones(stripes.shape, bool)
    masks = compute_tile_masks(rgb, valid, veg_cfg)
    assert masks.method == veg_cfg.method == "lab_a"
    assert masks.threshold == veg_cfg.threshold
    assert masks.veg_frac == pytest.approx(stripes.mean(), abs=0.02)
    assert masks.vis.shape == stripes.shape
    assert not masks.veg.flags.writeable


def test_compute_tile_masks_no_valid_pixels(veg_cfg: VegConfig) -> None:
    rgb = np.zeros((32, 32, 3), np.uint8)
    masks = compute_tile_masks(rgb, np.zeros((32, 32), bool), veg_cfg)
    assert masks.veg_frac == 0.0
    assert (masks.vis == VIS_NODATA).all()


def test_compute_tile_masks_otsu_fallback(veg_cfg: VegConfig) -> None:
    # everything is faintly green: the fixed threshold marks ~100% as vegetation
    rng = np.random.default_rng(3)
    rgb = np.full((128, 128, 3), (80, 140, 60), np.uint8)
    rgb[:, 64:] = (95, 130, 70)
    rgb = np.clip(rgb + rng.normal(0, 2, rgb.shape), 0, 255).astype(np.uint8)
    valid = np.ones((128, 128), bool)
    off = compute_tile_masks(rgb, valid, veg_cfg)
    assert off.method == "lab_a" and off.veg_frac > veg_cfg.otsu_fallback_veg_frac[1]
    on_cfg = veg_cfg.model_copy(update={"otsu_fallback_enabled": True})
    on = compute_tile_masks(rgb, valid, on_cfg)
    assert on.method == "otsu"
    assert on.threshold != veg_cfg.threshold
    assert 0.3 < on.veg_frac < 0.7


def test_compute_tile_masks_otsu_not_triggered_in_range(veg_cfg: VegConfig) -> None:
    stripes = striped_mask((256, 256), angle_deg=0.0, spacing_px=64.0, width_px=24.0)
    on_cfg = veg_cfg.model_copy(update={"otsu_fallback_enabled": True})
    masks = compute_tile_masks(rgb_from_mask(stripes), np.ones((256, 256), bool), on_cfg)
    assert masks.method == "lab_a"


def test_compute_tile_masks_morph_close(veg_cfg: VegConfig) -> None:
    stripes = striped_mask((128, 128), angle_deg=0.0, spacing_px=64.0, width_px=16.0)
    stripes[:, 60:62] = False  # a 2-px break across every stripe
    rgb = rgb_from_mask(stripes)
    valid = np.ones((128, 128), bool)
    base = compute_tile_masks(rgb, valid, veg_cfg.model_copy(update={"blur_sigma_px": 0.0}))
    closed = compute_tile_masks(rgb, valid, veg_cfg.model_copy(update={"blur_sigma_px": 0.0, "morph_close_px": 5}))
    assert closed.veg.sum() > base.veg.sum()


# ------------------------------------------------------------------ examples


@pytest.mark.examples
@pytest.mark.parametrize("tile_id", EXAMPLES)
def test_veg_mask_parity_with_prototype(example_tif, tile_id: str) -> None:
    path: Path = example_tif(tile_id)
    ours = veg_mask(read_tile(path), np.ones((2048, 2048), bool), blur_sigma_px=2.5, threshold=4.0)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)  # independent TIFF/JPEG decoder (OpenCV libtiff)
    assert bgr is not None and bgr.shape == (2048, 2048, 3)
    proto = _prototype_veg_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    agreement = float((ours == proto).mean())
    assert agreement >= PARITY_MIN, agreement


@pytest.mark.examples
def test_example_r021_veg_frac_plausible(example_tif, veg_cfg: VegConfig) -> None:
    rgb = read_tile(example_tif("siret3_r021_c012"))
    masks = compute_tile_masks(rgb, np.ones(rgb.shape[:2], bool), veg_cfg)
    assert 0.02 <= masks.veg_frac <= 0.3
    assert masks.method == "lab_a"
