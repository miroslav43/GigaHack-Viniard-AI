import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vineyard.contracts.enums import FusionVariant
from vineyard.errors import SchemaError
from vineyard.nn.fusion import FusedMask, fuse_masks
from vineyard.nn.probs import (
    CHANNELS,
    PROB_PX,
    ProbStats,
    corridor_prob_stats,
    load_canopy_prob,
    prob_png_path,
    read_prob_png,
    upsample_prob,
    write_prob_png,
)

TILE = "siret3_r021_c012"

# 4 pixels covering every (veg, nn) combination; corridor excludes the last column pair.
VEG = np.array([[True, True, False, False], [True, True, False, False]])
PROB = np.array([[0.9, 0.1, 0.9, 0.1], [0.9, 0.1, 0.9, 0.1]], dtype=np.float32)
CORR = np.array([[True, True, True, True], [False, False, False, False]])


def test_import_hygiene_subprocess() -> None:
    import subprocess

    code = "import sys, vineyard.nn.probs, vineyard.nn.fusion; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_constants() -> None:
    assert PROB_PX == 1024
    assert CHANNELS == ("canopy_prob", "axis_prob")


def test_prob_png_path(tmp_path: Path) -> None:
    p = prob_png_path(tmp_path, "vine-unet@v1+1a2b3c4d", "canopy_prob", TILE)
    assert p == tmp_path / "nn" / "vine-unet@v1+1a2b3c4d" / "canopy_prob" / f"{TILE}.png"
    with pytest.raises(SchemaError):
        prob_png_path(tmp_path, "v", "bogus", TILE)


def test_write_read_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    prob = rng.random((PROB_PX, PROB_PX), dtype=np.float32)
    path = tmp_path / "a" / "x.png"
    write_prob_png(path, prob)
    back = read_prob_png(path)
    assert back.dtype == np.uint8 and back.shape == (PROB_PX, PROB_PX)
    assert np.array_equal(back, np.rint(prob * 255).astype(np.uint8))
    assert list(path.parent.iterdir()) == [path]
    assert load_canopy_prob(path) is not None
    assert load_canopy_prob(tmp_path / "missing.png") is None


def test_write_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(SchemaError):
        write_prob_png(tmp_path / "x.png", np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(SchemaError):
        write_prob_png(tmp_path / "x.png", np.full((PROB_PX, PROB_PX), 1.5, dtype=np.float32))


def test_read_rejects_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    cv2.imwrite(str(path), np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(SchemaError):
        read_prob_png(path)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises(SchemaError):
        read_prob_png(bad)


def test_upsample_alignment() -> None:
    assert upsample_prob(None, 2048) is None
    prob = np.zeros((PROB_PX, PROB_PX), dtype=np.uint8)
    prob[10, 20] = 255
    up = upsample_prob(prob, 2048)
    assert up is not None and up.dtype == np.float32 and up.shape == (2048, 2048)
    # 1024 px k covers 2048 px [2k, 2k+2): the peak spreads symmetrically over rows 20..21, cols 40..41
    assert up[20, 40] == pytest.approx(up[21, 41])
    assert up[20, 40] == pytest.approx(0.5625)
    assert up.max() <= 1.0
    flat = upsample_prob(np.full((PROB_PX, PROB_PX), 255, dtype=np.uint8), 2048)
    assert flat is not None and np.allclose(flat, 1.0)
    with pytest.raises(SchemaError):
        upsample_prob(np.zeros((10, 12), dtype=np.uint8), 2048)
    with pytest.raises(SchemaError):
        upsample_prob(np.zeros((10, 10), dtype=np.float32), 2048)  # type: ignore[arg-type]


def test_corridor_prob_stats() -> None:
    prob = np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float32)
    corr = np.array([[True, True], [True, False]])
    stats = corridor_prob_stats(prob, corr)
    assert isinstance(stats, ProbStats)
    assert stats.n_px == 3
    assert stats.mean == pytest.approx(0.5)
    assert stats.p95 == pytest.approx(np.percentile([0.0, 0.5, 1.0], 95))
    empty = corridor_prob_stats(prob, np.zeros_like(corr))
    assert empty.n_px == 0 and np.isnan(empty.mean) and np.isnan(empty.p95)
    with pytest.raises(SchemaError):
        corridor_prob_stats(prob, np.ones((3, 3), dtype=bool))


@pytest.mark.parametrize(
    ("variant", "expected_row0"),
    [
        (FusionVariant.A, [True, True, False, False]),
        (FusionVariant.B, [True, False, True, False]),
        (FusionVariant.C, [True, False, False, False]),
        (FusionVariant.D, [True, True, True, False]),
    ],
)
def test_truth_tables_with_corridor(variant: FusionVariant, expected_row0: list[bool]) -> None:
    fused = fuse_masks(VEG, PROB, CORR, variant, 0.5)
    assert isinstance(fused, FusedMask)
    assert fused.variant_used is variant and not fused.fell_back
    assert fused.mask[0].tolist() == expected_row0
    assert not fused.mask[1].any()


def test_variant_e_ignores_corridor() -> None:
    fused = fuse_masks(VEG, PROB, CORR, FusionVariant.E, 0.5)
    assert fused.mask.tolist() == [[True, False, True, False], [True, False, True, False]]
    assert fuse_masks(VEG, PROB, None, FusionVariant.E, 0.5).mask.tolist() == fused.mask.tolist()


def test_threshold_is_inclusive() -> None:
    prob = np.full((1, 2), 0.5, dtype=np.float32)
    veg = np.zeros((1, 2), dtype=bool)
    assert fuse_masks(veg, prob, None, FusionVariant.B, 0.5).mask.all()


@pytest.mark.parametrize("variant", [FusionVariant.B, FusionVariant.C, FusionVariant.D, FusionVariant.E])
def test_fallback_to_classic_without_prob(variant: FusionVariant) -> None:
    fused = fuse_masks(VEG, None, CORR, variant, 0.5)
    assert fused.variant_used is FusionVariant.A and fused.fell_back
    assert np.array_equal(fused.mask, VEG & CORR)


def test_variant_a_without_prob_is_not_a_fallback() -> None:
    fused = fuse_masks(VEG, None, CORR, FusionVariant.A, 0.5)
    assert not fused.fell_back


def test_corridor_none_and_label_corridor() -> None:
    assert np.array_equal(fuse_masks(VEG, None, None, FusionVariant.A, 0.5).mask, VEG)
    labels = CORR.astype(np.int32) * 7
    assert np.array_equal(fuse_masks(VEG, PROB, labels, FusionVariant.A, 0.5).mask, VEG & CORR)


def test_result_is_read_only_and_inputs_untouched() -> None:
    veg = VEG.copy()
    prob = PROB.copy()
    fused = fuse_masks(veg, prob, CORR, FusionVariant.D, 0.5)
    assert not fused.mask.flags.writeable
    assert np.array_equal(veg, VEG) and np.array_equal(prob, PROB)


def test_fuse_validation() -> None:
    with pytest.raises(SchemaError):
        fuse_masks(VEG, PROB[:, :2], CORR, FusionVariant.B, 0.5)
    with pytest.raises(SchemaError):
        fuse_masks(VEG, PROB, CORR[:, :2], FusionVariant.A, 0.5)
    with pytest.raises(SchemaError):
        fuse_masks(VEG, PROB, CORR, FusionVariant.B, 1.5)
    with pytest.raises(SchemaError):
        fuse_masks(VEG.astype(np.uint8), PROB, CORR, FusionVariant.A, 0.5)
    assert fuse_masks(VEG, PROB, CORR, "C", 0.5).variant_used is FusionVariant.C  # type: ignore[arg-type]
