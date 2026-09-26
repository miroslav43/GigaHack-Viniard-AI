"""PatchDataset, deterministic augmentation, EpochSampler and loader (design 03 §3 N3, §5)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from vineyard.config import load_config
from vineyard.nn.dataset import (
    NO_AUGMENT,
    AugmentParams,
    EpochSampler,
    PatchDataset,
    augment_sample,
    build_loader,
    colour_jitter,
)
from vineyard.nn.patch_store import (
    IMG_NAME,
    LBL_NAME,
    HoldoutLeakError,
    PatchRecord,
    StoreManifest,
    TileEntry,
    read_patch,
    write_manifest,
)
from vineyard.nn.pseudolabels import PseudoLabelError

P = 16
TILES = ("siret3_r030_c001", "siret3_r030_c002")
MARK = (1, 3)  # asymmetric under every flip / rotation
FULL_AUG = AugmentParams(flips=True, rot90=True, colour_jitter=0.1)


def _write_store(root: Path) -> Path:
    tiles, patches = [], []
    rng = np.random.default_rng(7)
    for shard, tile_id in enumerate(TILES):
        img = rng.integers(40, 200, size=(2, P, P, 3), dtype=np.uint8)
        lbl = np.zeros((2, P, P), dtype=np.uint8)
        img[:, MARK[0], MARK[1]] = (255, 0, 0)
        img[:, :, :, 0] = np.minimum(img[:, :, :, 0], 250)
        img[:, MARK[0], MARK[1], 0] = 255
        lbl[:, MARK[0], MARK[1]] = 1
        lbl[:, -1, :] = 255
        np.save(root / IMG_NAME.format(shard), img)
        np.save(root / LBL_NAME.format(shard), lbl)
        tiles.append(TileEntry(tile_id=tile_id, kind="positive", shard=shard, n_patches=2, n_pos=2, n_neg=0,
                               n_ignore=0, n_small_px=0, n_clump_px=0, n_tree_px=0, n_ineligible_px=0,
                               n_overhang_px=0, seconds=0.0))
        patches += [PatchRecord(tile_id=tile_id, shard=shard, index=i, x0=0, y0=0, n_pos=1, n_neg=P * P - P - 1,
                                n_ignore=P) for i in range(2)]
    write_manifest(root, StoreManifest(key="k", format=1, run_id="r", created_at="t", patch_px=P, gsd_m=0.05,
                                       holdout=(), tiles=tuple(tiles), patches=tuple(patches), selection={},
                                       params={}, failed=(), build_s=0.0))
    return root


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _write_store(tmp_path)


def _full(store: Path, seed: int = 0, augment: AugmentParams = FULL_AUG) -> PatchDataset:
    return PatchDataset(store, seed=seed, augment=augment)


def test_read_patch(store: Path) -> None:
    img, lbl = read_patch(store, PatchRecord(TILES[1], 1, 1, 0, 0, 1, 0, 0))
    assert img.shape == (P, P, 3) and lbl.shape == (P, P) and img.dtype == np.uint8
    assert tuple(img[MARK]) == (255, 0, 0) and lbl[MARK] == 1
    with pytest.raises(PseudoLabelError, match="index"):
        read_patch(store, PatchRecord(TILES[1], 1, 5, 0, 0, 1, 0, 0))
    with pytest.raises(PseudoLabelError, match="shard"):
        read_patch(store, PatchRecord(TILES[1], 9, 0, 0, 0, 1, 0, 0))


def test_length_dtypes_and_ranges(store: Path) -> None:
    ds = _full(store)
    assert len(ds) == 4
    img, lbl = ds[(0, 2)]
    assert img.dtype == torch.float32 and lbl.dtype == torch.int64
    assert img.shape == (3, P, P) and lbl.shape == (P, P)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert set(torch.unique(lbl).tolist()) <= {0, 1, 255}


def test_same_seed_epoch_idx_gives_identical_samples(store: Path) -> None:
    a, b = _full(store), _full(store)
    for key in ((0, 1), (3, 1), (5, 3)):
        assert torch.equal(a[key][0], b[key][0]) and torch.equal(a[key][1], b[key][1])
    assert not torch.equal(a[(0, 1)][0], _full(store, seed=1)[(0, 1)][0])


def test_geometric_augmentation_moves_image_and_label_together(store: Path) -> None:
    ds = _full(store, augment=AugmentParams(True, True, 0.0))
    positions = set()
    for epoch in range(64):
        img, lbl = ds[(epoch, 0)]
        red = np.unravel_index(int(torch.argmax(img[0] - img[1] - img[2])), (P, P))
        lab = np.unravel_index(int(torch.argmax((lbl == 1).int())), (P, P))
        assert tuple(int(v) for v in red) == tuple(int(v) for v in lab)
        positions.add(lab)
    assert len(positions) == 8  # the whole dihedral group shows up


def test_jitter_never_changes_the_label(store: Path) -> None:
    plain = PatchDataset(store, seed=0, augment=NO_AUGMENT)
    jitter = PatchDataset(store, seed=0, augment=AugmentParams(False, False, 0.1))
    changed = 0
    for epoch in range(5):
        img0, lbl0 = plain[(epoch, 1)]
        img1, lbl1 = jitter[(epoch, 1)]
        assert torch.equal(lbl0, lbl1)
        changed += int(not torch.allclose(img0, img1))
    assert changed == 5


def test_no_augment_is_the_stored_patch(store: Path) -> None:
    ds = PatchDataset(store, seed=0, augment=NO_AUGMENT)
    img, lbl = ds[3]
    raw_img, raw_lbl = read_patch(store, PatchRecord(TILES[1], 1, 1, 0, 0, 0, 0, 0))
    assert torch.allclose(img, torch.from_numpy(raw_img.transpose(2, 0, 1).astype(np.float32) / 255.0))
    assert torch.equal(lbl, torch.from_numpy(raw_lbl.astype(np.int64)))


def test_colour_jitter_identity_and_range() -> None:
    x = np.random.default_rng(0).random((4, 4, 3), dtype=np.float32)
    assert np.allclose(colour_jitter(x, 1.0, 1.0, 1.0), x, atol=1e-6)
    out = colour_jitter(x, 1.1, 1.1, 1.1)
    assert out.dtype == np.float32 and out.min() >= 0.0 and out.max() <= 1.0
    assert np.allclose(x, np.random.default_rng(0).random((4, 4, 3), dtype=np.float32))  # input untouched


def test_augment_sample_validates_inputs() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="shape"):
        augment_sample(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5), np.uint8), rng, NO_AUGMENT)
    with pytest.raises(ValueError, match="colour_jitter"):
        AugmentParams(True, True, 1.5)


def test_augment_params_from_config() -> None:
    p = AugmentParams.from_config(load_config().nn.train)
    assert p.flips and p.rot90 and p.colour_jitter == pytest.approx(0.1)


def test_subset_keeps_manifest_level_randomness(store: Path) -> None:
    full = _full(store)
    sub = PatchDataset(store, seed=0, augment=AugmentParams(True, True, 0.1), indices=(1, 3))
    assert len(sub) == 2
    assert torch.equal(sub[(2, 1)][0], full[(2, 3)][0])
    with pytest.raises(IndexError):
        _ = sub[5]
    with pytest.raises(PseudoLabelError, match="indices"):
        PatchDataset(store, seed=0, augment=NO_AUGMENT, indices=(0, 9))


def test_holdout_is_checked_on_open(store: Path) -> None:
    with pytest.raises(HoldoutLeakError, match=TILES[0]):
        PatchDataset(store, seed=0, augment=NO_AUGMENT, holdout=(TILES[0],))


def test_dataset_pickles(store: Path) -> None:
    ds = _full(store)
    _ = ds[(1, 0)]
    # our own object, round-tripped in-process: DataLoader spawn workers receive the dataset this way
    clone = pickle.loads(pickle.dumps(ds))
    assert torch.equal(clone[(1, 0)][0], ds[(1, 0)][0])


def test_epoch_sampler_is_deterministic_per_epoch() -> None:
    s = EpochSampler(20, seed=3)
    first = list(s)
    assert len(s) == 20 and all(e == 0 for e, _ in first)
    assert sorted(i for _, i in first) == list(range(20))
    assert list(EpochSampler(20, seed=3)) == first
    s.set_epoch(1)
    second = list(s)
    assert all(e == 1 for e, _ in second) and [i for _, i in second] != [i for _, i in first]
    ordered = EpochSampler(4, seed=0, shuffle=False)
    assert list(ordered) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    with pytest.raises(ValueError, match="epoch"):
        s.set_epoch(-1)


def test_loader_batches(store: Path) -> None:
    ds = _full(store)
    loader = build_loader(ds, EpochSampler(len(ds), seed=0), batch_size=2, num_workers=0)
    batches = list(loader)
    assert len(batches) == 2
    img, lbl = batches[0]
    assert img.shape == (2, 3, P, P) and lbl.shape == (2, P, P) and lbl.dtype == torch.int64


@pytest.mark.slow
def test_loader_with_spawned_workers(store: Path) -> None:
    ds = _full(store)
    sampler = EpochSampler(len(ds), seed=0)
    loader = build_loader(ds, sampler, batch_size=2, num_workers=2)
    first = [b[0] for b in loader]
    inline = [b[0] for b in build_loader(ds, EpochSampler(len(ds), seed=0), batch_size=2, num_workers=0)]
    assert all(torch.equal(a, b) for a, b in zip(first, inline, strict=True))
    sampler.set_epoch(1)
    assert len(list(loader)) == 2  # persistent workers see the new epoch
