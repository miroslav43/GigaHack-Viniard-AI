"""Patch store (design 03 §3 N3, §5): offsets, patch cutting, manifest, holdout guard, synthetic end-to-end build."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import Source
from vineyard.geo.raster import write_mask_png
from vineyard.geo.tiling import CRS_EPSG, TILE_PX, px_to_utm, tile_ref
from vineyard.geo.vector_io import write_layer
from vineyard.nn.patch_store import (
    MANIFEST_NAME,
    HoldoutLeakError,
    PatchRecord,
    StoreManifest,
    TileEntry,
    build_store,
    cut_patches,
    load_manifest,
    patch_offsets,
    plan_store,
    store_stats,
    verify_store,
    write_manifest,
)
from vineyard.nn.pseudolabels import IGNORE, POSITIVE, PseudoLabelError
from vineyard.pipeline.cache import write_key
from vineyard.pipeline.tile_cache import stats_path, valid_mask_path, veg_mask_path

RUN_ID = "synth-run"
POS_TILE = "siret3_r020_c012"
EMPTY_TILE = "siret3_r023_c010"
HOLDOUT_TILE = "siret3_r021_c012"
ROW_VS = (500.0, 620.0, 740.0)
VEG_RGB = (70, 140, 60)
SOIL_RGB = (150, 118, 92)


# ------------------------------------------------------------------ pure helpers


def test_patch_store_is_torch_free() -> None:
    code = "import sys, vineyard.nn.patch_store; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_patch_offsets() -> None:
    assert patch_offsets(1024, 512, 512) == (0, 512)
    assert patch_offsets(1000, 512, 512) == (0, 488)  # the last patch is aligned to the end
    assert patch_offsets(512, 512, 256) == (0,)
    with pytest.raises(ValueError, match="patch"):
        patch_offsets(256, 512, 512)
    with pytest.raises(ValueError, match="stride"):
        patch_offsets(1024, 512, 0)


def test_cut_patches_gives_four_patches_with_offsets() -> None:
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    lbl = np.zeros((1024, 1024), dtype=np.uint8)
    img[600, 100] = 7  # inside the (x0=0, y0=512) patch at (88, 100)
    lbl[600, 100] = POSITIVE
    patches = cut_patches(img, lbl, 512, 512, 0.05)
    assert [(p.x0, p.y0) for p in patches] == [(0, 0), (512, 0), (0, 512), (512, 512)]
    third = patches[2]
    assert third.img.shape == (512, 512, 3) and third.lbl.shape == (512, 512)
    assert third.img[88, 100, 0] == 7 and third.lbl[88, 100] == POSITIVE
    assert third.counts.n_pos == 1


def test_cut_patches_skips_mostly_ignored_patches() -> None:
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    lbl = np.full((1024, 1024), IGNORE, dtype=np.uint8)
    lbl[:512, :512] = 0
    patches = cut_patches(img, lbl, 512, 512, 0.05)
    assert [(p.x0, p.y0) for p in patches] == [(0, 0)]


def test_cut_patches_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        cut_patches(np.zeros((8, 8, 3), np.uint8), np.zeros((4, 4), np.uint8), 4, 4, 0.0)


def _manifest(tile_ids: tuple[str, ...], holdout: tuple[str, ...] = (HOLDOUT_TILE,)) -> StoreManifest:
    tiles = tuple(TileEntry(tile_id=t, kind="positive", shard=i, n_patches=1, n_pos=1, n_neg=2, n_ignore=1,
                            n_small_px=0, n_clump_px=0, n_tree_px=0, n_ineligible_px=0, n_overhang_px=0,
                            seconds=0.1)
                  for i, t in enumerate(tile_ids))
    patches = tuple(PatchRecord(tile_id=t, shard=i, index=0, x0=0, y0=0, n_pos=1, n_neg=2, n_ignore=1)
                    for i, t in enumerate(tile_ids))
    return StoreManifest(key="abc", format=1, run_id="r", created_at="2026-09-26T00:00:00+03:00", patch_px=2,
                         gsd_m=0.05, holdout=holdout, tiles=tiles, patches=patches, selection={"source": "auto"},
                         params={"factor": 2}, failed=(), build_s=1.5)


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = _manifest((POS_TILE, EMPTY_TILE))
    write_manifest(tmp_path, m)
    assert load_manifest(tmp_path) == m
    assert json.loads((tmp_path / MANIFEST_NAME).read_text())["patches"][0]["tile_id"] == POS_TILE


def test_load_manifest_errors(tmp_path: Path) -> None:
    with pytest.raises(PseudoLabelError, match="manifest"):
        load_manifest(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(PseudoLabelError, match="manifest"):
        load_manifest(tmp_path)


def test_holdout_inside_a_store_raises_assertion_with_tile_id(tmp_path: Path) -> None:
    write_manifest(tmp_path, _manifest((POS_TILE, HOLDOUT_TILE)))
    with pytest.raises(AssertionError, match=HOLDOUT_TILE):
        verify_store(tmp_path, (HOLDOUT_TILE,))
    with pytest.raises(HoldoutLeakError):
        verify_store(tmp_path, (HOLDOUT_TILE,))


# ------------------------------------------------------------------ synthetic work dir


def _rows() -> gpd.GeoDataFrame:
    tile = tile_ref(POS_TILE)
    lines = [LineString(px_to_utm(tile, np.array([[-400.0, v], [1900.0, v]]))) for v in ROW_VS]
    n = len(lines)
    return gpd.GeoDataFrame(
        {"row_id": [f"V01-R{k + 1:03d}" for k in range(n)], "vineyard_id": "V01", "row_index": range(1, n + 1),
         "length_m": [g.length for g in lines], "extent_m": [g.length for g in lines], "n_pieces": 1,
         "tile_ids": POS_TILE, "angle_deg": 0.0, "spacing_prev_m": 3.0, "spacing_next_m": 3.0,
         "max_gap_m": np.nan, "n_gaps_ge5": None, "structure_any": None, "source": Source.MODEL.value,
         "run_id": RUN_ID, "model_version": "pipe@test", "confidence": 1.0, "qa_flags": ""},
        geometry=lines, crs=f"EPSG:{CRS_EPSG}",
    )


def _veg_mask() -> np.ndarray:
    veg = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    for v in ROW_VS:
        for u0 in range(100, 1900, 150):
            veg[int(v) - 8 : int(v) + 8, u0 : u0 + 80] = True
    veg[1400:1700, 200:600] = True  # grass field outside every corridor
    return veg


def _status() -> pd.DataFrame:
    return pd.DataFrame([
        dict(tile_id=POS_TILE, status="ok", has_vineyard=True, n_row_pieces=3, review_priority=2, issues="",
             veg_frac=0.1),
        dict(tile_id=HOLDOUT_TILE, status="ok", has_vineyard=True, n_row_pieces=20, review_priority=3, issues="",
             veg_frac=0.1),
        dict(tile_id=EMPTY_TILE, status="no_vineyard", has_vineyard=False, n_row_pieces=0, review_priority=1,
             issues="empty_tile_confirm", veg_frac=0.9),
    ])


def _write_tile(work: Path, tile_id: str, veg: np.ndarray, make_geotiff: Callable[..., Path]) -> None:
    rgb = np.where(veg[..., None], np.array(VEG_RGB, np.uint8), np.array(SOIL_RGB, np.uint8)).astype(np.uint8)
    make_geotiff(work / "tiles" / f"{tile_id}.tif", tile_id=tile_id, rgb=rgb, compress="DEFLATE")
    cache = work / "cache"
    valid = np.ones_like(veg)
    valid[:, :64] = tile_id != EMPTY_TILE  # the empty tile has a nodata strip
    for path, mask in ((veg_mask_path(cache, tile_id), veg & valid), (valid_mask_path(cache, tile_id), valid)):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_mask_png(path, mask)
    stats = stats_path(cache, tile_id)
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text("{}", encoding="utf-8")
    write_key(stats, f"prep-{tile_id}")


@pytest.fixture(scope="module")
def work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.conftest import write_geotiff

    root = tmp_path_factory.mktemp("nnwork")
    _write_tile(root, POS_TILE, _veg_mask(), write_geotiff)
    _write_tile(root, EMPTY_TILE, np.zeros((TILE_PX, TILE_PX), dtype=bool), write_geotiff)
    ids = [POS_TILE, EMPTY_TILE, HOLDOUT_TILE]
    pd.DataFrame({"tile_id": ids, "sha256": [f"{k}" * 64 for k in range(3)]}).to_parquet(root / "tile_index.parquet")
    layers = root / "runs" / RUN_ID / "layers"
    layers.mkdir(parents=True)
    write_layer(_rows(), "rows", layers / "rows.parquet")
    _status().to_parquet(layers / "tile_status.parquet")
    return root


def _cfg(work: Path, *extra: str) -> AppConfig:
    return load_config(overrides=[f"paths.work_dir={work}", "nn.pseudolabels.min_train_tiles=1",
                                  "nn.pseudolabels.max_empty_tile_frac=0.5", *extra])


def test_plan_selects_tiles_and_excludes_holdout(work: Path) -> None:
    plan = plan_store(_cfg(work), RUN_ID)
    assert plan.selection.positives == (POS_TILE,) and plan.selection.empties == (EMPTY_TILE,)
    assert HOLDOUT_TILE in plan.selection.holdout_removed
    assert [j.tile_id for j in plan.jobs] == [POS_TILE, EMPTY_TILE]
    assert plan.store_dir == work / "nn" / "stores" / plan.key
    assert len(plan.key) == 16


def test_plan_key_depends_on_config(work: Path) -> None:
    a = plan_store(_cfg(work), RUN_ID)
    b = plan_store(_cfg(work, "nn.pseudolabels.ignore_band_px=2"), RUN_ID)
    assert a.key != b.key
    assert plan_store(_cfg(work), RUN_ID).key == a.key


def test_plan_missing_run_raises(work: Path) -> None:
    with pytest.raises(PseudoLabelError, match="tile_status"):
        plan_store(_cfg(work), "no-such-run")


def test_build_store_end_to_end(work: Path) -> None:
    plan = plan_store(_cfg(work), RUN_ID)
    manifest = build_store(plan, workers=1)
    store = plan.store_dir
    assert (store / MANIFEST_NAME).is_file()
    assert {t.tile_id for t in manifest.tiles} == {POS_TILE, EMPTY_TILE}
    assert HOLDOUT_TILE not in {p.tile_id for p in manifest.patches}
    pos = next(t for t in manifest.tiles if t.tile_id == POS_TILE)
    assert pos.n_patches == 4 and pos.n_pos > 0
    img = np.load(store / "img_0000.npy", mmap_mode="r")
    lbl = np.load(store / "lbl_0000.npy", mmap_mode="r")
    assert img.shape == (4, 512, 512, 3) and img.dtype == np.uint8
    assert lbl.shape == (4, 512, 512) and lbl.dtype == np.uint8
    assert set(np.unique(lbl)) <= {0, 1, 255}
    first = lbl[0]  # patch (0, 0): rows at v = 500 / 620 / 740 native -> 250 / 310 / 370 label px
    assert (first[248:253, 60:80] == POSITIVE).any()
    # grass field (native rows 1400-1699, cols 200-599) -> patch (x0=0, y0=512) rows 188-337, cols 100-299
    assert (lbl[2][190:335, 105:295] == 0).all()
    assert (img[2][190:335, 105:295, 1] > img[2][190:335, 105:295, 0]).all()  # it is green in the image
    empty = next(t for t in manifest.tiles if t.tile_id == EMPTY_TILE)
    assert empty.n_pos == 0 and empty.n_ignore > 0  # nodata strip ignored


def test_build_store_second_call_is_cached(work: Path) -> None:
    plan = plan_store(_cfg(work), RUN_ID)
    first = build_store(plan, workers=1)
    second = build_store(plan, workers=1)
    assert second.created_at == first.created_at


def test_store_stats(work: Path) -> None:
    plan = plan_store(_cfg(work), RUN_ID)
    manifest = build_store(plan, workers=1)
    stats = store_stats(manifest, plan.store_dir)
    assert stats["n_patches"] == len(manifest.patches) == 8
    assert stats["n_tiles"] == 2 and stats["bytes"] > 0
    assert stats["pos_frac"] + stats["neg_frac"] + stats["ignore_frac"] == pytest.approx(1.0)
    assert 0 < stats["pos_frac_of_labeled"] < 1


def test_build_store_failure_raises_or_is_recorded(work: Path, tmp_path: Path) -> None:
    plan = plan_store(_cfg(work, "nn.pseudolabels.ignore_band_px=1"), RUN_ID)
    bad_jobs = tuple(replace(j, rows_path=tmp_path / "missing.parquet") if j.tile_id == POS_TILE else j
                     for j in plan.jobs)
    bad = replace(plan, jobs=bad_jobs, store_dir=tmp_path / "store")
    with pytest.raises(PseudoLabelError, match=POS_TILE):
        build_store(bad, workers=1)
    assert not (tmp_path / "store").exists()
    kept = build_store(replace(bad, allow_failures=True), workers=1)
    assert kept.failed == (POS_TILE,) and {t.tile_id for t in kept.tiles} == {EMPTY_TILE}


def test_build_store_refuses_holdout_job(work: Path, tmp_path: Path) -> None:
    plan = plan_store(_cfg(work), RUN_ID)
    leak = replace(plan.jobs[0], tile_id=HOLDOUT_TILE)
    with pytest.raises(HoldoutLeakError, match=HOLDOUT_TILE):
        build_store(replace(plan, jobs=(leak,), store_dir=tmp_path / "leak"), workers=1)


def test_manifest_without_overhang_field_still_loads(tmp_path: Path) -> None:
    m = _manifest((POS_TILE,))
    raw = m.to_dict()
    for tile in raw["tiles"]:
        del tile["n_overhang_px"]
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(raw), encoding="utf-8")
    assert load_manifest(tmp_path).tiles[0].n_overhang_px == 0
