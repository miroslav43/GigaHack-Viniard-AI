"""Stage `rows_detect` on a synthetic work dir (inline, workers=1): cache, key, forbidden mask, NN prob."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import box

from tests.helpers.synth import striped_mask
from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import Source, TileStatus
from vineyard.contracts.schemas import empty_layer
from vineyard.errors import StageError
from vineyard.geo.raster import write_mask_png, write_u8_png
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_box, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.nn.probs import PROB_PX, prob_png_path, write_prob_png
from vineyard.perception.types import VIS_OK, TileStats
from vineyard.pipeline.cache import read_key, write_key
from vineyard.pipeline.context import RunContext, ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.runner import TileTask
from vineyard.pipeline.stages import rows_detect as stage
from vineyard.pipeline.tile_cache import (
    stats_path,
    valid_mask_path,
    veg_mask_path,
    vis_path,
    write_tile_stats,
)

TILE_ID = "siret3_r021_c012"
EMPTY_TILE = "siret3_r021_c013"
TILE = tile_ref(TILE_ID)
N_ROWS = 20  # horizontal stripes every 100 px (2.5 m) -> rows at v = 0, 100, ..., 2000 (v=0 is a half stripe)


def _tile_index(tile_ids: list[str]) -> gpd.GeoDataFrame:
    refs = [tile_ref(t) for t in tile_ids]
    return gpd.GeoDataFrame(
        {"tile_id": tile_ids, "file_name": [f"{t}.tif" for t in tile_ids], "grid_row": [r.grid_row for r in refs],
         "grid_col": [r.grid_col for r in refs], "x0": [r.x0 for r in refs], "y0": [r.y0 for r in refs],
         "x1": [r.bounds[2] for r in refs], "y1": [r.bounds[1] for r in refs], "gsd_m": GSD_M,
         "width_px": TILE_PX, "height_px": TILE_PX, "src_zip": "z.zip", "path": "p", "sha256": "0" * 64,
         "file_size": 1, "nodata_frac": 0.0, "valid_area_m2": 2621.44},
        geometry=[tile_box(r) for r in refs], crs=f"EPSG:{CRS_EPSG}",
    )


def _tile_prep(ctx: RunContext, tile_id: str, veg: np.ndarray, status: TileStatus) -> None:
    cache = ctx.paths.cache_dir
    for path in (veg_mask_path(cache, tile_id), valid_mask_path(cache, tile_id), vis_path(cache, tile_id),
                 stats_path(cache, tile_id)):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_mask_png(veg_mask_path(cache, tile_id), veg)
    write_mask_png(valid_mask_path(cache, tile_id), np.ones_like(veg))
    write_u8_png(vis_path(cache, tile_id), np.full(veg.shape, VIS_OK, dtype=np.uint8))
    valid_frac = 1.0 if status == TileStatus.OK else 0.0
    write_tile_stats(cache, TileStats(tile_id, valid_frac, 1.0 - valid_frac, float(veg.mean()), status, 4.0, "lab_a"))
    write_key(stats_path(cache, tile_id), f"prep-{tile_id}")


def _forbidden(geoms: list[shapely.Geometry]) -> gpd.GeoDataFrame:
    if not geoms:
        return empty_layer("in_forbidden")
    return gpd.GeoDataFrame({"fid": list(range(1, len(geoms) + 1)), "type": "forbidden", "name": "f",
                             "source": "input"}, geometry=geoms, crs=f"EPSG:{CRS_EPSG}")


def _context(cfg: AppConfig) -> RunContext:
    context = new_run_context(cfg, source=Source.MODEL, run_id="test-run", workers=1)
    ensure_run_dirs(context.paths)
    return context


@pytest.fixture
def ctx(tmp_work: Path) -> RunContext:
    context = _context(load_config())
    tiles = [TILE_ID, EMPTY_TILE]
    write_layer(_tile_index(tiles), "tile_index", context.paths.tile_index)
    valid = gpd.GeoDataFrame({"tile_id": tiles, "valid_frac": [1.0, 0.0]},
                             geometry=[tile_box(TILE), shapely.Polygon()], crs=f"EPSG:{CRS_EPSG}")
    write_layer(valid, "tile_valid", context.paths.tile_valid)
    write_layer(_forbidden([]), "in_forbidden", context.paths.static_layers_dir / stage.FORBIDDEN_FILE)
    veg = striped_mask((TILE_PX, TILE_PX), angle_deg=0.0, spacing_px=100.0, width_px=16.0)
    _tile_prep(context, TILE_ID, veg, TileStatus.OK)
    _tile_prep(context, EMPTY_TILE, np.zeros_like(veg), TileStatus.EMPTY_NODATA)
    return context


def _accepted(ctx: RunContext, tile_id: str = TILE_ID) -> gpd.GeoDataFrame:
    frame = read_layer(stage.candidates_path(ctx, tile_id), stage.LAYER_NAME)
    return frame[frame["rejected_reason"].isna()]


def test_registry_loads_the_stage() -> None:
    assert load_stage("rows_detect") is stage.STAGE
    assert stage.STAGE.scope == "tile" and stage.STAGE.requires == ("tile_prep",)


def test_stage_writes_candidates_summary_and_reuses_cache(ctx: RunContext) -> None:
    res = stage.run(ctx)
    assert res.n_items == 2 and res.n_failed == 0 and res.n_cached == 0
    out = stage.candidates_path(ctx, TILE_ID)
    assert out == ctx.paths.cache_dir / "rows_detect" / f"{TILE_ID}.parquet"
    kept = _accepted(ctx)
    assert len(kept) >= N_ROWS
    assert (kept["source"] == "model").all() and (kept["run_id"] == "test-run").all()
    assert kept["model_version"].str.endswith(";nn=none;waste=none").all()
    summary = json.loads(ctx.paths.tile_cache("rows_detect", TILE_ID, "json").read_text())
    assert summary["status_hint"] == "ok" and summary["n_accepted"] == len(kept)
    empty = read_layer(stage.candidates_path(ctx, EMPTY_TILE), stage.LAYER_NAME)
    assert empty.empty
    key = read_key(out)
    assert key is not None
    again = stage.run(ctx)
    assert again.n_cached == 2 and read_key(out) == key


def test_forbidden_polygon_masks_rows_and_changes_only_that_key(ctx: RunContext) -> None:
    stage.run(ctx)
    key = read_key(stage.candidates_path(ctx, TILE_ID))
    empty_key = read_key(stage.candidates_path(ctx, EMPTY_TILE))
    band_px = np.array([[0.0, 950.0], [float(TILE_PX), 1250.0]])  # rows v=1000, 1100, 1200 forbidden
    xy = px_to_utm(TILE, band_px)
    polygon = box(xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
    write_layer(_forbidden([polygon]), "in_forbidden", ctx.paths.static_layers_dir / stage.FORBIDDEN_FILE)
    res = stage.run(ctx)
    assert res.n_cached == 1
    assert read_key(stage.candidates_path(ctx, TILE_ID)) != key
    assert read_key(stage.candidates_path(ctx, EMPTY_TILE)) == empty_key
    kept = _accepted(ctx)
    assert not any(kept.geometry.intersects(polygon.buffer(-0.5)))


def test_nn_prob_feeds_the_vine_score(tmp_work: Path) -> None:
    base = load_config(overrides=["nn.enabled=true", "nn.weights_sha256=" + "a" * 64])
    context = _context(base)
    write_layer(_tile_index([TILE_ID]), "tile_index", context.paths.tile_index)
    valid = gpd.GeoDataFrame({"tile_id": [TILE_ID], "valid_frac": [1.0]}, geometry=[tile_box(TILE)],
                             crs=f"EPSG:{CRS_EPSG}")
    write_layer(valid, "tile_valid", context.paths.tile_valid)
    write_layer(_forbidden([]), "in_forbidden", context.paths.static_layers_dir / stage.FORBIDDEN_FILE)
    veg = striped_mask((TILE_PX, TILE_PX), angle_deg=0.0, spacing_px=100.0, width_px=16.0)
    _tile_prep(context, TILE_ID, veg, TileStatus.OK)
    prob_path = prob_png_path(context.paths.cache_dir, base.nn.version, stage.PROB_CHANNEL, TILE_ID)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    write_prob_png(prob_path, np.full((PROB_PX, PROB_PX), 0.8, dtype=np.float32))
    res = stage.run(context)
    assert res.n_failed == 0
    kept = _accepted(context)
    assert len(kept) >= N_ROWS
    assert np.allclose(kept["vine_score"].to_numpy(), 0.8, atol=0.01)
    assert kept["model_version"].str.contains(f"nn={base.nn.version}").all()


def test_missing_prob_file_is_tolerated(ctx: RunContext) -> None:
    task = stage._make_task(ctx, TILE_ID)
    task_with_prob = replace(task, inputs={**task.inputs, "prob": ctx.paths.cache_dir / "nn" / "none.png"})
    out = stage.detect_tile(task_with_prob)
    assert out["status_hint"] == "ok"


def test_stage_fails_loudly_without_inputs(ctx: RunContext) -> None:
    (ctx.paths.static_layers_dir / stage.FORBIDDEN_FILE).unlink()
    with pytest.raises(StageError, match="in_forbidden"):
        stage.run(ctx)
    ctx.paths.tile_valid.unlink()
    with pytest.raises(StageError, match="tile_valid"):
        stage.run(ctx)


def test_forbidden_is_optional_when_masking_is_off(tmp_work: Path) -> None:
    context = _context(load_config(overrides=["rows.detect.mask_forbidden=false"]))
    write_layer(_tile_index([TILE_ID]), "tile_index", context.paths.tile_index)
    valid = gpd.GeoDataFrame({"tile_id": [TILE_ID], "valid_frac": [1.0]}, geometry=[tile_box(TILE)],
                             crs=f"EPSG:{CRS_EPSG}")
    write_layer(valid, "tile_valid", context.paths.tile_valid)
    veg = striped_mask((TILE_PX, TILE_PX), angle_deg=0.0, spacing_px=100.0, width_px=16.0)
    _tile_prep(context, TILE_ID, veg, TileStatus.OK)
    assert stage.run(context).n_failed == 0


def test_bad_task_cfg_is_a_stage_error(ctx: RunContext) -> None:
    task = stage._make_task(ctx, TILE_ID)
    bad = TileTask(tile_id=TILE_ID, tif_path=task.tif_path, key="", cfg=ctx.cfg, inputs=task.inputs,
                   outputs=task.outputs)
    with pytest.raises(StageError, match="RowsDetectTaskCfg"):
        stage.detect_tile(bad)


def test_helpers_clip_mask_and_digest(ctx: RunContext) -> None:
    tile_valid = read_layer(ctx.paths.tile_valid, stage.TILE_VALID_LAYER)
    clip = stage.tile_clip_px(tile_valid, TILE)
    assert clip.symmetric_difference(box(0.0, 0.0, float(TILE_PX), float(TILE_PX))).area < 1e-3
    assert stage.tile_clip_px(tile_valid, tile_ref(EMPTY_TILE)).is_empty
    with pytest.raises(StageError, match="missing from tile_valid"):
        stage.tile_clip_px(tile_valid, tile_ref("siret3_r005_c000"))
    assert not stage.forbidden_mask([], TILE).any()
    square = box(*px_to_utm(TILE, np.array([[0.0, 2048.0]]))[0], *px_to_utm(TILE, np.array([[1024.0, 1024.0]]))[0])
    mask = stage.forbidden_mask([square], TILE)
    assert mask[1500, 500] and not mask[500, 1500]
    assert stage.geometries_digest([]) == stage.NO_FORBIDDEN_DIGEST
    assert stage.geometries_digest([square]) == stage.geometries_digest([shapely.from_wkb(square.wkb)])
    far = box(0.0, 0.0, 1.0, 1.0)
    minx, miny, maxx, maxy = tile_box(TILE).bounds
    touching = box(maxx, miny, maxx + 5.0, maxy)
    assert stage.forbidden_in_tile(_forbidden([far, square, touching]), TILE) == [square]
