"""Stage `tile_prep` (tile): per tile read -> valid mask (contract §1.7) -> valid eroded by
nodata.veg_erode_px -> perception.vegmask.compute_tile_masks -> cache/{valid,veg,vis,stats} (+ .key),
then the static work/layers/tile_valid.parquet is rebuilt from the per-tile valid polygons.

Consumers read the cache only through vineyard.pipeline.tile_cache accessors.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Final

import cv2
import geopandas as gpd
import numpy as np
import shapely

from vineyard.config import AppConfig
from vineyard.contracts.enums import TileStatus
from vineyard.contracts.schemas import empty_layer
from vineyard.errors import StageError
from vineyard.geo.raster import read_tile, valid_mask, valid_polygon, write_mask_png, write_u8_png
from vineyard.geo.tiling import CRS_EPSG, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.types import BoolMask, TileStats
from vineyard.perception.vegmask import compute_tile_masks
from vineyard.pipeline.atomic import atomic_write_bytes, atomic_write_json
from vineyard.pipeline.cache import read_key
from vineyard.pipeline.context import RunContext
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult, TileTask, run_tile_stage
from vineyard.pipeline.tile_cache import (
    load_tile_stats,
    stats_path,
    valid_mask_path,
    veg_mask_path,
    vis_path,
)
from vineyard.pipeline.tile_index import indexed_tile_ids, tile_path

NAME: Final = "tile_prep"
VERSION: Final = "1"
CFG_KEYS: Final = ("grid", "nodata", "veg")
TILE_VALID_LAYER: Final = "tile_valid"
TILE_VALID_SUBDIR: Final = "tile_valid"
WKB_EXT: Final = ".wkb"
NODATA_REPORT_FRAC: Final = 0.01  # tiles above 1% nodata are listed in the stage summary
NODATA_TOP_N: Final = 3
EVENT_NODATA: Final = "tile_prep.nodata"

_log = get_logger("pipeline.stages.tile_prep")


def tile_valid_wkb_path(cache_dir: Path, tile_id: str) -> Path:
    return Path(cache_dir) / TILE_VALID_SUBDIR / f"{tile_id}{WKB_EXT}"


def tile_prep_outputs(cache_dir: Path, tile_id: str) -> dict[str, Path]:
    """Every artifact tile_prep writes for a tile (keys are written for all of them)."""
    return {"valid": valid_mask_path(cache_dir, tile_id), "veg": veg_mask_path(cache_dir, tile_id),
            "vis": vis_path(cache_dir, tile_id), "valid_wkb": tile_valid_wkb_path(cache_dir, tile_id),
            "stats": stats_path(cache_dir, tile_id)}


# ------------------------------------------------------------------ worker (top-level, picklable)


def _erode(mask: BoolMask, radius_px: int) -> BoolMask:
    if radius_px <= 0:
        return mask.copy()
    size = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask.astype(np.uint8), kernel) > 0  # default border: the tile edge does not erode


def _valid(rgb: np.ndarray, cfg: AppConfig) -> BoolMask:
    nd = cfg.nodata
    min_area_px = nd.min_area_m2 / (cfg.grid.gsd_m * cfg.grid.gsd_m)
    return valid_mask(rgb, max_rgb=nd.max_rgb, min_area_px=min_area_px, close_px=nd.close_px, dilate_px=nd.dilate_px)


def prep_tile(task: TileTask) -> Mapping[str, Any]:
    """tile_fn: writes valid/veg/vis PNGs, the valid polygon WKB, then the stats JSON (last)."""
    cfg = task.cfg
    if not isinstance(cfg, AppConfig):
        raise StageError("tile_prep task needs the AppConfig", stage=NAME, tile_id=task.tile_id)
    rgb = read_tile(task.tif_path)
    expected = (cfg.grid.tile_px, cfg.grid.tile_px)
    if rgb.shape[:2] != expected:
        raise StageError("tile has the wrong size", stage=NAME, tile_id=task.tile_id, shape=rgb.shape[:2])
    valid = _valid(rgb, cfg)
    masks = compute_tile_masks(rgb, _erode(valid, cfg.nodata.veg_erode_px), cfg.veg)
    out = task.outputs
    write_mask_png(out["valid"], valid)
    write_mask_png(out["veg"], masks.veg)
    write_u8_png(out["vis"], masks.vis)
    polygon = valid_polygon(valid, tile_ref(task.tile_id), approx_eps_px=cfg.nodata.approx_eps_px)
    atomic_write_bytes(out["valid_wkb"], shapely.to_wkb(polygon))
    valid_frac = float(valid.mean())
    status = TileStatus.EMPTY_NODATA if valid_frac < cfg.nodata.min_valid_frac else TileStatus.OK
    stats = TileStats(tile_id=task.tile_id, valid_frac=valid_frac, nodata_frac=1.0 - valid_frac,
                      veg_frac=masks.veg_frac, status=status, veg_threshold=masks.threshold, veg_method=masks.method)
    atomic_write_json(out["stats"], stats.to_dict())
    return {"valid_frac": round(valid_frac, 6), "veg_frac": round(masks.veg_frac, 6), "status": status.value,
            "veg_method": masks.method}


# ------------------------------------------------------------------ main process


def _input_keys(ctx: RunContext, tile_id: str) -> list[str]:
    key = read_key(tile_path(ctx, tile_id))
    if key is None:
        raise StageError("tile not ingested (no .key next to the tif); run `vineyard ingest`", stage=NAME,
                         tile_id=tile_id, path=str(tile_path(ctx, tile_id)))
    return [key]


def _make_task(ctx: RunContext, tile_id: str) -> TileTask:
    tif = tile_path(ctx, tile_id)
    return TileTask(tile_id=tile_id, tif_path=tif, key="", cfg=ctx.cfg, inputs={"tif": tif},
                    outputs=tile_prep_outputs(ctx.paths.cache_dir, tile_id))


def _valid_row(cache_dir: Path, tile_id: str) -> dict[str, Any]:
    stats = load_tile_stats(cache_dir, tile_id)
    geom = shapely.from_wkb(tile_valid_wkb_path(cache_dir, tile_id).read_bytes())
    return {"tile_id": tile_id, "valid_frac": stats.valid_frac, "geometry": geom}


def _write_tile_valid(ctx: RunContext, processed: Sequence[str], failed: Sequence[str]) -> Path:
    """Rows of the processed (ok) tiles replace theirs; rows of tiles outside this run are kept."""
    path = ctx.paths.tile_valid
    old = read_layer(path, TILE_VALID_LAYER) if path.is_file() else empty_layer(TILE_VALID_LAYER)
    keep = old[~old["tile_id"].isin(list(processed))]
    bad = set(failed)
    rows = [_valid_row(ctx.paths.cache_dir, t) for t in processed if t not in bad]
    fresh = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_EPSG) if rows else empty_layer(TILE_VALID_LAYER)
    kept = keep[["tile_id", "valid_frac", "geometry"]]
    merged = gpd.GeoDataFrame(
        {"tile_id": [*kept["tile_id"], *fresh["tile_id"]], "valid_frac": [*kept["valid_frac"], *fresh["valid_frac"]]},
        geometry=[*kept.geometry, *fresh.geometry], crs=CRS_EPSG,
    ).sort_values("tile_id", kind="mergesort").reset_index(drop=True)
    return write_layer(merged, TILE_VALID_LAYER, path)


def _nodata_summary(ctx: RunContext, processed: Sequence[str], failed: Sequence[str]) -> dict[str, float]:
    bad = set(failed)
    stats = [load_tile_stats(ctx.paths.cache_dir, t) for t in processed if t not in bad]
    heavy = sorted((s for s in stats if s.nodata_frac > NODATA_REPORT_FRAC), key=lambda s: (-s.nodata_frac, s.tile_id))
    empty = [s.tile_id for s in stats if s.status == TileStatus.EMPTY_NODATA]
    top = [f"{s.tile_id}={s.nodata_frac:.4f}" for s in heavy[:NODATA_TOP_N]]
    log_event(_log, EVENT_NODATA, level=logging.INFO, stage=NAME, n_tiles=len(stats), n_nodata_gt_1pct=len(heavy),
              n_empty_nodata=len(empty), top=top, empty=empty)
    veg = np.asarray([s.veg_frac for s in stats], dtype=np.float64)
    return {"n_nodata_gt_1pct": float(len(heavy)), "n_empty_nodata": float(len(empty)),
            "veg_frac_mean": round(float(veg.mean()), 6) if veg.size else 0.0}


def run(ctx: RunContext) -> StageResult:
    result = run_tile_stage(ctx, STAGE, make_task=partial(_make_task, ctx), tile_fn=prep_tile,
                            input_keys=partial(_input_keys, ctx))
    processed = ctx.selected_tiles(indexed_tile_ids(ctx))
    layer = _write_tile_valid(ctx, processed, result.failed)
    metrics = {**dict(result.metrics), **_nodata_summary(ctx, processed, result.failed)}
    return StageResult(stage=NAME, n_items=result.n_items, n_cached=result.n_cached, n_failed=result.n_failed,
                       failed=result.failed, outputs=(layer,), metrics=metrics)


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="tile", cfg_keys=CFG_KEYS, requires=("ingest",), run=run,
    description="valid / veg / vis masks + TileStats per tile (cache), static tile_valid layer.",
)
