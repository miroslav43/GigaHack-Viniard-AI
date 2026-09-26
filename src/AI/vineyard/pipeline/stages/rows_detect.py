"""Stage `rows_detect` (tile): tile_prep veg/valid (+ optional NN canopy prob, forbidden mask) ->
perception.rows_detect -> cache/rows_detect/<t>.parquet (row_candidates, UTM) + <t>.json summary.

The cache key covers the tile_prep key, the forbidden geometry touching the tile and the NN weights,
so the per-tile cache is shared by every run with the same rows/orchard config.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import numpy as np
import shapely
from pydantic import BaseModel, ConfigDict
from shapely.geometry.base import BaseGeometry

from vineyard.config import AppConfig
from vineyard.contracts.enums import TileStatus
from vineyard.errors import StageError
from vineyard.geo.raster import rasterize_utm
from vineyard.geo.tiling import TILE_PX, TileRef, tile_box, tile_ref, utm_to_px
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.nn.probs import load_canopy_prob, prob_png_path, upsample_prob
from vineyard.perception.rows_detect import (
    LAYER_NAME,
    DetectParams,
    TileDetection,
    detect_tile_rows,
    detection_to_candidates,
)
from vineyard.pipeline.atomic import atomic_write_json
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult, TileTask, run_tile_stage
from vineyard.pipeline.tile_cache import (
    load_tile_stats,
    load_valid_mask,
    load_veg_mask,
    stats_path,
    tile_prep_key,
    valid_mask_path,
    veg_mask_path,
)
from vineyard.pipeline.tile_index import tile_path

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext

NAME: Final = "rows_detect"
VERSION: Final = "2"
TILE_VALID_LAYER: Final = "tile_valid"
FORBIDDEN_LAYER: Final = "in_forbidden"
FORBIDDEN_FILE: Final = "in_forbidden.parquet"
PROB_CHANNEL: Final = "canopy_prob"
NO_FORBIDDEN_DIGEST: Final = "none"
EVENT_NO_PROB: Final = "rows_detect.nn_prob_missing"
CFG_KEYS: Final = ("rows", "orchard", "canopy.corridor_half_m", "row_structure.occ_bin_m", "runtime.seed",
                   "nn.enabled", "nn.use_in_rows", "nn.prob_threshold", "nn.version", "nn.weights_sha256")

_log = get_logger("pipeline.stages.rows_detect")


class RowsDetectTaskCfg(BaseModel):
    """TileTask.cfg of this stage: the app config plus the provenance stamped on the candidates."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    app: AppConfig
    run_id: str
    model_version: str


# ------------------------------------------------------------------ helpers (worker side)


def tile_clip_px(tile_valid: gpd.GeoDataFrame, tile: TileRef) -> BaseGeometry:
    """clip = tile_box ∩ tile_valid in CVAT px (contract §1.6); StageError if the tile has no row."""
    match = tile_valid[tile_valid["tile_id"] == tile.tile_id]
    if match.empty:
        raise StageError("tile missing from tile_valid", stage=NAME, tile_id=tile.tile_id)
    geom = match.geometry.iloc[0]
    if geom is None or geom.is_empty:
        return shapely.Polygon()
    clip = geom.intersection(tile_box(tile))
    return shapely.transform(clip, lambda xy: utm_to_px(tile, xy))


def forbidden_in_tile(forbidden: gpd.GeoDataFrame, tile: TileRef) -> list[BaseGeometry]:
    """Forbidden polygons overlapping the tile interior (touching an edge only does not count), in layer order."""
    box = tile_box(tile)
    return [g for g in forbidden.geometry
            if g is not None and not g.is_empty and g.intersects(box) and not g.touches(box)]


def forbidden_mask(geoms: list[BaseGeometry], tile: TileRef) -> np.ndarray:
    """Bool px mask of the forbidden area (index convention: a pixel is forbidden if its centre is)."""
    if not geoms:
        return np.zeros((TILE_PX, TILE_PX), dtype=bool)
    return rasterize_utm(geoms, tile, convention="index") > 0


def _prob(task: TileTask) -> np.ndarray | None:
    path = task.inputs.get("prob")
    if path is None:
        return None
    raw = load_canopy_prob(path)
    if raw is None:
        log_event(_log, EVENT_NO_PROB, stage=NAME, tile_id=task.tile_id, path=str(path))
    return upsample_prob(raw, TILE_PX)


def _task_cfg(task: TileTask) -> RowsDetectTaskCfg:
    if not isinstance(task.cfg, RowsDetectTaskCfg):
        raise StageError("rows_detect task needs a RowsDetectTaskCfg", stage=NAME, tile_id=task.tile_id,
                         got=type(task.cfg).__name__)
    return task.cfg


def _veg(task: TileTask, tile: TileRef, cfg: AppConfig) -> np.ndarray:
    veg = load_veg_mask(task.inputs["veg"].parents[1], task.tile_id)
    if not cfg.rows.detect.mask_forbidden:
        return veg
    forbidden = read_layer(task.inputs["forbidden"], FORBIDDEN_LAYER)
    return veg & ~forbidden_mask(forbidden_in_tile(forbidden, tile), tile)


def _write(task: TileTask, det: TileDetection, tcfg: RowsDetectTaskCfg, tile: TileRef) -> Mapping[str, Any]:
    frame = detection_to_candidates(det, tile, run_id=tcfg.run_id, model_version=tcfg.model_version)
    write_layer(frame, LAYER_NAME, task.outputs["candidates"])
    summary = det.summary()
    atomic_write_json(task.outputs["summary"], summary)
    return {k: summary[k] for k in ("n_candidates", "n_kept", "n_accepted", "status_hint")}


def detect_tile(task: TileTask) -> Mapping[str, Any]:
    """tile_fn: detect the tile's row candidates; empty_nodata tiles get an empty layer."""
    tcfg = _task_cfg(task)
    cfg = tcfg.app
    tile = tile_ref(task.tile_id)
    stats = load_tile_stats(task.inputs["stats"].parents[1], task.tile_id)
    params = DetectParams.from_config(cfg)
    if stats.status == TileStatus.EMPTY_NODATA:
        empty = detect_tile_rows(np.zeros((TILE_PX, TILE_PX), bool), shapely.box(0, 0, TILE_PX, TILE_PX),
                                 task.tile_id, params)
        return _write(task, empty, tcfg, tile)
    clip = tile_clip_px(read_layer(task.inputs["tile_valid"], TILE_VALID_LAYER), tile)
    valid = load_valid_mask(task.inputs["valid"].parents[1], task.tile_id)
    det = detect_tile_rows(_veg(task, tile, cfg), clip, task.tile_id, params, prob=_prob(task), valid=valid)
    return _write(task, det, tcfg, tile)


# ------------------------------------------------------------------ stage (main process)


def candidates_path(ctx: RunContext, tile_id: str) -> Path:
    """work/cache/rows_detect/<tile>.parquet (the row_candidates of one tile)."""
    return ctx.paths.tile_cache(NAME, tile_id, "parquet")


def _prob_path(ctx: RunContext, tile_id: str) -> Path | None:
    nn = ctx.cfg.nn
    return prob_png_path(ctx.paths.cache_dir, nn.version, PROB_CHANNEL, tile_id) if nn.enabled else None


def _nn_tag(ctx: RunContext) -> str:
    nn = ctx.cfg.nn
    return f"{nn.version}" if nn.enabled else "none"


def _make_task(ctx: RunContext, tile_id: str) -> TileTask:
    cache = ctx.paths.cache_dir
    inputs = {"veg": veg_mask_path(cache, tile_id), "valid": valid_mask_path(cache, tile_id),
              "stats": stats_path(cache, tile_id), "tile_valid": ctx.paths.tile_valid,
              "forbidden": ctx.paths.static_layers_dir / FORBIDDEN_FILE}
    prob = _prob_path(ctx, tile_id)
    inputs = {**inputs, "prob": prob} if prob is not None else inputs
    outputs = {"candidates": candidates_path(ctx, tile_id), "summary": ctx.paths.tile_cache(NAME, tile_id, "json")}
    tcfg = RowsDetectTaskCfg(app=ctx.cfg, run_id=ctx.run_id, model_version=ctx.model_version(nn=_nn_tag(ctx)))
    return TileTask(tile_id=tile_id, tif_path=tile_path(ctx, tile_id), key="", cfg=tcfg, inputs=inputs,
                    outputs=outputs)


def geometries_digest(geoms: list[BaseGeometry]) -> str:
    """sha1 over the 2-D WKB of the geometries, in order ("none" when empty)."""
    if not geoms:
        return NO_FORBIDDEN_DIGEST
    h = hashlib.sha1()
    for g in geoms:
        h.update(shapely.to_wkb(g, hex=False, output_dimension=2))
    return h.hexdigest()


def _input_keys(ctx: RunContext, forbidden: gpd.GeoDataFrame | None, tile_id: str) -> list[str]:
    keys = [f"tile_prep:{tile_prep_key(ctx.paths.cache_dir, tile_id)}"]
    if forbidden is not None:
        keys.append(f"forbidden:{geometries_digest(forbidden_in_tile(forbidden, tile_ref(tile_id)))}")
    nn = ctx.cfg.nn
    return keys + ([f"nn:{nn.version}:{nn.weights_sha256}"] if nn.enabled else [])


def _require(path: Path, what: str) -> Path:
    if not path.is_file():
        raise StageError(f"{what} missing; run ingest/tile_prep first", stage=NAME, path=str(path))
    return path


def run(ctx: RunContext) -> StageResult:
    _require(ctx.paths.tile_valid, TILE_VALID_LAYER)
    forbidden = None
    if ctx.cfg.rows.detect.mask_forbidden:
        path = _require(ctx.paths.static_layers_dir / FORBIDDEN_FILE, FORBIDDEN_LAYER)
        forbidden = read_layer(path, FORBIDDEN_LAYER)
    return run_tile_stage(ctx, STAGE, make_task=partial(_make_task, ctx), tile_fn=detect_tile,
                          input_keys=partial(_input_keys, ctx, forbidden))


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="tile", cfg_keys=CFG_KEYS, requires=("tile_prep",), run=run,
    description="per-tile row candidates: dominant angle, autocorr spacing + SNR, Huber band fits, "
                "vine/orchard features and soft/hard reject reasons",
)
