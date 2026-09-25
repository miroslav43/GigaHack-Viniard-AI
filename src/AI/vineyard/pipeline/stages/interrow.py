"""Stage `interrow` (block -> tile): global bands -> layers/interrows.parquet, then per-tile pieces with
cover classification -> cache/interrow/<t>.parquet (contract `interrow_pieces`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.schemas import empty_layer
from vineyard.errors import StageError
from vineyard.geo.tiling import GSD_M, tile_box, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.corridor import extend_censored_ends
from vineyard.perception.interrow import build_interrow_bands, classify_pieces, cut_interrows_to_tile
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult, TileTask, run_tile_stage
from vineyard.pipeline.stages.canopy import geometry_digest, tile_clip
from vineyard.pipeline.tile_cache import load_veg_mask, load_vis, tile_prep_key, veg_mask_path, vis_path
from vineyard.pipeline.tile_index import indexed_tile_ids, tile_path

if TYPE_CHECKING:
    from vineyard.config import AppConfig
    from vineyard.pipeline.context import RunContext

NAME: Final = "interrow"
VERSION: Final = "3"
BANDS_LAYER: Final = "interrows"
PIECES_LAYER: Final = "interrow_pieces"
ROWS_FILE: Final = "rows.parquet"
PAIRS_FILE: Final = "row_pairs.parquet"
FORBIDDEN_FILE: Final = "in_forbidden.parquet"
PROVENANCE: Final = ("source", "run_id", "model_version", "confidence", "qa_flags")
DIGEST_COLUMNS: Final = ("interrow_id", "row_left_id", "row_right_id", "n_notches", "qa_flags", "source")
BORDERLINE_CONFIDENCE: Final = 0.5
FULL_CONFIDENCE: Final = 1.0
CFG_KEYS: Final = ("interrow", "export.min_interrow_piece_m2", "export.cvat.notch_width_px")
EVENT_NO_FORBIDDEN: Final = "interrow.no_forbidden_layer"

_log = get_logger("pipeline.stages.interrow")


# ------------------------------------------------------------------ global bands


def _forbidden(ctx: RunContext) -> BaseGeometry | None:
    path = ctx.paths.static_layers_dir / FORBIDDEN_FILE
    if not path.is_file():
        log_event(_log, EVENT_NO_FORBIDDEN, path=str(path))
        return None
    layer = read_layer(path, "in_forbidden")
    return shapely.union_all(list(layer.geometry)) if len(layer) else None


def _pair_flags(pairs: pd.DataFrame, bands: gpd.GeoDataFrame) -> list[str]:
    if "qa_flags" not in pairs.columns:
        return [""] * len(bands)
    flags: dict[frozenset[str], str] = {}
    for a, b, f in zip(pairs["row_a"], pairs["row_b"], pairs["qa_flags"], strict=True):
        flags[frozenset((str(a), str(b)))] = "" if f is None or pd.isna(f) else str(f)
    return [flags.get(frozenset((str(a), str(b))), "") for a, b in zip(bands["row_left_id"], bands["row_right_id"],
                                                                        strict=True)]


def bands_with_provenance(ctx: RunContext, bands: gpd.GeoDataFrame, pairs: pd.DataFrame) -> gpd.GeoDataFrame:
    """Contract provenance for the global `interrows` layer (qa_flags from the row pair, e.g. skip-one)."""
    return bands.assign(source=ctx.source.value, run_id=ctx.run_id, model_version=ctx.model_version(),
                        confidence=FULL_CONFIDENCE, qa_flags=_pair_flags(pairs, bands))


def data_boundary(ctx: RunContext) -> BaseGeometry | None:
    """Boundary of the union of this run's tile clips: where rows stop because the data stops."""
    selected = set(ctx.selected_tiles(indexed_tile_ids(ctx)))
    valid = read_layer(ctx.paths.tile_valid, "tile_valid")
    clips = [g.intersection(tile_box(tile_ref(t))) for t, g in zip(valid["tile_id"].astype(str), valid.geometry,
                                                                    strict=True)
             if t in selected and g is not None and not g.is_empty]
    return shapely.union_all(clips).boundary if clips else None


def build_bands_layer(ctx: RunContext) -> tuple[gpd.GeoDataFrame, Path]:
    """Read rows / row_pairs / in_forbidden, build the bands and write layers/interrows.parquet."""
    layers = ctx.paths.layers_dir
    for path in (layers / ROWS_FILE, layers / PAIRS_FILE):
        if not path.is_file():
            raise StageError("input layer missing; run blocks first", stage=NAME, path=str(path))
    ir = ctx.cfg.interrow
    rows = extend_censored_ends(read_layer(layers / ROWS_FILE, "rows"), data_boundary(ctx),
                                extend_m=ir.edge_extend_m, tol_m=ir.edge_tol_m)
    pairs = read_layer(layers / PAIRS_FILE, "row_pairs")
    notch_m = ctx.cfg.export.cvat.notch_width_px * GSD_M
    bands = build_interrow_bands(rows, pairs, _forbidden(ctx), ir, notch_m)
    out = bands_with_provenance(ctx, bands, pairs)
    path = layers / f"{BANDS_LAYER}.parquet"
    write_layer(out, BANDS_LAYER, path)
    return out, path


# ------------------------------------------------------------------ tile function (worker)


def _local_bands(bands: gpd.GeoDataFrame, tile_id: str) -> gpd.GeoDataFrame:
    if bands.empty:
        return bands
    return bands[bands.intersects(tile_box(tile_ref(tile_id)))].reset_index(drop=True)


def _piece_provenance(pieces: gpd.GeoDataFrame, bands: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    by_id = bands.set_index("interrow_id")
    src = [by_id.loc[i] for i in pieces["interrow_id"]]
    flags = [";".join(f for f in (str(b["qa_flags"]), str(own)) if f) for b, own in zip(src, pieces["qa_flags"],
                                                                                      strict=True)]
    conf = [BORDERLINE_CONFIDENCE if own else FULL_CONFIDENCE for own in pieces["qa_flags"]]
    return pieces.assign(source=[str(b["source"]) for b in src], run_id=[str(b["run_id"]) for b in src],
                         model_version=[str(b["model_version"]) for b in src], confidence=conf, qa_flags=flags)


def interrow_tile(task: TileTask) -> Mapping[str, Any]:
    """Worker: cut the bands to one tile, classify the cover, write the interrow_pieces parquet."""
    cfg: AppConfig = task.cfg  # type: ignore[assignment]
    bands = _local_bands(read_layer(task.inputs["interrows"], BANDS_LAYER), task.tile_id)
    if bands.empty:
        write_layer(empty_layer(PIECES_LAYER), PIECES_LAYER, task.outputs["pieces"])
        return {"n_pieces": 0}
    tile = tile_ref(task.tile_id)
    clip = tile_clip(task.inputs["tile_valid"], tile)
    pieces = cut_interrows_to_tile(bands, tile, clip, cfg.export.min_interrow_piece_m2)
    if pieces.empty:
        write_layer(empty_layer(PIECES_LAYER), PIECES_LAYER, task.outputs["pieces"])
        return {"n_pieces": 0}
    cache = task.inputs["veg"].parents[1]
    classified = classify_pieces(pieces, load_veg_mask(cache, task.tile_id), load_vis(cache, task.tile_id), tile,
                                 cfg.interrow)
    out = _piece_provenance(classified, bands)
    write_layer(out, PIECES_LAYER, task.outputs["pieces"])
    return {"n_pieces": len(out)}


# ------------------------------------------------------------------ stage (main process)


def _make_task(ctx: RunContext, bands_path: Path, tile_id: str) -> TileTask:
    cache = ctx.paths.cache_dir
    inputs = {"interrows": bands_path, "veg": veg_mask_path(cache, tile_id), "vis": vis_path(cache, tile_id),
              "tile_valid": ctx.paths.tile_valid}
    return TileTask(tile_id=tile_id, tif_path=tile_path(ctx, tile_id), key="", cfg=ctx.cfg, inputs=inputs,
                    outputs={"pieces": ctx.paths.tile_cache(NAME, tile_id, "parquet")})


def _input_keys(ctx: RunContext, bands: gpd.GeoDataFrame, tile_id: str) -> list[str]:
    local = _local_bands(bands, tile_id)
    return [f"tile_prep:{tile_prep_key(ctx.paths.cache_dir, tile_id)}",
            f"bands:{geometry_digest(local, DIGEST_COLUMNS)}"]


def run(ctx: RunContext) -> StageResult:
    if not ctx.paths.tile_valid.is_file():
        raise StageError("tile_valid layer missing; run tile_prep first", stage=NAME, path=str(ctx.paths.tile_valid))
    bands, bands_path = build_bands_layer(ctx)
    result = run_tile_stage(ctx, STAGE, make_task=partial(_make_task, ctx, bands_path), tile_fn=interrow_tile,
                            input_keys=partial(_input_keys, ctx, bands))
    return replace(result, outputs=(bands_path, *result.outputs))


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="block", cfg_keys=CFG_KEYS, requires=("tile_prep", "blocks"), run=run,
    description="interrow bands between neighbour rows (global) and per-tile pieces with cover class",
)
