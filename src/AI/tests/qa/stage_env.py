"""A tiny two-tile work dir for the row_attrs / assemble / qa_previews stages (inline, workers=1).

Tile A carries three rows (R001..R003 at y = 40, 30, 20 m), canopies along them (R002 with a 6 m
gap) and two interrows; tile B is an empty tile. Caches are written directly with their keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np

from tests.conftest import write_geotiff
from tests.qa import annset_factory as af
from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import Source, TileStatus
from vineyard.contracts.schemas import empty_layer
from vineyard.geo.raster import write_mask_png, write_u8_png
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, tile_box, tile_ref
from vineyard.geo.vector_io import write_layer
from vineyard.perception.types import VIS_OK, TileStats
from vineyard.pipeline.cache import write_key
from vineyard.pipeline.context import RunContext, ensure_run_dirs, new_run_context
from vineyard.pipeline.stages.canopy import empty_evidence, evidence_path, write_evidence
from vineyard.pipeline.tile_cache import (
    stats_path,
    valid_mask_path,
    veg_mask_path,
    vis_path,
    write_tile_stats,
)

A, B = af.TILE_A, af.TILE_B
TILES = (A, B)
ROW_YS = (40.0, 30.0, 20.0)
GAP = (20.0, 26.0)  # R002 has no canopy between these along positions (m)
RUN_ID = "test-run"


def tile_index(tile_ids: tuple[str, ...]) -> gpd.GeoDataFrame:
    refs = [tile_ref(t) for t in tile_ids]
    return gpd.GeoDataFrame(
        {"tile_id": list(tile_ids), "file_name": [f"{t}.tif" for t in tile_ids],
         "grid_row": [r.grid_row for r in refs], "grid_col": [r.grid_col for r in refs],
         "x0": [r.x0 for r in refs], "y0": [r.y0 for r in refs], "x1": [r.bounds[2] for r in refs],
         "y1": [r.bounds[1] for r in refs], "gsd_m": GSD_M, "width_px": TILE_PX, "height_px": TILE_PX,
         "src_zip": "z.zip", "path": "p", "sha256": "0" * 64, "file_size": 1, "nodata_frac": 0.0,
         "valid_area_m2": 2621.44},
        geometry=[tile_box(r) for r in refs], crs=f"EPSG:{CRS_EPSG}",
    )


def rows_layer() -> gpd.GeoDataFrame:
    lines = [af.hline(A, y, -10.0, 61.2) for y in ROW_YS]
    n = len(lines)
    return gpd.GeoDataFrame(
        {"row_id": [f"V01-R{k + 1:03d}" for k in range(n)], "vineyard_id": "V01", "row_index": range(1, n + 1),
         "length_m": [g.length for g in lines], "extent_m": [g.length for g in lines], "n_pieces": 1,
         "tile_ids": A, "angle_deg": 0.0, "spacing_prev_m": 10.0, "spacing_next_m": 10.0, "max_gap_m": np.nan,
         "n_gaps_ge5": None, "structure_any": None, "source": Source.MODEL.value, "run_id": "old-run",
         "model_version": "pipe@old", "confidence": 1.0, "qa_flags": ""},
        geometry=lines, crs=f"EPSG:{CRS_EPSG}",
    )


def canopies_a() -> gpd.GeoDataFrame:
    spans = {ROW_YS[0]: [(0.0, 51.2)], ROW_YS[1]: [(0.0, GAP[0]), (GAP[1], 51.2)], ROW_YS[2]: [(0.0, 51.2)]}
    items = [(A, "V01", af.rect(A, x0, y - 0.2, x1, y + 0.2)) for y, parts in spans.items() for x0, x1 in parts]
    return af.canopies(items)


def interrows_a() -> gpd.GeoDataFrame:
    return af.interrow_pieces([(A, "V01-I001", af.rect(A, 0.0, 30.3, 51.2, 39.7)),
                               (A, "V01-I002", af.rect(A, 0.0, 20.3, 51.2, 29.7))])


def _tile_prep(ctx: RunContext, tile_id: str) -> None:
    cache = ctx.paths.cache_dir
    veg = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    write_mask_png(veg_mask_path(cache, tile_id), veg)
    write_mask_png(valid_mask_path(cache, tile_id), np.ones_like(veg))
    write_u8_png(vis_path(cache, tile_id), np.full(veg.shape, VIS_OK, dtype=np.uint8))
    write_tile_stats(cache, TileStats(tile_id, 1.0, 0.0, 0.25, TileStatus.OK, 4.0, "lab_a"))
    write_key(stats_path(cache, tile_id), f"prep-{tile_id}")


def write_cache(ctx: RunContext, stage: str, tile_id: str, frame: gpd.GeoDataFrame, layer: str) -> Path:
    path = write_layer(frame, layer, ctx.paths.tile_cache(stage, tile_id, "parquet"))
    write_key(path, f"{stage}-{tile_id}-{len(frame)}")
    return path


def write_candidates(ctx: RunContext, tile_id: str) -> Path:
    """rows_detect cache with one accepted and one rejected candidate (y = 40 m and y = 10 m)."""
    lines = [af.hline(tile_id, 40.0), af.hline(tile_id, 10.0)]
    frame = gpd.GeoDataFrame(
        {"cand_id": [f"{tile_id}:K01", f"{tile_id}:K02"], "tile_id": tile_id, "angle_deg": 0.0, "offset_m": 0.0,
         "length_m": 51.2, "support_frac": 0.9, "width_med_m": 0.4, "local_spacing_m": 10.0, "vine_score": 1.0,
         "rejected_reason": [None, "low_support"], "source": Source.MODEL.value, "run_id": RUN_ID,
         "model_version": "pipe@test", "confidence": 1.0, "qa_flags": ""},
        geometry=lines, crs=f"EPSG:{CRS_EPSG}",
    )
    path = write_layer(frame, "row_candidates", ctx.paths.tile_cache("rows_detect", tile_id, "parquet"))
    write_key(path, f"cands-{tile_id}")
    return path


def write_detect_summary(ctx: RunContext, tile_id: str, snr: float) -> Path:
    path = ctx.paths.tile_cache("rows_detect", tile_id, "json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tile_id": tile_id, "orientations": [{"snr": snr}, {"snr": None}]}))
    return path


def make_env(cfg: AppConfig | None = None) -> RunContext:
    """Work dir (under VINEYARD_WORK_DIR) with every input of row_attrs, assemble and qa_previews."""
    ctx = new_run_context(cfg or load_config(), source=Source.MODEL, run_id=RUN_ID, workers=1)
    ensure_run_dirs(ctx.paths)
    write_layer(tile_index(TILES), "tile_index", ctx.paths.tile_index)
    valid = gpd.GeoDataFrame({"tile_id": list(TILES), "valid_frac": 1.0},
                             geometry=[tile_box(tile_ref(t)) for t in TILES], crs=f"EPSG:{CRS_EPSG}")
    write_layer(valid, "tile_valid", ctx.paths.tile_valid)
    write_layer(rows_layer(), "rows", ctx.paths.layers_dir / "rows.parquet")
    for tile_id in TILES:
        _tile_prep(ctx, tile_id)
        write_geotiff(ctx.paths.tiles_dir / f"{tile_id}.tif", tile_id=tile_id, size=64)
    write_cache(ctx, "canopy", A, canopies_a(), "canopies")
    write_cache(ctx, "canopy", B, empty_layer("canopies"), "canopies")
    for tile_id in TILES:
        write_evidence(empty_evidence(f"EPSG:{CRS_EPSG}"), evidence_path(ctx.paths, tile_id))
    write_cache(ctx, "interrow", A, interrows_a(), "interrow_pieces")
    write_cache(ctx, "interrow", B, empty_layer("interrow_pieces"), "interrow_pieces")
    write_detect_summary(ctx, A, 12.0)
    write_candidates(ctx, A)
    return ctx
