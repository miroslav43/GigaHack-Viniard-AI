"""Stage `ingest` (global): the challenge ZIPs -> work/tiles + work/tile_index.parquet, and the
02_route GeoJSONs -> static work/layers/in_{passages,forbidden,study_area,start}.parquet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from vineyard.config import AppConfig
from vineyard.geo.vector_io import write_layer
from vineyard.ingest.route_inputs import ingest_route_inputs
from vineyard.ingest.tiles import TILE_INDEX_LAYER, ingest_tiles
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.context import RunContext
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult

NAME: Final = "ingest"
VERSION: Final = "1"
CFG_KEYS: Final = ("paths.data_root", "paths.tiles_zip_glob", "paths.route_dir", "grid")
EVENT_SUMMARY: Final = "ingest.summary"
EVENT_STALE: Final = "ingest.stale_tiles"
TIF_GLOB: Final = "*.tif"

_log = get_logger("pipeline.stages.ingest")


def tile_zip_paths(cfg: AppConfig) -> list[Path]:
    """The challenge tile ZIPs matched by paths.tiles_zip_glob under paths.data_root, sorted."""
    return sorted(Path(cfg.paths.data_root).glob(cfg.paths.tiles_zip_glob))


def _stale_tiles(tiles_dir: Path, known: set[str]) -> list[str]:
    return sorted(p.name for p in tiles_dir.glob(TIF_GLOB) if p.name not in known)


def run(ctx: RunContext) -> StageResult:
    cfg = ctx.cfg
    grid = cfg.grid
    zips = tile_zip_paths(cfg)
    index, summary = ingest_tiles(zips, ctx.paths.tiles_dir, expected_tiles=grid.expected_tiles,
                                  tile_px=grid.tile_px, tol_m=grid.tiepoint_tol_m)
    index_path = write_layer(index, TILE_INDEX_LAYER, ctx.paths.tile_index)
    route_layers, warnings = ingest_route_inputs(cfg.paths.route_dir, ctx.paths.static_layers_dir)
    stale = _stale_tiles(ctx.paths.tiles_dir, set(index["file_name"]))
    if stale:
        log_event(_log, EVENT_STALE, level=logging.WARNING, stage=NAME, n=len(stale), examples=stale[:5])
    log_event(_log, EVENT_SUMMARY, stage=NAME, n_zips=len(zips), n_tiles=summary.n_tiles,
              n_extracted=summary.n_extracted, n_skipped=summary.n_skipped, total_bytes=summary.total_bytes,
              duration_s=round(summary.duration_s, 3), n_route_warnings=len(warnings))
    return StageResult(
        stage=NAME, n_items=summary.n_tiles, n_cached=summary.n_skipped, n_failed=0,
        outputs=(index_path, *(route_layers[k] for k in sorted(route_layers))),
        metrics={"n_extracted": float(summary.n_extracted), "total_bytes": float(summary.total_bytes),
                 "tiles_s": round(summary.duration_s, 4), "n_route_warnings": float(len(warnings))},
    )


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="global", cfg_keys=CFG_KEYS, requires=(), run=run,
    description="Copy + verify the 311 tiles (sha256, names, tags vs grid), tile_index, in_* route layers.",
)
