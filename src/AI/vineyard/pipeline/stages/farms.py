"""Stage `farms` (post run): farms (groups of neighbouring blocks) and road classes, for the web map only.

Reads derive's `blocks` and passable's `cross_path_lines` (this run, else the newest post run of the same
AnnSet) and the OSM highway snapshot `farms.osm_highways` (committed, ODbL; refreshed by `vineyard osm-fetch`).
Writes layers/farms.parquet, layers/roads.parquet and metrics/farms.json. Never touches the AnnSet, the CVAT
export or measurements.csv: block ids and every scored output stay as they are.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd

from vineyard.errors import StageError
from vineyard.farms.grouping import Farm, FarmParams, group_blocks
from vineyard.farms.osm import COLUMNS, read_highways
from vineyard.farms.roads import Road, class_lengths, classify_roads, farms_frame, public_union, roads_frame
from vineyard.geo.tiling import CRS_EPSG
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.atomic import atomic_write_json
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.stages._post_io import find_post_run, layer_path, read_optional_layer

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext
    from vineyard.pipeline.runner import StageResult

NAME: Final = "farms"
VERSION: Final = "1"
CFG_KEYS: Final = ("farms",)
REQUIRES: Final = ("derive",)
BLOCKS_REL: Final = "layers/blocks.parquet"
CROSS_LINES_LAYER: Final = "cross_path_lines"
METRICS_FILE: Final = "farms.json"
FARM_CONFIDENCE: Final = 1.0
EVENT_DONE: Final = "farms.done"

_log = get_logger("pipeline.stages.farms")


def _provenance(ctx: RunContext) -> dict[str, Any]:
    return {"source": ctx.source.value, "run_id": ctx.run_id, "model_version": ctx.model_version(),
            "confidence": FARM_CONFIDENCE, "qa_flags": ""}


def load_highways(path: Path | None) -> gpd.GeoDataFrame:
    """The OSM snapshot (EPSG:32635); an empty frame when no snapshot is configured, an error when it is
    configured but missing (a silent empty road layer would look like a survey without roads)."""
    if path is None:
        return gpd.GeoDataFrame({c: [] for c in COLUMNS}, geometry=[], crs=CRS_EPSG)
    if not Path(path).is_file():
        raise StageError("OSM highway snapshot missing (run `vineyard osm-fetch`)", stage=NAME, path=str(path))
    return read_highways(path)


def build(ctx: RunContext) -> tuple[tuple[Farm, ...], tuple[Road, ...]]:
    cfg = ctx.cfg.farms
    run = find_post_run(ctx, (BLOCKS_REL,), NAME)
    blocks = read_layer(run.run_dir / BLOCKS_REL, "blocks")
    highways = load_highways(cfg.osm_highways)
    public = frozenset(cfg.public_highways)
    farms = group_blocks([str(v) for v in blocks["vineyard_id"]], list(blocks.geometry),
                         public_union(highways, public), FarmParams.from_config(cfg))
    cross = read_optional_layer(layer_path(run, CROSS_LINES_LAYER), CROSS_LINES_LAYER)
    roads = classify_roads(highways, farms, cross, public=public, min_internal_m=cfg.internal_min_len_m)
    return farms, roads


def metrics(farms: tuple[Farm, ...], roads: tuple[Road, ...], enabled: bool) -> dict[str, Any]:
    sizes = sorted((len(f.vineyard_ids) for f in farms), reverse=True)
    return {"enabled": enabled, "n_farms": len(farms), "n_blocks": sum(sizes), "farm_sizes": sizes,
            "n_single_block": sum(1 for s in sizes if s == 1), "n_roads": len(roads),
            "road_length_m": class_lengths(roads),
            "farms": [{"farm_id": f.farm_id, "vineyard_ids": list(f.vineyard_ids), "area_m2": round(f.area_m2, 2)}
                      for f in farms]}


def run(ctx: RunContext) -> StageResult:
    from vineyard.pipeline.runner import StageResult

    enabled = ctx.cfg.farms.enabled
    farms, roads = build(ctx) if enabled else ((), ())
    prov = _provenance(ctx)
    outputs = (write_layer(farms_frame(farms, prov), "farms", layer_path(ctx.paths, "farms")),
               write_layer(roads_frame(roads, prov), "roads", layer_path(ctx.paths, "roads")),
               atomic_write_json(ctx.paths.metrics_dir / METRICS_FILE, metrics(farms, roads, enabled)))
    lengths = class_lengths(roads)
    log_event(_log, EVENT_DONE, stage=NAME, farms=len(farms), roads=len(roads), **{f"{k}_m": v for k, v in
                                                                                   lengths.items()})
    return StageResult(stage=NAME, n_items=len(farms), n_cached=0, n_failed=0, outputs=outputs,
                       metrics={"n_farms": float(len(farms)), "n_roads": float(len(roads)),
                                **{f"{k}_m": float(v) for k, v in lengths.items()}})


STAGE: Final = StageSpec(name=NAME, version=VERSION, scope="global", cfg_keys=CFG_KEYS, requires=REQUIRES, run=run,
                         description="farms (blocks <= gap apart, not split by public roads) + road classes (web)")
