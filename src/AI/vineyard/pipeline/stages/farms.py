"""Stage `farms` (post run): farms (groups of neighbouring blocks) and road classes, for the web map only.

Reads derive's `blocks` and passable's `cross_path_lines` (this run, else the newest post run of the same
AnnSet) and the OSM highway snapshot `farms.osm_highways` (committed, ODbL; refreshed by `vineyard osm-fetch`).
With the optional AGCC parcel snapshot `farms.cadastre_parcels` it also counts the cadastral parcels per farm
and block, flags the roads an official road parcel covers and adds the road parcels no OSM road follows.
Writes layers/farms.parquet, layers/roads.parquet, layers/farm_blocks.parquet and metrics/farms.json. Never touches the AnnSet, the CVAT
export or measurements.csv: block ids and every scored output stay as they are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.errors import StageError
from vineyard.farms.cadastre import read_parcels
from vineyard.farms.frames import farm_blocks_frame, farms_frame
from vineyard.farms.grouping import Farm, FarmParams, group_blocks
from vineyard.farms.osm import COLUMNS, read_highways
from vineyard.farms.parcels import (
    ParcelIndex,
    ParcelParams,
    ParcelStats,
    cadastre_roads,
    mark_cadastral,
    road_parcels,
    stats_by_key,
)
from vineyard.farms.roads import Road, class_lengths, classify_roads, public_union, roads_frame
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
# 2: + AGCC cadastral parcels (optional snapshot): parcels per farm / block, cadastral roads, farm_blocks layer
VERSION: Final = "2"
CFG_KEYS: Final = ("farms",)
REQUIRES: Final = ("derive",)
BLOCKS_REL: Final = "layers/blocks.parquet"
CROSS_LINES_LAYER: Final = "cross_path_lines"
METRICS_FILE: Final = "farms.json"
FARM_CONFIDENCE: Final = 1.0
EVENT_DONE: Final = "farms.done"
EVENT_NO_CADASTRE: Final = "farms.cadastre_missing"

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


def load_parcels(path: Path | None) -> gpd.GeoDataFrame | None:
    """The AGCC snapshot, or None: it is local-only (gitignored), so its absence is a warning, not an error."""
    if path is None:
        return None
    if not Path(path).is_file():
        log_event(_log, EVENT_NO_CADASTRE, level=logging.WARNING, stage=NAME, path=str(path))
        return None
    return read_parcels(path)


@dataclass(frozen=True)
class FarmsOutput:
    farms: tuple[Farm, ...]
    roads: tuple[Road, ...]
    blocks: dict[str, BaseGeometry]
    farm_stats: dict[str, ParcelStats] | None = None
    block_stats: dict[str, ParcelStats] | None = None
    n_parcels: int | None = None


def _union(geoms: list[BaseGeometry]) -> BaseGeometry | None:
    return shapely.union_all(geoms) if geoms else None


def build(ctx: RunContext) -> FarmsOutput:
    cfg = ctx.cfg.farms
    run = find_post_run(ctx, (BLOCKS_REL,), NAME)
    frame = read_layer(run.run_dir / BLOCKS_REL, "blocks")
    blocks = {str(v): g for v, g in zip(frame["vineyard_id"], frame.geometry, strict=True)}
    highways = load_highways(cfg.osm_highways)
    parcels = load_parcels(cfg.cadastre_parcels)
    public, pparams = frozenset(cfg.public_highways), ParcelParams.from_config(cfg)
    extra = cadastre_roads(parcels, _union(list(highways.geometry)), pparams) if parcels is not None else ()
    barrier = _union([g for g in (public_union(highways, public), *(r.geometry for r in extra)) if g is not None])
    farms = group_blocks(list(blocks), list(blocks.values()), barrier, FarmParams.from_config(cfg))
    cross = read_optional_layer(layer_path(run, CROSS_LINES_LAYER), CROSS_LINES_LAYER)
    roads = classify_roads(highways, farms, cross, public=public, min_internal_m=cfg.internal_min_len_m)
    if parcels is None:
        return FarmsOutput(farms, roads, blocks)
    index = ParcelIndex(parcels)
    marked = mark_cadastral(roads, _union(list(road_parcels(parcels, pparams).geometry)), pparams)
    return FarmsOutput(farms, (*marked, *extra), blocks,
                       farm_stats=stats_by_key({f.farm_id: f.outline for f in farms}, index, pparams),
                       block_stats=stats_by_key(blocks, index, pparams), n_parcels=len(parcels))


def _cadastre_metrics(out: FarmsOutput) -> dict[str, Any] | None:
    if out.n_parcels is None:
        return None
    osm_public = [r for r in out.roads if r.origin.value == "osm" and r.road_class.value == "public"]
    confirmed = sum(r.length_m for r in osm_public if r.cadastral)
    extra = [r for r in out.roads if r.origin.value == "cadastre"]
    return {"n_parcels_snapshot": out.n_parcels,
            "osm_public_confirmed_frac": round(confirmed / max(sum(r.length_m for r in osm_public), 1e-9), 4),
            "n_cadastre_roads": len(extra), "cadastre_road_length_m": round(sum(r.length_m for r in extra), 2),
            "parcels_per_block": {k: s.n_parcels for k, s in (out.block_stats or {}).items()}}


def metrics(out: FarmsOutput, enabled: bool) -> dict[str, Any]:
    sizes = sorted((len(f.vineyard_ids) for f in out.farms), reverse=True)
    return {"enabled": enabled, "n_farms": len(out.farms), "n_blocks": sum(sizes), "farm_sizes": sizes,
            "n_single_block": sum(1 for s in sizes if s == 1), "n_roads": len(out.roads),
            "road_length_m": class_lengths(out.roads), "cadastre": _cadastre_metrics(out),
            "farms": [{"farm_id": f.farm_id, "vineyard_ids": list(f.vineyard_ids), "area_m2": round(f.area_m2, 2),
                       "n_parcels": None if out.farm_stats is None else out.farm_stats[f.farm_id].n_parcels}
                      for f in out.farms]}


def _write(ctx: RunContext, out: FarmsOutput, enabled: bool) -> tuple[Path, ...]:
    prov = _provenance(ctx)
    frames = {"farms": farms_frame(out.farms, prov, out.farm_stats), "roads": roads_frame(out.roads, prov),
              "farm_blocks": farm_blocks_frame(out.blocks, out.farms, prov, out.block_stats)}
    written = [write_layer(frame, name, layer_path(ctx.paths, name)) for name, frame in frames.items()]
    written.append(atomic_write_json(ctx.paths.metrics_dir / METRICS_FILE, metrics(out, enabled)))
    return tuple(written)


def run(ctx: RunContext) -> StageResult:
    from vineyard.pipeline.runner import StageResult

    enabled = ctx.cfg.farms.enabled
    out = build(ctx) if enabled else FarmsOutput((), (), {})
    outputs = _write(ctx, out, enabled)
    lengths = class_lengths(out.roads)
    log_event(_log, EVENT_DONE, stage=NAME, farms=len(out.farms), roads=len(out.roads),
              cadastre=out.n_parcels is not None, **{f"{k}_m": v for k, v in lengths.items()})
    return StageResult(stage=NAME, n_items=len(out.farms), n_cached=0, n_failed=0, outputs=outputs,
                       metrics={"n_farms": float(len(out.farms)), "n_roads": float(len(out.roads)),
                                **{f"{k}_m": float(v) for k, v in lengths.items()}})


STAGE: Final = StageSpec(name=NAME, version=VERSION, scope="global", cfg_keys=CFG_KEYS, requires=REQUIRES, run=run,
                         description="farms (blocks <= gap apart, not split by public roads) + road classes (web)")
