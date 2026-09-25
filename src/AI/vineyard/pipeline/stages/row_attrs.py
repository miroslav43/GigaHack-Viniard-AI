"""Stage `row_attrs` (tile): rows ∩ tile box + the tile's canopies + vis codes -> row structure (S2).

Writes cache/row_attrs/<t>.parquet (contract `row_pieces`) and cache/row_attrs/<t>.gaps.parquet
(internal: every gap with kind + censored, for targets/QA). The cache key covers the tile_prep key
(vis), the canopy cache key and a digest of the local row pieces; provenance is left out of the key
because assemble stamps the final run provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
from pydantic import BaseModel, ConfigDict

from vineyard.config import AppConfig
from vineyard.contracts.enums import RowStructure, Source
from vineyard.errors import StageError
from vineyard.geo.tiling import TileRef, tile_box, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.perception.attrs import row_pieces_attributes
from vineyard.perception.corridor import clip_rows_to_tile
from vineyard.pipeline.atomic import atomic_path
from vineyard.pipeline.cache import read_key
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult, TileTask, run_tile_stage
from vineyard.pipeline.stages.canopy import geometry_digest
from vineyard.pipeline.tile_cache import load_vis, tile_prep_key, vis_path
from vineyard.pipeline.tile_index import tile_path

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext, RunPaths

NAME: Final = "row_attrs"
VERSION: Final = "1"
LAYER: Final = "row_pieces"
ROWS_LAYER: Final = "rows"
ROWS_FILE: Final = "rows.parquet"
CANOPY_STAGE: Final = "canopy"
CANOPY_LAYER: Final = "canopies"
PIECES_EXT: Final = "parquet"
GAPS_EXT: Final = "gaps.parquet"
NN_OFF: Final = "none"
DIGEST_COLUMNS: Final = ("row_id", "vineyard_id", "qa_flags", "confidence")
CFG_KEYS: Final = ("row_structure", "canopy.corridor_half_m", "export.min_row_piece_m")


class RowAttrsTaskCfg(BaseModel):
    """TileTask.cfg of this stage: the app config plus the provenance stamped on the row pieces."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    app: AppConfig
    source: Source
    run_id: str
    model_version: str


# ------------------------------------------------------------------ paths (also read by assemble)


def row_pieces_path(paths: RunPaths, tile_id: str) -> Path:
    """work/cache/row_attrs/<tile>.parquet (contract `row_pieces` of one tile)."""
    return paths.tile_cache(NAME, tile_id, PIECES_EXT)


def gaps_path(paths: RunPaths, tile_id: str) -> Path:
    """work/cache/row_attrs/<tile>.gaps.parquet (internal gaps of one tile)."""
    return paths.tile_cache(NAME, tile_id, GAPS_EXT)


def canopy_cache_path(paths: RunPaths, tile_id: str) -> Path:
    """work/cache/canopy/<tile>.parquet (written by the canopy stage)."""
    return paths.tile_cache(CANOPY_STAGE, tile_id, PIECES_EXT)


def run_model_version(ctx: RunContext) -> str:
    """Contract model_version of this run (NN tag when the NN is enabled)."""
    nn = ctx.cfg.nn
    return ctx.model_version(nn=nn.version if nn.enabled else NN_OFF)


def local_row_pieces(rows: gpd.GeoDataFrame, tile: TileRef) -> gpd.GeoDataFrame:
    """Row pieces of one tile, cut by the tile box (contract §1.6), with piece ids <row_id>@<tile>[#k]."""
    box = tile_box(tile)
    near = rows[rows.intersects(box).to_numpy()]
    return clip_rows_to_tile(near, tile, box, margin_m=0.0)


# ------------------------------------------------------------------ tile function (worker)


def _task_cfg(task: TileTask) -> RowAttrsTaskCfg:
    if not isinstance(task.cfg, RowAttrsTaskCfg):
        raise StageError("row_attrs task needs a RowAttrsTaskCfg", stage=NAME, tile_id=task.tile_id,
                         got=type(task.cfg).__name__)
    return task.cfg


def write_gaps(gaps: gpd.GeoDataFrame, path: Path) -> Path:
    """Internal gaps layer (no contract schema): GeoParquet written atomically."""
    with atomic_path(Path(path)) as tmp:
        gaps.to_parquet(tmp, index=False)
    return Path(path)


def row_attrs_tile(task: TileTask) -> Mapping[str, Any]:
    """Worker: measure the row structure of every row piece of one tile and write both parquets."""
    tcfg = _task_cfg(task)
    cfg = tcfg.app
    tile = tile_ref(task.tile_id)
    pieces = local_row_pieces(read_layer(task.inputs["rows"], ROWS_LAYER), tile)
    canopies = read_layer(task.inputs["canopy"], CANOPY_LAYER)
    vis = load_vis(task.inputs["vis"].parents[1], task.tile_id) if not pieces.empty else None
    res = row_pieces_attributes(
        pieces, canopies, tile, cfg.row_structure, half_m=cfg.canopy.corridor_half_m,
        min_piece_m=cfg.export.min_row_piece_m, vis=vis, source=tcfg.source, run_id=tcfg.run_id,
        model_version=tcfg.model_version,
    )
    write_layer(res.row_pieces, LAYER, task.outputs["row_pieces"])
    write_gaps(res.gaps, task.outputs["gaps"])
    structures = list(res.row_pieces["row_structure"])
    return {"n_pieces": len(structures), "n_disrupted": structures.count(RowStructure.DISRUPTED.value),
            "n_unassessable": structures.count(RowStructure.UNASSESSABLE.value), "n_gaps": len(res.gaps)}


# ------------------------------------------------------------------ stage (main process)


def _make_task(ctx: RunContext, tile_id: str) -> TileTask:
    paths = ctx.paths
    inputs = {"rows": paths.layers_dir / ROWS_FILE, "canopy": canopy_cache_path(paths, tile_id),
              "vis": vis_path(paths.cache_dir, tile_id)}
    outputs = {"row_pieces": row_pieces_path(paths, tile_id), "gaps": gaps_path(paths, tile_id)}
    tcfg = RowAttrsTaskCfg(app=ctx.cfg, source=ctx.source, run_id=ctx.run_id, model_version=run_model_version(ctx))
    return TileTask(tile_id=tile_id, tif_path=tile_path(ctx, tile_id), key="", cfg=tcfg, inputs=inputs,
                    outputs=outputs)


def _canopy_key(paths: RunPaths, tile_id: str) -> str:
    key = read_key(canopy_cache_path(paths, tile_id))
    if key is None:
        raise StageError("canopy cache missing; run canopy first", stage=NAME, tile_id=tile_id,
                         path=str(canopy_cache_path(paths, tile_id)))
    return key


def _input_keys(ctx: RunContext, rows: gpd.GeoDataFrame, tile_id: str) -> list[str]:
    pieces = local_row_pieces(rows, tile_ref(tile_id))
    return [f"tile_prep:{tile_prep_key(ctx.paths.cache_dir, tile_id)}",
            f"canopy:{_canopy_key(ctx.paths, tile_id)}",
            f"rows:{geometry_digest(pieces, DIGEST_COLUMNS)}"]


def run(ctx: RunContext) -> StageResult:
    rows_path = ctx.paths.layers_dir / ROWS_FILE
    if not rows_path.is_file():
        raise StageError("rows layer missing; run blocks first", stage=NAME, path=str(rows_path))
    rows = read_layer(rows_path, ROWS_LAYER)
    return run_tile_stage(ctx, STAGE, make_task=partial(_make_task, ctx), tile_fn=row_attrs_tile,
                          input_keys=partial(_input_keys, ctx, rows))


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="tile", cfg_keys=CFG_KEYS, requires=("tile_prep", "blocks", "canopy"),
    run=run,
    description="row_structure per row piece from canopy pixel-set gaps (1-px bins, ends included, nodata unknown)",
)
