"""Stage `blocks` (global, 02 §3.6 + plan S3): layers/rows_raw.parquet -> row overrides (delete, extend, add)
-> row graph -> layers/{rows,blocks,row_pairs,rows_rejected}.parquet and qa/qa_blocks.parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from shapely.geometry import LineString

from vineyard.errors import StageError
from vineyard.farms.osm import read_highways
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.perception.blocks import BlockResult, BlockSettings, build_blocks
from vineyard.perception.overrides import apply_row_overrides
from vineyard.perception.road_split import road_lines
from vineyard.perception.row_evidence import VegEvidence
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult
from vineyard.pipeline.stages.rows_link import (
    ROWS_RAW_LAYER,
    load_clips,
    load_overrides_cfg,
    load_passages,
    nn_tag,
    write_issues,
)
from vineyard.pipeline.tile_cache import load_valid_mask, load_veg_mask

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext

NAME: Final = "blocks"
VERSION: Final = "3"  # 2: row-frame regularisation (blocks.regularize: headlands, lattice, strays)
# 3: rows cut where an OSM road / track crosses them on a vine-free stretch (blocks.road_split)
OUTPUT_LAYERS: Final = ("rows", "blocks", "row_pairs", "rows_rejected")
CFG_KEYS: Final = (
    "blocks", "orchard", "rows.detect.spacing_min_m", "rows.link", "canopy.corridor_half_m",
    "export.min_row_piece_m", "paths.overrides", "farms.osm_highways",
)


def evidence_of(ctx: RunContext) -> VegEvidence | None:
    """Veg-mask evidence for blocks.regularize / blocks.road_split (None when both are disabled)."""
    rc = ctx.cfg.blocks.regularize
    if not rc.enabled and not ctx.cfg.blocks.road_split.enabled:
        return None
    return VegEvidence(ctx.paths.cache_dir, rc.evidence_band_m, rc.evidence_step_m, load_veg_mask, load_valid_mask)


def load_roads(ctx: RunContext) -> tuple[LineString, ...]:
    """The OSM snapshot's road lines for blocks.road_split (none when disabled or not configured)."""
    path = ctx.cfg.farms.osm_highways
    if not ctx.cfg.blocks.road_split.enabled or path is None:
        return ()
    if not Path(path).is_file():
        raise StageError("OSM highway snapshot missing (blocks.road_split needs it)", stage=NAME, path=str(path))
    return road_lines(list(read_highways(path).geometry), ctx.cfg.blocks.road_split.min_line_m)


def blocks_stage(ctx: RunContext) -> tuple[BlockResult, tuple]:
    """(BlockResult, override issues) from layers/rows_raw.parquet of this run."""
    path = ctx.paths.layers_dir / f"{ROWS_RAW_LAYER}.parquet"
    if not path.is_file():
        raise StageError("rows_raw layer missing; run rows_link first", stage=NAME, path=str(path))
    ov = apply_row_overrides(read_layer(path, ROWS_RAW_LAYER), load_overrides_cfg(ctx),
                             default_tol_m=ctx.cfg.blocks.override_match_tol_m)
    res = build_blocks(ov.frame, load_passages(ctx), load_clips(ctx), BlockSettings.from_config(ctx.cfg),
                       run_id=ctx.run_id, model_version=ctx.model_version(nn=nn_tag(ctx)), evidence=evidence_of(ctx),
                       roads=load_roads(ctx))
    return res, ov.issues


def run(ctx: RunContext) -> StageResult:
    res, ov_issues = blocks_stage(ctx)
    ctx.paths.layers_dir.mkdir(parents=True, exist_ok=True)
    ctx.paths.qa_dir.mkdir(parents=True, exist_ok=True)
    frames = {"rows": res.rows, "blocks": res.blocks, "row_pairs": res.row_pairs, "rows_rejected": res.rows_rejected}
    outputs = tuple(write_layer(frames[n], n, ctx.paths.layers_dir / f"{n}.parquet") for n in OUTPUT_LAYERS)
    qa = write_issues(ctx, (*ov_issues, *res.issues), NAME)
    metrics = {"n_blocks": float(len(res.blocks)), "n_rows": float(len(res.rows)),
               "n_row_pairs": float(len(res.row_pairs)), "n_rows_rejected": float(len(res.rows_rejected))}
    return StageResult(stage=NAME, n_items=len(res.blocks), n_cached=0, n_failed=0, outputs=(*outputs, qa),
                       metrics=metrics)


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="global", cfg_keys=CFG_KEYS, requires=("rows_link",), run=run,
    description="row graph -> blocks V01.., rows R001.. (n·c order), row_pairs (incl. skip-one), rows_rejected",
)
