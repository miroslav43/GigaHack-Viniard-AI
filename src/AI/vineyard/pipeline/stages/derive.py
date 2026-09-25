"""Stage `derive`: AnnSet(--annset) -> layers rows, blocks, interrows, interrow_pieces_linked + qa (design 04 §2).

Runs in a post run directory (runs/<ts>-post-<h6>/) and refuses to write into the run that owns the AnnSet,
so model-run layers are never overwritten (critique I9). Inputs: the AnnSet, the static `tile_valid`
(required), `in_passages` / `in_forbidden` (static parquet, else the 02_route GeoJSON).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

from shapely.geometry.base import BaseGeometry

from vineyard.annset.derive import (
    DeriveInputs,
    DeriveParams,
    DeriveResult,
    LayerProv,
    build_coverage,
    derive,
    total_length_by_block,
)
from vineyard.annset.io import ANNSET_DIRNAME, read_annset, resolve_run_dir
from vineyard.errors import StageError
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.atomic import atomic_write_json
from vineyard.pipeline.registry import StageSpec, module_available
from vineyard.pipeline.stages._post_io import static_frame
from vineyard.route.target_gaps import ENGINE_MODULE, GapFn, engine_gap_fn

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext
    from vineyard.pipeline.runner import StageResult

STAGE_NAME: Final = "derive"
STAGE_VERSION: Final = "1"
CFG_KEYS: Final = (
    "derive", "blocks.outline_buffer_m", "blocks.min_rows_per_block", "canopy.corridor_half_m",
    "row_structure.gap_disrupted_m", "import.require_all_tiles", "import.accept_enum_synonyms",
    "import.enum_synonyms", "export.cvat.max_canopy_interrow_overlap_m2", "route.domain.seam_close_m",
)
QA_FILE: Final = "issues_derive.parquet"
METRICS_FILE: Final = "derive.json"
ROUTE_INPUTS: Final = ("passages", "forbidden")

_log = get_logger(__name__)


def annset_dir_of(ctx: RunContext) -> Path:
    """annset/ directory of --annset; StageError when missing or when it is this run's own directory."""
    if not ctx.annset_ref:
        raise StageError("derive needs --annset (run id, LATEST_MODEL, LATEST_MARCAJ ...)", run_id=ctx.run_id)
    source_run = resolve_run_dir(ctx.paths.work_dir, ctx.annset_ref)
    if source_run.resolve() == ctx.paths.run_dir.resolve():
        raise StageError("post stages never write into the AnnSet's own run", run_dir=str(source_run))
    return source_run / ANNSET_DIRNAME


def read_route_input(ctx: RunContext, name: str) -> BaseGeometry:
    """Union of the static `in_<name>` layer, or of 02_route/<name>.geojson when ingest has not run."""
    return static_frame(ctx, f"in_{name}", STAGE_NAME).geometry.union_all()


def load_inputs(ctx: RunContext) -> DeriveInputs:
    annset = read_annset(annset_dir_of(ctx))
    if not ctx.paths.tile_valid.is_file():
        raise StageError("tile_valid missing: run tile_prep first", path=str(ctx.paths.tile_valid))
    tile_valid = read_layer(ctx.paths.tile_valid, "tile_valid")
    coverage = build_coverage(annset.meta.tile_ids, tile_valid)
    passages, forbidden = (read_route_input(ctx, name) for name in ROUTE_INPUTS)
    return DeriveInputs(annset, coverage, passages, forbidden)


def gap_function(ctx: RunContext) -> GapFn | None:
    """The perception gap engine when installed; else None (rows keep max_gap_m = NaN), logged."""
    if module_available(ENGINE_MODULE):
        return engine_gap_fn(ctx.cfg.row_structure, ctx.cfg.canopy.corridor_half_m)
    log_event(_log, "gap_engine_missing", level=logging.WARNING, stage=STAGE_NAME, module=ENGINE_MODULE)
    return None


def write_outputs(ctx: RunContext, result: DeriveResult, prov: LayerProv) -> tuple[Path, ...]:
    """layers/{rows,blocks,interrows,interrow_pieces_linked}.parquet, qa/issues_derive.parquet, metrics."""
    outputs = []
    for name in ("rows", "blocks", "interrows", "interrow_pieces_linked"):
        path = ctx.paths.layers_dir / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        outputs.append(write_layer(result.layer(name), name, path))
    qa_path = ctx.paths.qa_dir / QA_FILE
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    outputs.append(write_layer(prov.qa_layer(result.issues), "qa_issues", qa_path))
    metrics_path = ctx.paths.metrics_dir / METRICS_FILE
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"metrics": result.metrics(), "row_length_m_by_block": total_length_by_block(result.rows),
           "annset_ref": ctx.annset_ref}
    outputs.append(atomic_write_json(metrics_path, doc))
    return tuple(outputs)


def run(ctx: RunContext) -> StageResult:
    from vineyard.pipeline.runner import StageResult  # the runner imports the stages, never the reverse

    inputs = load_inputs(ctx)
    params = DeriveParams.from_config(ctx.cfg)
    result = derive(inputs, params, run_id=ctx.run_id, gap_fn=gap_function(ctx))
    outputs = write_outputs(ctx, result, LayerProv.of(inputs.annset, ctx.run_id))
    metrics = result.metrics()
    log_event(_log, "derive_done", stage=STAGE_NAME, metrics=metrics)
    return StageResult(stage=STAGE_NAME, n_items=len(result.rows), n_cached=0, n_failed=0, outputs=outputs,
                       metrics=metrics)


STAGE: Final = StageSpec(
    name=STAGE_NAME, version=STAGE_VERSION, scope="global", cfg_keys=CFG_KEYS, requires=(), run=run,
    description="Rânduri contopite, blocuri, legarea inter-rândurilor și verificări QA ale AnnSet-ului.",
)
