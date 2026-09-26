"""Stage `targets`: AnnSet + derive `rows` (+ `passable_domain`, `cross_paths`, tile_valid, in_forbidden) -> targets.

Writes layers/targets.parquet, layers/target_extents.parquet, qa/issues_targets.parquet and
metrics/targets.json in the current post run. Gaps come from the single row-gap engine
(`vineyard.perception.attrs`) through the `route.target_gaps` adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.atomic import atomic_write_json
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.stages._cross_paths_io import strips_union
from vineyard.pipeline.stages._post_io import (
    coverage,
    find_post_run,
    layer_path,
    load_annset,
    stage_result,
    static_geometry,
    walking_domain,
    write_issues,
)
from vineyard.route.target_gaps import GapFn, engine_gap_fn
from vineyard.route.targets import (
    TargetInputs,
    TargetProvenance,
    TargetSettings,
    TargetsResult,
    build_targets,
)

if TYPE_CHECKING:
    from vineyard.annset.model import AnnSet
    from vineyard.pipeline.context import RunContext, RunPaths

STAGE_NAME: Final = "targets"
# 3: the stretches of row gaps inside a cross-path strip (passable's cross_paths layer) are no targets
STAGE_VERSION: Final = "3"
CFG_KEYS: Final = ("targets", "canopy.corridor_half_m", "route.candidate_radius_m", "row_structure", "cross_paths")
ROWS_LAYER: Final = "rows"
QA_NAME: Final = "issues_targets.parquet"
METRICS_NAME: Final = "targets.json"
EVENT_WRITTEN: Final = "targets.written"

_log = get_logger("pipeline.stages.targets")


def make_gap_fn(ctx: RunContext) -> GapFn:
    return engine_gap_fn(ctx.cfg.row_structure, ctx.cfg.canopy.corridor_half_m)


def _inputs(ctx: RunContext, annset: AnnSet, paths: RunPaths) -> tuple[TargetInputs, bool]:
    covered, from_tile_valid = coverage(ctx, sorted(annset.tile_ids()))
    domain = walking_domain(paths, ctx)
    cross = ctx.cfg.cross_paths
    inputs = TargetInputs(rows=read_layer(layer_path(paths, ROWS_LAYER), ROWS_LAYER), canopies=annset.canopies,
                          waste=annset.waste, coverage=covered,
                          reach_domain=None if domain is None else domain.inner,
                          forbidden=static_geometry(ctx, "in_forbidden"),
                          cross_paths=strips_union(paths) if cross.enabled and cross.drop_targets else None)
    return inputs, from_tile_valid


def _metrics(result: TargetsResult, from_tile_valid: bool, has_domain: bool) -> dict[str, float]:
    return {**{f"n_{k}": float(v) for k, v in result.counts.items()},
            "tile_valid_used": float(from_tile_valid), "reach_checked": float(has_domain)}


def run(ctx: RunContext) -> Any:
    annset = load_annset(ctx, STAGE_NAME)
    paths = find_post_run(ctx, (f"layers/{ROWS_LAYER}.parquet",), STAGE_NAME)
    inputs, from_tile_valid = _inputs(ctx, annset, paths)
    prov = TargetProvenance(annset.meta.source, ctx.run_id, annset.meta.model_version)
    result = build_targets(inputs, TargetSettings.from_config(ctx.cfg), prov, make_gap_fn(ctx))
    outputs: list[Path] = [
        write_layer(result.targets, "targets", layer_path(ctx.paths, "targets")),
        write_layer(result.extents, "target_extents", layer_path(ctx.paths, "target_extents")),
        write_issues(ctx, result.issues, annset.meta, QA_NAME),
    ]
    metrics = _metrics(result, from_tile_valid, inputs.reach_domain is not None)
    outputs.append(atomic_write_json(ctx.paths.metrics_dir / METRICS_NAME,
                                     {"counts": dict(result.counts), "rows_run": paths.run_dir.name,
                                      "tile_valid_used": from_tile_valid,
                                      "reach_checked": inputs.reach_domain is not None}))
    log_event(_log, EVENT_WRITTEN, stage=STAGE_NAME, metrics=metrics)
    return stage_result(STAGE_NAME, n_items=len(result.targets), outputs=outputs, metrics=metrics)


STAGE: Final = StageSpec(
    name=STAGE_NAME, version=STAGE_VERSION, scope="global", cfg_keys=CFG_KEYS, requires=("derive", "passable"),
    run=run, description="inspection targets (GAP/END/MRW/MSP/SPR/WST) + target_extents on the global rows",
)
