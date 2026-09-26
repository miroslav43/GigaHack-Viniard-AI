"""Stage `measure`: AnnSet (+ derive `rows`, `interrow_pieces_linked`, `targets` when present) ->
exports/measurements.{csv,json}.

The CSV is exactly the web data contract (src/Web/CLAUDE.md §6.4); it is self-checked with the same
checker `publish` uses before it is written. The JSON carries the same values plus extras.

Interrow areas come from overlap-free pieces, so the block lines add up to the survey line (the checker
enforces it): derive's `interrow_pieces_linked`, read together with `rows` and `targets` from this run, else
from the newest post run of the same AnnSet that holds it (a standalone `vineyard measure --annset X`).
When no post run of the AnnSet has it, the AnnSet's pieces lose their cross-block overlap here, with the
function and config derive uses (warning `measure.interrow_overlap_removed_here`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd

from vineyard.contracts.qa import QaIssue
from vineyard.errors import StageError
from vineyard.geo.vector_io import read_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.measure.csv_format import (
    check_measurements_csv,
    decimals_for,
    describe_failed,
    format_measurements_csv,
)
from vineyard.measure.measurements import MeasureInputs, Measurements, compute_measurements, measurements_json
from vineyard.perception.block_overlap import BlockOverlapParams, remove_block_overlap
from vineyard.pipeline.atomic import atomic_write_json, atomic_write_text
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.stages._post_io import (
    find_optional_post_run,
    layer_path,
    load_annset,
    read_optional_layer,
    stage_result,
    write_issues,
)

if TYPE_CHECKING:
    from vineyard.annset.model import AnnSet
    from vineyard.pipeline.context import RunContext, RunPaths

STAGE_NAME: Final = "measure"
# 2: interrow areas from derive's interrow_pieces_linked; 3: that layer from the newest post run of the
# AnnSet, else the overlap removed here; block interrow_area_m2 sum check
STAGE_VERSION: Final = "3"
CFG_KEYS: Final = ("measure", "publish.sum_check_tol_m", "publish.sum_check_tol_m2",
                   "derive.interrow_overlap_min_m2", "derive.interrow_overlap_support_m",
                   "derive.interrow_overlap_min_width_m", "export.min_interrow_piece_m2")
CSV_NAME: Final = "measurements.csv"
JSON_NAME: Final = "measurements.json"
METRICS_NAME: Final = "measure.json"
QA_NAME: Final = "issues_measure.parquet"
EVENT_WRITTEN: Final = "measure.written"
EVENT_OVERLAP_HERE: Final = "measure.interrow_overlap_removed_here"
LINKED_LAYER: Final = "interrow_pieces_linked"
LINKED_PATH: Final = f"layers/{LINKED_LAYER}.parquet"

_log = get_logger("pipeline.stages.measure")


@dataclass(frozen=True)
class _Interrows:
    pieces: gpd.GeoDataFrame  # overlap-free interrow pieces
    derive_run: RunPaths | None  # the post run whose derive layers are read; None = overlap removed here
    issues: tuple[QaIssue, ...] = ()  # the overlap removal's warnings when it ran here


def _interrows(ctx: RunContext, annset: AnnSet) -> _Interrows:
    """derive's overlap-free pieces (this run, else the newest post run of the AnnSet); the AnnSet's pieces
    without their cross-block overlap (warned) when no post run of the AnnSet has them."""
    derive_run = find_optional_post_run(ctx, (LINKED_PATH,))
    if derive_run is not None:
        return _Interrows(read_layer(layer_path(derive_run, LINKED_LAYER), LINKED_LAYER), derive_run)
    params = BlockOverlapParams.from_config(ctx.cfg)
    resolved = remove_block_overlap(annset.interrow_pieces, annset.canopies, params)
    log_event(_log, EVENT_OVERLAP_HERE, level=logging.WARNING, stage=STAGE_NAME, missing=LINKED_PATH,
              annset_ref=ctx.annset_ref, metrics=resolved.metrics())
    return _Interrows(resolved.pieces, None, resolved.issues)


def _inputs(ctx: RunContext, annset: AnnSet, interrows: _Interrows) -> MeasureInputs:
    paths = ctx.paths if interrows.derive_run is None else interrows.derive_run
    return MeasureInputs(canopies=annset.canopies, row_pieces=annset.row_pieces,
                         interrow_pieces=interrows.pieces, waste=annset.waste,
                         rows=read_optional_layer(layer_path(paths, "rows"), "rows"),
                         targets=read_optional_layer(layer_path(paths, "targets"), "targets"))


def _meta(ctx: RunContext, annset: AnnSet, interrows: _Interrows) -> dict[str, Any]:
    run = interrows.derive_run
    return {"run_id": ctx.run_id, "source": annset.meta.source.value, "annset_ref": ctx.annset_ref,
            "annset_run_id": annset.meta.run_id, "model_version": annset.meta.model_version,
            "contract_version": annset.meta.contract_version, "format": "src/Web/CLAUDE.md#6.4",
            "derive_run_id": None if run is None else run.run_dir.name}


def _metrics(m: Measurements, interrows: _Interrows) -> dict[str, float]:
    s = m.survey
    return {"block_count": float(s.block_count or 0), "row_count": float(s.row_count or 0),
            "row_length_m": float(s.row_length_m or 0.0), "canopy_area_m2": float(s.canopy_area_m2 or 0.0),
            "interrow_area_m2": float(s.interrow_area_m2 or 0.0), "plant_count": float(s.plant_count or 0),
            "n_issues": float(len(m.issues) + len(interrows.issues)),
            "interrow_overlap_removed_here": float(interrows.derive_run is None)}


def run(ctx: RunContext) -> Any:
    cfg = ctx.cfg
    annset = load_annset(ctx, STAGE_NAME)
    interrows = _interrows(ctx, annset)
    m = compute_measurements(_inputs(ctx, annset, interrows),
                             union_sum_warn_frac=cfg.measure.area_union_sum_warn_frac)
    m_dec, ha_dec = decimals_for(cfg.measure.round_m), decimals_for(cfg.measure.round_ha)
    text = format_measurements_csv(m.records(), m_decimals=m_dec, ha_decimals=ha_dec)
    failed = check_measurements_csv(text, sum_tol_m=cfg.publish.sum_check_tol_m,
                                    sum_tol_m2=cfg.publish.sum_check_tol_m2)
    if failed:
        raise StageError("measurements.csv failed its own checks", stage=STAGE_NAME, failed=describe_failed(failed))
    exports = ctx.paths.exports_dir
    outputs: list[Path] = [
        atomic_write_text(exports / CSV_NAME, text),
        atomic_write_json(exports / JSON_NAME, measurements_json(m, _meta(ctx, annset, interrows),
                                                                 m_decimals=m_dec, ha_decimals=ha_dec)),
        write_issues(ctx, (*m.issues, *interrows.issues), annset.meta, QA_NAME),
    ]
    metrics = _metrics(m, interrows)
    outputs.append(atomic_write_json(ctx.paths.metrics_dir / METRICS_NAME, metrics))
    log_event(_log, EVENT_WRITTEN, stage=STAGE_NAME, metrics=metrics)
    return stage_result(STAGE_NAME, n_items=len(m.records()), outputs=outputs, metrics=metrics)


STAGE: Final = StageSpec(
    name=STAGE_NAME, version=STAGE_VERSION, scope="global", cfg_keys=CFG_KEYS, requires=("derive", "targets"),
    run=run, description="measurements.csv (web contract §6.4) + measurements.json from the AnnSet",
)
