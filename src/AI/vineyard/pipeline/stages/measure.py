"""Stage `measure`: AnnSet (+ derive `rows`, `targets` when present) -> exports/measurements.{csv,json}.

The CSV is exactly the web data contract (src/Web/CLAUDE.md §6.4); it is self-checked with the same
checker `publish` uses before it is written. The JSON carries the same values plus extras.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from vineyard.errors import StageError
from vineyard.logging_setup import get_logger, log_event
from vineyard.measure.csv_format import (
    check_measurements_csv,
    decimals_for,
    describe_failed,
    format_measurements_csv,
)
from vineyard.measure.measurements import MeasureInputs, Measurements, compute_measurements, measurements_json
from vineyard.pipeline.atomic import atomic_write_json, atomic_write_text
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.stages._post_io import (
    layer_path,
    load_annset,
    read_optional_layer,
    stage_result,
    write_issues,
)

if TYPE_CHECKING:
    from vineyard.annset.model import AnnSet
    from vineyard.pipeline.context import RunContext

STAGE_NAME: Final = "measure"
STAGE_VERSION: Final = "1"
CFG_KEYS: Final = ("measure", "publish.sum_check_tol_m")
CSV_NAME: Final = "measurements.csv"
JSON_NAME: Final = "measurements.json"
METRICS_NAME: Final = "measure.json"
QA_NAME: Final = "issues_measure.parquet"
EVENT_WRITTEN: Final = "measure.written"

_log = get_logger("pipeline.stages.measure")


def _inputs(ctx: RunContext, annset: AnnSet) -> MeasureInputs:
    return MeasureInputs(canopies=annset.canopies, row_pieces=annset.row_pieces,
                         interrow_pieces=annset.interrow_pieces, waste=annset.waste,
                         rows=read_optional_layer(layer_path(ctx.paths, "rows"), "rows"),
                         targets=read_optional_layer(layer_path(ctx.paths, "targets"), "targets"))


def _meta(ctx: RunContext, annset: AnnSet) -> dict[str, Any]:
    return {"run_id": ctx.run_id, "source": annset.meta.source.value, "annset_ref": ctx.annset_ref,
            "annset_run_id": annset.meta.run_id, "model_version": annset.meta.model_version,
            "contract_version": annset.meta.contract_version, "format": "src/Web/CLAUDE.md#6.4"}


def _metrics(m: Measurements) -> dict[str, float]:
    s = m.survey
    return {"block_count": float(s.block_count or 0), "row_count": float(s.row_count or 0),
            "row_length_m": float(s.row_length_m or 0.0), "canopy_area_m2": float(s.canopy_area_m2 or 0.0),
            "interrow_area_m2": float(s.interrow_area_m2 or 0.0), "plant_count": float(s.plant_count or 0),
            "n_issues": float(len(m.issues))}


def run(ctx: RunContext) -> Any:
    cfg = ctx.cfg
    annset = load_annset(ctx, STAGE_NAME)
    m = compute_measurements(_inputs(ctx, annset), union_sum_warn_frac=cfg.measure.area_union_sum_warn_frac)
    m_dec, ha_dec = decimals_for(cfg.measure.round_m), decimals_for(cfg.measure.round_ha)
    text = format_measurements_csv(m.records(), m_decimals=m_dec, ha_decimals=ha_dec)
    failed = check_measurements_csv(text, sum_tol_m=cfg.publish.sum_check_tol_m)
    if failed:
        raise StageError("measurements.csv failed its own checks", stage=STAGE_NAME, failed=describe_failed(failed))
    exports = ctx.paths.exports_dir
    outputs: list[Path] = [
        atomic_write_text(exports / CSV_NAME, text),
        atomic_write_json(exports / JSON_NAME, measurements_json(m, _meta(ctx, annset), m_decimals=m_dec,
                                                                 ha_decimals=ha_dec)),
        write_issues(ctx, m.issues, annset.meta, QA_NAME),
    ]
    metrics = _metrics(m)
    outputs.append(atomic_write_json(ctx.paths.metrics_dir / METRICS_NAME, metrics))
    log_event(_log, EVENT_WRITTEN, stage=STAGE_NAME, metrics=metrics)
    return stage_result(STAGE_NAME, n_items=len(m.records()), outputs=outputs, metrics=metrics)


STAGE: Final = StageSpec(
    name=STAGE_NAME, version=STAGE_VERSION, scope="global", cfg_keys=CFG_KEYS, requires=("derive", "targets"),
    run=run, description="measurements.csv (web contract §6.4) + measurements.json from the AnnSet",
)
