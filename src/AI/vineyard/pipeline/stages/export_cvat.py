"""Stage export_cvat (`vineyard export-cvat [--annset REF]`): AnnSet -> validated Marcaj upload ZIPs.

The AnnSet is --annset (LATEST_MODEL | run id | path), the current run's own AnnSet when no ref is
given (`vineyard all`), or the special ref EMPTY: an empty AnnSet over every selected tile (M2 test).
Output: <AnnSet run>/exports/marcaj_upload/ (EMPTY: this run's exports/). AnnSet qa errors block the
export unless export.cvat.allow_qa_errors (--allow-qa-errors).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import geopandas as gpd

from vineyard.annset.io import ANNSET_DIRNAME, read_annset, resolve_run_dir
from vineyard.annset.model import AnnSet, empty_annset, make_meta
from vineyard.cvat.export import UploadResult, export_upload, require_no_qa_errors
from vineyard.errors import StageError
from vineyard.geo.vector_io import read_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult
from vineyard.pipeline.tile_index import read_tile_index

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext

STAGE_NAME: Final = "export_cvat"
STAGE_VERSION: Final = "1"
CFG_KEYS: Final = ("export.cvat", "waste.block_assign_max_m")
EMPTY_REF: Final = "EMPTY"
UPLOAD_DIRNAME: Final = "marcaj_upload"
EXPORTS_DIRNAME: Final = "exports"
TILE_STATUS_PATH: Final = Path("layers") / "tile_status.parquet"
QA_ISSUES_PATH: Final = Path("qa") / "qa_issues.parquet"
EVENT_DONE: Final = "export_cvat.done"
EVENT_NO_STATUS: Final = "export_cvat.no_tile_status"

_log = get_logger("pipeline.stages.export_cvat")


@dataclass(frozen=True)
class ExportInput:
    annset: AnnSet
    out_dir: Path
    tile_status: gpd.GeoDataFrame | None
    qa_issues: gpd.GeoDataFrame | None


def _optional_layer(path: Path, name: str) -> gpd.GeoDataFrame | None:
    return read_layer(path, name) if path.is_file() else None


def _empty_input(ctx: RunContext, tile_ids: tuple[str, ...]) -> ExportInput:
    meta = make_meta(ctx.source, ctx.run_id, ctx.model_version(), tile_ids)
    return ExportInput(empty_annset(meta), ctx.paths.exports_dir / UPLOAD_DIRNAME, None, None)


def _run_dir(ctx: RunContext) -> Path:
    if ctx.annset_ref is None:
        return ctx.paths.run_dir
    return resolve_run_dir(ctx.paths.work_dir, ctx.annset_ref)


def load_export_input(ctx: RunContext, tile_ids: tuple[str, ...]) -> ExportInput:
    """What to export and where, from --annset (see the module docstring)."""
    if ctx.annset_ref == EMPTY_REF:
        return _empty_input(ctx, tile_ids)
    run_dir = _run_dir(ctx)
    annset_dir = run_dir / ANNSET_DIRNAME
    if not annset_dir.is_dir():
        raise StageError("AnnSet not found", ref=ctx.annset_ref, run_dir=str(run_dir),
                         hint="run `vineyard run` first or pass --annset")
    status = _optional_layer(run_dir / TILE_STATUS_PATH, "tile_status")
    if status is None:
        log_event(_log, EVENT_NO_STATUS, level=30, run_dir=str(run_dir))
    return ExportInput(read_annset(annset_dir), run_dir / EXPORTS_DIRNAME / UPLOAD_DIRNAME, status,
                       _optional_layer(run_dir / QA_ISSUES_PATH, "qa_issues"))


def _metrics(result: UploadResult) -> dict[str, float]:
    sizes = [p.size for p in result.parts]
    return {"n_zips": float(len(sizes)), "max_zip_bytes": float(max(sizes, default=0)),
            "total_bytes": float(sum(sizes)), "n_tiles": float(sum(p.n_tiles for p in result.parts)),
            "n_warnings": float(result.report.n_warnings), "n_failed_tiles": float(len(result.failed_tiles))}


def _outputs(result: UploadResult) -> tuple[Path, ...]:
    return (*result.zips, result.manifest, result.empty_tiles, result.id_registry, result.validation_report,
            result.summary)


def run(ctx: RunContext) -> StageResult:
    tile_index = read_tile_index(ctx)
    tile_ids = ctx.selected_tiles(str(t) for t in tile_index["tile_id"])
    if not tile_ids:
        raise StageError("no tile selected for the export", tile_filter=list(ctx.tile_filter))
    source = load_export_input(ctx, tile_ids)
    if ctx.annset_ref != EMPTY_REF:
        require_no_qa_errors(source.qa_issues, allow=ctx.cfg.export.cvat.allow_qa_errors)
    result = export_upload(source.annset, tile_index, ctx.cfg, source.out_dir, source.tile_status,
                           run_id=ctx.run_id, tile_ids=tile_ids)
    metrics = _metrics(result)
    log_event(_log, EVENT_DONE, stage=STAGE_NAME, out_dir=str(result.out_dir),
              zips=[p.zip_name for p in result.parts], sizes=[p.size for p in result.parts],
              tiles_per_zip=[p.n_tiles for p in result.parts], **metrics)
    return StageResult(stage=STAGE_NAME, n_items=len(tile_ids), n_cached=0, n_failed=len(result.failed_tiles),
                       failed=result.failed_tiles, outputs=_outputs(result), metrics=metrics)


STAGE: Final = StageSpec(
    name=STAGE_NAME,
    version=STAGE_VERSION,
    scope="global",
    cfg_keys=CFG_KEYS,
    requires=(),
    run=run,
    description="AnnSet -> validated, self-checked Marcaj upload ZIPs (<= max_zip_bytes) + manifests",
)
