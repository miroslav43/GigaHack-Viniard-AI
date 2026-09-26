"""Stage web_bundle: the post run's layers -> `<web.out_dir>/surveys/<survey_id>/pipeline/` (Web CLAUDE.md §6).

Reads the AnnSet named by `--annset` and the optional derive / targets / route / measure outputs of the post
run: this run when it already holds layers, else the newest earlier post run of the same AnnSet (so
`vineyard post --from web_bundle` rebuilds the bundle without recomputing the route). tiles.geojson also reads
the AnnSet run's layers/tile_status.parquet (model runs only), the tile_prep cache (stats, veg masks) and
`web.tile_review`. Never writes into model-run directories.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from zoneinfo import ZoneInfo

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from vineyard.annset.io import ANNSET_DIRNAME, read_annset, resolve_run_dir
from vineyard.errors import SchemaError, StageError
from vineyard.geo.vector_io import CRS_URN_32635, read_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.atomic import atomic_write_json
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.stages._post_io import other_post_runs, run_annset_ref, same_annset
from vineyard.web.bundle import WebInputs, build_web_bundle, bundle_dir, web_params
from vineyard.web.manifest import generated_at_from_run_id
from vineyard.web.objects_export import RouteInfo
from vineyard.web.tile_review import read_tile_review

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext
    from vineyard.pipeline.runner import StageResult

STAGE_NAME: Final = "web_bundle"
# 2: computed measurements use derive's overlap-free interrows; 3: block interrow_area_m2 sum check, computed
# measurements without derive's pieces remove the cross-block overlap here; 4: tiles.geojson + masks/ + manifest
# counts.tiles / masks
# 5: + cross_paths.geojson (passable's tracks across the rows) + its manifest count
STAGE_VERSION: Final = "5"
CFG_KEYS: Final = ("web", "measure", "publish.sum_check_tol_m", "publish.sum_check_tol_m2",
                   "route.visit_radius_m", "route.walking_speed_kmh", "blocks.outline_buffer_m", "grid.gsd_m",
                   "grid.expected_tiles", "project.crs", "logging.tz", "derive.interrow_overlap_min_m2",
                   "derive.interrow_overlap_support_m", "derive.interrow_overlap_min_width_m",
                   "export.min_interrow_piece_m2")
# WebInputs field -> (layer file in <run>/layers, contract layer name)
OPTIONAL_LAYERS: Final = {
    "rows": ("rows.parquet", "rows"), "blocks": ("blocks.parquet", "blocks"),
    "interrows": ("interrow_pieces_linked.parquet", "interrow_pieces_linked"),
    "targets": ("targets.parquet", "targets"), "stops": ("route_stops.parquet", "route_stops"),
    "visits": ("target_visits.parquet", "target_visits"),
    "cross_paths": ("cross_paths.parquet", "cross_paths"),
}
ROUTE_EXPORT: Final = "route.geojson"
MEASUREMENTS_EXPORT: Final = "measurements.csv"
EXPORT_FILES: Final = (ROUTE_EXPORT, MEASUREMENTS_EXPORT)
TILE_STATUS_FILE: Final = "tile_status.parquet"
TILE_STATUS_COLUMNS: Final = ("tile_id", "veg_frac", "review_priority")
LAYERS_DIRNAME: Final = "layers"
EXPORTS_DIRNAME: Final = "exports"
METRICS_FILE: Final = "web_bundle.json"
EVENT_WRITTEN: Final = "web_bundle.written"

_log = get_logger("pipeline.stages.web_bundle")


# ------------------------------------------------------------------ locating the post run


def _has_outputs(run_dir: Path) -> bool:
    layers = run_dir / LAYERS_DIRNAME
    exports = run_dir / EXPORTS_DIRNAME
    return (layers.is_dir() and any(layers.glob("*.parquet"))) or any((exports / f).is_file() for f in EXPORT_FILES)


def find_layers_run(ctx: RunContext) -> Path:
    """This run's directory when it holds post outputs, else the newest other post run of the same AnnSet
    (same `--annset`, or a run id / LATEST_* spelling of the same run) that does; else this run's directory."""
    own = ctx.paths.run_dir
    if _has_outputs(own) or not ctx.annset_ref:
        return own
    for run_dir in other_post_runs(ctx):
        if same_annset(ctx, run_annset_ref(run_dir)) and _has_outputs(run_dir):
            return run_dir
    return own


# ------------------------------------------------------------------ reading the inputs


def _single_feature(doc: Any, path: Path) -> dict[str, Any]:
    if isinstance(doc, dict) and doc.get("type") == "Feature":
        return doc
    features = doc.get("features") if isinstance(doc, dict) and doc.get("type") == "FeatureCollection" else None
    if not isinstance(features, list) or len(features) != 1:
        raise StageError("route export must hold exactly one Feature", stage=STAGE_NAME, path=str(path))
    return features[0]


def _optional_float(props: dict[str, Any], key: str) -> float | None:
    value = props.get(key)
    return None if value is None else float(value)


def read_route_export(path: Path) -> RouteInfo | None:
    """RouteInfo from the route stage's exports/route.geojson (EPSG:32635 crs member); None when absent."""
    source = Path(path)
    if not source.is_file():
        return None
    try:
        doc = json.loads(source.read_bytes())
    except json.JSONDecodeError as exc:
        raise StageError("route export is not valid JSON", stage=STAGE_NAME, path=str(source)) from exc
    crs = ((doc.get("crs") or {}).get("properties") or {}).get("name") if isinstance(doc, dict) else None
    if crs != CRS_URN_32635:
        raise StageError("route export must declare EPSG:32635", stage=STAGE_NAME, path=str(source), crs=crs)
    feature = _single_feature(doc, source)
    props = dict(feature.get("properties") or {})
    try:
        return RouteInfo(shape(feature["geometry"]), baseline_length_m=_optional_float(props, "baseline_length_m"),
                         outside_share=_optional_float(props, "outside_share"))
    except (SchemaError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise StageError("route export has no usable LineString", stage=STAGE_NAME, path=str(source),
                         error=f"{type(exc).__name__}: {exc}") from exc


def _optional_layer(layers_dir: Path, file_name: str, name: str) -> gpd.GeoDataFrame | None:
    path = layers_dir / file_name
    return read_layer(path, name) if path.is_file() else None


def read_tile_status(run_dir: Path) -> pd.DataFrame | None:
    """The AnnSet run's layers/tile_status (model runs; None for Marcaj / reference sets without one)."""
    path = run_dir / LAYERS_DIRNAME / TILE_STATUS_FILE
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise StageError("tile_status layer is unreadable", stage=STAGE_NAME, path=str(path),
                         error=f"{type(exc).__name__}: {exc}") from exc
    missing = [c for c in TILE_STATUS_COLUMNS if c not in frame.columns]
    if missing:
        raise StageError("tile_status layer lacks columns", stage=STAGE_NAME, path=str(path), missing=missing)
    return frame[list(TILE_STATUS_COLUMNS)]


def load_web_inputs(ctx: RunContext, layers_run: Path) -> WebInputs:
    if not ctx.annset_ref:
        raise StageError("web_bundle needs --annset (the AnnSet the post run was built from)", stage=STAGE_NAME)
    annset_run = resolve_run_dir(ctx.paths.work_dir, ctx.annset_ref)
    annset = read_annset(annset_run / ANNSET_DIRNAME)
    layers = {field: _optional_layer(layers_run / LAYERS_DIRNAME, file_name, name)
              for field, (file_name, name) in OPTIONAL_LAYERS.items()}
    exports = layers_run / EXPORTS_DIRNAME
    csv_path = exports / MEASUREMENTS_EXPORT
    return WebInputs(annset=annset, route=read_route_export(exports / ROUTE_EXPORT),
                     measurements_csv=csv_path.read_bytes() if csv_path.is_file() else None,
                     tile_status=read_tile_status(annset_run), cache_dir=ctx.paths.cache_dir,
                     tile_review=read_tile_review(ctx.cfg.web.tile_review), **layers)


# ------------------------------------------------------------------ run


def _generated_at(ctx: RunContext) -> str:
    stamped = generated_at_from_run_id(ctx.run_id, ctx.cfg.logging.tz)
    return stamped or datetime.now(ZoneInfo(ctx.cfg.logging.tz)).isoformat(timespec="seconds")


def run(ctx: RunContext) -> StageResult:
    from vineyard.pipeline.runner import StageResult

    layers_run = find_layers_run(ctx)
    inputs = load_web_inputs(ctx, layers_run)
    params = web_params(ctx.cfg)
    out = bundle_dir(ctx.cfg.web.out_dir, params.survey.survey_id)
    result = build_web_bundle(inputs, out, params, generated_at=_generated_at(ctx), pipeline_version=ctx.git_sha,
                              run_id=ctx.run_id)
    report = {"out_dir": str(out), "layers_run": str(layers_run), "counts": dict(result.counts),
              "written": [p.name for p in result.written], "removed": [p.name for p in result.removed],
              "measurements": "measure" if inputs.measurements_csv is not None else "computed",
              "has_route": inputs.route is not None, "n_masks": len(result.masks),
              "n_review_tiles": len(inputs.tile_review)}
    atomic_write_json(ctx.paths.metrics_dir / METRICS_FILE, report)
    log_event(_log, EVENT_WRITTEN, stage=STAGE_NAME, out_dir=str(out), layers_run=layers_run.name,
              counts=dict(result.counts))
    return StageResult(stage=STAGE_NAME, n_items=len(result.written), n_cached=0, n_failed=0,
                       outputs=result.written, metrics={f"n_{k}": float(v) for k, v in result.counts.items()})


STAGE: Final = StageSpec(
    name=STAGE_NAME,
    version=STAGE_VERSION,
    scope="global",
    cfg_keys=CFG_KEYS,
    requires=("derive", "targets", "route", "measure"),
    run=run,
    description="post run layers -> src/Web/data/surveys/<survey_id>/pipeline/ (EPSG:32635 web data bundle)",
)
