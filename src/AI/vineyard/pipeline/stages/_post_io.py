"""Shared I/O of the post stages `targets`, `measure` and `publish` (not a stage itself).

- the AnnSet behind `ctx.annset_ref`;
- the post run holding a stage's inputs: the current run when it has them (the `post` chain runs every
  stage in one run dir), otherwise the newest post run whose run.json names the same `annset_ref`
  (a standalone `vineyard targets|measure|publish --annset X`);
- static inputs: `work/layers/in_*.parquet` (ingest) with the organizers' `02_route/*.geojson` as fallback,
  and the imaged coverage from `tile_valid` (full tile squares when tile_valid is absent).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.annset.io import ANNSET_DIRNAME, ANNSET_JSON, read_annset, resolve_run_dir
from vineyard.annset.model import AnnSet, AnnSetMeta
from vineyard.contracts.qa import QaIssue, issues_to_gdf
from vineyard.errors import SchemaError, StageError
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.geo.vector_io import read_geojson, read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.context import POST_KIND, make_run_paths
from vineyard.route.domain import DomainParams, DomainSet, domain_from_raw

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext, RunPaths

RUN_JSON: Final = "run.json"
POST_RUN_MARK: Final = f"-{POST_KIND}-"
STATIC_GEOJSON: Final[Mapping[str, str]] = {
    "in_passages": "passages.geojson", "in_forbidden": "forbidden.geojson",
    "in_study_area": "study_area.geojson", "in_start": "start.geojson",
}
PASSABLE_DOMAIN: Final = "passable_domain"
EVENT_STATIC_FALLBACK: Final = "post.static_geojson_fallback"

_log = get_logger("pipeline.stages.post_io")


def stage_result(stage: str, *, n_items: int, outputs: Sequence[Path], metrics: Mapping[str, float]) -> Any:
    """A runner StageResult (imported lazily: the runner module is merged separately)."""
    from vineyard.pipeline.runner import StageResult

    return StageResult(stage=stage, n_items=n_items, n_cached=0, n_failed=0, outputs=tuple(outputs),
                       metrics=dict(metrics))


def layer_path(paths: RunPaths, name: str) -> Path:
    return paths.layers_dir / f"{name}.parquet"


# ------------------------------------------------------------------ AnnSet and post runs


def annset_dir(ctx: RunContext, stage: str) -> Path:
    if not ctx.annset_ref:
        raise StageError("post stage needs --annset (run id, LATEST_MARCAJ, ...)", stage=stage, run_id=ctx.run_id)
    return resolve_run_dir(ctx.cfg.paths.work_dir, ctx.annset_ref) / ANNSET_DIRNAME


def load_annset(ctx: RunContext, stage: str) -> AnnSet:
    return read_annset(annset_dir(ctx, stage))


def load_annset_meta(ctx: RunContext, stage: str) -> AnnSetMeta:
    path = annset_dir(ctx, stage) / ANNSET_JSON
    try:
        return AnnSetMeta.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise StageError("cannot read annset.json", stage=stage, path=str(path), error=str(exc)) from exc


def run_annset_ref(run_dir: Path) -> str | None:
    path = run_dir / RUN_JSON
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageError("unreadable run record", path=str(path), error=str(exc)) from exc
    ref = doc.get("annset_ref") if isinstance(doc, dict) else None
    return str(ref) if ref else None


def other_post_runs(ctx: RunContext) -> list[Path]:
    runs = ctx.paths.runs_dir
    if not runs.is_dir():
        return []
    found = [p for p in runs.iterdir() if p.is_dir() and not p.is_symlink() and POST_RUN_MARK in p.name
             and p.resolve() != ctx.paths.run_dir.resolve()]
    return sorted(found, key=lambda p: p.name, reverse=True)  # run ids start with YYYYMMDDTHHMM


def _annset_run(ctx: RunContext, ref: str) -> Path | None:
    """The AnnSet run dir a reference points at today (None when it no longer resolves)."""
    plain = ctx.paths.runs_dir / ref
    if not (Path(ref).is_absolute() or plain.exists() or plain.is_symlink()):
        return None
    return resolve_run_dir(ctx.cfg.paths.work_dir, ref).resolve()


def same_annset(ctx: RunContext, recorded: str | None) -> bool:
    """Same --annset string, or two spellings (run id / LATEST_* link) of the same AnnSet run."""
    if recorded is None or ctx.annset_ref is None:
        return False
    if recorded == ctx.annset_ref:
        return True
    mine, theirs = _annset_run(ctx, ctx.annset_ref), _annset_run(ctx, recorded)
    return mine is not None and mine == theirs


def find_optional_post_run(ctx: RunContext, required: Sequence[str]) -> RunPaths | None:
    """Paths of the run holding every `required` path (relative to the run dir): this run, else the newest
    post run of the same AnnSet; None when no such run exists."""
    if all((ctx.paths.run_dir / rel).exists() for rel in required):
        return ctx.paths
    if ctx.annset_ref:
        for run in other_post_runs(ctx):
            if all((run / rel).exists() for rel in required) and same_annset(ctx, run_annset_ref(run)):
                return make_run_paths(ctx.cfg, run.name)
    return None


def find_post_run(ctx: RunContext, required: Sequence[str], stage: str) -> RunPaths:
    """find_optional_post_run, as a StageError when no run holds the inputs."""
    paths = find_optional_post_run(ctx, required)
    if paths is None:
        raise StageError("no post run of this annset holds the stage inputs (run the earlier stages first)",
                         stage=stage, annset_ref=ctx.annset_ref, required=", ".join(required))
    return paths


def read_optional_layer(path: Path, name: str) -> gpd.GeoDataFrame | None:
    return read_layer(path, name) if path.is_file() else None


def write_issues(ctx: RunContext, issues: Sequence[QaIssue], meta: AnnSetMeta, file_name: str) -> Path:
    gdf = issues_to_gdf(issues, source=meta.source, run_id=ctx.run_id, model_version=meta.model_version)
    return write_layer(gdf, "qa_issues", ctx.paths.qa_dir / file_name)


# ------------------------------------------------------------------ static inputs


def _static_paths(ctx: RunContext, layer: str) -> tuple[Path, Path]:
    if layer not in STATIC_GEOJSON:
        raise StageError("unknown static route layer", layer=layer, known=", ".join(STATIC_GEOJSON))
    return ctx.paths.static_layers_dir / f"{layer}.parquet", Path(ctx.cfg.paths.route_dir) / STATIC_GEOJSON[layer]


def _static_frame(ctx: RunContext, layer: str) -> gpd.GeoDataFrame | None:
    parquet, geojson = _static_paths(ctx, layer)
    if parquet.is_file():
        return read_layer(parquet, layer)
    if not geojson.is_file():
        return None
    log_event(_log, EVENT_STATIC_FALLBACK, level=logging.WARNING, layer=layer, path=str(geojson))
    return read_geojson(geojson)


def static_frame(ctx: RunContext, layer: str, stage: str) -> gpd.GeoDataFrame:
    """A required static route layer: work/layers/<layer>.parquet (ingest), else the organizers' GeoJSON."""
    frame = _static_frame(ctx, layer)
    if frame is None:
        parquet, geojson = _static_paths(ctx, layer)
        raise StageError("static route input missing (run ingest)", stage=stage, layer=layer,
                         tried=f"{parquet}; {geojson}")
    return frame


def static_geometry(ctx: RunContext, layer: str) -> BaseGeometry | None:
    """Union of a static route layer (None when neither the layer nor the organizers' file exists)."""
    frame = _static_frame(ctx, layer)
    if frame is None:
        return None
    geoms = [g for g in frame.geometry if g is not None and not g.is_empty]
    return shapely.union_all(geoms) if geoms else None


def start_xy(ctx: RunContext, stage: str) -> tuple[float, float]:
    start = static_geometry(ctx, "in_start")
    if not isinstance(start, Point):
        raise StageError("START must be one point (in_start / 02_route/start.geojson)", stage=stage,
                         got=None if start is None else start.geom_type)
    return float(start.x), float(start.y)


def coverage(ctx: RunContext, tile_ids: Sequence[str]) -> tuple[BaseGeometry, bool]:
    """(imaged area of `tile_ids`, whether tile_valid was used)."""
    wanted = sorted(set(tile_ids))
    if not ctx.paths.tile_valid.is_file():
        return shapely.union_all([tile_box(tile_ref(t)) for t in wanted] or [Polygon()]), False
    valid = read_layer(ctx.paths.tile_valid, "tile_valid")
    valid = valid[valid["tile_id"].isin(wanted)]
    known = set(valid["tile_id"])
    geoms = [g for g in valid.geometry if g is not None and not g.is_empty]
    geoms += [tile_box(tile_ref(t)) for t in wanted if t not in known]
    return shapely.union_all(geoms or [Polygon()]), True


def walking_domain(paths: RunPaths, ctx: RunContext) -> DomainSet | None:
    """DomainSet from the run's `passable_domain` layer (None when absent)."""
    frame = read_optional_layer(layer_path(paths, PASSABLE_DOMAIN), PASSABLE_DOMAIN)
    if frame is None or frame.empty:
        return None
    erosion = float(frame["erosion_m"].max())
    params = DomainParams.from_config(ctx.cfg.route.domain)
    if erosion < 0.0:
        raise SchemaError("passable_domain erosion_m must be >= 0", erosion_m=erosion)
    # The layer normally stores the raw domain (erosion 0); an eroded one only needs the remainder.
    params = replace(params, inner_buffer_m=max(0.0, params.inner_buffer_m - erosion),
                     eroded_buffer_m=max(0.0, params.eroded_buffer_m - erosion))
    return domain_from_raw(shapely.union_all(list(frame.geometry)), params)


__all__ = [
    "annset_dir", "coverage", "find_post_run", "layer_path", "load_annset", "load_annset_meta",
    "other_post_runs", "read_optional_layer", "run_annset_ref", "same_annset", "stage_result", "start_xy", "static_frame", "static_geometry", "walking_domain",
    "write_issues",
]
