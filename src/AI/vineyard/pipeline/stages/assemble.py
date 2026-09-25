"""Stage `assemble` (global): tile caches -> AnnSet(model) + tile_status + qa_issues + review CSVs.

Reads cache/{canopy,interrow,row_attrs}/<t>.parquet, layers/waste.parquet (optional, plan S8),
tile_valid (clips), cache/stats + rows_detect/canopy JSONs (vine evidence), overrides
(force_empty_tiles), layers/rows.parquet (interrow invariant), this run's tile failures and the
issues other stages left in qa/qa_<stage>.parquet (e.g. rows_link, blocks). A stage that failed on
a tile this run does not contribute its (possibly stale) cache.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.annset.assemble import (
    AssembleInputs,
    AssembleResult,
    AssembleSettings,
    TileEvidence,
    assemble_annset,
    write_assemble_outputs,
)
from vineyard.annset.model import make_meta
from vineyard.contracts.enums import Severity, TileStatus
from vineyard.contracts.qa import QA_LAYER, QaIssue
from vineyard.contracts.schemas import empty_layer
from vineyard.errors import StageError
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.geo.vector_io import read_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.overrides import load_overrides
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult, TileFailure, load_tile_failures, record_tile_failures
from vineyard.pipeline.stages.row_attrs import run_model_version
from vineyard.pipeline.tile_cache import load_tile_stats, stats_path
from vineyard.pipeline.tile_index import indexed_tile_ids
from vineyard.qa.review import issues_from_frame

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext, RunPaths

NAME: Final = "assemble"
VERSION: Final = "1"
# AnnSet layer -> the tile stage whose cache holds it.
LAYER_STAGES: Final[Mapping[str, str]] = MappingProxyType(
    {"canopies": "canopy", "row_pieces": "row_attrs", "interrow_pieces": "interrow"}
)
PARQUET_EXT: Final = "parquet"
JSON_EXT: Final = "json"
ROWS_FILE: Final = "rows.parquet"
WASTE_FILE: Final = "waste.parquet"
TILE_VALID_LAYER: Final = "tile_valid"
ROWS_DETECT_STAGE: Final = "rows_detect"
CANOPY_STAGE: Final = "canopy"
ORIENTATIONS_KEY: Final = "orientations"
SNR_KEY: Final = "snr"
VINE_SCORE_KEY: Final = "vineyard_score_p95"
PROVENANCE_LAYERS: Final = ("canopies", "row_pieces", "interrow_pieces")
STAGE_QA_GLOB: Final = "qa_*.parquet"  # qa/qa_<stage>.parquet written by upstream global stages
OWN_QA_FILE: Final = f"{QA_LAYER}.parquet"
EVENT_NO_OVERRIDES: Final = "assemble.no_overrides"
EVENT_DONE: Final = "assemble.done"
CFG_KEYS: Final = ("export.cvat.max_canopy_interrow_overlap_m2", "export.min_interrow_piece_m2",
                   "rows.detect.periodicity_min_snr", "qa.snr_review_max", "rows.filter.min_vine_score",
                   "paths.overrides")

_log = get_logger("pipeline.stages.assemble")


@dataclass(frozen=True, eq=False)
class TileLayers:
    """Concatenated per-tile caches and the tiles whose cache was missing."""

    frames: Mapping[str, gpd.GeoDataFrame]
    missing: tuple[TileFailure, ...]


# ------------------------------------------------------------------ loading


def _concat(parts: Sequence[gpd.GeoDataFrame], layer: str) -> gpd.GeoDataFrame:
    non_empty = [p for p in parts if not p.empty]
    if not non_empty:
        return empty_layer(layer)
    return gpd.GeoDataFrame(pd.concat(non_empty, ignore_index=True), geometry="geometry", crs=non_empty[0].crs)


def load_tile_layers(paths: RunPaths, tile_ids: Sequence[str], failed: Mapping[str, frozenset[str]]) -> TileLayers:
    """Read every tile cache of the three perception layers; a missing cache is a tile failure."""
    frames: dict[str, gpd.GeoDataFrame] = {}
    missing: list[TileFailure] = []
    for layer, stage in LAYER_STAGES.items():
        parts = []
        for tile_id in tile_ids:
            if stage in failed.get(tile_id, frozenset()):
                continue
            path = paths.tile_cache(stage, tile_id, PARQUET_EXT)
            if not path.is_file():
                missing.append(TileFailure(stage=NAME, tile_id=tile_id, error=f"{stage} cache missing: {path}"))
                continue
            parts.append(read_layer(path, layer))
        frames[layer] = _concat(parts, layer)
    return TileLayers(MappingProxyType(frames), tuple(sorted(missing, key=lambda f: (f.tile_id, f.error))))


def load_waste(paths: RunPaths, tile_ids: Sequence[str]) -> gpd.GeoDataFrame | None:
    """layers/waste.parquet restricted to the tiles, or None when the waste stage did not run."""
    path = paths.layers_dir / WASTE_FILE
    if not path.is_file():
        return None
    waste = read_layer(path, "waste")
    return waste[waste["tile_id"].isin(list(tile_ids)).to_numpy()].reset_index(drop=True)


def load_rows(paths: RunPaths) -> gpd.GeoDataFrame | None:
    path = paths.layers_dir / ROWS_FILE
    return read_layer(path, "rows") if path.is_file() else None


def tile_clips(paths: RunPaths, tile_ids: Sequence[str]) -> dict[str, BaseGeometry]:
    """clip = tile_box ∩ tile_valid per tile (contract §1.6); fails loudly without tile_valid."""
    if not paths.tile_valid.is_file():
        raise StageError("tile_valid layer missing; run tile_prep first", stage=NAME, path=str(paths.tile_valid))
    valid = read_layer(paths.tile_valid, TILE_VALID_LAYER)
    by_id = dict(zip(valid["tile_id"].astype(str), valid.geometry, strict=True))
    clips = {}
    for tile_id in tile_ids:
        geom = by_id.get(tile_id)
        if geom is not None:
            box = tile_box(tile_ref(tile_id))
            clips[tile_id] = shapely.Polygon() if geom.is_empty else geom.intersection(box)
    return clips


def load_stage_issues(paths: RunPaths, tile_ids: Sequence[str]) -> tuple[QaIssue, ...]:
    """Issues of qa/qa_<stage>.parquet on the selected tiles (tile-less issues are kept)."""
    selected = frozenset(tile_ids)
    issues: list[QaIssue] = []
    for path in sorted(paths.qa_dir.glob(STAGE_QA_GLOB)):
        if path.name != OWN_QA_FILE:
            issues.extend(i for i in issues_from_frame(read_layer(path, QA_LAYER))
                          if not i.tile_id or i.tile_id in selected)
    return tuple(issues)


# ------------------------------------------------------------------ evidence


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageError("unreadable tile summary JSON", stage=NAME, path=str(path), error=str(exc)) from exc


def _finite(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def tile_snr(summary: Mapping[str, Any]) -> float:
    """Best periodicity SNR over the tile's detected orientations (NaN when none)."""
    values = [_finite(o.get(SNR_KEY)) for o in summary.get(ORIENTATIONS_KEY, []) if isinstance(o, Mapping)]
    finite = [v for v in values if not math.isnan(v)]
    return max(finite) if finite else math.nan


def tile_evidence(paths: RunPaths, tile_id: str) -> TileEvidence:
    """veg_frac / nodata status (tile_prep), SNR (rows_detect) and NN vine score (canopy, when present)."""
    stats = load_tile_stats(paths.cache_dir, tile_id) if stats_path(paths.cache_dir, tile_id).is_file() else None
    detect = _read_json(paths.tile_cache(ROWS_DETECT_STAGE, tile_id, JSON_EXT))
    canopy = _read_json(paths.tile_cache(CANOPY_STAGE, tile_id, JSON_EXT))
    return TileEvidence(
        veg_frac=stats.veg_frac if stats is not None else math.nan,
        snr=tile_snr(detect),
        vineyard_score=_finite(canopy.get(VINE_SCORE_KEY)),
        empty_nodata=stats is not None and stats.status == TileStatus.EMPTY_NODATA,
    )


# ------------------------------------------------------------------ overrides, failures, provenance


def force_empty_tiles(ctx: RunContext) -> tuple[str, ...]:
    path = Path(ctx.cfg.paths.overrides)
    if not path.is_file():
        log_event(_log, EVENT_NO_OVERRIDES, level=logging.WARNING, stage=NAME, path=str(path))
        return ()
    return tuple(sorted(load_overrides(path).force_empty_tiles))


def _failed_stages(failures: Sequence[TileFailure]) -> dict[str, frozenset[str]]:
    by_tile: dict[str, set[str]] = {}
    for failure in failures:
        by_tile.setdefault(failure.tile_id, set()).add(failure.stage)
    return {t: frozenset(s) for t, s in by_tile.items()}


def stamp_provenance(frame: gpd.GeoDataFrame, ctx: RunContext, model_version: str) -> gpd.GeoDataFrame:
    """New frame carrying this run's source / run_id / model_version (tile caches are shared by runs)."""
    return frame.assign(source=ctx.source.value, run_id=ctx.run_id, model_version=model_version)


# ------------------------------------------------------------------ stage


def build_inputs(ctx: RunContext, tile_ids: Sequence[str], model_version: str
                 ) -> tuple[AssembleInputs, tuple[TileFailure, ...]]:
    """AssembleInputs of the selected tiles, plus the tiles whose caches were missing."""
    selected = frozenset(tile_ids)
    upstream = [f for f in load_tile_failures(ctx.paths) if f.tile_id in selected and f.stage != NAME]
    loaded = load_tile_layers(ctx.paths, tile_ids, _failed_stages(upstream))
    frames = {name: stamp_provenance(f, ctx, model_version) if name in PROVENANCE_LAYERS else f
              for name, f in loaded.frames.items()}
    failures = sorted([*upstream, *loaded.missing], key=lambda f: (f.tile_id, f.stage, f.error))
    inputs = AssembleInputs(
        canopies=frames["canopies"], row_pieces=frames["row_pieces"], interrow_pieces=frames["interrow_pieces"],
        waste=load_waste(ctx.paths, tile_ids), clips=tile_clips(ctx.paths, tile_ids),
        evidence={t: tile_evidence(ctx.paths, t) for t in tile_ids},
        failed_tiles=tuple(sorted({f.tile_id for f in failures})), force_empty_tiles=force_empty_tiles(ctx),
        upstream_issues=(*(f.to_issue() for f in failures), *load_stage_issues(ctx.paths, tile_ids)),
        rows=load_rows(ctx.paths),
    )
    return inputs, loaded.missing


def _inputs_list(paths: RunPaths) -> tuple[str, ...]:
    names = [f"layers/{ROWS_FILE}", *(f"cache/{s}" for s in LAYER_STAGES.values())]
    return tuple(names + ([f"layers/{WASTE_FILE}"] if (paths.layers_dir / WASTE_FILE).is_file() else []))


def _metrics(result: AssembleResult) -> dict[str, float]:
    counts = result.annset.counts()
    priority = result.tile_status["review_priority"].astype(int)
    return {
        "n_errors": float(result.n_errors),
        "n_warnings": float(sum(i.severity == Severity.WARNING for i in result.issues)),
        **{f"n_{name}": float(n) for name, n in counts.items()},
        "n_fixed_overlap_tiles": float(len(result.fixed_overlap_tiles)),
        "n_priority_1": float((priority == 1).sum()),
    }


def run(ctx: RunContext) -> StageResult:
    tile_ids = ctx.selected_tiles(indexed_tile_ids(ctx))
    model_version = run_model_version(ctx)
    inputs, missing = build_inputs(ctx, tile_ids, model_version)
    record_tile_failures(ctx.paths, NAME, tile_ids, missing)
    meta = make_meta(ctx.source, ctx.run_id, model_version, tile_ids, inputs=_inputs_list(ctx.paths))
    result = assemble_annset(inputs, AssembleSettings.from_config(ctx.cfg), meta)
    written = write_assemble_outputs(result, annset_dir=ctx.paths.annset_dir, layers_dir=ctx.paths.layers_dir,
                                     qa_dir=ctx.paths.qa_dir)
    metrics = _metrics(result)
    log_event(_log, EVENT_DONE, stage=NAME, **{k: int(v) for k, v in metrics.items()})
    failed = tuple(sorted({f.tile_id for f in missing}))
    return StageResult(stage=NAME, n_items=len(tile_ids), n_cached=0, n_failed=len(failed), failed=failed,
                       outputs=written, metrics=metrics)


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="global", cfg_keys=CFG_KEYS,
    requires=("canopy", "interrow", "row_attrs"), run=run,
    description="AnnSet(model) + tile_status + qa_issues + review_queue from the tile caches (S5/S8/S9)",
)
