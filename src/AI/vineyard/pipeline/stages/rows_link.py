"""Stage `rows_link` (global, 02 §3.5 + plan S4): cache/rows_detect/<t>.parquet of the selected tiles ->
candidate overrides (exclude_areas, force_empty_tiles) -> support-line linking across tiles ->
layers/rows_raw.parquet, layers/row_candidates.parquet (+ link_decision, chain_id) and qa/qa_rows_link.parquet.

Tiles whose rows_detect output has no cache key (failed or stale) are skipped with a warning issue.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import Severity
from vineyard.contracts.qa import QaIssue, issues_to_gdf
from vineyard.contracts.schemas import coerce_layer, empty_layer
from vineyard.errors import StageError
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.overrides import (
    EMPTY_OVERRIDES,
    Overrides,
    apply_candidate_overrides,
    load_overrides,
)
from vineyard.perception.rows_link import LinkSettings, link_candidates
from vineyard.pipeline.cache import read_key
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult
from vineyard.pipeline.tile_index import indexed_tile_ids

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext

NAME: Final = "rows_link"
VERSION: Final = "1"
DETECT_STAGE: Final = "rows_detect"
CANDIDATES_LAYER: Final = "row_candidates"
ROWS_RAW_LAYER: Final = "rows_raw"
QA_LAYER: Final = "qa_issues"
PASSAGES_FILE: Final = "in_passages.parquet"
PASSAGES_LAYER: Final = "in_passages"
TILE_VALID_LAYER: Final = "tile_valid"
CODE_STALE: Final = "tile_failed"
EVENT_NO_PASSAGES: Final = "rows_link.passages_missing"
EVENT_NO_OVERRIDES: Final = "rows_link.overrides_missing"
CFG_KEYS: Final = (
    "rows.link", "rows.detect.residual_split_m", "rows.detect.dp_tolerance_m", "rows.detect.track_vertex_m",
    "rows.detect.snap_to_edge_m", "rows.detect.gap_record_min_m", "blocks.collinear_gap_max_m",
    "blocks.neighbour_max_m", "blocks.parallel_max_deg", "export.min_row_piece_m", "paths.overrides",
)

_log = get_logger("pipeline.stages.rows_link")


# ------------------------------------------------------------------ shared inputs (also used by `blocks`)


def nn_tag(ctx: RunContext) -> str:
    nn = ctx.cfg.nn
    return f"{nn.version}" if nn.enabled else "none"


def load_passages(ctx: RunContext) -> BaseGeometry | None:
    """Union of work/layers/in_passages (None + a logged warning when ingest has not written it)."""
    path = ctx.paths.static_layers_dir / PASSAGES_FILE
    if not path.is_file():
        log_event(_log, EVENT_NO_PASSAGES, stage=NAME, path=str(path))
        return None
    geoms = [g for g in read_layer(path, PASSAGES_LAYER).geometry if g is not None and not g.is_empty]
    return shapely.union_all(geoms) if geoms else None


def load_clips(ctx: RunContext) -> Mapping[str, BaseGeometry]:
    """tile_id -> tile_box ∩ tile_valid (contract §1.6), for every tile of the tile_valid layer."""
    if not ctx.paths.tile_valid.is_file():
        raise StageError("tile_valid layer missing; run tile_prep first", stage=NAME, path=str(ctx.paths.tile_valid))
    valid = read_layer(ctx.paths.tile_valid, TILE_VALID_LAYER)
    clips = {}
    for tile_id, geom in sorted(zip(valid.tile_id.astype(str), valid.geometry, strict=True)):
        box = tile_box(tile_ref(tile_id))
        clips[tile_id] = shapely.Polygon() if geom is None or geom.is_empty else geom.intersection(box)
    return clips


def load_overrides_cfg(ctx: RunContext) -> Overrides:
    path = Path(ctx.cfg.paths.overrides)
    path = path if path.is_absolute() else ctx.paths.project_root / path
    if not path.is_file():
        log_event(_log, EVENT_NO_OVERRIDES, stage=NAME, path=str(path))
        return EMPTY_OVERRIDES
    return load_overrides(path)


def write_issues(ctx: RunContext, issues: Sequence[QaIssue], stage: str) -> Path:
    """qa/qa_<stage>.parquet (qa_issues schema; ids Q00001.. local to the stage)."""
    frame = issues_to_gdf(issues, source=ctx.source, run_id=ctx.run_id, model_version=ctx.model_version(nn=nn_tag(ctx)))
    return write_layer(frame, QA_LAYER, ctx.paths.qa_dir / f"qa_{stage}.parquet")


def provenance(frame: gpd.GeoDataFrame, ctx: RunContext) -> gpd.GeoDataFrame:
    return frame.assign(source=ctx.source.value, run_id=ctx.run_id, model_version=ctx.model_version(nn=nn_tag(ctx)))


# ------------------------------------------------------------------ candidates


def candidates_path(ctx: RunContext, tile_id: str) -> Path:
    return ctx.paths.tile_cache(DETECT_STAGE, tile_id, "parquet")


def _stale_issue(tile_id: str) -> QaIssue:
    centre = tile_box(tile_ref(tile_id)).centroid
    return QaIssue(Severity.WARNING, CODE_STALE, tile_id, NAME, "rows_detect output missing or without cache key",
                   centre.x, centre.y)


def collect_candidates(ctx: RunContext, tile_ids: Sequence[str]) -> tuple[gpd.GeoDataFrame, list[QaIssue]]:
    """Concatenated row_candidates of the tiles with a keyed rows_detect cache file, sorted by cand_id."""
    frames, issues = [], []
    for tile_id in tile_ids:
        path = candidates_path(ctx, tile_id)
        if not path.is_file() or read_key(path) is None:
            issues.append(_stale_issue(tile_id))
            continue
        frame = read_layer(path, CANDIDATES_LAYER)
        if len(frame):
            frames.append(frame)
    if not frames:
        return empty_layer(CANDIDATES_LAYER), issues
    joined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    return joined.sort_values("cand_id", kind="stable").reset_index(drop=True), issues


def _with_decisions(cands: gpd.GeoDataFrame, decisions: pd.DataFrame) -> gpd.GeoDataFrame:
    dec = decisions.set_index("cand_id")
    ids = cands.cand_id.astype(str)
    return cands.assign(link_decision=[str(dec.link_decision.get(c, "override")) for c in ids],
                        chain_id=[dec.chain_id.get(c) if c in dec.index else None for c in ids])


def link_stage(ctx: RunContext) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[QaIssue]]:
    """(rows_raw, candidates with decisions, issues) of the selected tiles."""
    tile_ids = ctx.selected_tiles(indexed_tile_ids(ctx))
    cands, issues = collect_candidates(ctx, tile_ids)
    ov = apply_candidate_overrides(cands, load_overrides_cfg(ctx))
    res = link_candidates(ov.frame, load_passages(ctx), load_clips(ctx), LinkSettings.from_config(ctx.cfg))
    rows_raw = provenance(res.rows_raw, ctx)
    kept = _with_decisions(ov.frame, res.decisions) if len(ov.frame) else ov.frame
    return rows_raw, kept, [*issues, *ov.issues, *res.issues]


def run(ctx: RunContext) -> StageResult:
    rows_raw, cands, issues = link_stage(ctx)
    layers = ctx.paths.layers_dir
    layers.mkdir(parents=True, exist_ok=True)
    ctx.paths.qa_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        write_layer(rows_raw, ROWS_RAW_LAYER, layers / f"{ROWS_RAW_LAYER}.parquet"),
        write_layer(coerce_layer(cands, CANDIDATES_LAYER), CANDIDATES_LAYER, layers / f"{CANDIDATES_LAYER}.parquet"),
        write_issues(ctx, issues, NAME),
    )
    n_tiles = len(ctx.selected_tiles(indexed_tile_ids(ctx)))
    metrics = {"n_candidates": float(len(cands)), "n_chains": float(len(rows_raw)), "n_issues": float(len(issues))}
    return StageResult(stage=NAME, n_items=n_tiles, n_cached=0, n_failed=0, outputs=outputs, metrics=metrics)


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="global", cfg_keys=CFG_KEYS, requires=("rows_detect",), run=run,
    description="global support-line linking of per-tile row candidates into chains (rows_raw)",
)
