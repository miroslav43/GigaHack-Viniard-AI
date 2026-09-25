"""Stage `waste` (tile pool + global finalize; plan S8, design 03 W1/W3a, arch §4.9).

1. Per tile (spawn pool, no torch): candidates + hard filters -> cache/waste/<t>.parquet (+ .key), every
   candidate with its reject reason. Axes = rows clipped to the tile grown by waste.axis_margin_m.
2. Main process: rule-only verification (level L3), decision (auto-accept disabled by config), NMS by rank,
   vineyard_id, layers/waste_candidates.parquet, qa/waste_candidates.csv + crops + waste_review.html.
3. The FINAL layers/waste.parquet (AnnSet waste schema) holds only the rows confirmed in
   paths.waste_confirmed (plus auto candidates: none while auto-accept is disabled).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiPolygon
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import Source
from vineyard.contracts.schemas import empty_layer
from vineyard.errors import StageError
from vineyard.geo.ops import clip_polygonal
from vineyard.geo.raster import read_tile
from vineyard.geo.tiling import TileRef, tile_box, tile_ref, utm_to_px
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.corridor import clip_rows_to_tile
from vineyard.perception.waste.assign import Provenance, candidates_layer
from vineyard.perception.waste.candidates import CandidateParams, find_candidates
from vineyard.perception.waste.confirm import MergeParams, merge_confirmed, read_confirmations
from vineyard.perception.waste.decide import DecideParams
from vineyard.perception.waste.filters import FilterParams, TileWasteContext, apply_filters, reject_counts
from vineyard.perception.waste.review import (
    CROPS_DIRNAME,
    CSV_NAME,
    HTML_NAME,
    review_items,
    write_review_crops,
    write_review_csv,
    write_review_html,
)
from vineyard.perception.waste.types import (
    RECORD_COLUMNS,
    Candidate,
    candidate_from_record,
    candidate_to_record,
)
from vineyard.perception.waste.verify import (
    RankedSet,
    VerifyStatus,
    load_verifier,
    rank_and_select,
    waste_version_string,
)
from vineyard.pipeline.atomic import atomic_path, atomic_write_json
from vineyard.pipeline.cache import read_key
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult, TileTask, run_tile_stage
from vineyard.pipeline.tile_cache import load_valid_mask, tile_prep_key, valid_mask_path
from vineyard.pipeline.tile_index import indexed_tile_ids, tile_path, tile_refs

if TYPE_CHECKING:
    from vineyard.config import AppConfig
    from vineyard.pipeline.context import RunContext

NAME: Final = "waste"
VERSION: Final = "1"
CFG_KEYS: Final = ("waste", "grid")
ROWS_FILE: Final = "rows.parquet"
BLOCKS_FILE: Final = "blocks.parquet"
FORBIDDEN_FILE: Final = "in_forbidden.parquet"
CANDIDATES_LAYER: Final = "waste_candidates"
WASTE_LAYER: Final = "waste"
STATUS_FILE: Final = "waste.json"
METRIC_LEVEL: Final = "degradation_level"  # 0..3 for L0..L3
LEVEL_PREFIX: Final = "L"

_log = get_logger("pipeline.stages.waste")


# ------------------------------------------------------------------ geometry helpers (both processes)


def local_axes(rows: gpd.GeoDataFrame, tile: TileRef, margin_m: float) -> gpd.GeoDataFrame:
    """Row pieces within the tile grown by margin_m (waste.axis_margin_m, UTM)."""
    grown = tile_box(tile).buffer(margin_m, join_style="mitre")
    return clip_rows_to_tile(rows[rows.intersects(grown)], tile, tile_box(tile), margin_m=margin_m)


def local_forbidden(forbidden: gpd.GeoDataFrame | None, tile: TileRef) -> BaseGeometry | None:
    """Polygonal part of forbidden zone ∩ tile box in UTM (a zone only touching the tile is None)."""
    if forbidden is None or forbidden.empty:
        return None
    parts = clip_polygonal(shapely.union_all(forbidden.geometry.to_numpy()), tile_box(tile))
    return MultiPolygon(parts) if parts else None


def _digest(parts: Sequence[bytes]) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part)
        h.update(b"\x1f")
    return h.hexdigest()


def _geom_digest(frame: gpd.GeoDataFrame | None, zone: BaseGeometry | None = None) -> str:
    geoms = [] if frame is None else list(frame.geometry)
    ids = [] if frame is None or "row_id" not in frame.columns else [str(r) for r in frame["row_id"]]
    parts = [shapely.to_wkb(g, output_dimension=2) for g in [*geoms, *([zone] if zone is not None else [])]]
    return _digest([*parts, *(i.encode("utf-8") for i in ids)])


# ------------------------------------------------------------------ tile function (worker)


def _read_forbidden(path: Path | None) -> gpd.GeoDataFrame | None:
    return read_layer(path, "in_forbidden") if path is not None and Path(path).is_file() else None


def _write_records(cands: Sequence[Candidate], path: Path) -> Path:
    frame = pd.DataFrame([candidate_to_record(c) for c in cands], columns=list(RECORD_COLUMNS))
    with atomic_path(path) as tmp:
        frame.to_parquet(tmp, index=False)
    return path


def filter_tile(
    rgb: np.ndarray,
    valid: np.ndarray,
    tile: TileRef,
    axes_utm: Sequence[LineString],
    zone_utm: BaseGeometry | None,
    cfg: AppConfig,
) -> tuple[Candidate, ...]:
    """Candidates of one tile with their filter verdicts (axes / zone in UTM)."""
    gsd = cfg.grid.gsd_m
    cands = find_candidates(rgb, valid, tile.tile_id, CandidateParams.from_config(cfg.waste, gsd))
    ctx = TileWasteContext(
        axes_px=tuple(utm_to_px(tile, np.asarray(g.coords)) for g in axes_utm),
        forbidden_px=None
        if zone_utm is None
        else shapely.transform(zone_utm, lambda xy: utm_to_px(tile, xy)),
        gsd_m=gsd,
    )
    return apply_filters(cands, ctx, FilterParams.from_config(cfg.waste, gsd))


def waste_tile(task: TileTask) -> Mapping[str, Any]:
    """Worker: candidates + filters of one tile -> cache/waste/<t>.parquet."""
    cfg: AppConfig = task.cfg  # type: ignore[assignment]
    tile = tile_ref(task.tile_id)
    valid = load_valid_mask(task.inputs["valid"].parents[1], task.tile_id)
    axes = local_axes(read_layer(task.inputs["rows"], "rows"), tile, cfg.waste.axis_margin_m)
    zone = local_forbidden(_read_forbidden(task.inputs.get("forbidden")), tile)
    out = filter_tile(read_tile(task.tif_path), valid, tile, list(axes.geometry), zone, cfg)
    _write_records(out, task.outputs["candidates"])
    counts = reject_counts(out)
    return {"n_candidates": len(out), "n_kept": counts.get("kept", 0), "n_axes": len(axes)}


# ------------------------------------------------------------------ stage (main process)


def _make_task(ctx: RunContext, tile_id: str) -> TileTask:
    forbidden = ctx.paths.static_layers_dir / FORBIDDEN_FILE
    inputs = {
        "rows": ctx.paths.layers_dir / ROWS_FILE,
        "valid": valid_mask_path(ctx.paths.cache_dir, tile_id),
    }
    inputs = {**inputs, "forbidden": forbidden} if forbidden.is_file() else inputs
    return TileTask(
        tile_id=tile_id,
        tif_path=tile_path(ctx, tile_id),
        key="",
        cfg=ctx.cfg,
        inputs=inputs,
        outputs={"candidates": ctx.paths.tile_cache(NAME, tile_id, "parquet")},
    )


def _input_keys(
    ctx: RunContext, rows: gpd.GeoDataFrame, forbidden: gpd.GeoDataFrame | None, tile_id: str
) -> list[str]:
    tile = tile_ref(tile_id)
    return [
        f"tile_prep:{tile_prep_key(ctx.paths.cache_dir, tile_id)}",
        f"tile:{read_key(tile_path(ctx, tile_id)) or ''}",
        f"rows:{_geom_digest(local_axes(rows, tile, ctx.cfg.waste.axis_margin_m))}",
        f"forbidden:{_geom_digest(None, local_forbidden(forbidden, tile))}",
    ]


def _load_cached(ctx: RunContext, tile_ids: Sequence[str]) -> tuple[Candidate, ...]:
    out: list[Candidate] = []
    for t in tile_ids:
        frame = pd.read_parquet(ctx.paths.tile_cache(NAME, t, "parquet"))
        out.extend(candidate_from_record(r) for r in frame.to_dict("records"))
    return tuple(out)


def _read_blocks(ctx: RunContext) -> gpd.GeoDataFrame:
    path = ctx.paths.layers_dir / BLOCKS_FILE
    if path.is_file():
        return read_layer(path, "blocks")
    log_event(
        _log,
        "waste.blocks_missing",
        level=logging.WARNING,
        stage=NAME,
        path=str(path),
        message="blocks layer absent: every waste vineyard_id is empty",
    )
    return empty_layer("blocks")


def write_review_files(
    ctx: RunContext, layer: gpd.GeoDataFrame, summary: Mapping[str, object]
) -> tuple[Path, ...]:
    qa, review = ctx.paths.qa_dir, ctx.cfg.waste
    items = review_items(layer.drop(columns="geometry")) if len(layer) else ()
    keep = {qa / it.crop for it in items}
    for stale in sorted((qa / CROPS_DIRNAME).glob("*.jpg")):
        if stale not in keep:
            stale.unlink()
    crops = write_review_crops(
        items,
        lambda t: read_tile(tile_path(ctx, t)),
        qa,
        context=review.probe.crop_context,
        min_px=review.probe.crop_min_px,
        out_px=review.review.crop_px,
    )
    return (write_review_csv(items, qa / CSV_NAME), *crops, write_review_html(items, qa / HTML_NAME, summary))


def _final_waste(
    ctx: RunContext, layer: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, tiles: Sequence[str], prov: Provenance
) -> gpd.GeoDataFrame:
    refs = tile_refs(ctx)
    confs = read_confirmations(ctx.cfg.paths.waste_confirmed, frozenset(refs))
    selected = frozenset(tiles)
    waste = ctx.cfg.waste
    p = MergeParams(
        match_iou=waste.review.confirm_match_iou,
        block_assign_max_m=waste.block_assign_max_m,
        edge_tol_px=waste.review.edge_tol_px,
        run_id=prov.run_id,
        model_version=prov.model_version,
        source=prov.source,
    )
    return merge_confirmed(layer, [c for c in confs if c.tile_id in selected], blocks, refs, p)


def _per_tile_counts(cands: Sequence[Candidate], tile_ids: Sequence[str]) -> dict[str, dict[str, int]]:
    by_tile: dict[str, list[Candidate]] = {t: [] for t in tile_ids}
    for c in cands:
        by_tile.setdefault(c.tile_id, []).append(c)
    return {t: reject_counts(by_tile[t]) for t in sorted(by_tile)}


def _metrics(ranked: RankedSet, n_exported: int, status: VerifyStatus) -> dict[str, float]:
    return {
        "n_candidates": float(len(ranked.candidates)),
        "n_survivors": float(reject_counts(ranked.candidates).get("kept", 0)),
        "n_review": float(len(ranked.review_keys)),
        "n_exported": float(n_exported),
        "n_auto": float(sum(d.auto for d in ranked.decisions.values())),
        METRIC_LEVEL: float(status.level.removeprefix(LEVEL_PREFIX)),
    }


def _write_status(
    ctx: RunContext,
    ranked: RankedSet,
    metrics: Mapping[str, float],
    status: VerifyStatus,
    tile_ids: Sequence[str],
) -> Path:
    doc = {
        "level": status.level,
        "reason": status.reason,
        "counts": reject_counts(ranked.candidates),
        "metrics": dict(metrics),
        "per_tile": _per_tile_counts(ranked.candidates, tile_ids),
    }
    return atomic_write_json(ctx.paths.metrics_dir / STATUS_FILE, doc)


def _finalize(
    ctx: RunContext, tile_ids: Sequence[str], status: VerifyStatus
) -> tuple[dict[str, float], tuple[Path, ...]]:
    cfg = ctx.cfg
    verifier, _ = load_verifier(cfg.waste, cfg.grid.gsd_m)
    decide_p = DecideParams.from_config(cfg.waste.decide)
    ranked = rank_and_select(_load_cached(ctx, tile_ids), verifier, status, decide_p, cfg.waste.nms_iou)
    blocks = _read_blocks(ctx)
    prov = Provenance(Source.MODEL.value, ctx.run_id, ctx.model_version(waste=waste_version_string(status)))
    layer, dropped = candidates_layer(
        ranked.candidates,
        ranked.decisions,
        ranked.review_keys,
        blocks,
        tile_refs(ctx),
        cfg.waste.block_assign_max_m,
        prov,
        gsd_m=cfg.grid.gsd_m,
    )
    if dropped:
        log_event(_log, "waste.candidates_truncated", level=logging.WARNING, stage=NAME, n_dropped=dropped)
    layer = layer if len(layer) else empty_layer(CANDIDATES_LAYER)
    cand_path = write_layer(layer, CANDIDATES_LAYER, ctx.paths.layers_dir / f"{CANDIDATES_LAYER}.parquet")
    waste = _final_waste(ctx, layer, blocks, tile_ids, prov)
    waste_path = write_layer(waste, WASTE_LAYER, ctx.paths.layers_dir / f"{WASTE_LAYER}.parquet")
    metrics = _metrics(ranked, len(waste), status)
    summary = {
        "level": status.level,
        "tiles": len(tile_ids),
        **{k: int(v) for k, v in metrics.items() if k != METRIC_LEVEL},
    }
    review = write_review_files(ctx, layer, summary)
    status_path = _write_status(ctx, ranked, metrics, status, tile_ids)
    return metrics, (cand_path, waste_path, status_path, *review)


def run(ctx: RunContext) -> StageResult:
    rows_path = ctx.paths.layers_dir / ROWS_FILE
    if not rows_path.is_file():
        raise StageError("rows layer missing; run blocks first", stage=NAME, path=str(rows_path))
    rows = read_layer(rows_path, "rows")
    forbidden = _read_forbidden(ctx.paths.static_layers_dir / FORBIDDEN_FILE)
    tiles_result = run_tile_stage(
        ctx,
        STAGE,
        make_task=partial(_make_task, ctx),
        tile_fn=waste_tile,
        input_keys=partial(_input_keys, ctx, rows, forbidden),
    )
    ids = ctx.selected_tiles(indexed_tile_ids(ctx))
    ok = [t for t in ids if t not in set(tiles_result.failed)]
    _, status = load_verifier(ctx.cfg.waste, ctx.cfg.grid.gsd_m)
    metrics, outputs = _finalize(ctx, ok, status)
    log_event(_log, "waste.summary", stage=NAME, verify_level=status.level, **metrics)
    return replace(tiles_result, outputs=outputs, metrics={**tiles_result.metrics, **metrics})


STAGE: Final = StageSpec(
    name=NAME,
    version=VERSION,
    scope="tile",
    cfg_keys=CFG_KEYS,
    requires=("tile_prep", "blocks"),
    run=run,
    description="waste: rule candidates + filters per tile, review files, final layer = confirmed rows only (S8)",
)
