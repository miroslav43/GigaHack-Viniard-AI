"""rows_link helper: seeded rows from reviewed tile seeds (configs/row_seeds.csv, perception.rows_seeded).

Runs after the guided pass and before overrides/linking: the seeds of the selected tiles are fitted in the
process pool, the rows are appended as accepted candidates (method "seeded", soft flag "seeded") with K
numbers after the tile's own, and after linking every chain with a seeded member carries qa flag "seeded".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import TileStatus
from vineyard.contracts.ids import parse_row_candidate_id
from vineyard.errors import StageError
from vineyard.geo.tiling import tile_ref, utm_to_px
from vineyard.geo.vector_io import read_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.row_features import FLAG_SEP
from vineyard.perception.row_seeds import RowSeed, SeedFileError, load_row_seeds, usable_seeds
from vineyard.perception.rows_detect import DetectParams, TileDetection, detection_to_candidates
from vineyard.perception.rows_link import LIST_SEP
from vineyard.perception.rows_seeded import (
    FLAG_SEEDED,
    METHOD_SEEDED,
    SeededInputs,
    SeededResult,
    seeded_detect,
)
from vineyard.pipeline.parallel import parallel_map
from vineyard.pipeline.stages.rows_detect import forbidden_in_tile, forbidden_mask
from vineyard.pipeline.tile_cache import (
    load_tile_stats,
    load_valid_mask,
    load_veg_mask,
    stats_path,
    valid_mask_path,
    veg_mask_path,
)
from vineyard.pipeline.tile_index import indexed_tile_ids

if TYPE_CHECKING:
    from vineyard.config.sections_seeded import RowsSeededConfig
    from vineyard.pipeline.context import RunContext

STAGE: Final = "rows_link"
FORBIDDEN_FILE: Final = "in_forbidden.parquet"
FORBIDDEN_LAYER: Final = "in_forbidden"
MAX_K: Final = 999
EVENT_NO_FILE: Final = "rows_link.seeds_missing"
EVENT_SEEDS: Final = "rows_link.seeded_rows"
EVENT_SEED_FAIL: Final = "rows_link.seed_region_failed"
EVENT_NO_PREP: Final = "rows_link.seeded_no_tile_prep"

_log = get_logger("pipeline.stages.rows_link")


def seeds_file(ctx: RunContext) -> Path:
    path = Path(ctx.cfg.rows_seeded.path)
    return path if path.is_absolute() else ctx.paths.project_root / path


def load_seeds(ctx: RunContext) -> tuple[RowSeed, ...]:
    """Usable seeds of the configured file (validated against the tile index); () without a file."""
    path = seeds_file(ctx)
    if not path.is_file():
        log_event(_log, EVENT_NO_FILE, stage=STAGE, path=str(path))
        return ()
    try:
        seeds = load_row_seeds(path, set(indexed_tile_ids(ctx)))
    except SeedFileError as exc:
        raise StageError(str(exc), stage=STAGE, path=str(path)) from exc
    return usable_seeds(seeds, ctx.cfg.rows_seeded.use_confidence)


@dataclass(frozen=True, eq=False)
class SeededJob:
    """Picklable inputs of one tile's seeded pass (runs in a pool worker)."""

    cache_dir: Path
    tile_id: str
    seeds: tuple[RowSeed, ...]
    clip_px: BaseGeometry
    forbidden: tuple[BaseGeometry, ...]
    existing: tuple[shapely.LineString, ...]
    params: DetectParams
    s: RowsSeededConfig


def run_seeded_job(job: SeededJob) -> tuple[SeededResult, int] | None:
    """(seeded result, n veg px) of one tile; None without tile_prep output or on an all-nodata tile."""
    cache = job.cache_dir
    if not all(f(cache, job.tile_id).is_file() for f in (stats_path, veg_mask_path, valid_mask_path)):
        return None
    if load_tile_stats(cache, job.tile_id).status == TileStatus.EMPTY_NODATA:
        return None
    tile = tile_ref(job.tile_id)
    veg = load_veg_mask(cache, job.tile_id)
    if job.forbidden:
        veg = veg & ~forbidden_mask(list(job.forbidden), tile)
    inp = SeededInputs(veg=veg, clip_px=job.clip_px, tile=tile, valid=load_valid_mask(cache, job.tile_id),
                       existing_utm=job.existing)
    return seeded_detect(inp, job.seeds, job.params, job.s), int(veg.sum())


def _clip_px(clips: Mapping[str, BaseGeometry], tile_id: str) -> BaseGeometry:
    geom = clips.get(tile_id)
    if geom is None or geom.is_empty:
        return shapely.Polygon()
    tile = tile_ref(tile_id)
    return shapely.transform(geom, lambda xy: utm_to_px(tile, xy))


def _next_k(cands: gpd.GeoDataFrame, tile_id: str) -> int:
    ids = cands.cand_id[cands.tile_id.astype(str) == tile_id].astype(str) if len(cands) else []
    return max((parse_row_candidate_id(c)[1] for c in ids), default=0) + 1


def _frame(ctx: RunContext, cands: gpd.GeoDataFrame, tile_id: str, res: SeededResult, n_veg: int,
           model_version: str) -> gpd.GeoDataFrame:
    k0 = _next_k(cands, tile_id)
    if k0 + len(res.candidates) - 1 > MAX_K:
        raise StageError("seeded rows overflow the candidate K range", stage=STAGE, tile_id=tile_id)
    numbered = tuple(replace(c, k=k0 + i) for i, c in enumerate(res.candidates))
    det = TileDetection(tile_id=tile_id, status_hint="ok", method=METHOD_SEEDED, n_veg_px=n_veg,
                        orientations=res.orientations, candidates=numbered)
    return detection_to_candidates(det, tile_ref(tile_id), run_id=ctx.run_id, model_version=model_version)


def _jobs(ctx: RunContext, cands: gpd.GeoDataFrame, seeds: tuple[RowSeed, ...], tile_ids: Sequence[str],
          clips: Mapping[str, BaseGeometry]) -> list[SeededJob]:
    forbidden = None
    path = ctx.paths.static_layers_dir / FORBIDDEN_FILE
    if ctx.cfg.rows.detect.mask_forbidden and path.is_file():
        forbidden = read_layer(path, FORBIDDEN_LAYER)
    accepted = cands[cands.rejected_reason.isna()] if len(cands) else cands
    selected = set(tile_ids)
    by_tile: dict[str, list[RowSeed]] = {}
    for seed in seeds:
        if seed.tile_id in selected:
            by_tile.setdefault(seed.tile_id, []).append(seed)
    params = DetectParams.from_config(ctx.cfg)
    jobs = []
    for tile_id in sorted(by_tile):
        tile = tile_ref(tile_id)
        own = accepted.geometry[accepted.tile_id.astype(str) == tile_id] if len(accepted) else []
        jobs.append(SeededJob(
            cache_dir=ctx.paths.cache_dir, tile_id=tile_id, seeds=tuple(by_tile[tile_id]),
            clip_px=_clip_px(clips, tile_id),
            forbidden=tuple(forbidden_in_tile(forbidden, tile)) if forbidden is not None else (),
            existing=tuple(own), params=params, s=ctx.cfg.rows_seeded))
    return jobs


def _log_outcomes(tile_id: str, res: SeededResult) -> None:
    for o in res.outcomes:
        event = EVENT_SEEDS if o.reason is None else EVENT_SEED_FAIL
        log_event(_log, event, stage=STAGE, tile_id=tile_id, line=o.line_no, n_rows=o.n_rows, reason=o.reason,
                  angle_px_deg=round(o.angle_px_deg, 2), spacing_m=round(o.spacing_m, 3))


def add_seeded_candidates(ctx: RunContext, cands: gpd.GeoDataFrame, tile_ids: Sequence[str],
                          clips: Mapping[str, BaseGeometry], model_version: str) -> gpd.GeoDataFrame:
    """cands plus the seeded rows of the selected tiles (unchanged when rows_seeded.enabled is false)."""
    if not ctx.cfg.rows_seeded.enabled:
        return cands
    seeds = load_seeds(ctx)
    if not seeds:
        return cands
    jobs = _jobs(ctx, cands, seeds, tile_ids, clips)
    frames = []
    for job, out in zip(jobs, parallel_map(run_seeded_job, jobs, key=lambda j: j.tile_id, workers=ctx.workers,
                                           chunksize=1, maxtasksperchild=ctx.cfg.runtime.maxtasksperchild),
                        strict=True):
        if not out.ok:
            raise StageError("seeded pass failed", stage=STAGE, tile_id=job.tile_id, error=out.error,
                             traceback=out.traceback)
        if out.value is None:
            log_event(_log, EVENT_NO_PREP, stage=STAGE, tile_id=job.tile_id)
            continue
        res, n_veg = out.value
        _log_outcomes(job.tile_id, res)
        if res.candidates:
            frames.append(_frame(ctx, cands, job.tile_id, res, n_veg, model_version))
    if not frames:
        return cands
    base = [cands] if len(cands) else []
    joined = pd.concat([*base, *frames], ignore_index=True)
    return gpd.GeoDataFrame(joined, geometry="geometry", crs=frames[0].crs).sort_values(
        "cand_id", kind="stable").reset_index(drop=True)


def mark_seeded_chains(rows_raw: gpd.GeoDataFrame, cands: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """rows_raw with qa flag "seeded" on every chain that has a seeded member candidate."""
    if rows_raw.empty or cands.empty or "method" not in cands.columns:
        return rows_raw
    seeded = set(cands.cand_id[cands.method.astype(str) == METHOD_SEEDED].astype(str))
    if not seeded:
        return rows_raw

    def flags(members: object, current: object) -> str:
        has = any(m in seeded for m in str(members).split(LIST_SEP) if m)
        old = [f for f in str(current or "").split(FLAG_SEP) if f and f != "<NA>"]
        return FLAG_SEP.join(dict.fromkeys([*old, FLAG_SEEDED])) if has else FLAG_SEP.join(old)

    new = [flags(m, q) for m, q in zip(rows_raw.member_cand_ids, rows_raw.qa_flags, strict=True)]
    return rows_raw.assign(qa_flags=pd.Series(new, index=rows_raw.index, dtype=rows_raw.qa_flags.dtype))
