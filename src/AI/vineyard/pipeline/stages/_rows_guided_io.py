"""rows_link helper: the neighbour-guided second pass (perception.rows_guided) on the collected candidates.

Round 1 takes its priors from the rows_detect accepted candidates; round k (rows_guided.rounds) from the
rows accepted by the end of round k-1, and only retries tiles next to a tile that gained rows. Inside a
round every tile sees the same frozen state, so the result does not depend on the tile order (tiles run
in the process pool). Guided rows are appended as accepted candidates (method "guided", soft flag
"guided") with K numbers after the tile's own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import TileStatus
from vineyard.contracts.ids import parse_row_candidate_id
from vineyard.errors import StageError
from vineyard.geo.tiling import tile_ref, utm_to_px
from vineyard.geo.vector_io import read_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.rows_detect import DetectParams, TileDetection, detection_to_candidates
from vineyard.perception.rows_guided import (
    METHOD_GUIDED,
    GuidedInputs,
    GuidedResult,
    LatticePrior,
    guided_detect,
    lattice_prior,
    neighbour_tile_ids,
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

if TYPE_CHECKING:
    from vineyard.config import AppConfig
    from vineyard.config.sections_perception import RowsGuidedConfig
    from vineyard.pipeline.context import RunContext

FORBIDDEN_FILE: Final = "in_forbidden.parquet"
FORBIDDEN_LAYER: Final = "in_forbidden"
MAX_K: Final = 999
EVENT_GUIDED: Final = "rows_link.guided_rows"
EVENT_NO_PREP: Final = "rows_link.guided_no_tile_prep"
EVENT_ROUND: Final = "rows_link.guided_round"

_log = get_logger("pipeline.stages.rows_link")


def tile_priors(accepted: gpd.GeoDataFrame, cfg: AppConfig) -> dict[str, LatticePrior]:
    """tile_id -> lattice prior of its accepted candidates (tiles under prior_min_rows left out)."""
    d = cfg.rows.detect
    out: dict[str, LatticePrior] = {}
    for tile_id, grp in accepted.groupby("tile_id", sort=True):
        prior = lattice_prior(str(tile_id), list(grp.geometry), grp.angle_deg.to_numpy(dtype=np.float64),
                              grp.local_spacing_m.to_numpy(dtype=np.float64), cfg.rows_guided,
                              (d.spacing_min_m, d.spacing_max_m))
        if prior is not None:
            out[str(tile_id)] = prior
    return out


def target_tiles(cands: gpd.GeoDataFrame, tile_ids: Sequence[str], priors: Mapping[str, LatticePrior],
                 max_accepted: int, *, self_prior: bool = False
                 ) -> list[tuple[str, tuple[LatticePrior, ...]]]:
    """(tile, priors) of the selected tiles with <= max_accepted accepted rows and >= 1 prior: the priors of
    its 8 neighbours, plus its own lattice with self_prior (a partial tile completes its own block)."""
    accepted = cands[cands.rejected_reason.isna()].tile_id.astype(str).value_counts().to_dict()
    out = []
    for tile_id in sorted(tile_ids):
        if accepted.get(tile_id, 0) > max_accepted:
            continue
        found = [priors[n] for n in neighbour_tile_ids(tile_id) if n in priors]
        if self_prior and tile_id in priors:
            found.append(priors[tile_id])
        if found:
            out.append((tile_id, tuple(found)))
    return out


def neighbourhood(tile_ids: Sequence[str]) -> set[str]:
    """The tiles and their 8 neighbours (round k retries only the neighbourhood of the tiles that gained rows)."""
    return {n for t in tile_ids for n in (t, *neighbour_tile_ids(t))}


def _clip_px(clips: Mapping[str, BaseGeometry], tile_id: str) -> BaseGeometry:
    geom = clips.get(tile_id)
    if geom is None or geom.is_empty:
        return shapely.Polygon()
    tile = tile_ref(tile_id)
    return shapely.transform(geom, lambda xy: utm_to_px(tile, xy))


def _next_k(cands: gpd.GeoDataFrame, tile_id: str) -> int:
    ids = cands.cand_id[cands.tile_id.astype(str) == tile_id].astype(str)
    return max((parse_row_candidate_id(c)[1] for c in ids), default=0) + 1


@dataclass(frozen=True, eq=False)
class GuidedJob:
    """Picklable inputs of one tile's guided pass (runs in a pool worker)."""

    cache_dir: Path
    tile_id: str
    priors: tuple[LatticePrior, ...]
    clip_px: BaseGeometry
    forbidden: tuple[BaseGeometry, ...]
    neighbours: tuple[shapely.LineString, ...]
    existing: tuple[shapely.LineString, ...]
    params: DetectParams
    g: RowsGuidedConfig


def run_guided_job(job: GuidedJob) -> tuple[GuidedResult, int] | None:
    """(guided result, n veg px) of one tile; None without tile_prep output or on an all-nodata tile."""
    cache = job.cache_dir
    if not all(f(cache, job.tile_id).is_file() for f in (stats_path, veg_mask_path, valid_mask_path)):
        return None
    if load_tile_stats(cache, job.tile_id).status == TileStatus.EMPTY_NODATA:
        return None
    tile = tile_ref(job.tile_id)
    veg = load_veg_mask(cache, job.tile_id)
    if job.forbidden:
        veg = veg & ~forbidden_mask(list(job.forbidden), tile)
    inp = GuidedInputs(veg=veg, clip_px=job.clip_px, tile=tile, valid=load_valid_mask(cache, job.tile_id),
                       neighbour_utm=job.neighbours)
    return guided_detect(inp, job.priors, list(job.existing), job.params, job.g), int(veg.sum())


def _job(ctx: RunContext, accepted: gpd.GeoDataFrame, tile_id: str, priors: tuple[LatticePrior, ...],
         clips: Mapping[str, BaseGeometry], forbidden: gpd.GeoDataFrame | None) -> GuidedJob:
    tiles = accepted.tile_id.astype(str)
    tile = tile_ref(tile_id)
    return GuidedJob(
        cache_dir=ctx.paths.cache_dir, tile_id=tile_id, priors=priors, clip_px=_clip_px(clips, tile_id),
        forbidden=tuple(forbidden_in_tile(forbidden, tile)) if forbidden is not None else (),
        neighbours=tuple(accepted.geometry[tiles.isin(neighbour_tile_ids(tile_id))]),
        existing=tuple(accepted.geometry[tiles == tile_id]), params=DetectParams.from_config(ctx.cfg),
        g=ctx.cfg.rows_guided)


def _frame(ctx: RunContext, cands: gpd.GeoDataFrame, tile_id: str, res: GuidedResult, n_veg: int,
           model_version: str) -> gpd.GeoDataFrame:
    k0 = _next_k(cands, tile_id)
    if k0 + len(res.candidates) - 1 > MAX_K:
        raise StageError("guided rows overflow the candidate K range", stage="rows_link", tile_id=tile_id)
    numbered = tuple(replace(c, k=k0 + i) for i, c in enumerate(res.candidates))
    det = TileDetection(tile_id=tile_id, status_hint="ok", method=METHOD_GUIDED, n_veg_px=n_veg,
                        orientations=res.orientations, candidates=numbered)
    return detection_to_candidates(det, tile_ref(tile_id), run_id=ctx.run_id, model_version=model_version)


def _round(ctx: RunContext, cands: gpd.GeoDataFrame, targets: list[tuple[str, tuple[LatticePrior, ...]]],
           clips: Mapping[str, BaseGeometry], forbidden: gpd.GeoDataFrame | None,
           model_version: str) -> list[gpd.GeoDataFrame]:
    accepted = cands[cands.rejected_reason.isna()]
    jobs = [_job(ctx, accepted, t, pr, clips, forbidden) for t, pr in targets]
    frames = []
    for job, out in zip(jobs, parallel_map(run_guided_job, jobs, key=lambda j: j.tile_id, workers=ctx.workers,
                                           chunksize=1, maxtasksperchild=ctx.cfg.runtime.maxtasksperchild),
                        strict=True):
        if not out.ok:
            raise StageError("guided pass failed", stage="rows_link", tile_id=job.tile_id, error=out.error,
                             traceback=out.traceback)
        if out.value is None:
            log_event(_log, EVENT_NO_PREP, stage="rows_link", tile_id=job.tile_id)
            continue
        res, n_veg = out.value
        if res.candidates:
            frames.append(_frame(ctx, cands, job.tile_id, res, n_veg, model_version))
            log_event(_log, EVENT_GUIDED, stage="rows_link", tile_id=job.tile_id, n_rows=len(res.candidates))
    return frames


def add_guided_candidates(ctx: RunContext, cands: gpd.GeoDataFrame, tile_ids: Sequence[str],
                          clips: Mapping[str, BaseGeometry], model_version: str) -> gpd.GeoDataFrame:
    """cands plus the guided rows of the target tiles (unchanged when rows_guided.enabled is false)."""
    g = ctx.cfg.rows_guided
    if not g.enabled or cands.empty:
        return cands
    forbidden = None
    path = ctx.paths.static_layers_dir / FORBIDDEN_FILE
    if ctx.cfg.rows.detect.mask_forbidden and path.is_file():
        forbidden = read_layer(path, FORBIDDEN_LAYER)
    selected: set[str] = set(tile_ids)
    for round_no in range(1, g.rounds + 1):
        priors = tile_priors(cands[cands.rejected_reason.isna()], ctx.cfg)
        targets = [(t, pr) for t, pr in target_tiles(cands, sorted(selected), priors, g.target_max_accepted,
                                                      self_prior=g.self_prior)]
        frames = _round(ctx, cands, targets, clips, forbidden, model_version)
        log_event(_log, EVENT_ROUND, stage="rows_link", round=round_no, n_targets=len(targets),
                  n_tiles=len(frames), n_rows=int(sum(len(f) for f in frames)))
        if not frames:
            break
        joined = pd.concat([cands, *frames], ignore_index=True)
        cands = gpd.GeoDataFrame(joined, geometry="geometry", crs=cands.crs).sort_values(
            "cand_id", kind="stable").reset_index(drop=True)
        gained = [str(f.tile_id.iloc[0]) for f in frames]
        selected = neighbourhood(gained) & set(tile_ids)
    return cands
