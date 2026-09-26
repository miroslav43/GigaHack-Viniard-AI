"""rows_link helper: the neighbour-guided second pass (perception.rows_guided) on the collected candidates.

Priors come from the rows_detect accepted candidates only, so the pass never propagates its own rows and
the result does not depend on the tile order. Guided rows are appended as accepted candidates (method
"guided", soft flag "guided") with K numbers after the tile's own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
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
    LatticePrior,
    guided_detect,
    lattice_prior,
    neighbour_tile_ids,
)
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
    from vineyard.pipeline.context import RunContext

FORBIDDEN_FILE: Final = "in_forbidden.parquet"
FORBIDDEN_LAYER: Final = "in_forbidden"
MAX_K: Final = 999
EVENT_GUIDED: Final = "rows_link.guided_rows"
EVENT_NO_PREP: Final = "rows_link.guided_no_tile_prep"

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
                 max_accepted: int) -> list[tuple[str, tuple[LatticePrior, ...]]]:
    """(tile, neighbour priors) of the selected tiles with <= max_accepted accepted rows and >= 1 prior."""
    accepted = cands[cands.rejected_reason.isna()].tile_id.astype(str).value_counts().to_dict()
    out = []
    for tile_id in sorted(tile_ids):
        if accepted.get(tile_id, 0) > max_accepted:
            continue
        found = [priors[n] for n in neighbour_tile_ids(tile_id) if n in priors]
        if found:
            out.append((tile_id, tuple(found)))
    return out


def _clip_px(clips: Mapping[str, BaseGeometry], tile_id: str) -> BaseGeometry:
    geom = clips.get(tile_id)
    if geom is None or geom.is_empty:
        return shapely.Polygon()
    tile = tile_ref(tile_id)
    return shapely.transform(geom, lambda xy: utm_to_px(tile, xy))


def _veg(ctx: RunContext, tile_id: str, forbidden: gpd.GeoDataFrame | None) -> np.ndarray:
    veg = load_veg_mask(ctx.paths.cache_dir, tile_id)
    if forbidden is None:
        return veg
    tile = tile_ref(tile_id)
    return veg & ~forbidden_mask(forbidden_in_tile(forbidden, tile), tile)


def _next_k(cands: gpd.GeoDataFrame, tile_id: str) -> int:
    ids = cands.cand_id[cands.tile_id.astype(str) == tile_id].astype(str)
    return max((parse_row_candidate_id(c)[1] for c in ids), default=0) + 1


def _tile_frame(ctx: RunContext, cands: gpd.GeoDataFrame, tile_id: str, priors: tuple[LatticePrior, ...],
                clips: Mapping[str, BaseGeometry], forbidden: gpd.GeoDataFrame | None,
                model_version: str) -> gpd.GeoDataFrame | None:
    cfg = ctx.cfg
    cache = ctx.paths.cache_dir
    if not all(f(cache, tile_id).is_file() for f in (stats_path, veg_mask_path, valid_mask_path)):
        log_event(_log, EVENT_NO_PREP, stage="rows_link", tile_id=tile_id)
        return None
    if load_tile_stats(cache, tile_id).status == TileStatus.EMPTY_NODATA:
        return None
    g = cfg.rows_guided
    tile = tile_ref(tile_id)
    accepted = cands[cands.rejected_reason.isna()]
    tiles = accepted.tile_id.astype(str)
    neighbours = tuple(accepted.geometry[tiles.isin(neighbour_tile_ids(tile_id))])
    inp = GuidedInputs(veg=_veg(ctx, tile_id, forbidden), clip_px=_clip_px(clips, tile_id), tile=tile,
                       valid=load_valid_mask(cache, tile_id), neighbour_utm=neighbours)
    existing = list(accepted.geometry[tiles == tile_id])
    res = guided_detect(inp, priors, existing, DetectParams.from_config(cfg), g)
    if not res.candidates:
        return None
    k0 = _next_k(cands, tile_id)
    if k0 + len(res.candidates) - 1 > MAX_K:
        raise StageError("guided rows overflow the candidate K range", stage="rows_link", tile_id=tile_id)
    numbered = tuple(replace(c, k=k0 + i) for i, c in enumerate(res.candidates))
    det = TileDetection(tile_id=tile_id, status_hint="ok", method=METHOD_GUIDED, n_veg_px=int(inp.veg.sum()),
                        orientations=res.orientations, candidates=numbered)
    return detection_to_candidates(det, tile, run_id=ctx.run_id, model_version=model_version)


def add_guided_candidates(ctx: RunContext, cands: gpd.GeoDataFrame, tile_ids: Sequence[str],
                          clips: Mapping[str, BaseGeometry], model_version: str) -> gpd.GeoDataFrame:
    """cands plus the guided rows of the target tiles (unchanged when rows_guided.enabled is false)."""
    g = ctx.cfg.rows_guided
    if not g.enabled or cands.empty:
        return cands
    priors = tile_priors(cands[cands.rejected_reason.isna()], ctx.cfg)
    forbidden = None
    path = ctx.paths.static_layers_dir / FORBIDDEN_FILE
    if ctx.cfg.rows.detect.mask_forbidden and path.is_file():
        forbidden = read_layer(path, FORBIDDEN_LAYER)
    frames = []
    for tile_id, found in target_tiles(cands, tile_ids, priors, g.target_max_accepted):
        frame = _tile_frame(ctx, cands, tile_id, found, clips, forbidden, model_version)
        if frame is not None and len(frame):
            frames.append(frame)
            log_event(_log, EVENT_GUIDED, stage="rows_link", tile_id=tile_id, n_rows=len(frame))
    if not frames:
        return cands
    joined = pd.concat([cands, *frames], ignore_index=True)
    out = gpd.GeoDataFrame(joined, geometry="geometry", crs=cands.crs)
    return out.sort_values("cand_id", kind="stable").reset_index(drop=True)
