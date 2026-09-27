"""Vetoes for predicted waste boxes (consistent with the src/AI rule stage).

(a) forbidden: the box intersects the forbidden zone (village core, buildings) - src/AI rejects a
    candidate whose box polygon intersects the zone, same here;
(b) look-alike: the box matches a cached src/AI candidate rejected as tube_shape / near_axis /
    periodic_tube / hose / vehicle: IoU >= 0.3, or the box centre lies inside that candidate's box and the
    box is at most 2x its area (a heap containing a tube is kept).
The CLIP probe re-scoring is not wired (see the report).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

from fte.convert.waste_negatives import load_candidates
from fte.paths import REPO_ROOT
from fte.waste.boxes import box_iou, centres

FORBIDDEN_PATH: Final = REPO_ROOT / "data & info" / "02_route" / "forbidden.geojson"
LOOKALIKE_REASONS: Final = ("tube_shape", "near_axis", "periodic_tube", "hose", "vehicle")
CRS_EPSG: Final = 32635


@dataclass(frozen=True)
class VetoParams:
    reasons: tuple[str, ...] = LOOKALIKE_REASONS
    iou: float = 0.3
    centre_area_ratio: float = 2.0
    forbidden_path: Path = FORBIDDEN_PATH


@lru_cache(maxsize=2)
def load_forbidden(path: Path = FORBIDDEN_PATH) -> BaseGeometry | None:
    """Union of the forbidden polygons in EPSG:32635 (None when the file is missing)."""
    import geopandas as gpd

    if not Path(path).is_file():
        return None
    frame = gpd.read_file(path)
    if frame.crs is not None and frame.crs.to_epsg() != CRS_EPSG:
        frame = frame.to_crs(CRS_EPSG)
    return shapely.union_all(frame.geometry.to_numpy())


def forbidden_in_tile(zone_utm: BaseGeometry | None, tile_id: str) -> BaseGeometry | None:
    """Forbidden zone clipped to the tile, in tile px (None when it misses the tile)."""
    from vineyard.geo.tiling import tile_ref, utm_to_px

    if zone_utm is None:
        return None
    t = tile_ref(tile_id)
    tile_utm = shapely_box(t.x0, t.y0 - 2048 * 0.025, t.x0 + 2048 * 0.025, t.y0)
    local = zone_utm.intersection(tile_utm)
    if local.is_empty or local.area <= 0:
        return None
    return shapely.transform(local, lambda xy: utm_to_px(t, xy))


def veto_forbidden(boxes: np.ndarray, zone_px: BaseGeometry | None) -> np.ndarray:
    if zone_px is None or len(boxes) == 0:
        return np.zeros(len(boxes), dtype=bool)
    polys = shapely.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    return shapely.intersects(polys, zone_px)


def veto_lookalike(boxes: np.ndarray, cands: np.ndarray, p: VetoParams) -> np.ndarray:
    """Bool per box: matches a rejected look-alike candidate (``cands`` xyxy)."""
    if len(boxes) == 0 or len(cands) == 0:
        return np.zeros(len(boxes), dtype=bool)
    iou_hit = (box_iou(boxes, cands) >= p.iou).any(axis=1)
    c = centres(boxes)
    inside = ((c[:, 0:1] >= cands[None, :, 0]) & (c[:, 0:1] <= cands[None, :, 2])
              & (c[:, 1:2] >= cands[None, :, 1]) & (c[:, 1:2] <= cands[None, :, 3]))
    area_b = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[:, None]
    area_c = ((cands[:, 2] - cands[:, 0]) * (cands[:, 3] - cands[:, 1]))[None, :]
    small = area_b <= p.centre_area_ratio * area_c
    return iou_hit | (inside & small).any(axis=1)


def tile_lookalikes(tile_id: str, reasons: tuple[str, ...]) -> np.ndarray:
    cands = load_candidates(tile_id)
    sub = cands[cands["reject_reason"].isin(reasons)]
    return sub[["xtl", "ytl", "xbr", "ybr"]].to_numpy(dtype=np.float64)


def apply_vetoes(frame: pd.DataFrame, p: VetoParams = VetoParams()) -> pd.DataFrame:
    """New frame with a ``veto`` column: '' (kept), 'forbidden' or 'lookalike'."""
    zone = load_forbidden(p.forbidden_path)
    parts = []
    for tile_id, sub in frame.groupby("tile_id", sort=True):
        boxes = sub[["xtl", "ytl", "xbr", "ybr"]].to_numpy(dtype=np.float64)
        forb = veto_forbidden(boxes, forbidden_in_tile(zone, str(tile_id)))
        look = veto_lookalike(boxes, tile_lookalikes(str(tile_id), p.reasons), p)
        veto = np.where(forb, "forbidden", np.where(look, "lookalike", ""))
        parts.append(sub.assign(veto=veto))
    if not parts:
        return frame.assign(veto=pd.Series(dtype=str))
    return pd.concat(parts, ignore_index=True)
