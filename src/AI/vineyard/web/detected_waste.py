"""Waste detections inside the vineyard blocks, for the web map only (not annotated, measured or routed).

Every waste candidate of a model run (layers/waste_candidates.parquet), whatever the rule filters decided,
whose box is at least `min_short_px` x `min_long_px` and whose centre lies inside a block outline. Boxes that
overlap an annotated waste box are left out (the annotated one is already on the map). Ids D00001... in
(tile, top, left) order, vineyard_id = the block holding the centre, confidence = the probe score (0 when
the candidate was never scored).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import geopandas as gpd
import numpy as np
import shapely

DETECTED_PREFIX: Final = "D"
ID_DIGITS: Final = 5
NMS_REASON: Final = "nms"  # a duplicate box of the same blob, not a filter decision
CANDIDATE_COLUMNS: Final = ("tile_id", "px_xtl", "px_ytl", "px_xbr", "px_ybr", "probe_p", "reject_reason")


@dataclass(frozen=True)
class DetectedWasteParams:
    min_short_px: float
    min_long_px: float

    def __post_init__(self) -> None:
        if not 0 < self.min_short_px <= self.min_long_px:
            raise ValueError(f"need 0 < min_short_px <= min_long_px, got {self.min_short_px}, {self.min_long_px}")


def _big_enough(cands: gpd.GeoDataFrame, p: DetectedWasteParams) -> np.ndarray:
    w = cands["px_xbr"].to_numpy(float) - cands["px_xtl"].to_numpy(float)
    h = cands["px_ybr"].to_numpy(float) - cands["px_ytl"].to_numpy(float)
    return (np.minimum(w, h) >= p.min_short_px) & (np.maximum(w, h) >= p.min_long_px)


def _confidence(value: object) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, v)) if math.isfinite(v) else 0.0


def detected_waste(
    cands: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, annotated: gpd.GeoDataFrame, p: DetectedWasteParams
) -> gpd.GeoDataFrame:
    """waste_id, vineyard_id, tile, confidence + box geometry (the candidates' CRS) of the kept detections."""
    missing = [c for c in CANDIDATE_COLUMNS if c not in cands.columns]
    if missing:
        raise ValueError(f"waste candidates lack columns {missing}")
    keep = _big_enough(cands, p) & (cands["reject_reason"].fillna("").astype(str) != NMS_REASON).to_numpy()
    sub = cands[keep]
    centres = gpd.GeoDataFrame(geometry=sub.geometry.centroid, crs=cands.crs, index=sub.index)
    hits = gpd.sjoin(centres, blocks[["vineyard_id", "geometry"]], predicate="within", how="inner")
    block_of = hits[~hits.index.duplicated()]["vineyard_id"]
    sub = sub.loc[block_of.index]
    taken = [g for g in annotated.geometry if g is not None and not g.is_empty]
    if taken and len(sub):
        tree = shapely.STRtree(taken)
        overlapping = np.zeros(len(sub), dtype=bool)
        overlapping[np.unique(tree.query(sub.geometry.to_numpy(), predicate="intersects")[0])] = True
        sub = sub[~overlapping]
    sub = sub.sort_values(["tile_id", "px_ytl", "px_xtl"], kind="stable")
    return gpd.GeoDataFrame({
        "waste_id": [f"{DETECTED_PREFIX}{k + 1:0{ID_DIGITS}d}" for k in range(len(sub))],
        "vineyard_id": [str(block_of[i]) for i in sub.index],
        "tile": [str(t) for t in sub["tile_id"]],
        "confidence": [_confidence(v) for v in sub["probe_p"]],
    }, geometry=list(sub.geometry), crs=cands.crs)
