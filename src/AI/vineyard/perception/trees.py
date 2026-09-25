"""Tree-in-row filter (02 §3.7, P1, behind canopy.tree_filter_enabled = false).

A vegetation blob touching a row corridor is a tree when it is large (> orchard.tree_blob_area_m2),
compact (minAreaRect axis ratio < orchard.tree_blob_axis_ratio_max) and wider across than
canopy.max_perp_width_tree_m. Elongated grass along the row never qualifies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import cv2
import geopandas as gpd
import numpy as np

from vineyard.geo.tiling import GSD_M, TileRef
from vineyard.perception.corridor import corridor_labels
from vineyard.perception.types import F32, BoolMask

if TYPE_CHECKING:
    from vineyard.config.sections_perception import CanopyConfig, OrchardConfig

_CONNECTIVITY: Final = 8


@dataclass(frozen=True)
class TreeParams:
    min_area_m2: float
    max_axis_ratio: float
    min_width_m: float
    max_prob: float | None = None  # with an NN probability, trees must also be "not vine"


def blob_is_tree(points_px: np.ndarray, area_px: int, p: TreeParams) -> bool:
    """Geometric tree test on one blob's pixel coordinates (N, 2) and pixel count."""
    if area_px * GSD_M * GSD_M <= p.min_area_m2 or len(points_px) < 3:
        return False
    (_, _), (w, h), _ = cv2.minAreaRect(points_px.astype(np.float32))
    short, long_ = sorted((float(w), float(h)))
    if short <= 0:
        return False
    return long_ / short < p.max_axis_ratio and short * GSD_M > p.min_width_m


def tree_mask_from_labels(veg: BoolMask, labels: np.ndarray, p: TreeParams, prob: F32 | None = None) -> BoolMask:
    """Bool mask of tree blobs among the veg components that touch a corridor (labels > 0)."""
    n, comp, stats, _ = cv2.connectedComponentsWithStats(veg.astype(np.uint8), connectivity=_CONNECTIVITY)
    out = np.zeros(veg.shape, dtype=bool)
    min_px = p.min_area_m2 / (GSD_M * GSD_M)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > min_px]
    for i in big:
        x, y, w, h = (int(v) for v in stats[i, :4])
        crop = comp[y : y + h, x : x + w] == i
        if not (labels[y : y + h, x : x + w][crop] > 0).any():
            continue
        if prob is not None and p.max_prob is not None and float(prob[y : y + h, x : x + w][crop].mean()) >= p.max_prob:
            continue
        ys, xs = np.nonzero(crop)
        if blob_is_tree(np.column_stack((xs + x, ys + y)), int(stats[i, cv2.CC_STAT_AREA]), p):
            out[y : y + h, x : x + w] |= crop
    return out


def tree_mask(veg: BoolMask, pieces: gpd.GeoDataFrame, tile: TileRef, cfg: CanopyConfig, orchard: OrchardConfig,
              prob: F32 | None = None, prob_threshold: float | None = None) -> BoolMask:
    """Tree-in-row mask of one tile (bool, tile shape)."""
    params = TreeParams(min_area_m2=orchard.tree_blob_area_m2, max_axis_ratio=orchard.tree_blob_axis_ratio_max,
                        min_width_m=cfg.max_perp_width_tree_m, max_prob=prob_threshold)
    labels = corridor_labels(pieces, tile, cfg.corridor_half_m)
    return tree_mask_from_labels(veg, labels, params, prob)
