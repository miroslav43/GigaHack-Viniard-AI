"""Optional CLIP-probe re-scoring of selected boxes with src/AI's ``models/waste-probe/v1``.

Crops follow the pipeline's probe policy (``crops.crop_window``: box + context, >= min px, square, resized
to the CLIP input). final = 0.7 * p_model + 0.3 * probe. The final score only re-ranks (caps, export note);
it never removes a box, in line with the recall-oriented policy.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

from fte.paths import AI_ROOT, AI_TILES

log = logging.getLogger(__name__)

MODEL_WEIGHT: Final = 0.7
PROBE_WEIGHT: Final = 0.3
TILE_EXTENT_PX: Final = 2048


def combine(p_model: np.ndarray, probe: np.ndarray) -> np.ndarray:
    return MODEL_WEIGHT * np.asarray(p_model, float) + PROBE_WEIGHT * np.asarray(probe, float)


def _crops(frame: pd.DataFrame, image_px: int, context: float, min_px: int) -> np.ndarray:
    from vineyard.geo.raster import read_tile
    from vineyard.perception.waste.crops import crop_window, extract_crop
    from vineyard.perception.waste.types import BoxPx

    crops = np.zeros((len(frame), image_px, image_px, 3), np.uint8)
    for tile_id, sub in frame.groupby("tile_id", sort=False):
        rgb = read_tile(AI_TILES / f"{tile_id}.tif")
        for pos, r in zip(frame.index.get_indexer(sub.index), sub.itertuples(index=False), strict=True):
            box = BoxPx(float(r.xtl), float(r.ytl), float(r.xbr), float(r.ybr))
            crops[pos] = extract_crop(rgb, crop_window(box, context, min_px, TILE_EXTENT_PX), image_px)
    return crops


def probe_scores(frame: pd.DataFrame, device: str = "cpu", batch: int = 64) -> np.ndarray:
    """P(waste) of the probe for each box of ``frame`` (tile_id, xtl, ytl, xbr, ybr)."""
    from vineyard.config.loader import load_config
    from vineyard.perception.waste.clip_embed import ClipParams, load_openclip
    from vineyard.perception.waste.probe import load_probe, score_probe

    if frame.empty:
        return np.zeros(0)
    cfg = load_config().waste.probe
    embedder = load_openclip(ClipParams.from_config(cfg), device)
    probe = load_probe(AI_ROOT / "models", "waste-probe", "v1")
    crops = _crops(frame.reset_index(drop=True), embedder.image_px, cfg.crop_context, cfg.crop_min_px)
    parts = [score_probe(probe, embedder.embed(crops[k:k + batch])) for k in range(0, len(crops), batch)]
    return np.concatenate(parts)


def rescore(frame: pd.DataFrame, device: str = "cpu") -> pd.DataFrame:
    """New frame: p_model (old score), probe, score = combined."""
    base = frame.reset_index(drop=True)
    probe = probe_scores(base, device)
    return base.assign(p_model=base["score"], probe=probe, score=combine(base["score"], probe))
