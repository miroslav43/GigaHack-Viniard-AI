"""Per-epoch validation: DroneWaste val-site box F1 (threshold grid), synthetic Sireț3 recall, and the
number of boxes on the 2 example tiles (which have no waste: every box is a false positive).

Heatmaps are computed once per image; boxes are re-extracted per threshold (components depend on t).
GT boxes below the minimum area and DroneWaste `ignore` objects are don't-care regions: a prediction
whose centre falls inside one is dropped instead of counted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import nn

from fte.convert.coco import IGNORE_ROLE, WASTE, StoreImage, WasteStore, read_rgb, xywh_to_xyxy
from fte.paths import AI_CACHE, AI_TILES, EXAMPLE_TILES
from fte.waste.boxes import BoxParams, MatchScore, drop_in_regions, heatmap_to_boxes, match_boxes
from fte.waste.predict import PredictSpec, predict_heat
from fte.waste.synval import SynvalItem

THRESHOLDS: Final = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SMALL_SPEC: Final = PredictSpec(window=512, overlap=64, tta_hflip=False)  # one input shape for val images
TILE_SPEC: Final = PredictSpec(window=1024, overlap=128, tta_hflip=False)


@dataclass(frozen=True)
class EvalCase:
    key: str
    heat: np.ndarray  # uint8 0..255
    gt: np.ndarray  # (N, 4) xyxy
    dont_care: np.ndarray  # (M, 4) xyxy


def to_u8(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.round(prob * 255.0), 0, 255).astype(np.uint8)


def split_gt(gt: np.ndarray, min_area_px: float) -> tuple[np.ndarray, np.ndarray]:
    """(kept, tiny) GT boxes by bbox area."""
    gt = gt.reshape(-1, 4)
    area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    return gt[area >= min_area_px].copy(), gt[area < min_area_px].copy()


def make_case(key: str, heat: np.ndarray, gt: np.ndarray, ignore: np.ndarray, min_area_px: float) -> EvalCase:
    kept, tiny = split_gt(gt, min_area_px)
    return EvalCase(key, heat, kept, np.concatenate([ignore.reshape(-1, 4), tiny]))


def score_cases(cases: Sequence[EvalCase], thresholds: Sequence[float],
                base: BoxParams = BoxParams()) -> dict[float, MatchScore]:
    out: dict[float, MatchScore] = {}
    for t in thresholds:
        p = BoxParams(threshold=t, min_area_m2=base.min_area_m2, max_area_m2=base.max_area_m2,
                      gsd_m=base.gsd_m, nms_iou=base.nms_iou)
        acc = MatchScore(0, 0, 0)
        for c in cases:
            acc = acc + match_boxes(drop_in_regions(heatmap_to_boxes(c.heat, p), c.dont_care), c.gt)
        out[float(t)] = acc
    return out


def best_threshold(scores: Mapping[float, MatchScore]) -> tuple[float, float]:
    """(t, F1) with the highest F1; ties go to the lower (recall-friendlier) threshold."""
    if not scores:
        raise ValueError("no scores")
    t = min(scores, key=lambda k: (-round(scores[k].f1, 6), k))
    return t, scores[t].f1


def _store_boxes(img: StoreImage, role: str) -> np.ndarray:
    boxes = [xywh_to_xyxy(o.bbox) for o in img.objects if o.role == role]
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def dronewaste_cases(model: nn.Module, device: torch.device, store: WasteStore,
                     images: Sequence[StoreImage], min_area_px: float) -> list[EvalCase]:
    spec = SMALL_SPEC
    cases = []
    for img in images:
        heat = to_u8(predict_heat(model, read_rgb(store.image_path(img)), device, spec))
        cases.append(make_case(img.file_name, heat, _store_boxes(img, WASTE), _store_boxes(img, IGNORE_ROLE),
                               min_area_px))
    return cases


def synval_cases(model: nn.Module, device: torch.device, items: Sequence[SynvalItem],
                 images: Mapping[str, np.ndarray], min_area_px: float) -> list[EvalCase]:
    spec = SMALL_SPEC
    return [make_case(it.key, to_u8(predict_heat(model, images[it.key], device, spec)), it.gt, it.ignore,
                      min_area_px) for it in items]


def example_heats(model: nn.Module, device: torch.device, crop: int | None = None) -> dict[str, np.ndarray]:
    """uint8 heatmaps of the 2 example tiles (nodata zeroed); ``crop`` limits to the top-left square."""
    from vineyard.geo.raster import read_tile
    from vineyard.pipeline.tile_cache import load_valid_mask

    spec = TILE_SPEC
    out = {}
    for tile_id in EXAMPLE_TILES:
        rgb = read_tile(AI_TILES / f"{tile_id}.tif")
        valid = load_valid_mask(AI_CACHE, tile_id)
        if crop:
            rgb, valid = rgb[:crop, :crop], valid[:crop, :crop]
        heat = predict_heat(model, rgb, device, spec)
        out[tile_id] = to_u8(np.where(valid, heat, 0.0))
    return out


def count_boxes(heats: Mapping[str, np.ndarray], thresholds: Sequence[float]) -> dict[float, int]:
    return {float(t): int(sum(len(heatmap_to_boxes(h, BoxParams(threshold=t))) for h in heats.values()))
            for t in thresholds}


def summarise(scores: Mapping[float, MatchScore]) -> dict[str, dict[str, float]]:
    return {f"{t:.2f}": {"f1": round(s.f1, 4), "precision": round(s.precision, 4), "recall": round(s.recall, 4),
                         "tp": s.tp, "n_pred": s.n_pred, "n_ref": s.n_ref} for t, s in scores.items()}
