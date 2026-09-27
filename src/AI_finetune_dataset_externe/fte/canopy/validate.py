"""Validation of the canopy model.

(a) Official canopy metric (0.6·IoU + 0.4·F1@0.5) on the two example tiles through vineyard's own
    extraction (fusion B = p >= t, C = ExG ∧ p) with the base run's model row axes as corridors.
(b) ICAERUS val instance F1@0.5: predicted plants = connected components of (c0 >= 0.5 ∧ c1 < 0.5)
    (>= 20 px) vs the GT instances, pixel-IoU greedy 1:1 matching inside the labelled cell (alpha > 0).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fte.canopy.icaerus_samples import IcaerusSample
from fte.canopy.infer import predict_image, read_rgb
from fte.paths import AI_WORK, BASE_RUN, EXAMPLE_TILES

log = logging.getLogger(__name__)

PROB_PX = 1024
MIN_INSTANCE_PX = 20
MATCH_IOU = 0.5
VARIANTS = ("B", "C")


# ------------------------------------------------------------------ instance F1 (pure)


def instances_from_probs(c0: np.ndarray, c1: np.ndarray, min_px: int = MIN_INSTANCE_PX,
                         thr: float = 0.5) -> np.ndarray:
    """int32 label map of plant instances: components of (c0 >= thr ∧ c1 < thr) with >= min_px pixels."""
    core = ((c0 >= thr) & (c1 < thr)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
    keep = np.zeros(n, np.int32)
    ids = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_px]
    keep[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return keep[lab]


@dataclass(frozen=True)
class MatchCounts:
    tp: int
    fp: int
    fn: int

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else 1.0

    def __add__(self, other: MatchCounts) -> MatchCounts:
        return MatchCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def match_instances(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, iou_thr: float = MATCH_IOU,
                    min_px: int = MIN_INSTANCE_PX) -> MatchCounts:
    """Greedy 1:1 matching by pixel IoU inside ``valid``; predictions mostly outside ``valid`` are dropped."""
    v = valid.astype(bool)
    p_all = np.bincount(pred.ravel(), minlength=1)
    p_in = np.bincount(pred[v].ravel(), minlength=p_all.size)
    pred_ids = [k for k in range(1, p_all.size) if p_all[k] > 0 and p_in[k] >= 0.5 * p_all[k]]
    g_in = np.bincount(gt[v].ravel().astype(np.int64), minlength=1)
    gt_ids = [k for k in range(1, g_in.size) if g_in[k] >= min_px]
    if not pred_ids or not gt_ids:
        return MatchCounts(0, len(pred_ids), len(gt_ids))
    pv, gv = pred[v].astype(np.int64), gt[v].astype(np.int64)
    both = (pv > 0) & (gv > 0)
    pairs, inter = np.unique(pv[both] * (int(gv.max()) + 1) + gv[both], return_counts=True)
    cand = []
    for code, n in zip(pairs, inter, strict=True):
        pi, gi = divmod(int(code), int(gv.max()) + 1)
        union = p_in[pi] + g_in[gi] - n
        cand.append((n / union, pi, gi))
    pset, gset, used_p, used_g = set(pred_ids), set(gt_ids), set(), set()
    for iou, pi, gi in sorted(cand, reverse=True):
        if iou < iou_thr:
            break
        if pi in pset and gi in gset and pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    tp = len(used_p)
    return MatchCounts(tp, len(pred_ids) - tp, len(gt_ids) - tp)


# ------------------------------------------------------------------ model forward on small crops


@torch.no_grad()
def predict_crop(model: torch.nn.Module, rgb: np.ndarray, device: torch.device, multiple: int = 32) -> np.ndarray:
    """(K, H, W) probabilities of a small crop, zero-padded to a multiple of 32."""
    h, w = rgb.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)[None]
    ph, pw = (-h) % multiple, (-w) % multiple
    x = F.pad(x, (0, pw, 0, ph))
    p = torch.sigmoid(model(x.to(device)))[0, :, :h, :w]
    return p.cpu().numpy()


def icaerus_instance_f1(model: torch.nn.Module, samples: Sequence[IcaerusSample], device: torch.device) -> dict:
    total = MatchCounts(0, 0, 0)
    for s in samples:
        probs = predict_crop(model, s.rgb, device)
        total = total + match_instances(instances_from_probs(probs[0], probs[1]), s.inst, s.valid)
    return {"f1": total.f1, "tp": total.tp, "fp": total.fp, "fn": total.fn, "n_crops": len(samples)}


# ------------------------------------------------------------------ official metric on the examples


class ExampleValidator:
    """Loads the example tiles once (RGB, masks, corridors of the base run's rows, reference canopies)."""

    def __init__(self, tile_ids: Sequence[str] = EXAMPLE_TILES, base_run: str = BASE_RUN,
                 work_dir: Path = AI_WORK) -> None:
        from vineyard.config import load_config
        from vineyard.nn.validate import load_model_rows, model_axes, reference_val_tiles

        self.cfg = load_config()
        tiles = reference_val_tiles(self.cfg, list(tile_ids), "LATEST_REFERENCE")
        self.tiles = model_axes(tiles, load_model_rows(work_dir, base_run), self.cfg.canopy.rows_margin_m)
        self.rgb = {t: read_rgb(t) for t in tile_ids}

    def score_probs(self, c0_by_tile: dict[str, np.ndarray], variants: Sequence[str] = VARIANTS) -> dict:
        """c0 probabilities (2048² float or uint8) per tile -> per-variant per-tile scores and means."""
        from vineyard.contracts.enums import FusionVariant
        from vineyard.nn.validate import score_tile

        out: dict[str, dict] = {}
        for v in variants:
            scores = {}
            for tile in self.tiles:
                prob = c0_by_tile[tile.tile_id]
                u8 = prob if prob.dtype == np.uint8 else np.rint(np.clip(prob, 0, 1) * 255).astype(np.uint8)
                small = cv2.resize(u8, (PROB_PX, PROB_PX), interpolation=cv2.INTER_AREA)
                s = score_tile(tile, small, FusionVariant(v), self.cfg)
                scores[tile.tile_id] = {"iou": s.iou, "f1": s.f1, "score": s.score, "n_pred": s.n_pred,
                                        "n_ref": s.n_ref}
            out[v] = {"tiles": scores, "mean": float(np.mean([s["score"] for s in scores.values()]))}
        return out

    def baseline(self) -> dict:
        """Classical ExG-only extraction (variant A) for reference."""
        from vineyard.contracts.enums import FusionVariant
        from vineyard.nn.validate import score_tile

        scores = {t.tile_id: score_tile(t, None, FusionVariant.A, self.cfg).score for t in self.tiles}
        return {"tiles": scores, "mean": float(np.mean(list(scores.values())))}

    def evaluate(self, model: torch.nn.Module, device: torch.device, tta: int = 1, batch: int = 6) -> dict:
        c0 = {t: predict_image(model, rgb, device, tta=tta, batch=batch)[0] for t, rgb in self.rgb.items()}
        return self.score_probs(c0)
