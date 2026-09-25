"""Greedy NMS of waste boxes (A§4.9 2d): one box per cluster, since duplicates are FP at F1@IoU 0.3.

Boxes of different tiles never suppress each other (tile px coordinates are per tile).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from vineyard.perception.waste.types import Candidate, RejectReason


def box_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """(N,N) IoU of (N,4) xyxy boxes."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    ix = np.clip(
        np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0.0, None
    )
    iy = np.clip(
        np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0.0, None
    )
    inter = ix * iy
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area[:, None] + area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _rank_order(cands: Sequence[Candidate], rank: Mapping[str, float]) -> list[int]:
    missing = sorted(c.cand_key for c in cands if c.cand_key not in rank)
    if missing:
        raise KeyError(f"no rank score for candidates {missing[:5]}")
    return sorted(range(len(cands)), key=lambda i: (-rank[cands[i].cand_key], cands[i].cand_key))


def nms_keep(cands: Sequence[Candidate], rank: Mapping[str, float], iou_max: float) -> tuple[bool, ...]:
    """Keep flags (input order): a box is dropped when IoU > iou_max with a better-ranked kept box
    of the same tile (ties broken by cand_key)."""
    keep = [False] * len(cands)
    kept_by_tile: dict[str, list[int]] = {}
    for i in _rank_order(cands, rank):
        tile = cands[i].tile_id
        others = kept_by_tile.get(tile, [])
        if all(cands[i].box.iou(cands[j].box) <= iou_max for j in others):
            keep[i] = True
            kept_by_tile[tile] = [*others, i]
    return tuple(keep)


def nms(cands: Sequence[Candidate], rank: Mapping[str, float], iou_max: float) -> tuple[Candidate, ...]:
    """Survivors of greedy NMS, in input order."""
    keep = nms_keep(cands, rank, iou_max)
    return tuple(c for c, k in zip(cands, keep, strict=True) if k)


def mark_suppressed(
    cands: Sequence[Candidate], rank: Mapping[str, float], iou_max: float
) -> tuple[Candidate, ...]:
    """Same order as `cands`; NMS runs on the not-yet-rejected ones and the losers get reason `nms`."""
    live = [i for i, c in enumerate(cands) if not c.rejected]
    keep = nms_keep([cands[i] for i in live], rank, iou_max)
    lost = {i for i, k in zip(live, keep, strict=True) if not k}
    return tuple(replace(c, reject_reason=RejectReason.NMS) if i in lost else c for i, c in enumerate(cands))
