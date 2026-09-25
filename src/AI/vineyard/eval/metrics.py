"""Official challenge metrics, vector-based (shapely, UTM metres) — cerinta_wine.txt, design 01 §2.6/§3.4.

Conventions for empty inputs: nothing to find and nothing found scores 1.0; one side empty scores 0.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields
from functools import reduce
from typing import Any, Final

import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.config import EvalConfig
from vineyard.contracts.enums import InterrowCover, RowStructure
from vineyard.eval.matching import (
    LayerObjects,
    Pair,
    TileObjects,
    line_candidates,
    match_polygons,
    optimal_one_to_one,
)

# Buffer resolution for the 0.4 m row tolerance: the polygonal arc deviates < 0.5 mm from the circle.
BUFFER_QUAD_SEGS: Final = 32


@dataclass(frozen=True)
class AreaOverlap:
    """Areas of the union intersection and union of two polygon sets."""

    inter_m2: float
    union_m2: float

    @property
    def iou(self) -> float:
        return 1.0 if self.union_m2 <= 0 else float(self.inter_m2 / self.union_m2)

    def merge(self, other: AreaOverlap) -> AreaOverlap:
        """Pooled overlap of disjoint areas (tiles)."""
        return AreaOverlap(self.inter_m2 + other.inter_m2, self.union_m2 + other.union_m2)


@dataclass(frozen=True)
class CanopyScore:
    iou: float
    f1: float
    score: float
    tp: int
    n_pred: int
    n_ref: int
    inter_m2: float = 0.0
    union_m2: float = 0.0


@dataclass(frozen=True)
class RowScore:
    """One-to-one match summary (rows, waste, interrows)."""

    f1: float
    tp: int
    n_pred: int
    n_ref: int
    matches: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class AttrScore:
    accuracy: float
    macro_f1: float
    score: float
    n_ref: int


# ------------------------------------------------------------------ helpers


def f1_from_counts(tp: int, n_pred: int, n_ref: int) -> float:
    """2TP / (n_pred + n_ref); 1.0 when both sets are empty."""
    total = n_pred + n_ref
    return 1.0 if total == 0 else float(2 * tp / total)


def weighted_canopy_score(iou: float, f1: float, *, w_iou: float, w_f1: float) -> float:
    return float(w_iou * iou + w_f1 * f1)


def _union(geoms: Sequence[BaseGeometry]) -> BaseGeometry:
    return shapely.union_all(list(geoms)) if len(geoms) else shapely.Polygon()


def union_area(geoms: Sequence[BaseGeometry]) -> float:
    """Area of the union (overlaps counted once)."""
    return float(_union(geoms).area)


def area_overlap(pred: Sequence[BaseGeometry], ref: Sequence[BaseGeometry]) -> AreaOverlap:
    """Intersection and union areas of union(pred) and union(ref)."""
    pu, ru = _union(pred), _union(ref)
    if pu.is_empty or ru.is_empty:
        return AreaOverlap(0.0, float(pu.area + ru.area))
    if pu.equals(ru):
        return AreaOverlap(float(pu.area), float(pu.area))
    inter = float(shapely.intersection(pu, ru).area)
    return AreaOverlap(inter, float(pu.area + ru.area - inter))


def class_iou(pred: Sequence[BaseGeometry], ref: Sequence[BaseGeometry]) -> float:
    """Class IoU on union areas (1.0 when both are empty)."""
    return area_overlap(pred, ref).iou


def _match_summary(matches: tuple[Pair, ...], n_pred: int, n_ref: int) -> RowScore:
    tp = len(matches)
    return RowScore(f1_from_counts(tp, n_pred, n_ref), tp, n_pred, n_ref, tuple((i, j) for i, j, _ in matches))


# ------------------------------------------------------------------ canopy, waste, penalty


def canopy_metrics(
    pred: Sequence[BaseGeometry], ref: Sequence[BaseGeometry], *, match_iou: float, w_iou: float, w_f1: float
) -> CanopyScore:
    """w_iou * class IoU + w_f1 * F1 of one-to-one matches at IoU >= match_iou (greedy is exact here)."""
    overlap = area_overlap(pred, ref)
    matches = match_polygons(pred, ref, min_iou=match_iou, method="greedy")
    f1 = f1_from_counts(len(matches), len(pred), len(ref))
    score = weighted_canopy_score(overlap.iou, f1, w_iou=w_iou, w_f1=w_f1)
    return CanopyScore(overlap.iou, f1, score, len(matches), len(pred), len(ref), overlap.inter_m2,
                       overlap.union_m2)


def waste_f1(pred: Sequence[BaseGeometry], ref: Sequence[BaseGeometry], *, match_iou: float) -> RowScore:
    """Box F1 with optimal one-to-one matching at IoU >= match_iou (duplicates become false positives)."""
    matches = match_polygons(pred, ref, min_iou=match_iou, method="optimal")
    return _match_summary(matches, len(pred), len(ref))


def fp_canopy_penalty(pred: Sequence[BaseGeometry], tile_area_m2: float, *, factor: float) -> float:
    """factor * share of the tile covered by predicted canopies (tiles without reference vineyard)."""
    if tile_area_m2 <= 0:
        raise ValueError(f"tile_area_m2 must be > 0, got {tile_area_m2}")
    return float(factor * union_area(pred) / tile_area_m2)


# ------------------------------------------------------------------ rows


def mutual_cover(a: BaseGeometry, b: BaseGeometry, tol_m: float) -> tuple[float, float]:
    """(share of a within tol of b, share of b within tol of a), exact on the line geometry."""
    if a.is_empty or b.is_empty or a.length <= 0 or b.length <= 0:
        return 0.0, 0.0
    a_in_b = a.intersection(b.buffer(tol_m, quad_segs=BUFFER_QUAD_SEGS)).length / a.length
    b_in_a = b.intersection(a.buffer(tol_m, quad_segs=BUFFER_QUAD_SEGS)).length / b.length
    return float(a_in_b), float(b_in_a)


def row_matches(
    pred: Sequence[BaseGeometry], ref: Sequence[BaseGeometry], *, tol_m: float, min_cover: float
) -> tuple[Pair, ...]:
    """Optimal one-to-one row matches (i_pred, j_ref, min mutual cover) where both covers >= min_cover."""
    i_idx, j_idx = line_candidates(pred, ref, tol_m)
    pairs = []
    for i, j in zip(i_idx.tolist(), j_idx.tolist(), strict=True):
        cov = min(mutual_cover(pred[i], ref[j], tol_m))
        if cov >= min_cover:
            pairs.append((i, j, cov))
    return optimal_one_to_one(pairs, min_score=min_cover)


def row_axis_f1(
    pred: Sequence[BaseGeometry], ref: Sequence[BaseGeometry], *, tol_m: float, min_cover: float
) -> RowScore:
    """Row axis F1: a match needs each axis to lie >= min_cover within tol_m of the other."""
    return _match_summary(row_matches(pred, ref, tol_m=tol_m, min_cover=min_cover), len(pred), len(ref))


# ------------------------------------------------------------------ attributes, grouping, values


def _classes(y_ref: Sequence[str], y_pred: Sequence[str | None], labels: Sequence[str]) -> list[str]:
    present = set(y_ref) | {p for p in y_pred if p in labels}
    known = [c for c in labels if c in present]
    return known + sorted(present - set(known))


def attribute_scores(y_ref: Sequence[str], y_pred: Sequence[str | None], labels: Sequence[str]) -> AttrScore:
    """mean(accuracy, macro-F1) over reference objects; a missing prediction (None) is always wrong."""
    if len(y_ref) != len(y_pred):
        raise ValueError(f"y_ref/y_pred length mismatch: {len(y_ref)} != {len(y_pred)}")
    if not y_ref:
        return AttrScore(1.0, 1.0, 1.0, 0)
    hits = [r == p for r, p in zip(y_ref, y_pred, strict=True)]
    accuracy = sum(hits) / len(y_ref)
    f1s = []
    for c in _classes(y_ref, y_pred, labels):
        tp = sum(1 for r, p in zip(y_ref, y_pred, strict=True) if r == c and p == c)
        n_ref_c, n_pred_c = sum(1 for r in y_ref if r == c), sum(1 for p in y_pred if p == c)
        f1s.append(f1_from_counts(tp, n_pred_c, n_ref_c))
    macro = sum(f1s) / len(f1s)
    return AttrScore(float(accuracy), float(macro), float((accuracy + macro) / 2), len(y_ref))


def _same_pairs(counts: Counter) -> int:
    return sum(n * (n - 1) // 2 for n in counts.values())


def grouping_consistency(ref_ids: Sequence[str | None], pred_ids: Sequence[str | None]) -> float:
    """Pair-counting F1 of two partitions of the same objects; blank/None ids are singletons."""
    if len(ref_ids) != len(pred_ids):
        raise ValueError(f"ref_ids/pred_ids length mismatch: {len(ref_ids)} != {len(pred_ids)}")
    keep = [k for k, (r, p) in enumerate(zip(ref_ids, pred_ids, strict=True)) if r and p]
    same_ref = _same_pairs(Counter(r for r in ref_ids if r))
    same_pred = _same_pairs(Counter(p for p in pred_ids if p))
    both = _same_pairs(Counter((ref_ids[k], pred_ids[k]) for k in keep))
    total = same_ref + same_pred
    return 1.0 if total == 0 else float(2 * both / total)


def relative_error(pred: float, ref: float) -> float:
    """|pred - ref| / ref; 0 when both are 0, +inf when only ref is 0."""
    if ref == 0:
        return 0.0 if pred == 0 else math.inf
    return float(abs(pred - ref) / abs(ref))


def value_score(pred: float, ref: float, tol: float) -> float:
    """max(0, 1 - relative error / tol)."""
    return float(max(0.0, 1.0 - relative_error(pred, ref) / tol))


def line_length(geoms: Sequence[BaseGeometry]) -> float:
    return float(sum(g.length for g in geoms))


# ------------------------------------------------------------------ tile-level sufficient statistics

ROW_STRUCTURE_LABELS: Final = tuple(v.value for v in RowStructure)
INTERROW_COVER_LABELS: Final = tuple(v.value for v in InterrowCover)
AttrPair = tuple[str, str | None]  # (reference value, predicted value or None when unmatched)
IdPair = tuple[str | None, str | None]  # (reference vineyard_id, predicted vineyard_id) of a matched object


@dataclass(frozen=True)
class EvalParams:
    """Metric parameters (eval.* config keys)."""

    canopy_match_iou: float
    canopy_w_iou: float
    canopy_w_f1: float
    row_tol_m: float
    row_min_cover: float
    interrow_match_iou: float
    waste_match_iou: float
    count_tol: float
    length_tol: float
    fp_canopy_factor: float

    @classmethod
    def from_config(cls, cfg: EvalConfig) -> EvalParams:
        return cls(**{f.name: float(getattr(cfg, f.name)) for f in fields(cls)})


@dataclass(frozen=True)
class Counts:
    tp: int = 0
    n_pred: int = 0
    n_ref: int = 0

    @property
    def f1(self) -> float:
        return f1_from_counts(self.tp, self.n_pred, self.n_ref)

    def merge(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.n_pred + other.n_pred, self.n_ref + other.n_ref)


@dataclass(frozen=True)
class PredRef:
    pred: float = 0.0
    ref: float = 0.0

    def merge(self, other: PredRef) -> PredRef:
        return PredRef(self.pred + other.pred, self.ref + other.ref)


@dataclass(frozen=True)
class IdSets:
    pred: tuple[str, ...] = ()
    ref: tuple[str, ...] = ()

    def merge(self, other: IdSets) -> IdSets:
        return IdSets(tuple(sorted(set(self.pred) | set(other.pred))), tuple(sorted(set(self.ref) | set(other.ref))))


@dataclass(frozen=True)
class TileStats:
    """Additive statistics of one tile; pooling several tiles is `merge_stats`."""

    canopy_overlap: AreaOverlap
    canopy: Counts
    canopy_tiles: int  # tiles scored by IoU/F1 (reference has canopies)
    fp_penalty: float  # summed over tiles without reference canopies
    fp_tiles: int
    rows: Counts
    interrow_overlap: AreaOverlap
    interrows: Counts
    waste: Counts
    row_structure: tuple[AttrPair, ...]
    interrow_cover: tuple[AttrPair, ...]
    groups: tuple[IdPair, ...]
    blocks: IdSets
    row_ids: IdSets
    canopy_area: PredRef
    interrow_area: PredRef
    row_length: PredRef


def _combine(a: Any, b: Any) -> Any:
    if isinstance(a, tuple | int | float):
        return a + b
    return a.merge(b)


def merge_stats(stats: Sequence[TileStats]) -> TileStats:
    if not stats:
        raise ValueError("merge_stats needs at least one TileStats")
    return reduce(lambda a, b: TileStats(**{f.name: _combine(getattr(a, f.name), getattr(b, f.name))
                                            for f in fields(TileStats)}), stats)


def _attr_pairs(pred: LayerObjects, ref: LayerObjects, matches: tuple[Pair, ...]) -> tuple[AttrPair, ...]:
    pred_of = {j: i for i, j, _ in matches}
    return tuple((r, pred.attrs[pred_of[j]] if j in pred_of else None)
                 for j, r in enumerate(ref.attrs) if r is not None)


def _group_pairs(pred: LayerObjects, ref: LayerObjects, matches: tuple[Pair, ...]) -> tuple[IdPair, ...]:
    return tuple((ref.vineyard_ids[j], pred.vineyard_ids[i]) for i, j, _ in sorted(matches, key=lambda m: m[1]))


def _ids(values: Sequence[str | None]) -> tuple[str, ...]:
    return tuple(sorted({v for v in values if v}))


def _counts(matches: tuple[Pair, ...], pred: LayerObjects, ref: LayerObjects) -> Counts:
    return Counts(len(matches), len(pred.geoms), len(ref.geoms))


def _canopy_part(pred: LayerObjects, ref: LayerObjects, p: EvalParams,
                 tile_area_m2: float) -> tuple[dict[str, Any], tuple[Pair, ...]]:
    """(TileStats canopy fields, matches); a tile without reference canopies only gets the FP penalty."""
    if not ref.geoms:
        penalty = fp_canopy_penalty(pred.geoms, tile_area_m2, factor=p.fp_canopy_factor)
        return {"canopy_overlap": AreaOverlap(0.0, 0.0), "canopy": Counts(0, 0, 0), "canopy_tiles": 0,
                "fp_penalty": penalty, "fp_tiles": 1}, ()
    matches = match_polygons(pred.geoms, ref.geoms, min_iou=p.canopy_match_iou, method="greedy")
    return {"canopy_overlap": area_overlap(pred.geoms, ref.geoms), "canopy": _counts(matches, pred, ref),
            "canopy_tiles": 1, "fp_penalty": 0.0, "fp_tiles": 0}, matches


def _measures(pred: TileObjects, ref: TileObjects) -> dict[str, Any]:
    def vids(t: TileObjects) -> tuple[str, ...]:
        return _ids([v for lay in (t.canopies, t.rows, t.interrows, t.waste) for v in lay.vineyard_ids])

    return {"blocks": IdSets(vids(pred), vids(ref)),
            "row_ids": IdSets(_ids(pred.rows.object_ids), _ids(ref.rows.object_ids)),
            "canopy_area": PredRef(union_area(pred.canopies.geoms), union_area(ref.canopies.geoms)),
            "interrow_area": PredRef(union_area(pred.interrows.geoms), union_area(ref.interrows.geoms)),
            "row_length": PredRef(line_length(pred.rows.geoms), line_length(ref.rows.geoms))}


def evaluate_tile(pred: TileObjects, ref: TileObjects, p: EvalParams, *, tile_area_m2: float) -> TileStats:
    """All official metrics of one tile as additive statistics."""
    canopy, c_match = _canopy_part(pred.canopies, ref.canopies, p, tile_area_m2)
    r_match = row_matches(pred.rows.geoms, ref.rows.geoms, tol_m=p.row_tol_m, min_cover=p.row_min_cover)
    i_match = match_polygons(pred.interrows.geoms, ref.interrows.geoms, min_iou=p.interrow_match_iou)
    w_match = match_polygons(pred.waste.geoms, ref.waste.geoms, min_iou=p.waste_match_iou, method="optimal")
    layers = ((pred.canopies, ref.canopies, c_match), (pred.rows, ref.rows, r_match),
              (pred.interrows, ref.interrows, i_match), (pred.waste, ref.waste, w_match))
    return TileStats(
        **canopy,
        rows=_counts(r_match, pred.rows, ref.rows),
        interrow_overlap=area_overlap(pred.interrows.geoms, ref.interrows.geoms),
        interrows=_counts(i_match, pred.interrows, ref.interrows),
        waste=_counts(w_match, pred.waste, ref.waste),
        row_structure=_attr_pairs(pred.rows, ref.rows, r_match),
        interrow_cover=_attr_pairs(pred.interrows, ref.interrows, i_match),
        groups=tuple(g for pl, rl, m in layers for g in _group_pairs(pl, rl, m)),
        **_measures(pred, ref),
    )

