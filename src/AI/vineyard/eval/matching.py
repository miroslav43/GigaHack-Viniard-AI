"""Per-tile objects to match, candidate pairs (STRtree), pairwise IoU and one-to-one matching.

Indices always refer to the input sequences; matches are returned sorted by (pred, ref) index.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

Pair = tuple[int, int, float]
MatchMethod = Literal["greedy", "optimal"]
MATCH_METHODS: Final[tuple[str, ...]] = ("greedy", "optimal")
# Score tie-break weight in the assignment cost: small enough that cardinality always dominates.
_TIEBREAK_WEIGHT: Final = 1e-6
_EMPTY_IDX: Final = np.zeros(0, dtype=np.int64)


@dataclass(frozen=True)
class LayerObjects:
    """One layer of one tile: geometries plus aligned vineyard ids, object ids (row_id) and attribute values."""

    geoms: tuple[BaseGeometry, ...] = ()
    vineyard_ids: tuple[str | None, ...] = ()
    object_ids: tuple[str | None, ...] = ()
    attrs: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.geoms)
        object.__setattr__(self, "geoms", tuple(self.geoms))
        for name in ("vineyard_ids", "object_ids", "attrs"):
            values = tuple(getattr(self, name)) or (None,) * n
            if len(values) != n:
                raise ValueError(f"LayerObjects.{name} has {len(values)} values for {n} geometries")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class TileObjects:
    tile_id: str
    canopies: LayerObjects = field(default_factory=LayerObjects)
    rows: LayerObjects = field(default_factory=LayerObjects)
    interrows: LayerObjects = field(default_factory=LayerObjects)
    waste: LayerObjects = field(default_factory=LayerObjects)


def clean_polygonal(geom: BaseGeometry | None) -> BaseGeometry:
    """Valid polygonal version of `geom` (make_valid, non-polygonal parts dropped); may be empty."""
    if geom is None or geom.is_empty:
        return Polygon()
    fixed = geom if geom.is_valid else shapely.make_valid(geom, method="structure", keep_collapsed=False)
    parts = [p for p in shapely.get_parts(fixed) if isinstance(p, Polygon) and not p.is_empty]
    if not parts:
        return Polygon()
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def pair_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    """Area IoU of two polygonal geometries (0 when either is empty or they are disjoint)."""
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return float(inter / union) if union > 0 else 0.0


def _as_array(geoms: Sequence[BaseGeometry]) -> np.ndarray:
    return np.asarray(list(geoms), dtype=object)


def iou_pairs(
    pred: Sequence[BaseGeometry], ref: Sequence[BaseGeometry]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i_pred, j_ref, iou) for every intersecting pair with a positive intersection area."""
    if not len(pred) or not len(ref):
        return _EMPTY_IDX, _EMPTY_IDX, np.zeros(0)
    p_arr, r_arr = _as_array(pred), _as_array(ref)
    tree = shapely.STRtree(r_arr)
    i, j = tree.query(p_arr, predicate="intersects")
    inter = shapely.area(shapely.intersection(p_arr[i], r_arr[j]))
    union = shapely.area(p_arr[i]) + shapely.area(r_arr[j]) - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    keep = inter > 0
    order = np.lexsort((j[keep], i[keep]))
    return i[keep][order].astype(np.int64), j[keep][order].astype(np.int64), iou[keep][order]


def line_candidates(
    pred: Sequence[BaseGeometry], ref: Sequence[BaseGeometry], tol_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """(i_pred, j_ref) of line pairs that come within `tol_m` of each other somewhere."""
    if not len(pred) or not len(ref):
        return _EMPTY_IDX, _EMPTY_IDX
    tree = shapely.STRtree(_as_array(ref))
    i, j = tree.query(_as_array(pred), predicate="dwithin", distance=tol_m)
    order = np.lexsort((j, i))
    return i[order].astype(np.int64), j[order].astype(np.int64)


def _feasible(pairs: Iterable[Pair], min_score: float) -> list[Pair]:
    return [(int(i), int(j), float(s)) for i, j, s in pairs if s >= min_score]


def greedy_one_to_one(pairs: Iterable[Pair], *, min_score: float) -> tuple[Pair, ...]:
    """Greedy by score descending (ties: lower pred, then lower ref index); exact when one side is disjoint."""
    used_p: set[int] = set()
    used_r: set[int] = set()
    chosen: list[Pair] = []
    for i, j, s in sorted(_feasible(pairs, min_score), key=lambda p: (-p[2], p[0], p[1])):
        if i not in used_p and j not in used_r:
            used_p.add(i)
            used_r.add(j)
            chosen.append((i, j, s))
    return tuple(sorted(chosen))


def optimal_one_to_one(pairs: Iterable[Pair], *, min_score: float) -> tuple[Pair, ...]:
    """Maximum-cardinality one-to-one matching (then maximum total score) via linear_sum_assignment."""
    feasible = _feasible(pairs, min_score)
    if not feasible:
        return ()
    rows = sorted({p[0] for p in feasible})
    cols = sorted({p[1] for p in feasible})
    r_pos = {v: k for k, v in enumerate(rows)}
    c_pos = {v: k for k, v in enumerate(cols)}
    cost = np.zeros((len(rows), len(cols)))
    score = {}
    for i, j, s in feasible:
        cost[r_pos[i], c_pos[j]] = -(1.0 + _TIEBREAK_WEIGHT * s)
        score[(i, j)] = s
    ri, ci = linear_sum_assignment(cost)
    chosen = [(rows[a], cols[b]) for a, b in zip(ri, ci, strict=True)]
    return tuple(sorted((i, j, score[(i, j)]) for i, j in chosen if (i, j) in score))


def match_polygons(
    pred: Sequence[BaseGeometry], ref: Sequence[BaseGeometry], *, min_iou: float, method: str = "greedy"
) -> tuple[Pair, ...]:
    """One-to-one polygon matches (i_pred, j_ref, iou) with iou >= min_iou."""
    if method not in MATCH_METHODS:
        raise ValueError(f"unknown match method {method!r}, expected one of {MATCH_METHODS}")
    i, j, iou = iou_pairs(pred, ref)
    pairs = zip(i.tolist(), j.tolist(), iou.tolist(), strict=True)
    if method == "greedy":
        return greedy_one_to_one(pairs, min_score=min_iou)
    return optimal_one_to_one(pairs, min_score=min_iou)
