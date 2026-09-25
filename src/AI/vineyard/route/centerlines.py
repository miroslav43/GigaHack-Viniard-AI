"""Interrow centerlines (arch §4.11.2, design 04 §3.7).

The centerline of an interrow is the geometric mean of its two consecutive row axes: both axes are
sampled at the same positions along their mean direction (only where they overlap) and the
midpoints are joined. The line is clipped to `domain.eroded`; pieces shorter than `min_len_m` are
dropped. Interrow ids follow the geometric `row_index` order inside a block (never the row-id
suffix): the k-th consecutive pair of a block is `<vid>-I<k:03d>`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from vineyard.route.cells import CellIndex
from vineyard.route.skeleton import SkeletonParams, skeleton_lines

if TYPE_CHECKING:
    from vineyard.config.sections_post import RouteGraphConfig

# CONFIG-REQUEST: route.graph.centerline_fallback_near_m = 0.5
FALLBACK_NEAR_M: Final = 0.5
MIN_OVERLAP_M: Final = 1e-6
INTERROW_ID_FMT: Final = "{vid}-I{k:03d}"
REQUIRED_ROW_COLUMNS: Final = ("row_id", "vineyard_id", "row_index", "geometry")
Source = Literal["midline", "skeleton"]


@dataclass(frozen=True)
class CenterlineParams:
    step_m: float
    min_len_m: float
    fallback_near_m: float = FALLBACK_NEAR_M

    def __post_init__(self) -> None:
        if self.step_m <= 0.0:
            raise ValueError(f"step_m must be > 0, got {self.step_m}")
        if self.min_len_m < 0.0 or self.fallback_near_m < 0.0:
            raise ValueError("min_len_m and fallback_near_m must be >= 0")

    @classmethod
    def from_config(cls, cfg: RouteGraphConfig) -> CenterlineParams:
        return cls(step_m=cfg.centerline_step_m, min_len_m=cfg.centerline_min_len_m)


@dataclass(frozen=True, eq=False)
class RowPair:
    vineyard_id: str
    interrow_id: str
    row_left_id: str
    row_right_id: str
    left: LineString
    right: LineString


@dataclass(frozen=True, eq=False)
class Centerline:
    """One walkable interrow piece; `geom` runs along the rows' mean direction."""

    interrow_id: str
    vineyard_id: str
    row_left_id: str
    row_right_id: str
    piece: int
    source: Source
    geom: LineString


def row_pairs(rows: gpd.GeoDataFrame) -> tuple[RowPair, ...]:
    """Consecutive rows per block in (row_index, row_id) order; blocks sorted by vineyard_id."""
    missing = [c for c in REQUIRED_ROW_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"row_pairs: rows layer lacks columns {missing}")
    frame = rows.assign(_idx=rows["row_index"].astype(np.int64), _rid=rows["row_id"].astype(str),
                        _vid=rows["vineyard_id"].astype(str))
    pairs: list[RowPair] = []
    for vid, block in sorted(frame.groupby("_vid", sort=False), key=lambda kv: str(kv[0])):
        ordered = block.sort_values(["_idx", "_rid"], kind="mergesort")
        ids, geoms = list(ordered["_rid"]), list(ordered.geometry)
        for k in range(1, len(ids)):
            pairs.append(RowPair(str(vid), INTERROW_ID_FMT.format(vid=vid, k=k), ids[k - 1], ids[k],
                                 geoms[k - 1], geoms[k]))
    return tuple(pairs)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.hypot(*v))
    return v / norm if norm > 0.0 else v


def _mean_direction(a: LineString, b: LineString) -> np.ndarray:
    ca, cb = np.asarray(a.coords)[:, :2], np.asarray(b.coords)[:, :2]
    da, db = _unit(ca[-1] - ca[0]), _unit(cb[-1] - cb[0])
    d = _unit(da + (db if float(np.dot(da, db)) >= 0.0 else -db))
    # canonical axial orientation (angle in [0, 180)), so the result does not depend on digitising order
    return -d if d[1] < 0.0 or (d[1] == 0.0 and d[0] < 0.0) else d


def _along(line: LineString, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(line.coords)[:, :2]
    proj = xy @ d
    if proj[-1] < proj[0]:
        xy, proj = xy[::-1], proj[::-1]
    return np.maximum.accumulate(proj), xy


def _at(proj: np.ndarray, xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t, proj, xy[:, 0]), np.interp(t, proj, xy[:, 1])])


def midline(a: LineString, b: LineString, step_m: float) -> LineString | None:
    """Midpoints of `a` and `b` at equal positions along their mean direction, over their overlap only."""
    d = _mean_direction(a, b)
    pa, xa = _along(a, d)
    pb, xb = _along(b, d)
    lo, hi = max(pa[0], pb[0]), min(pa[-1], pb[-1])
    if hi - lo <= MIN_OVERLAP_M:
        return None
    n = max(1, math.ceil((hi - lo) / step_m - 1e-9))
    t = np.linspace(lo, hi, n + 1)
    return LineString((_at(pa, xa, t) + _at(pb, xb, t)) / 2.0)


def _as_index(region: BaseGeometry | CellIndex) -> CellIndex:
    return region if isinstance(region, CellIndex) else CellIndex.build(region)


def _ordered_pieces(mid: LineString, parts: Sequence[LineString], min_len: float) -> list[LineString]:
    start = np.asarray(mid.coords[0])
    d = _unit(np.asarray(mid.coords[-1]) - start)
    kept = [p for p in parts if p.length >= min_len]
    oriented = [p if float(np.dot(np.asarray(p.coords[-1]) - p.coords[0], d)) >= 0.0 else p.reverse()
                for p in kept]
    return sorted(oriented, key=lambda p: float(np.dot(np.asarray(p.coords[0]) - start, d)))


def interrow_centerlines(rows: gpd.GeoDataFrame, eroded: BaseGeometry | CellIndex,
                         params: CenterlineParams) -> tuple[Centerline, ...]:
    """Midline pieces of every consecutive row pair, clipped to the eroded domain."""
    index = _as_index(eroded)
    out: list[Centerline] = []
    for pair in row_pairs(rows):
        mid = midline(pair.left, pair.right, params.step_m)
        if mid is None:
            continue
        for k, piece in enumerate(_ordered_pieces(mid, index.clip_line(mid), params.min_len_m), start=1):
            out.append(Centerline(pair.interrow_id, pair.vineyard_id, pair.row_left_id, pair.row_right_id, k,
                                  "midline", piece))
    return tuple(out)


def _unserved(pieces: Sequence[BaseGeometry], lines: Sequence[Centerline], near_m: float) -> list[int]:
    if not pieces:
        return []
    if not lines:
        return list(range(len(pieces)))
    tree = shapely.STRtree([c.geom for c in lines])
    hit_pieces, _ = tree.query(np.asarray(list(pieces), dtype=object), predicate="dwithin", distance=near_m)
    served = set(int(i) for i in hit_pieces)
    return [i for i in range(len(pieces)) if i not in served]


def fallback_centerlines(
    piece_geoms: Sequence[BaseGeometry],
    piece_ids: Sequence[str],
    piece_refs: Sequence[str | None],
    lines: Sequence[Centerline],
    eroded: BaseGeometry | CellIndex,
    skeleton: SkeletonParams,
    params: CenterlineParams,
) -> tuple[Centerline, ...]:
    """Skeleton lines of the interrow pieces that no midline passes within `fallback_near_m` of.

    The skeleton of the piece polygon is clipped to the eroded domain (canopies are not cut out of
    the piece itself). The interrow id is the piece's link (`piece_refs`) or else its piece id.
    """
    if not len(piece_geoms) == len(piece_ids) == len(piece_refs):
        raise ValueError(f"fallback_centerlines: {len(piece_geoms)} pieces, {len(piece_ids)} ids, "
                         f"{len(piece_refs)} refs")
    index = _as_index(eroded)
    out: list[Centerline] = []
    for i in _unserved(piece_geoms, lines, params.fallback_near_m):
        ref = piece_refs[i] if piece_refs[i] else str(piece_ids[i])
        clipped = [part for ln in skeleton_lines(piece_geoms[i], skeleton) for part in index.clip_line(ln)]
        found = sorted((p for p in clipped if p.length >= params.min_len_m),
                       key=lambda p: (round(p.coords[0][0], 3), round(p.coords[0][1], 3)))
        out.extend(Centerline(str(ref), "", "", "", k, "skeleton", ln) for k, ln in enumerate(found, start=1))
    return tuple(out)


__all__ = [
    "FALLBACK_NEAR_M", "Centerline", "CenterlineParams", "RowPair", "fallback_centerlines", "interrow_centerlines",
    "midline", "row_pairs",
]
