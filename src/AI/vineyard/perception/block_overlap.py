"""Interrow pieces of different blocks must not overlap, so block areas add up to the survey union.

Interrow bands are built per block, so where the rows of two blocks share ground (a block boundary, or a
spurious row family laid over a neighbouring block) the bands of both blocks cover it and the overlap is
counted in both block totals. Every cross-block overlap is given to ONE block:

- pieces linked by cross-block overlaps above `min_overlap_m2` form clusters (one contested area each);
- in a cluster, the blocks are ranked by their own canopy area within `support_buffer_m` of the contested
  area (the vines flanking the ground tell whose inter-row it is), ties by the lower vineyard_id;
- each piece loses what it shares with the overlapping pieces of higher-ranked blocks; parts below
  `min_piece_m2` are dropped. The top block of every overlap keeps it, so the union of all pieces is
  unchanged (up to the dropped parts) and no two blocks share area afterwards.

Pieces of one block are never cut against each other: their overlaps do not double count a block total.
Deterministic: the result does not depend on the order of the input rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import Severity
from vineyard.contracts.qa import QaIssue
from vineyard.perception.topology import cut_parts, split_piece_id, used_piece_dups

if TYPE_CHECKING:
    from vineyard.config import AppConfig

BLOCK_OVERLAP_CODE: Final = "interrow_block_overlap"
SUPPORT_DECIMALS: Final = 6  # support compared at 1e-6 m²: float summation order never decides the winner
PIECE_ID: Final = "piece_id"
BLOCK: Final = "vineyard_id"
TILE: Final = "tile_id"


@dataclass(frozen=True)
class BlockOverlapParams:
    min_overlap_m2: float  # cross-block intersections up to this are shared edges or float noise
    support_buffer_m: float  # canopies within this of the contested area vote for their block
    min_piece_m2: float  # parts of a cut piece smaller than this are dropped

    @classmethod
    def from_config(cls, cfg: AppConfig) -> BlockOverlapParams:
        return cls(min_overlap_m2=cfg.derive.interrow_overlap_min_m2,
                   support_buffer_m=cfg.derive.interrow_overlap_support_m,
                   min_piece_m2=cfg.export.min_interrow_piece_m2)


@dataclass(frozen=True)
class BlockOverlapResult:
    pieces: gpd.GeoDataFrame  # input rows in input order; cut pieces replaced by their parts
    issues: tuple[QaIssue, ...]  # one warning per piece that lost area, sorted by piece_id
    n_cut: int = 0  # pieces that lost area and kept at least one part
    n_dropped: int = 0  # pieces with no part left of at least min_piece_m2
    n_split: int = 0  # extra parts (new `#k` ids)
    moved_m2: float = 0.0  # overlap area left to the higher-ranked block
    dropped_m2: float = 0.0  # area of the parts dropped below min_piece_m2

    def metrics(self) -> dict[str, float]:
        return {"n_interrow_overlap_cut": float(self.n_cut), "n_interrow_overlap_dropped": float(self.n_dropped),
                "n_interrow_overlap_split": float(self.n_split), "interrow_overlap_moved_m2": self.moved_m2,
                "interrow_overlap_dropped_m2": self.dropped_m2}


@dataclass(frozen=True)
class _Cut:
    position: int  # row in the piece_id-sorted frame
    winners: tuple[int, ...]  # overlapping pieces of higher-ranked blocks, piece_id order
    lost: BaseGeometry  # piece ∩ union(winners)
    parts: tuple[Polygon, ...]  # what the piece keeps


# ------------------------------------------------------------------ pairs, clusters, ranking


def cross_block_pairs(vids: np.ndarray, geoms: np.ndarray, min_overlap_m2: float) -> list[tuple[int, int]]:
    """Sorted (i, j), i < j: pieces of different blocks whose intersection is larger than min_overlap_m2."""
    left, right = STRtree(geoms).query(geoms, predicate="intersects")
    keep = (left < right) & (vids[left] != vids[right])
    left, right = left[keep], right[keep]
    area = shapely.area(shapely.intersection(geoms[left], geoms[right]))
    return sorted((int(a), int(b)) for a, b, x in zip(left, right, area, strict=True) if x > min_overlap_m2)


def overlap_clusters(n: int, pairs: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """The pairs grouped by connected component of the overlap graph, ordered by their first pair."""
    ends = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(ends)), (ends[:, 0], ends[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    groups: dict[int, list[tuple[int, int]]] = {}
    for pair in pairs:
        groups.setdefault(int(labels[pair[0]]), []).append(pair)
    return sorted(groups.values(), key=lambda g: g[0])


def block_support(zone: BaseGeometry, blocks: Sequence[str], canopies: gpd.GeoDataFrame,
                  tree: STRtree) -> dict[str, float]:
    """vineyard_id -> area of its canopies inside `zone`, for `blocks` only."""
    support = dict.fromkeys(blocks, 0.0)
    geoms = canopies.geometry.to_numpy()
    vids = canopies[BLOCK].to_numpy()
    for i in sorted(int(k) for k in tree.query(zone, predicate="intersects")):
        vid = vids[i]
        if isinstance(vid, str) and vid in support:
            support[vid] += float(shapely.intersection(geoms[i], zone).area)
    return support


def rank_blocks(support: Mapping[str, float]) -> dict[str, int]:
    """vineyard_id -> rank (0 = keeps every overlap): most support first, then the lower vineyard_id."""
    order = sorted(support, key=lambda v: (-round(support[v], SUPPORT_DECIMALS), v))
    return {vid: k for k, vid in enumerate(order)}


# ------------------------------------------------------------------ cutting


def _cluster_cuts(pairs: Sequence[tuple[int, int]], vids: np.ndarray, geoms: np.ndarray,
                  canopies: gpd.GeoDataFrame, tree: STRtree, params: BlockOverlapParams) -> list[_Cut]:
    contested = shapely.union_all([shapely.intersection(geoms[a], geoms[b]) for a, b in pairs])
    zone = contested.buffer(params.support_buffer_m) if params.support_buffer_m > 0 else contested
    rank = rank_blocks(block_support(zone, sorted({str(vids[i]) for pair in pairs for i in pair}), canopies, tree))
    winners: dict[int, set[int]] = {}
    for a, b in pairs:
        loser, winner = (a, b) if rank[str(vids[a])] > rank[str(vids[b])] else (b, a)
        winners.setdefault(loser, set()).add(winner)
    cuts = []
    for pos in sorted(winners):
        won = tuple(sorted(winners[pos]))
        cutter = shapely.union_all(geoms[list(won)])
        cuts.append(_Cut(pos, won, shapely.intersection(geoms[pos], cutter),
                         tuple(cut_parts(geoms[pos], cutter, params.min_piece_m2))))
    return cuts


def _issue(cut: _Cut, frame: gpd.GeoDataFrame, min_piece_m2: float) -> QaIssue:
    pid, vid, tile = (str(frame[c].iloc[cut.position]) for c in (PIECE_ID, BLOCK, TILE))
    others = ", ".join(sorted({str(frame[BLOCK].iloc[w]) for w in cut.winners}))
    msg = (f"Inter-rândul {pid} (bloc {vid}) se suprapunea {cut.lost.area:.2f} m² cu blocul {others}; "
           f"suprapunerea a rămas blocului {others}")
    if not cut.parts:
        msg += f"; restul, sub {min_piece_m2} m², a fost eliminat"
    point = None if cut.lost.is_empty else cut.lost.representative_point()
    return QaIssue(Severity.WARNING, BLOCK_OVERLAP_CODE, tile, pid, msg,
                   None if point is None else float(point.x), None if point is None else float(point.y))


def _split_ids(frame: gpd.GeoDataFrame, cuts: Mapping[int, _Cut]) -> dict[str, list[tuple[str, BaseGeometry]]]:
    """piece_id -> [(id, part)] of every cut piece; the largest part keeps the id, the others get the next
    free `#k`, assigned in piece_id order (`frame` is sorted by piece_id)."""
    used = used_piece_dups(list(frame[PIECE_ID]))
    out: dict[str, list[tuple[str, BaseGeometry]]] = {}
    for pos in sorted(cuts):
        pid = str(frame[PIECE_ID].iloc[pos])
        out[pid] = [(pid if k == 1 else split_piece_id(pid, k, used), part)
                    for k, part in enumerate(cuts[pos].parts, start=1)]
    return out


def _rebuild(pieces: gpd.GeoDataFrame, split: Mapping[str, list[tuple[str, BaseGeometry]]]) -> gpd.GeoDataFrame:
    """Rows in `pieces` order; a cut piece becomes its parts (area_m2 recomputed, width_mean_m NaN so that
    linking measures it again)."""
    frame = pieces.reset_index(drop=True)
    positions, geoms, ids, changed = [], [], [], []
    for pos, (pid, geom) in enumerate(zip(frame[PIECE_ID].astype(str), frame.geometry, strict=True)):
        parts = split.get(pid, [(pid, geom)])
        positions += [pos] * len(parts)
        ids += [i for i, _ in parts]
        geoms += [g for _, g in parts]
        changed += [pid in split] * len(parts)
    base = frame.iloc[positions].drop(columns=frame.geometry.name).reset_index(drop=True)
    mask = np.asarray(changed, dtype=bool)
    area = np.where(mask, shapely.area(np.asarray(geoms, dtype=object)), base["area_m2"].to_numpy())
    width = np.where(mask, math.nan, base["width_mean_m"].to_numpy())
    return gpd.GeoDataFrame(base.assign(**{PIECE_ID: ids, "area_m2": area, "width_mean_m": width}),
                            geometry=gpd.GeoSeries(geoms, crs=pieces.crs), crs=pieces.crs)


def _dropped_m2(cut: _Cut, geom: BaseGeometry) -> float:
    return max(float(geom.area - cut.lost.area - sum(p.area for p in cut.parts)), 0.0)


def remove_block_overlap(pieces: gpd.GeoDataFrame, canopies: gpd.GeoDataFrame,
                         params: BlockOverlapParams) -> BlockOverlapResult:
    """Interrow pieces without cross-block overlap (module docstring); `pieces` itself when there is none.

    Inputs are not modified. Needs the columns piece_id, vineyard_id, tile_id, area_m2, width_mean_m.
    """
    if pieces.empty:
        return BlockOverlapResult(pieces, ())
    frame = pieces.sort_values(PIECE_ID, kind="stable").reset_index(drop=True)  # input order never decides
    geoms = frame.geometry.to_numpy()
    vids = frame[BLOCK].astype(str).to_numpy()
    pairs = cross_block_pairs(vids, geoms, params.min_overlap_m2)
    if not pairs:
        return BlockOverlapResult(pieces, ())
    tree = STRtree(canopies.geometry.to_numpy())
    cuts = {c.position: c for cluster in overlap_clusters(len(frame), pairs)
            for c in _cluster_cuts(cluster, vids, geoms, canopies, tree, params)}
    kept = [cuts[pos] for pos in sorted(cuts)]
    return BlockOverlapResult(
        pieces=_rebuild(pieces, _split_ids(frame, cuts)),
        issues=tuple(_issue(c, frame, params.min_piece_m2) for c in kept),
        n_cut=sum(1 for c in kept if c.parts), n_dropped=sum(1 for c in kept if not c.parts),
        n_split=sum(max(len(c.parts) - 1, 0) for c in kept),
        moved_m2=float(sum(c.lost.area for c in kept)),
        dropped_m2=float(sum(_dropped_m2(c, geoms[c.position]) for c in kept)),
    )
