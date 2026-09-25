"""Link interrow pieces to their bounding rows (contract §4.5.7, critique C4) and build global `interrows`.

Each piece looks for the nearest row piece on each side of its centroid, measured along the canonical
block normal, among the row pieces of the same tile and block. A side still missing falls back to the
block's merged rows whose extent covers the centroid. Interrow ids follow the geometric row_index order
of the bounding pair inside the block (never the min(k) of the raw row ids, which collides when manual
rows such as R900 are inserted); `k_min_ids` keeps the min(k) value as a debug column only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.annset.lines import along_range, oriented, points_at, project
from vineyard.annset.row_order import BlockAxis, OrderedRow
from vineyard.contracts.enums import Severity
from vineyard.contracts.ids import row_index_of
from vineyard.contracts.qa import QaIssue
from vineyard.geo.ops import make_valid_polygonal, orient_ccw

METHOD_TILE: Final = "tile"  # both sides found among the same-tile row pieces
METHOD_BLOCK: Final = "block"  # at least one side came from the block's merged rows
METHOD_NONE: Final = "none"
_NO_ID: Final = None
K_DTYPE: Final = "Int16"


@dataclass(frozen=True)
class LinkParams:
    max_offset_m: float  # a bounding row is at most this far from the piece centroid, along the normal
    along_tol_m: float  # fallback rows must cover the centroid's along position within this tolerance
    seam_close_m: float  # pieces of one interrow split at a tile edge are joined across gaps up to 2x this


@dataclass(frozen=True)
class _Side:
    row_id: str
    offset_m: float  # |n · (row point - centroid)|


@dataclass(frozen=True)
class PieceLink:
    piece_id: str
    vineyard_id: str
    left: _Side | None  # +n side: the row with the larger n·c, i.e. the smaller row_index
    right: _Side | None
    method: str

    @property
    def linked(self) -> bool:
        return self.left is not None and self.right is not None


@dataclass(frozen=True)
class LinkResult:
    pieces: gpd.GeoDataFrame  # interrow pieces + link columns, input order (derive coerces the layer)
    interrows: tuple[dict[str, Any], ...]
    issues: tuple[QaIssue, ...]


RowLines = Mapping[tuple[str, str], tuple[tuple[str, Any], ...]]  # (tile_id, vineyard_id) -> (row_id, line)


def _str(value: object) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def tile_row_lines(row_pieces: gpd.GeoDataFrame) -> RowLines:
    """Row piece lines grouped by (tile_id, vineyard_id); blank row ids are skipped."""
    groups: dict[tuple[str, str], list[tuple[str, Any]]] = {}
    for rid, vid, tile, geom in zip(row_pieces["row_id"], row_pieces["vineyard_id"], row_pieces["tile_id"],
                                    row_pieces.geometry, strict=True):
        if _str(rid).strip():
            groups.setdefault((_str(tile), _str(vid)), []).append((_str(rid), geom))
    return {key: tuple(sorted(val, key=lambda item: item[0])) for key, val in groups.items()}


def _offset(line: Any, centre: np.ndarray, axis: BlockAxis) -> float:
    point = points_at(oriented(line, axis.direction), axis.direction, project(centre[None, :], axis.direction))[0]
    return float(np.dot(point - centre, axis.normal))


def _nearest(offsets: Sequence[tuple[str, float]], sign: float, max_m: float) -> _Side | None:
    side = [(abs(o), rid) for rid, o in offsets if o * sign > 0 and abs(o) <= max_m]
    if not side:
        return None
    dist, rid = min(side)
    return _Side(rid, dist)


def _covers(line: Any, centre: np.ndarray, axis: BlockAxis, tol_m: float) -> bool:
    lo, hi = along_range(line, axis.direction)
    s = float(project(centre[None, :], axis.direction)[0])
    return lo - tol_m <= s <= hi + tol_m


def _block_offsets(rows: Sequence[OrderedRow], centre: np.ndarray, axis: BlockAxis,
                   params: LinkParams) -> list[tuple[str, float]]:
    return [(o.row.row_id, _offset(o.row.line, centre, axis)) for o in rows
            if _covers(o.row.line, centre, axis, params.along_tol_m)]


def link_piece(piece_id: str, vid: str, tile_id: str, geom: BaseGeometry, lines: RowLines,
               block_rows: Sequence[OrderedRow], axis: BlockAxis | None, params: LinkParams) -> PieceLink:
    """Left/right rows of one interrow piece (same tile first, then the block's merged rows)."""
    if axis is None:
        return PieceLink(piece_id, vid, None, None, METHOD_NONE)
    c = geom.centroid
    centre = np.array([c.x, c.y])
    tile_offsets = [(rid, _offset(line, centre, axis)) for rid, line in lines.get((tile_id, vid), ())]
    left = _nearest(tile_offsets, 1.0, params.max_offset_m)
    right = _nearest(tile_offsets, -1.0, params.max_offset_m)
    if left is not None and right is not None:
        return PieceLink(piece_id, vid, left, right, METHOD_TILE)
    block_offsets = _block_offsets(block_rows, centre, axis, params)
    left = left or _nearest(block_offsets, 1.0, params.max_offset_m)
    right = right or _nearest(block_offsets, -1.0, params.max_offset_m)
    method = METHOD_BLOCK if left is not None and right is not None else METHOD_NONE
    return PieceLink(piece_id, vid, left, right, method)


def interrow_numbers(pairs: Sequence[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Number distinct (i, j) row_index pairs of one block: k = max(previous k + 1, min(i, j)).

    Adjacent pairs keep k = the upper row's index (I00k between R k and R k+1); a non-adjacent pair
    (a row that does not reach this tile) takes the next free number, so ids never collide.
    """
    numbers: dict[tuple[int, int], int] = {}
    last = 0
    for pair in sorted(set(pairs), key=lambda p: (min(p), max(p))):
        last = max(last + 1, min(pair))
        numbers[pair] = last
    return numbers


def assign_interrow_ids(links: Sequence[PieceLink], index_of: Mapping[str, int]) -> dict[str, str]:
    """piece_id -> interrow_id for the linked pieces, numbered per block by geometric row_index."""
    per_block: dict[str, list[tuple[int, int]]] = {}
    for link in links:
        if link.linked:
            per_block.setdefault(link.vineyard_id, []).append(_pair(link, index_of))
    numbers = {vid: interrow_numbers(pairs) for vid, pairs in per_block.items()}
    return {link.piece_id: f"{link.vineyard_id}-I{numbers[link.vineyard_id][_pair(link, index_of)]:03d}"
            for link in links if link.linked}


def _pair(link: PieceLink, index_of: Mapping[str, int]) -> tuple[int, int]:
    a, b = index_of[link.left.row_id], index_of[link.right.row_id]  # type: ignore[union-attr]
    return (min(a, b), max(a, b))


def k_min_of_ids(link: PieceLink) -> int | None:
    """Debug only: min(k) of the raw `-R<k>` suffixes of both rows (contract §4.5.7 rule)."""
    if not link.linked:
        return None
    ks = [row_index_of(link.left.row_id), row_index_of(link.right.row_id)]  # type: ignore[union-attr]
    return None if any(k is None for k in ks) else min(ks)  # type: ignore[type-var]


def unlinked_issue(link: PieceLink, tile_id: str, geom: BaseGeometry) -> QaIssue:
    found = [s.row_id for s in (link.left, link.right) if s is not None]
    detail = f"un singur rând vecin găsit ({found[0]})" if found else "niciun rând vecin găsit"
    p = geom.representative_point()
    return QaIssue(Severity.WARNING, "interrow_unlinked", tile_id, link.piece_id,
                   f"Inter-rândul {link.piece_id} nu poate fi legat de două rânduri: {detail}", p.x, p.y)


def along_extent_m(geom: BaseGeometry, axis: BlockAxis | None) -> float:
    """Extent of the geometry's vertices along the block direction (NaN without an axis)."""
    if axis is None or geom.is_empty:
        return math.nan
    along = project(shapely.get_coordinates(geom), axis.direction)
    return float(along.max() - along.min())


def closed_union(geoms: Sequence[BaseGeometry], seam_close_m: float) -> list[Polygon]:
    """Union with a mitred closing, so tile-seam slivers and float noise do not split one interrow."""
    union = shapely.union_all(np.asarray(geoms, dtype=object))
    if seam_close_m > 0:
        union = union.buffer(seam_close_m, join_style="mitre").buffer(-seam_close_m, join_style="mitre")
    return sorted(make_valid_polygonal(union), key=lambda p: (-p.area, p.representative_point().coords[0]))


def interrow_record(interrow_id: str, members: gpd.GeoDataFrame, axes: Mapping[str, BlockAxis],
                    seam_close_m: float) -> dict[str, Any]:
    """Global `interrows` record: union of the linked pieces sharing `interrow_id` (contract §2.5.9)."""
    first = members.iloc[0]
    axis = axes.get(_str(first["vineyard_id"]))
    parts = closed_union(list(members.geometry), seam_close_m)
    geom: Polygon | MultiPolygon = orient_ccw(parts[0]) if len(parts) == 1 \
        else MultiPolygon([orient_ccw(p) for p in parts])
    length = along_extent_m(geom, axis)
    return {"interrow_id": interrow_id, "vineyard_id": _str(first["vineyard_id"]),
            "row_left_id": _str(first["row_left_id"]), "row_right_id": _str(first["row_right_id"]),
            "area_m2": float(geom.area), "length_m": length,
            "width_mean_m": float(geom.area) / length if length > 0 else math.nan,
            "n_pieces": len(members), "tile_ids": ",".join(sorted(set(members["tile_id"]))), "geometry": geom}


def _link_columns(links: Sequence[PieceLink], ids: Mapping[str, str], widths: Sequence[float]) -> dict[str, Any]:
    return {
        "interrow_id": [ids.get(link.piece_id, _NO_ID) for link in links],
        "row_left_id": [None if link.left is None else link.left.row_id for link in links],
        "row_right_id": [None if link.right is None else link.right.row_id for link in links],
        "width_mean_m": list(widths),
        "link_method": [link.method for link in links],
        "left_offset_m": [math.nan if link.left is None else link.left.offset_m for link in links],
        "right_offset_m": [math.nan if link.right is None else link.right.offset_m for link in links],
        "k_min_ids": pd.array([k_min_of_ids(link) for link in links], dtype=K_DTYPE),
    }


def _piece_width(geom: BaseGeometry, current: float, axis: BlockAxis | None) -> float:
    if not math.isnan(current):
        return float(current)
    length = along_extent_m(geom, axis)
    return float(geom.area) / length if length > 0 else math.nan


def link_interrows(interrow_pieces: gpd.GeoDataFrame, row_pieces: gpd.GeoDataFrame,
                   ordered: Sequence[OrderedRow], axes: Mapping[str, BlockAxis],
                   params: LinkParams) -> LinkResult:
    """Linked pieces (input order, new frame), global interrows sorted by id, interrow_unlinked issues."""
    lines = tile_row_lines(row_pieces)
    block_rows: dict[str, list[OrderedRow]] = {}
    for o in ordered:
        block_rows.setdefault(o.row.vineyard_id, []).append(o)
    index_of = {o.row.row_id: o.row_index for o in ordered}
    frame = interrow_pieces.reset_index(drop=True)
    rows = list(zip(frame["piece_id"], frame["vineyard_id"], frame["tile_id"], frame.geometry, strict=True))
    links = [link_piece(_str(pid), _str(vid), _str(tile), geom, lines, block_rows.get(_str(vid), ()),
                        axes.get(_str(vid)), params) for pid, vid, tile, geom in rows]
    ids = assign_interrow_ids(links, index_of)
    widths = [_piece_width(geom, float(w), axes.get(_str(vid)))
              for (_, vid, _, geom), w in zip(rows, frame["width_mean_m"], strict=True)]
    linked = frame.assign(**_link_columns(links, ids, widths))
    issues = tuple(unlinked_issue(link, _str(tile), geom)
                   for link, (_, _, tile, geom) in zip(links, rows, strict=True) if not link.linked)
    interrows = tuple(interrow_record(iid, linked[linked["interrow_id"] == iid], axes, params.seam_close_m)
                      for iid in sorted(set(ids.values())))
    return LinkResult(linked, interrows, issues)
