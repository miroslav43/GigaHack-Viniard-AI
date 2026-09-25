"""Merge row pieces per row_id into physical rows (contract §4.5 steps 1-6, design 04 §3.3).

Pieces are grouped by their raw row_id (organizers count raw values; blank ids are left to import_checks).
The merged line runs through every vertex ordered by projection on the row's PCA axis.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import geopandas as gpd
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from vineyard.annset.lines import (
    axial_angle_of,
    endpoint_direction,
    lateral_offset,
    oriented,
    pca_direction,
    project,
)
from vineyard.contracts.enums import RowStructure, Severity
from vineyard.contracts.qa import QaIssue
from vineyard.errors import SchemaError

Coverage = Mapping[str, BaseGeometry]  # tile_id -> imaged part of the tile (tile box ∩ tile_valid)


@dataclass(frozen=True)
class MergeParams:
    join_lateral_max_m: float
    join_support_min_m: float
    row_missing_min_len_m: float
    vertex_dedupe_m: float


@dataclass(frozen=True)
class RowPiece:
    piece_id: str
    tile_id: str
    vineyard_id: str
    structure: str
    line: LineString  # oriented along the row axis
    length_m: float
    along_mid_m: float


@dataclass(frozen=True)
class MergedRow:
    row_id: str
    vineyard_id: str  # majority over the pieces
    line: LineString
    length_m: float  # Σ piece lengths (contract §2.5.5)
    extent_m: float  # merged line length
    n_pieces: int
    tile_ids: tuple[str, ...]  # in axis order
    angle_deg: float
    structure_any: str
    piece_ids: tuple[str, ...]  # in axis order
    vineyard_ids: tuple[str, ...]  # distinct raw ids of the pieces, sorted
    tile_structures: tuple[tuple[str, str], ...]  # (tile_id, aggregated structure), sorted by tile


def structure_any(values: Iterable[str]) -> str:
    """disrupted if any piece is, unassessable if all are, else regular (contract §2.5.5)."""
    vals = [str(v) for v in values]
    if RowStructure.DISRUPTED.value in vals:
        return RowStructure.DISRUPTED.value
    if vals and all(v == RowStructure.UNASSESSABLE.value for v in vals):
        return RowStructure.UNASSESSABLE.value
    return RowStructure.REGULAR.value


def _majority(values: Sequence[str], weights: Sequence[float]) -> str:
    """Most frequent value; ties -> larger total weight, then the smallest value."""
    counts = Counter(values)
    totals: dict[str, float] = {}
    for value, weight in zip(values, weights, strict=True):
        totals[value] = totals.get(value, 0.0) + float(weight)
    return min(counts, key=lambda v: (-counts[v], -totals[v], v))


def _pieces(group: gpd.GeoDataFrame, direction: np.ndarray) -> tuple[RowPiece, ...]:
    out = []
    for piece_id, tile_id, vid, structure, geom in zip(group["piece_id"], group["tile_id"], group["vineyard_id"],
                                                       group["row_structure"], group.geometry, strict=True):
        line = oriented(geom, direction)
        mid = project(np.asarray(line.interpolate(0.5, normalized=True).coords), direction)[0]
        out.append(RowPiece(str(piece_id), str(tile_id), str(vid), str(structure), line, line.length, float(mid)))
    return tuple(sorted(out, key=lambda p: (p.along_mid_m, p.piece_id)))


def merge_piece_lines(lines: Sequence[LineString], direction: np.ndarray, dedupe_m: float) -> LineString:
    """LineString through all vertices ordered by projection; near-equal projections are dropped."""
    coords = np.concatenate([np.asarray(line.coords)[:, :2] for line in lines])
    along = project(coords, direction)
    order = np.argsort(along, kind="stable")
    kept = [order[0]]
    for idx in order[1:]:
        if along[idx] > along[kept[-1]] + dedupe_m:
            kept.append(idx)
    if len(kept) < 2:
        raise SchemaError("merged row collapses to a point", n_vertices=len(coords), dedupe_m=dedupe_m)
    return LineString(coords[kept])


def _issue(severity: Severity, code: str, tile_id: str, row_id: str, message: str,
           at: BaseGeometry) -> QaIssue:
    point = at.representative_point() if not at.is_empty else None
    return QaIssue(severity, code, tile_id, row_id, message,
                   None if point is None else point.x, None if point is None else point.y)


def _junction_issues(row_id: str, pieces: Sequence[RowPiece], direction: np.ndarray,
                     params: MergeParams) -> list[QaIssue]:
    issues = []
    for a, b in pairwise(pieces):
        support = endpoint_direction(a.line) if a.length_m >= params.join_support_min_m else direction
        a_end, b_start = np.asarray(a.line.coords[-1]), np.asarray(b.line.coords[0])
        offset = lateral_offset(b_start, a_end, support)
        if offset > params.join_lateral_max_m:
            issues.append(_issue(Severity.WARNING, "row_piece_misaligned", b.tile_id, row_id,
                                 f"Bucățile rândului {row_id} nu se aliniază: abatere laterală {offset:.2f} m "
                                 f"> {params.join_lateral_max_m:.2f} m", LineString([a_end, b_start]).centroid))
    return issues


def _dup_issues(row_id: str, pieces: Sequence[RowPiece]) -> list[QaIssue]:
    per_tile = Counter(p.tile_id for p in pieces)
    issues = []
    for tile_id in sorted(t for t, n in per_tile.items() if n > 1):
        second = [p for p in pieces if p.tile_id == tile_id][1]
        issues.append(_issue(Severity.WARNING, "dup_row_in_tile", tile_id, row_id,
                             f"Rândul {row_id} apare de {per_tile[tile_id]} ori în tile-ul {tile_id}",
                             second.line.interpolate(0.5, normalized=True)))
    return issues


def _multi_block_issue(row_id: str, pieces: Sequence[RowPiece], major: str) -> list[QaIssue]:
    ids = sorted({p.vineyard_id for p in pieces})
    if len(ids) < 2:
        return []
    other = next(p for p in pieces if p.vineyard_id != major)
    return [_issue(Severity.ERROR, "row_multi_block", other.tile_id, row_id,
                   f"Rândul {row_id} are bucăți în blocuri diferite ({', '.join(ids)}); se folosește {major}",
                   other.line.interpolate(0.5, normalized=True))]


@dataclass(frozen=True)
class _CoverageIndex:
    tile_ids: tuple[str, ...]
    geoms: tuple[BaseGeometry, ...]
    tree: STRtree

    @classmethod
    def build(cls, coverage: Coverage) -> _CoverageIndex:
        tiles = tuple(sorted(coverage))
        geoms = tuple(coverage[t] for t in tiles)
        return cls(tiles, geoms, STRtree(geoms))


def _missing_tile_issues(row_id: str, line: LineString, piece_tiles: frozenset[str], index: _CoverageIndex,
                         min_len_m: float) -> list[QaIssue]:
    issues = []
    for i in sorted(int(k) for k in index.tree.query(line)):
        tile_id = index.tile_ids[i]
        if tile_id in piece_tiles:
            continue
        inside = line.intersection(index.geoms[i])
        if inside.length >= min_len_m:
            issues.append(_issue(Severity.WARNING, "row_missing_in_tile", tile_id, row_id,
                                 f"Rândul {row_id} trece prin tile-ul {tile_id} ({inside.length:.1f} m) "
                                 f"fără bucată adnotată", inside.interpolate(0.5, normalized=True)))
    return issues


def _tile_structures(pieces: Sequence[RowPiece]) -> tuple[tuple[str, str], ...]:
    tiles = sorted({p.tile_id for p in pieces})
    return tuple((t, structure_any(p.structure for p in pieces if p.tile_id == t)) for t in tiles)


def _merge_one(row_id: str, group: gpd.GeoDataFrame, params: MergeParams,
               index: _CoverageIndex | None) -> tuple[MergedRow, list[QaIssue]]:
    coords = np.concatenate([np.asarray(g.coords)[:, :2] for g in group.geometry])
    direction = pca_direction(coords)
    pieces = _pieces(group, direction)
    major = _majority([p.vineyard_id for p in pieces], [p.length_m for p in pieces])
    line = merge_piece_lines([p.line for p in pieces], direction, params.vertex_dedupe_m)
    row = MergedRow(
        row_id=row_id, vineyard_id=major, line=line, length_m=float(sum(p.length_m for p in pieces)),
        extent_m=line.length, n_pieces=len(pieces), tile_ids=tuple(dict.fromkeys(p.tile_id for p in pieces)),
        angle_deg=axial_angle_of(direction), structure_any=structure_any(p.structure for p in pieces),
        piece_ids=tuple(p.piece_id for p in pieces), vineyard_ids=tuple(sorted({p.vineyard_id for p in pieces})),
        tile_structures=_tile_structures(pieces),
    )
    issues = _multi_block_issue(row_id, pieces, major) + _dup_issues(row_id, pieces) \
        + _junction_issues(row_id, pieces, direction, params)
    if index is not None:
        piece_tiles = frozenset(p.tile_id for p in pieces)
        issues += _missing_tile_issues(row_id, line, piece_tiles, index, params.row_missing_min_len_m)
    return row, issues


def _usable(row_pieces: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ids = row_pieces["row_id"]
    blank = ids.isna() | (ids.astype(str).str.strip() == "")
    return row_pieces[~blank.to_numpy(dtype=bool)]


def merge_rows(row_pieces: gpd.GeoDataFrame, params: MergeParams,
               coverage: Coverage | None = None) -> tuple[tuple[MergedRow, ...], tuple[QaIssue, ...]]:
    """Physical rows sorted by row_id, plus row_multi_block / dup_row_in_tile / row_piece_misaligned /
    row_missing_in_tile issues. `coverage=None` skips the row_missing_in_tile check."""
    usable = _usable(row_pieces)
    index = _CoverageIndex.build(coverage) if coverage else None
    rows: list[MergedRow] = []
    issues: list[QaIssue] = []
    for row_id, group in usable.groupby(usable["row_id"].astype(str), sort=True):
        try:
            row, row_issues = _merge_one(str(row_id), group.sort_values("piece_id", kind="stable"), params, index)
        except SchemaError as exc:
            raise SchemaError("cannot merge row pieces", row_id=str(row_id),
                              pieces=list(group["piece_id"])[:5], error=str(exc)) from exc
        rows.append(row)
        issues += row_issues
    return tuple(rows), tuple(issues)
