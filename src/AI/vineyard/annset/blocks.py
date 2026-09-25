"""Blocks from any AnnSet (contract §2.5.6, §4.5.8): outline = union(buffer(objects, b)).buffer(-b).

A block is a non-blank vineyard_id carried by a canopy, a row piece or an interrow piece. An id seen only on
waste boxes is not a block (`waste_block_unknown`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import geopandas as gpd
import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.annset.model import AnnSet
from vineyard.annset.row_order import BlockAxis, OrderedRow
from vineyard.contracts.enums import Severity
from vineyard.contracts.qa import QaIssue
from vineyard.geo.ops import make_valid_polygonal, orient_ccw
from vineyard.geo.tiling import tile_of_point

BLOCK_LAYERS: Final = ("canopies", "row_pieces", "interrow_pieces")
ALL_LAYERS: Final = (*BLOCK_LAYERS, "waste")


@dataclass(frozen=True)
class BlockParams:
    outline_buffer_m: float
    merge_warn_m: float
    garden_forbidden_dist_m: float
    garden_max_rows: int
    min_rows_per_block: int
    lone_row_buffer_m: float  # outline of a block whose closing is empty (a single row, no polygons)


@dataclass(frozen=True)
class Block:
    vineyard_id: str
    outline: Polygon | MultiPolygon
    n_rows: int
    n_row_pieces: int
    n_canopies: int
    n_tiles: int
    row_length_m: float
    canopy_area_m2: float
    interrow_area_m2: float
    angle_deg: float
    spacing_med_m: float
    is_garden: bool

    @property
    def outline_area_m2(self) -> float:
        return float(self.outline.area)


def _ids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    values = gdf["vineyard_id"].astype(object).to_numpy()
    return np.array(["" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in values],
                    dtype=object)


def _members(annset: AnnSet, vid: str, layers: Sequence[str]) -> list[gpd.GeoDataFrame]:
    return [annset.layer(name)[_ids(annset.layer(name)) == vid] for name in layers]


def block_ids(annset: AnnSet) -> tuple[str, ...]:
    """Sorted non-blank vineyard_ids of canopies, row pieces and interrow pieces."""
    ids = {v for name in BLOCK_LAYERS for v in _ids(annset.layer(name)) if v.strip()}
    return tuple(sorted(ids))


def _polygonal(geom: BaseGeometry) -> Polygon | MultiPolygon:
    parts = sorted(make_valid_polygonal(geom), key=lambda p: (-p.area, p.representative_point().coords[0]))
    oriented = [orient_ccw(p) for p in parts]
    return oriented[0] if len(oriented) == 1 else MultiPolygon(oriented)


def outline_of(geoms: Sequence[BaseGeometry], buffer_m: float, lone_buffer_m: float) -> Polygon | MultiPolygon:
    """unary_union(buffer(objects, b)).buffer(-b), the contract formula (a closing of the objects).

    Buffering each object before the union is ~100x faster than buffering the union of thousands of
    canopies. Lines alone close to nothing (a lone row): such a block falls back to buffer(lone_buffer_m).
    """
    parts = np.asarray(geoms, dtype=object)
    closed = shapely.union_all(shapely.buffer(parts, buffer_m)).buffer(-buffer_m)
    if not make_valid_polygonal(closed):
        closed = shapely.union_all(shapely.buffer(parts, lone_buffer_m))
    return _polygonal(closed)


def _union_area(gdf: gpd.GeoDataFrame) -> float:
    return float(shapely.union_all(gdf.geometry.to_numpy()).area) if len(gdf) else 0.0


def _tile_of(geom: BaseGeometry) -> str:
    p = geom.representative_point()
    return tile_of_point(p.x, p.y).tile_id


def _issue(severity: Severity, code: str, object_id: str, message: str, at: BaseGeometry) -> QaIssue:
    p = at.representative_point()
    return QaIssue(severity, code, _tile_of(p), object_id, message, p.x, p.y)


def _block(annset: AnnSet, vid: str, rows: Sequence[OrderedRow], axis: BlockAxis | None,
           params: BlockParams, forbidden: BaseGeometry | None) -> Block:
    canopies, pieces, interrows = _members(annset, vid, BLOCK_LAYERS)
    outline = outline_of([*canopies.geometry, *pieces.geometry, *interrows.geometry], params.outline_buffer_m,
                         params.lone_row_buffer_m)
    tiles = set(canopies["tile_id"]) | set(pieces["tile_id"]) | set(interrows["tile_id"])
    spacings = [r.spacing_next_m for r in rows if not math.isnan(r.spacing_next_m)]
    near_village = forbidden is not None and not forbidden.is_empty \
        and outline.distance(forbidden) <= params.garden_forbidden_dist_m
    return Block(
        vineyard_id=vid, outline=outline, n_rows=len(rows), n_row_pieces=len(pieces), n_canopies=len(canopies),
        n_tiles=len(tiles), row_length_m=float(pieces.geometry.length.sum()),
        canopy_area_m2=_union_area(canopies), interrow_area_m2=_union_area(interrows),
        angle_deg=math.nan if axis is None else axis.angle_deg,
        spacing_med_m=float(np.median(spacings)) if spacings else math.nan,
        is_garden=bool(near_village and len(rows) <= params.garden_max_rows),
    )


def _split_issue(block: Block) -> list[QaIssue]:
    if not isinstance(block.outline, MultiPolygon):
        return []
    parts = list(block.outline.geoms)
    gap = min(parts[0].distance(p) for p in parts[1:])
    return [_issue(Severity.WARNING, "block_split", block.vineyard_id,
                   f"Blocul {block.vineyard_id} are {len(parts)} componente separate (cel puțin {gap:.1f} m)",
                   parts[1])]


def _few_rows_issue(block: Block, params: BlockParams) -> list[QaIssue]:
    if block.n_rows >= params.min_rows_per_block:
        return []
    return [_issue(Severity.WARNING, "block_too_few_rows", block.vineyard_id,
                   f"Blocul {block.vineyard_id} are doar {block.n_rows} rânduri", block.outline)]


def merge_issues(blocks: Sequence[Block], params: BlockParams,
                 passages: BaseGeometry | None) -> list[QaIssue]:
    """blocks_should_merge: two ids closer than merge_warn_m with no passage crossing the gap."""
    outlines = np.asarray([b.outline for b in blocks], dtype=object)
    if len(outlines) < 2:
        return []
    left, right = STRtree(outlines).query(outlines, predicate="dwithin", distance=params.merge_warn_m)
    issues = []
    for i, j in sorted({(int(a), int(b)) for a, b in zip(left, right, strict=True) if a < b}):
        link = shapely.shortest_line(outlines[i], outlines[j])
        crossing = passages is not None and link.length > 0 and passages.intersection(link).length > 0
        if crossing:
            continue
        a, b = blocks[i].vineyard_id, blocks[j].vineyard_id
        issues.append(_issue(Severity.WARNING, "blocks_should_merge", f"{a}+{b}",
                             f"Blocurile {a} și {b} sunt la {link.length:.1f} m fără pasaj între ele",
                             link.centroid if link.length > 0 else link))
    return issues


def waste_issues(annset: AnnSet, blocks: Sequence[str]) -> list[QaIssue]:
    """waste_block_unknown: a waste box names a vineyard_id that no canopy/row/interrow carries."""
    known = set(blocks)
    waste = annset.waste
    issues = []
    for waste_id, vid, geom in sorted(zip(waste["waste_id"], _ids(waste), waste.geometry, strict=True)):
        if vid.strip() and vid not in known:
            issues.append(_issue(Severity.WARNING, "waste_block_unknown", str(waste_id),
                                 f"Deșeul {waste_id} are vineyard_id={vid}, care nu există pe alte obiecte", geom))
    return issues


def build_blocks(annset: AnnSet, ordered: Sequence[OrderedRow], axes: Mapping[str, BlockAxis],
                 params: BlockParams, *, passages: BaseGeometry | None = None,
                 forbidden: BaseGeometry | None = None) -> tuple[tuple[Block, ...], tuple[QaIssue, ...]]:
    """Blocks sorted by vineyard_id plus block_split / block_too_few_rows / blocks_should_merge /
    waste_block_unknown issues."""
    ids = block_ids(annset)
    blocks = tuple(
        _block(annset, vid, [r for r in ordered if r.row.vineyard_id == vid], axes.get(vid), params, forbidden)
        for vid in ids
    )
    issues = [i for b in blocks for i in _split_issue(b) + _few_rows_issue(b, params)]
    issues += merge_issues(blocks, params, passages) + waste_issues(annset, ids)
    return blocks, tuple(issues)


def block_record(block: Block) -> dict[str, Any]:
    """`blocks` layer record (contract §2.5.6)."""
    return {
        "vineyard_id": block.vineyard_id, "n_rows": block.n_rows, "n_row_pieces": block.n_row_pieces,
        "n_canopies": block.n_canopies, "n_tiles": block.n_tiles, "row_length_m": block.row_length_m,
        "canopy_area_m2": block.canopy_area_m2, "interrow_area_m2": block.interrow_area_m2,
        "outline_area_m2": block.outline_area_m2, "angle_deg": block.angle_deg,
        "spacing_med_m": block.spacing_med_m, "is_garden": block.is_garden, "geometry": block.outline,
    }
