"""Contract §2.7 topological invariants as QaIssue errors, and the conditional canopy subtraction (plan S5).

Canopies are subtracted from interrow pieces only in tiles whose canopy ∩ interrow area exceeds the
limit: the reference itself overlaps by 0.033-0.037 m² per tile and its interrows are clean quads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.annset.model import AnnSet
from vineyard.contracts.enums import Severity
from vineyard.contracts.ids import format_interrow_piece_id, parse_interrow_piece_id
from vineyard.contracts.qa import QaIssue
from vineyard.errors import SchemaError
from vineyard.geo.ops import make_valid_polygonal, orient_ccw
from vineyard.geo.tiling import tile_box, tile_ref

CLIP_TOL_M: Final = 0.001  # contract §2.7: objects lie inside their tile clip within 1 mm
OVERLAP_CODE: Final = "canopy_interrow_overlap"
ROW_MULTI_BLOCK_CODE: Final = "row_multi_block"
WASTE_BLOCK_CODE: Final = "waste_block_unknown"
OUTSIDE_CLIP_CODE: Final = "object_outside_clip"
INTERROW_OUTSIDE_CODE: Final = "interrow_outside_rows"
TILE = "tile_id"
# Rows and waste boxes are cut by the tile box, polygons by tile_box ∩ tile_valid (contract §1.6).
CLIP_LAYERS: Final[Mapping[str, tuple[str, bool]]] = {
    "canopies": ("canopy_id", True), "interrow_pieces": ("piece_id", True),
    "row_pieces": ("piece_id", False), "waste": ("waste_id", False),
}
_SIDE_EPS_M: Final = 1e-9


def _issue(code: str, tile_id: str, object_id: str, message: str, where: BaseGeometry | None) -> QaIssue:
    point = None if where is None or where.is_empty else where.representative_point()
    return QaIssue(Severity.ERROR, code, tile_id, object_id, message,
                   None if point is None else float(point.x), None if point is None else float(point.y))


def _union(frame: gpd.GeoDataFrame, tile_id: str) -> BaseGeometry:
    geoms = frame.geometry[(frame[TILE] == tile_id).to_numpy()].to_numpy()
    return shapely.union_all(geoms) if len(geoms) else Polygon()


def _tile_ids(*frames: gpd.GeoDataFrame) -> list[str]:
    return sorted({str(t) for f in frames for t in f[TILE].dropna().unique()})


# ------------------------------------------------------------------ canopy ∩ interrow


def overlap_by_tile(canopies: gpd.GeoDataFrame, interrows: gpd.GeoDataFrame) -> dict[str, float]:
    """Area (m²) of union(canopies) ∩ union(interrow pieces), per tile present in either frame."""
    out = {}
    for tile_id in _tile_ids(canopies, interrows):
        out[tile_id] = float(shapely.intersection(_union(canopies, tile_id), _union(interrows, tile_id)).area)
    return out


def _cut(geom: BaseGeometry, cutter: BaseGeometry, min_piece_m2: float) -> list[Polygon]:
    parts = [orient_ccw(p) for p in make_valid_polygonal(shapely.difference(geom, cutter)) if p.area >= min_piece_m2]
    return sorted(parts, key=lambda p: (-round(p.area, 9), p.representative_point().x, p.representative_point().y))


def _next_ids(piece_ids: Sequence[str]) -> dict[tuple[str, str], int]:
    used: dict[tuple[str, str], int] = {}
    for pid in piece_ids:
        try:
            base, tile_id, dup = parse_interrow_piece_id(pid)
        except SchemaError:
            continue
        used[(base, tile_id)] = max(used.get((base, tile_id), 0), dup)
    return used


def _split_id(pid: str, k: int, used: dict[tuple[str, str], int]) -> str:
    try:
        base, tile_id, _ = parse_interrow_piece_id(pid)
    except SchemaError:
        return f"{pid}#{k}"  # relaxed (Marcaj/reference) ids: keep the raw id plus a suffix
    used[(base, tile_id)] = used.get((base, tile_id), 1) + 1
    return format_interrow_piece_id(base, tile_id, used[(base, tile_id)])


def remove_canopy_overlap(pieces: gpd.GeoDataFrame, canopies: gpd.GeoDataFrame, *, max_overlap_m2: float,
                          min_piece_m2: float, clearance_m: float) -> tuple[gpd.GeoDataFrame, tuple[str, ...]]:
    """New interrow pieces with canopies subtracted in the tiles over the limit; also those tile ids.

    Split pieces keep their id on the largest part; the other parts get the next free `#k`.
    """
    overlap = overlap_by_tile(canopies, pieces)
    fixed = tuple(t for t in sorted(overlap) if overlap[t] > max_overlap_m2)
    if not fixed:
        return pieces, ()
    # A clearance keeps the shared edges apart, so 0.1 px rounding at export cannot re-create overlap.
    cutters = {t: _union(canopies, t).buffer(clearance_m) if clearance_m > 0.0 else _union(canopies, t)
               for t in fixed}
    used = _next_ids(list(pieces["piece_id"]))
    positions, geoms, ids = [], [], []
    for pos, (pid, tile_id, geom) in enumerate(zip(pieces["piece_id"], pieces[TILE], pieces.geometry, strict=True)):
        parts = _cut(geom, cutters[tile_id], min_piece_m2) if tile_id in cutters else [geom]
        for k, part in enumerate(parts, start=1):
            positions.append(pos)
            geoms.append(part)
            ids.append(pid if k == 1 else _split_id(pid, k, used))
    base = pieces.iloc[positions].drop(columns=pieces.geometry.name).reset_index(drop=True)
    out = gpd.GeoDataFrame(base.assign(piece_id=ids, area_m2=[g.area for g in geoms]), geometry=geoms,
                           crs=pieces.crs)
    return out, fixed


def _overlap_issues(annset: AnnSet, max_overlap_m2: float) -> list[QaIssue]:
    issues = []
    for tile_id, area in overlap_by_tile(annset.canopies, annset.interrow_pieces).items():
        if area > max_overlap_m2:
            where = shapely.intersection(_union(annset.canopies, tile_id), _union(annset.interrow_pieces, tile_id))
            msg = f"canopy ∩ interrow = {area:.4f} m² > {max_overlap_m2} m²"
            issues.append(_issue(OVERLAP_CODE, tile_id, "", msg, where))
    return issues


# ------------------------------------------------------------------ ids and blocks


def _row_block_issues(annset: AnnSet) -> list[QaIssue]:
    """One vineyard_id per row_id, over row pieces and the canopies that carry a row_id."""
    rows = annset.row_pieces.sort_values([TILE, "piece_id"], kind="mergesort")
    can = annset.canopies[annset.canopies["row_id"].notna().to_numpy()]
    blocks: dict[str, set[str]] = {}
    first: dict[str, tuple[str, BaseGeometry]] = {}
    for frame in (rows, can):
        for row_id, vid, tile_id, geom in zip(frame["row_id"], frame["vineyard_id"], frame[TILE], frame.geometry,
                                              strict=True):
            blocks.setdefault(row_id, set()).add(vid)
            first.setdefault(row_id, (tile_id, geom))
    return [_issue(ROW_MULTI_BLOCK_CODE, first[r][0], r, f"row on blocks {sorted(v)}", first[r][1])
            for r, v in sorted(blocks.items()) if len(v) > 1]


def _waste_issues(annset: AnnSet) -> list[QaIssue]:
    known = set(annset.canopies["vineyard_id"].dropna()) | set(annset.row_pieces["vineyard_id"].dropna())
    wst = annset.waste
    return [_issue(WASTE_BLOCK_CODE, t, w, f"waste on unknown block {v!r}", g)
            for w, t, v, g in zip(wst["waste_id"], wst[TILE], wst["vineyard_id"], wst.geometry, strict=True)
            if isinstance(v, str) and v and v not in known]


def _region(tile_id: str, clips: Mapping[str, BaseGeometry], use_clip: bool) -> BaseGeometry:
    box_ = tile_box(tile_ref(tile_id))
    return clips.get(tile_id, box_) if use_clip else box_


def _clip_issues(annset: AnnSet, clips: Mapping[str, BaseGeometry]) -> list[QaIssue]:
    issues = []
    for layer, (id_col, use_clip) in CLIP_LAYERS.items():
        frame = annset.layer(layer)
        for tile_id in _tile_ids(frame):
            sub = frame[(frame[TILE] == tile_id).to_numpy()]
            region = _region(tile_id, clips, use_clip).buffer(CLIP_TOL_M)
            inside = shapely.covers(region, sub.geometry.to_numpy())
            issues.extend(_issue(OUTSIDE_CLIP_CODE, tile_id, oid, f"{layer} object outside its tile clip", g)
                          for oid, g, ok in zip(sub[id_col], sub.geometry, inside, strict=True) if not ok)
    return issues


# ------------------------------------------------------------------ interrows between extreme rows


def _between_rows(lines: Sequence[LineString], points: np.ndarray) -> np.ndarray:
    """True where a point has at least one row of the block on each side."""
    pos = np.zeros(len(points), dtype=bool)
    neg = np.zeros(len(points), dtype=bool)
    ref: np.ndarray | None = None
    xy = shapely.get_coordinates(points)
    for line in lines:
        coords = np.asarray(line.coords)[:, :2]
        d = coords[-1] - coords[0]
        ref = d if ref is None else ref
        d = -d if float(d @ ref) < 0 else d
        near = shapely.get_coordinates(shapely.line_interpolate_point(line, shapely.line_locate_point(line, points)))
        cross = d[0] * (xy[:, 1] - near[:, 1]) - d[1] * (xy[:, 0] - near[:, 0])
        pos |= cross > _SIDE_EPS_M
        neg |= cross < -_SIDE_EPS_M
    return pos & neg


def _block_lines(annset: AnnSet, rows: gpd.GeoDataFrame | None) -> dict[str, list[LineString]]:
    source = rows if rows is not None else annset.row_pieces
    ordered = source.sort_values("row_id", kind="mergesort")
    lines: dict[str, list[LineString]] = {}
    for vid, geom in zip(ordered["vineyard_id"], ordered.geometry, strict=True):
        if isinstance(geom, LineString) and not geom.is_empty:
            lines.setdefault(vid, []).append(geom)
    return lines


def _interrow_issues(annset: AnnSet, rows: gpd.GeoDataFrame | None) -> list[QaIssue]:
    irs = annset.interrow_pieces
    if irs.empty:
        return []
    lines = _block_lines(annset, rows)
    points = np.asarray(shapely.point_on_surface(irs.geometry.to_numpy()))
    ok = np.ones(len(irs), dtype=bool)
    for vid in sorted(set(irs["vineyard_id"].dropna())):
        sel = (irs["vineyard_id"] == vid).to_numpy()
        ok[sel] = _between_rows(lines.get(vid, []), points[sel])
    return [_issue(INTERROW_OUTSIDE_CODE, t, pid, "interrow piece outside the extreme rows of its block", g)
            for pid, t, g, good in zip(irs["piece_id"], irs[TILE], irs.geometry, ok, strict=True) if not good]


def check_invariants(annset: AnnSet, clips: Mapping[str, BaseGeometry], *, max_overlap_m2: float,
                     rows: gpd.GeoDataFrame | None = None) -> tuple[QaIssue, ...]:
    """Every contract §2.7 invariant violation as an error, sorted by QaIssue.sort_key.

    clips: tile_box ∩ tile_valid per tile (the tile box when absent); rows: the global `rows`
    layer for the interrow side test (the AnnSet row pieces when None).
    """
    issues = (_overlap_issues(annset, max_overlap_m2) + _row_block_issues(annset) + _waste_issues(annset)
              + _clip_issues(annset, clips) + _interrow_issues(annset, rows))
    return tuple(sorted(issues, key=lambda i: i.sort_key))
