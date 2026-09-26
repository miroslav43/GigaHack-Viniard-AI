"""Per-tile interrow pieces from TILE-LOCAL row pairs (annotation rule 5.1, tune-implement 2026-09-26).

An annotator draws, per tile, one polygon between each pair of neighbouring rows OF THAT TILE, cut at the
tile edge, and nothing outside the tile's outermost rows (the reference has exactly n_rows - 1 pieces per
tile). Cutting the global bands to the tile instead lets rows of neighbouring tiles and unlinked row
fragments add corner triangles and slivers (r006_c004: 34 pieces vs 25, r021_c012: 31 vs 24).

Per block, every row piece of the tile is paired with the nearest along-overlapping piece on each side
(overlap >= min_overlap_frac of the shorter, lateral >= COLLINEAR_M so the two pieces of one row line are
never paired); a pair is kept when its spacing is <= max_spacing_m or when it is a global pair (skip-one).
Piece ends on the clip boundary (tile edge or nodata edge: the row goes on, or the data stops) are
extended by edge_extend_m first, so the band reaches the clip. Bands are punched with the forbidden holes
and cut to the clip. Each piece keeps the id of its global band; a pair without one takes the id of the
global band it overlaps most, else a fresh id after the block's last one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.ids import format_interrow_id, format_interrow_piece_id, interrow_index_of
from vineyard.geo.ops import make_valid_polygonal, orient_ccw
from vineyard.geo.tiling import GSD_M, TileRef
from vineyard.perception.corridor import extend_censored_ends
from vineyard.perception.interrow import PIECE_COLUMNS, band_polygon, punch_holes

if TYPE_CHECKING:
    from vineyard.config import AppConfig

COLLINEAR_M: Final = 0.5  # |lateral| below this: the two pieces lie on one row line
_FAR_M: Final = 1.0e3
BASE_COLUMNS: Final = tuple(c for c in PIECE_COLUMNS if c not in ("interrow_cover", "veg_frac", "shadow_frac",
                                                                  "qa_flags"))


@dataclass(frozen=True)
class LocalPairOptions:
    offset_m: float
    max_spacing_m: float
    min_overlap_frac: float
    edge_extend_m: float
    edge_tol_m: float
    hole_min_area_m2: float
    notch_width_m: float
    min_piece_m2: float
    exclusive: bool = False  # label audit 2026-09-26: no band across a third row, no overlapping pieces
    through_row_min_m: float = 1.0

    @classmethod
    def from_config(cls, cfg: AppConfig) -> LocalPairOptions:
        ir = cfg.interrow
        return cls(offset_m=ir.offset_m, max_spacing_m=cfg.blocks.neighbour_max_m,
                   min_overlap_frac=cfg.blocks.min_overlap_frac, edge_extend_m=ir.edge_extend_m,
                   edge_tol_m=ir.edge_tol_m, hole_min_area_m2=ir.hole_min_area_m2,
                   notch_width_m=cfg.export.cvat.notch_width_px * GSD_M,
                   min_piece_m2=cfg.export.min_interrow_piece_m2, exclusive=ir.exclusive_pieces,
                   through_row_min_m=ir.through_row_min_m)


# ------------------------------------------------------------------ pairing


def _frame_of(line: LineString) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(line.coords)[:, :2]
    d = xy[-1] - xy[0]
    u = d / float(np.hypot(*d))
    return xy, u, np.array([-u[1], u[0]])


def pair_geometry(a: LineString, b: LineString) -> tuple[float, float, float]:
    """(signed lateral offset of b from a at the middle of their along-overlap, overlap length, shorter
    length), measured in a's chord frame; the overlap is 0 when they do not overlap along a."""
    ca, u, n = _frame_of(a)
    cb = np.asarray(b.coords)[:, :2]
    ta, tb = (ca - ca[0]) @ u, (cb - ca[0]) @ u
    lo, hi = max(ta.min(), tb.min()), min(ta.max(), tb.max())
    shorter = min(a.length, b.length)
    if hi <= lo:
        return 0.0, 0.0, shorter
    mid = ca[0] + u * 0.5 * (lo + hi)
    pa = np.asarray(a.interpolate(a.project(shapely.Point(mid))).coords[0])
    hit = b.intersection(LineString([mid - n * _FAR_M, mid + n * _FAR_M]))
    pts = shapely.get_coordinates(hit)
    pb = pts[0] if len(pts) else np.asarray(b.interpolate(b.project(shapely.Point(mid))).coords[0])
    return float((pb - pa) @ n), float(hi - lo), shorter


def adjacent_pairs(lines: list[LineString], ids: list[str], global_pairs: set[frozenset[str]],
                   opts: LocalPairOptions) -> list[tuple[int, int]]:
    """Sorted (i, j), i < j: each piece with its nearest along-overlapping piece on either side."""
    pairs: set[tuple[int, int]] = set()
    for i, li in enumerate(lines):
        best: dict[int, tuple[float, int]] = {}
        for j, lj in enumerate(lines):
            if i == j:
                continue
            off, overlap, shorter = pair_geometry(li, lj)
            if overlap <= 0 or overlap < opts.min_overlap_frac * shorter or abs(off) < COLLINEAR_M:
                continue
            side = 1 if off > 0 else -1
            if side not in best or abs(off) < best[side][0]:
                best[side] = (abs(off), j)
        for dist, j in best.values():
            if dist <= opts.max_spacing_m or frozenset((ids[i], ids[j])) in global_pairs:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


# ------------------------------------------------------------------ ids


@dataclass(frozen=True)
class _BandIndex:
    by_pair: Mapping[frozenset[str], str]
    local: gpd.GeoDataFrame  # bands near the tile (interrow_id, vineyard_id, geometry)
    last_k: Mapping[str, int]  # vineyard_id -> largest band index of the block


def _band_index(bands: gpd.GeoDataFrame, clip: BaseGeometry) -> _BandIndex:
    if bands.empty:
        return _BandIndex({}, bands, {})
    by_pair = {frozenset((str(a), str(b))): str(i)
               for a, b, i in zip(bands["row_left_id"], bands["row_right_id"], bands["interrow_id"], strict=True)}
    last: dict[str, int] = {}
    for vid, iid in zip(bands["vineyard_id"].astype(str), bands["interrow_id"], strict=True):
        last[vid] = max(last.get(vid, 0), interrow_index_of(str(iid)) or 0)
    return _BandIndex(by_pair, bands[bands.intersects(clip)], last)


def _pick_id(vid: str, left: str, right: str, geom: BaseGeometry, index: _BandIndex, fresh: dict[str, int]) -> str:
    exact = index.by_pair.get(frozenset((left, right)))
    if exact is not None:
        return exact
    same = index.local[index.local["vineyard_id"].astype(str) == vid] if not index.local.empty else index.local
    if not same.empty:
        overlap = same.geometry.intersection(geom).area.to_numpy()
        if overlap.max() > 0:
            return str(same["interrow_id"].iloc[int(np.argmax(overlap))])
    fresh[vid] = fresh.get(vid, index.last_k.get(vid, 0)) + 1
    return format_interrow_id(vid, fresh[vid])


# ------------------------------------------------------------------ pieces


def _ordered(a: str, b: str, positions: Mapping[str, int]) -> tuple[str, str]:
    ka, kb = positions.get(a), positions.get(b)
    if ka is not None and kb is not None and ka != kb:
        return (a, b) if ka < kb else (b, a)
    return (a, b) if a <= b else (b, a)


def _band_parts(left: LineString, right: LineString, clip: BaseGeometry, forbidden: BaseGeometry | None,
                opts: LocalPairOptions) -> tuple[list[Polygon], int]:
    band = band_polygon(left, right, opts.offset_m)
    if band is None:
        return [], 0
    holed, n_notches = punch_holes(band, forbidden, opts.hole_min_area_m2, opts.notch_width_m)
    parts = [orient_ccw(p) for h in holed for p in make_valid_polygonal(h.intersection(clip))
             if p.area >= opts.min_piece_m2]
    return parts, n_notches


@dataclass(frozen=True)
class _Part:
    vineyard_id: str
    left: str
    right: str
    n_notches: int
    along: float  # centroid position along the left row (sort key)
    width_mean_m: float
    geometry: Polygon
    interrow_id: str = ""


def _block_parts(vid: str, block: gpd.GeoDataFrame, clip: BaseGeometry, forbidden: BaseGeometry | None,
                 global_pairs: set[frozenset[str]], positions: Mapping[str, int], opts: LocalPairOptions) -> list[_Part]:
    lines = list(block.geometry)
    ids = [str(r) for r in block["row_id"]]
    out: list[_Part] = []
    for i, j in adjacent_pairs(lines, ids, global_pairs, opts):
        left, right = _ordered(ids[i], ids[j], positions)
        a, b = (lines[i], lines[j]) if left == ids[i] else (lines[j], lines[i])
        parts, n_notches = _band_parts(a, b, clip, forbidden, opts)
        if opts.exclusive and _spans_other_row(parts, [ln for k, ln in enumerate(lines) if k not in (i, j)],
                                                opts.through_row_min_m):
            continue  # a band across a third row of the block is not an inter-row (label audit #5)
        _, u, _ = _frame_of(a)
        for part in parts:
            t = shapely.get_coordinates(part) @ u
            span = float(t.max() - t.min())
            out.append(_Part(vid, left, right, n_notches, float(np.asarray(part.centroid.coords)[0] @ u),
                             float(part.area / span) if span > 0 else 0.0, part))
    return out


def _spans_other_row(parts: list[Polygon], others: list[LineString], min_m: float) -> bool:
    """True when another row axis runs >= min_m inside one of the band parts."""
    return any(ln.intersection(p).length >= min_m for p in parts for ln in others if ln.intersects(p))


def exclusive_parts(parts: list[_Part], min_piece_m2: float) -> list[_Part]:
    """Non-overlapping parts: narrowest band first (the true neighbour pair), later parts lose the area
    already taken; pieces under min_piece_m2 are dropped. Deterministic (width, along, ids)."""
    taken: BaseGeometry | None = None
    out: list[_Part] = []
    for part in sorted(parts, key=lambda p: (p.vineyard_id, round(p.width_mean_m, 6), p.left, p.right, p.along)):
        geom = part.geometry if taken is None or not part.geometry.intersects(taken) else \
            part.geometry.difference(taken)
        pieces = [orient_ccw(g) for g in make_valid_polygonal(geom) if g.area >= min_piece_m2]
        out.extend(replace(part, geometry=g) for g in pieces)
        taken = part.geometry if taken is None else shapely.union(taken, part.geometry)
    return out


def tile_interrow_pieces(row_pieces: gpd.GeoDataFrame, bands: gpd.GeoDataFrame, tile: TileRef,
                         clip: BaseGeometry, forbidden: BaseGeometry | None, positions: Mapping[str, int],
                         opts: LocalPairOptions) -> gpd.GeoDataFrame:
    """Interrow pieces of one tile (BASE_COLUMNS + geometry) from its row pieces (clipped to the tile box,
    with row_id and vineyard_id), the global bands (ids, skip-one pairs) and `positions` (row_id -> order
    in its block, e.g. interrow.row_positions of the rows layer). Inputs are not modified."""
    crs = row_pieces.crs if row_pieces.crs is not None else bands.crs
    if len(row_pieces) < 2 or clip.is_empty:
        return _frame([], crs)
    ext = extend_censored_ends(row_pieces, clip.boundary, extend_m=opts.edge_extend_m, tol_m=opts.edge_tol_m)
    index = _band_index(bands, clip)
    global_pairs = set(index.by_pair)
    fresh: dict[str, int] = {}
    parts: list[_Part] = []
    for vid, block in ext.groupby("vineyard_id", sort=True):
        block_parts = _block_parts(str(vid), block, clip, forbidden, global_pairs, positions, opts)
        if opts.exclusive:
            block_parts = exclusive_parts(block_parts, opts.min_piece_m2)
        for part in block_parts:
            iid = _pick_id(part.vineyard_id, part.left, part.right, part.geometry, index, fresh)
            parts.append(replace(part, interrow_id=iid))
    return _frame(parts, crs, tile)


def _frame(parts: list[_Part], crs: object, tile: TileRef | None = None) -> gpd.GeoDataFrame:
    if not parts or tile is None:
        return gpd.GeoDataFrame({c: [] for c in BASE_COLUMNS}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    seen: dict[str, int] = {}
    rows = []
    for p in sorted(parts, key=lambda q: (q.interrow_id, q.along, -q.geometry.area)):
        seen[p.interrow_id] = seen.get(p.interrow_id, 0) + 1
        rows.append({"piece_id": format_interrow_piece_id(p.interrow_id, tile.tile_id, seen[p.interrow_id]),
                     "interrow_id": p.interrow_id, "vineyard_id": p.vineyard_id, "tile_id": tile.tile_id,
                     "row_left_id": p.left, "row_right_id": p.right, "area_m2": float(p.geometry.area),
                     "width_mean_m": p.width_mean_m, "n_notches": p.n_notches, "geometry": p.geometry})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)[[*BASE_COLUMNS, "geometry"]]
