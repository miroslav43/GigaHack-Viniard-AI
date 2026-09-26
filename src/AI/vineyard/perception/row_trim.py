"""Row ends at the last vine (annotation rules 4.1 / 4.4, label audit 2026-09-26).

A row polyline runs "from the first to the last vine of the row in this tile, or to the tile edge". The
global rows are fitted lines that can overshoot the planting (a fitted end, a guided row carried to the
headland, a line across a track into a ploughed field). Per tile, every piece end that is also an end of
the whole row (no continuation into the next tile) is pulled back to the last canopy within `band_m` of
the axis plus `margin_m`, when it overshoots by more than `min_overshoot_m`. Ends where the row goes on
into the next tile, and ends on the tile edge (rule 4.1: "or to the tile edge"; the neighbour may simply
have missed the row), are kept. A piece without any canopy is left unchanged.

Used identically by row_attrs (exported row pieces, row_structure) and interrow (the pieces the inter-row
polygons are built between), so both see the same row ends.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import geopandas as gpd
import shapely
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

if TYPE_CHECKING:
    from vineyard.config.sections_perception import RowStructureConfig

END_EPS_M: Final = 1e-3  # a piece end within this of the row end is the row end
EDGE_TOL_M: Final = 0.05  # a piece end this close to the tile box boundary lies on the tile edge
ALONG_FROM: Final = "along_from_m"
ALONG_TO: Final = "along_to_m"


@dataclass(frozen=True)
class TrimOptions:
    enabled: bool
    band_m: float
    margin_m: float
    min_overshoot_m: float
    min_occupancy: float = 0.0

    @classmethod
    def from_config(cls, cfg: RowStructureConfig) -> TrimOptions:
        return cls(enabled=cfg.trim_ends_enabled, band_m=cfg.trim_band_m, margin_m=cfg.trim_margin_m,
                   min_overshoot_m=cfg.trim_min_overshoot_m, min_occupancy=cfg.trim_min_occupancy)


def canopy_intervals(line: LineString, canopies: Sequence[BaseGeometry], band_m: float
                     ) -> list[tuple[float, float]]:
    """Sorted, merged along intervals on `line` of the canopy parts within band_m of it."""
    if not canopies or line.length <= 0:
        return []
    corridor = line.buffer(band_m, cap_style="flat")
    spans = []
    for g in canopies:
        if g is None or g.is_empty or not g.intersects(corridor):
            continue
        pts = shapely.get_coordinates(g.intersection(corridor))
        if len(pts):
            pos = [line.project(shapely.Point(p)) for p in pts]
            spans.append((min(pos), max(pos)))
    merged: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def canopy_span(line: LineString, canopies: Sequence[BaseGeometry], band_m: float) -> tuple[float, float] | None:
    """(first, last) along position on `line` of canopy parts within band_m of it; None without any."""
    iv = canopy_intervals(line, canopies, band_m)
    return (iv[0][0], iv[-1][1]) if iv else None


def trimmed_span(line: LineString, head_free: bool, tail_free: bool, canopies: Sequence[BaseGeometry],
                 opts: TrimOptions) -> tuple[float, float]:
    """(from, to) along `line` after pulling the free ends back to the canopy (unchanged: (0, length)).

    Only rows whose canopies cover >= min_occupancy of the kept span are trimmed: in young plantings the
    canopy stage misses most vines, and there the fitted row is better evidence than the canopies."""
    length = float(line.length)
    iv = canopy_intervals(line, canopies, opts.band_m) if (head_free or tail_free) else []
    if not iv:
        return 0.0, length
    first, last = iv[0][0], iv[-1][1]
    covered = sum(b - a for a, b in iv)
    if last - first <= END_EPS_M or covered / (last - first) < opts.min_occupancy:
        return 0.0, length
    lo, hi = 0.0, length
    if head_free and first > opts.min_overshoot_m:
        lo = max(0.0, first - opts.margin_m)
    if tail_free and length - last > opts.min_overshoot_m:
        hi = min(length, last + opts.margin_m)
    return (lo, hi) if hi - lo > END_EPS_M else (0.0, length)


def _off_edge(geom: LineString, first: bool, edge: BaseGeometry | None) -> bool:
    """True when the piece end is not on the tile edge (an end on the edge may go on in the next tile)."""
    if edge is None:
        return True
    xy = geom.coords[0] if first else geom.coords[-1]
    return edge.distance(shapely.Point(xy)) > EDGE_TOL_M


def trim_row_pieces(pieces: gpd.GeoDataFrame, row_lengths: dict[str, float], canopies: gpd.GeoDataFrame,
                    opts: TrimOptions, tile_edge: BaseGeometry | None = None) -> gpd.GeoDataFrame:
    """New frame of `pieces` (clip_rows_to_tile output: row_id, along_from_m, along_to_m) with the ends that
    are row ends (and not on `tile_edge`, the tile box boundary) trimmed to the tile's canopies;
    along_from_m / along_to_m follow the trim."""
    if not opts.enabled or pieces.empty or canopies is None or canopies.empty:
        return pieces
    polys = [g for g in canopies.geometry if g is not None and not g.is_empty]
    tree = shapely.STRtree(polys) if polys else None
    geoms, froms, tos = [], [], []
    for rid, geom, a0, a1 in zip(pieces["row_id"].astype(str), pieces.geometry, pieces[ALONG_FROM],
                                 pieces[ALONG_TO], strict=True):
        full = row_lengths.get(rid, float("inf"))
        head_free = float(a0) <= END_EPS_M and _off_edge(geom, True, tile_edge)
        tail_free = float(a1) >= full - END_EPS_M and _off_edge(geom, False, tile_edge)
        near = [polys[int(i)] for i in tree.query(geom, predicate="dwithin", distance=opts.band_m)] if tree else []
        lo, hi = trimmed_span(geom, head_free, tail_free, near, opts)
        if lo <= 0.0 and hi >= geom.length:
            geoms.append(geom)
            froms.append(float(a0))
            tos.append(float(a1))
            continue
        geoms.append(substring(geom, lo, hi))
        froms.append(float(a0) + lo)
        tos.append(float(a0) + hi)
    return gpd.GeoDataFrame(pieces.drop(columns=pieces.geometry.name).assign(**{ALONG_FROM: froms, ALONG_TO: tos}),
                            geometry=gpd.GeoSeries(geoms, crs=pieces.crs), crs=pieces.crs)


def row_lengths_of(rows: gpd.GeoDataFrame) -> dict[str, float]:
    """row_id -> full length of the global row polyline."""
    return {str(r): float(g.length) for r, g in zip(rows["row_id"], rows.geometry, strict=True)}

