"""Row corridors (02 §2.2). P0 stub with frozen signatures; P1-CAN implements the bodies.

Consumers (waste, derive, NN pseudo-labels) code against these signatures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.geo.tiling import TileRef
from vineyard.perception.types import I32

if TYPE_CHECKING:
    import geopandas as gpd


def clip_rows_to_tile(
    rows: gpd.GeoDataFrame, tile: TileRef, clip: BaseGeometry, margin_m: float = 0.0
) -> gpd.GeoDataFrame:
    """One polyline per row per tile (first to last valid vertex), clipped to `clip` grown by margin_m."""
    raise NotImplementedError("perception.corridor.clip_rows_to_tile is implemented in P1-CAN")


def corridor_polygon(axis: LineString, half_m: float) -> Polygon:
    """Flat-capped corridor of half-width `half_m` around `axis`."""
    raise NotImplementedError("perception.corridor.corridor_polygon is implemented in P1-CAN")


def corridor_labels(pieces: gpd.GeoDataFrame, tile: TileRef, half_m: float) -> I32:
    """int32 raster (tile px): label k+1 inside the corridor of piece k, 0 elsewhere."""
    raise NotImplementedError("perception.corridor.corridor_labels is implemented in P1-CAN")
