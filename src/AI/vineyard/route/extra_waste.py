"""Confirmed waste (configs/waste_confirmed.csv) as extra route targets of a post run.

The waste boxes of the AnnSet (Marcaj) stay the only waste that is annotated and measured. Every confirmed
box farther than `min_sep_m` from all of them is added as one more waste object for `targets`, so the route
also collects the waste confirmed after the Marcaj upload. Ids continue from W9001 and skip ids in use.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Final

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box

from vineyard.contracts.ids import format_waste_id
from vineyard.geo.tiling import TileRef
from vineyard.perception.waste.types import Confirmation

FIRST_EXTRA_ID: Final = 9001
LAST_WASTE_ID: Final = 9999


def confirmation_box(conf: Confirmation, tile: TileRef, gsd_m: float) -> shapely.Polygon:
    """Tile px box -> EPSG:32635 polygon (x = x0 + px * gsd, y = y0 - py * gsd)."""
    b = conf.box
    return box(tile.x0 + b.xtl * gsd_m, tile.y0 - b.ybr * gsd_m, tile.x0 + b.xbr * gsd_m, tile.y0 - b.ytl * gsd_m)


def _free_ids(used: set[str]) -> Iterator[str]:
    for k in range(FIRST_EXTRA_ID, LAST_WASTE_ID + 1):
        wid = format_waste_id(k)
        if wid not in used and f"{wid}a" not in used:
            yield wid


def extra_waste(
    waste: gpd.GeoDataFrame,
    confs: Sequence[Confirmation],
    tiles: Mapping[str, TileRef],
    gsd_m: float,
    min_sep_m: float,
) -> gpd.GeoDataFrame:
    """The confirmed boxes not within `min_sep_m` of an AnnSet waste box (or of each other), as waste rows."""
    taken = [g for g in waste.geometry if g is not None and not g.is_empty]
    ids = _free_ids({str(w) for w in waste["waste_id"]})
    rows: list[dict[str, object]] = []
    for conf in confs:
        if conf.tile_id not in tiles:
            continue
        geom = confirmation_box(conf, tiles[conf.tile_id], gsd_m)
        if any(geom.distance(g) <= min_sep_m for g in taken):
            continue
        wid = next(ids, None)
        if wid is None:
            break
        taken.append(geom)
        rows.append({"waste_id": wid, "vineyard_id": "", "category": conf.category.value, "geometry": geom})
    return gpd.GeoDataFrame(rows, columns=["waste_id", "vineyard_id", "category", "geometry"],
                            geometry="geometry", crs=waste.crs)


def with_extra_waste(waste: gpd.GeoDataFrame, extra: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """AnnSet waste plus the extra rows (columns missing on either side are left empty)."""
    if extra.empty:
        return waste
    merged = pd.concat([waste, extra.to_crs(waste.crs) if waste.crs else extra], ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=waste.crs)
