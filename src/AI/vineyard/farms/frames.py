"""Layer frames of the `farms` stage: farms (outlines) and farm_blocks (block -> farm + cadastral parcels)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from vineyard.farms.grouping import Farm, farm_of_block
from vineyard.farms.parcels import ParcelStats
from vineyard.geo.tiling import CRS_EPSG


def _parcel_columns(keys: Sequence[str], stats: Mapping[str, ParcelStats] | None) -> dict[str, Any]:
    """n_parcels / cadastral_codes / landuse_counts per key (nulls without a cadastre snapshot)."""
    found = [None if stats is None else stats.get(k) for k in keys]
    return {"n_parcels": pd.array([None if s is None else s.n_parcels for s in found], dtype="Int32"),
            "cadastral_codes": [None if s is None else s.codes_text for s in found],
            "landuse_counts": [None if s is None else s.landuse_json for s in found]}


def farms_frame(farms: Sequence[Farm], provenance: Mapping[str, Any],
                stats: Mapping[str, ParcelStats] | None = None) -> gpd.GeoDataFrame:
    """The `farms` layer: one outline per farm; vineyard_ids comma-joined (V03,V07)."""
    ids = [f.farm_id for f in farms]
    data = {"farm_id": ids, "vineyard_ids": [",".join(f.vineyard_ids) for f in farms],
            "n_blocks": [len(f.vineyard_ids) for f in farms], "area_m2": [f.area_m2 for f in farms],
            **_parcel_columns(ids, stats), **{k: [v] * len(farms) for k, v in provenance.items()}}
    return gpd.GeoDataFrame(data, geometry=gpd.GeoSeries([f.outline for f in farms], crs=CRS_EPSG), crs=CRS_EPSG)


def farm_blocks_frame(blocks: Mapping[str, BaseGeometry], farms: Sequence[Farm], provenance: Mapping[str, Any],
                      stats: Mapping[str, ParcelStats] | None = None) -> gpd.GeoDataFrame:
    """The `farm_blocks` layer: every grouped block with its farm and its cadastral parcels."""
    farm_of = farm_of_block(farms)
    ids = [vid for vid in blocks if vid in farm_of]
    data = {"vineyard_id": ids, "farm_id": [farm_of[v] for v in ids], **_parcel_columns(ids, stats),
            **{k: [v] * len(ids) for k, v in provenance.items()}}
    return gpd.GeoDataFrame(data, geometry=gpd.GeoSeries([blocks[v] for v in ids], crs=CRS_EPSG), crs=CRS_EPSG)
