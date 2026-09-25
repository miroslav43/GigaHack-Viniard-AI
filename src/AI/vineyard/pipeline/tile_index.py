"""Read access to work/tile_index.parquet (written by ingest) joined with the static tile_valid layer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd

from vineyard.contracts.ids import file_name_from_tile_id
from vineyard.errors import StageError
from vineyard.geo.tiling import TileRef, tile_ref_from_grid
from vineyard.geo.vector_io import read_layer
from vineyard.pipeline.context import RunContext, RunPaths

TILE_INDEX_LAYER: Final = "tile_index"
TILE_VALID_LAYER: Final = "tile_valid"
TILE_ID: Final = "tile_id"
MSG_RUN_INGEST: Final = "tile_index missing; run `vineyard ingest` first"


def _paths(where: RunContext | RunPaths) -> RunPaths:
    return where.paths if isinstance(where, RunContext) else where


def _require_index(paths: RunPaths) -> Path:
    if not paths.tile_index.is_file():
        raise StageError(MSG_RUN_INGEST, path=str(paths.tile_index))
    return paths.tile_index


def indexed_tile_ids(where: RunContext | RunPaths) -> tuple[str, ...]:
    """Sorted tile ids of the tile index (reads only the tile_id column)."""
    frame = pd.read_parquet(_require_index(_paths(where)), columns=[TILE_ID])
    return tuple(sorted(str(t) for t in frame[TILE_ID]))


def _join_valid(index: gpd.GeoDataFrame, paths: RunPaths) -> gpd.GeoDataFrame:
    if not paths.tile_valid.is_file():
        return index
    valid = read_layer(paths.tile_valid, TILE_VALID_LAYER)
    frac = dict(zip(valid[TILE_ID], valid["valid_frac"].astype(np.float64), strict=True))
    area = dict(zip(valid[TILE_ID], valid.geometry.area.astype(np.float64), strict=True))
    ids = index[TILE_ID].astype(str)
    nodata = np.array([1.0 - frac[t] if t in frac else np.nan for t in ids], dtype=np.float32)
    valid_area = np.array([area.get(t, np.nan) for t in ids], dtype=np.float64)
    return index.assign(nodata_frac=nodata, valid_area_m2=valid_area)


def read_tile_index(where: RunContext | RunPaths) -> gpd.GeoDataFrame:
    """The tile index sorted by tile_id; nodata_frac / valid_area_m2 come from tile_valid when present."""
    paths = _paths(where)
    index = read_layer(_require_index(paths), TILE_INDEX_LAYER)
    ordered = index.sort_values(TILE_ID, kind="mergesort").reset_index(drop=True)
    return _join_valid(ordered, paths)


def tile_refs(where: RunContext | RunPaths) -> Mapping[str, TileRef]:
    """tile_id -> TileRef of every indexed tile (grid formula; tags were verified at ingest)."""
    frame = pd.read_parquet(_require_index(_paths(where)), columns=[TILE_ID, "grid_row", "grid_col"])
    refs = {str(t): tile_ref_from_grid(int(r), int(c))
            for t, r, c in zip(frame[TILE_ID], frame["grid_row"], frame["grid_col"], strict=True)}
    return MappingProxyType(dict(sorted(refs.items())))


def tile_path(where: RunContext | RunPaths, tile_id: str) -> Path:
    """work/tiles/<tile_id>.tif (the ingested copy)."""
    return _paths(where).tiles_dir / file_name_from_tile_id(tile_id)
