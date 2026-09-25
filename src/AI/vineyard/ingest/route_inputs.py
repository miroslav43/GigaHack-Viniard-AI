"""02_route GeoJSON -> static in_* layers (work/layers/in_*.parquet) with sanity checks (01 §3.6).

Structural problems (missing file, wrong crs member, no features, wrong geometry type, study area
not the 311 tiles) are IngestErrors; START placement problems are warnings.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Point
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.ids import tile_grid_ids
from vineyard.contracts.schemas import coerce_layer, get_schema
from vineyard.errors import IngestError, SchemaError
from vineyard.geo.ops import make_valid_polygonal, orient_ccw
from vineyard.geo.tiling import CRS_EPSG, TILE_M, tile_of_point, tile_ref, utm_to_px
from vineyard.geo.vector_io import CRS_URN_32635, read_geojson, write_layer
from vineyard.logging_setup import get_logger, log_event

ROUTE_FILES: Final[Mapping[str, str]] = MappingProxyType({
    "in_passages": "passages.geojson",
    "in_forbidden": "forbidden.geojson",
    "in_study_area": "study_area.geojson",
    "in_start": "start.geojson",
})
EXPECTED_STUDY_AREA_M2: Final = len(tile_grid_ids()) * TILE_M * TILE_M
STUDY_AREA_TOL_M2: Final = 1.0
START_TILE: Final = "siret3_r018_c010"
START_PX: Final = (28.0, 2002.0)
START_PX_TOL: Final = 0.01
PROPERTY_COLUMNS: Final = ("type", "name", "source")
FID_START: Final = 1
EVENT_REPAIRED: Final = "route_input.repaired"
EVENT_WARNING: Final = "route_input.warning"

_log = get_logger("ingest.route_inputs")


def _load_doc(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IngestError(f"route input missing: {path.name}", path=str(path))
    try:
        doc = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise IngestError(f"{path.name}: invalid JSON: {exc}", path=str(path)) from exc
    member = (doc.get("crs") or {}).get("properties", {}).get("name") if isinstance(doc, dict) else None
    if member != CRS_URN_32635:
        raise IngestError(f"{path.name}: crs member {member!r} != {CRS_URN_32635!r}", path=str(path))
    return doc


def _fix_polygonal(geom: BaseGeometry, path: Path, index: int) -> BaseGeometry:
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise IngestError(f"{path.name}: feature {index} geometry {geom.geom_type} is not polygonal", path=str(path))
    if geom.is_valid:
        return geom
    parts = [orient_ccw(p) for p in make_valid_polygonal(geom)]
    log_event(_log, EVENT_REPAIRED, level=logging.WARNING, file=path.name, feature=index,
              n_parts=len(parts), area_before=round(float(geom.area), 3))
    return MultiPolygon(parts)


def _fix_geometry(geom: BaseGeometry | None, layer: str, path: Path, index: int) -> BaseGeometry:
    if geom is None or geom.is_empty:
        raise IngestError(f"{path.name}: feature {index} has no geometry", path=str(path))
    if "Point" in get_schema(layer).geom_types:
        if geom.geom_type != "Point":
            raise IngestError(f"{path.name}: feature {index} geometry {geom.geom_type} is not a Point",
                              path=str(path))
        return geom
    return _fix_polygonal(geom, path, index)


def read_route_input(path: Path, layer: str) -> gpd.GeoDataFrame:
    """One 02_route file as an `in_*` layer (fid 1..n, type/name/source kept, geometry repaired)."""
    source = Path(path)
    _load_doc(source)
    try:
        raw = read_geojson(source, expect_epsg=CRS_EPSG)
    except SchemaError as exc:
        raise IngestError(f"{source.name}: {exc}", path=str(source)) from exc
    if raw.empty:
        raise IngestError(f"{source.name}: no features", path=str(source))
    geoms = [_fix_geometry(g, layer, source, i) for i, g in enumerate(raw.geometry)]
    data: dict[str, Any] = {"fid": np.arange(FID_START, FID_START + len(raw), dtype=np.int64)}
    for col in PROPERTY_COLUMNS:
        data[col] = [None if v is None or v != v else str(v) for v in raw[col]] if col in raw else [None] * len(raw)
    return coerce_layer(gpd.GeoDataFrame(data, geometry=geoms, crs=CRS_EPSG), layer)


def _start_warnings(start: Point, passages: gpd.GeoDataFrame) -> list[str]:
    warnings: list[str] = []
    if not passages.geometry.union_all().covers(start):
        warnings.append(f"START {start.x:.2f},{start.y:.2f} is not inside the passages")
    ref = tile_of_point(start.x, start.y)
    uv = utm_to_px(tile_ref(START_TILE), np.array([[start.x, start.y]]))[0]
    if ref.tile_id != START_TILE or np.abs(uv - np.asarray(START_PX)).max() > START_PX_TOL:
        warnings.append(f"START is in {ref.tile_id} at px ({uv[0]:.2f}, {uv[1]:.2f}), expected "
                        f"{START_TILE} at {START_PX}")
    return warnings


def check_route_sanity(layers: Mapping[str, gpd.GeoDataFrame]) -> tuple[str, ...]:
    """Warnings for START placement; IngestError when the study area is not the 311 tiles."""
    area = float(layers["in_study_area"].area.sum())
    if abs(area - EXPECTED_STUDY_AREA_M2) > STUDY_AREA_TOL_M2:
        raise IngestError(f"study_area is {area:.2f} m², expected {EXPECTED_STUDY_AREA_M2:.2f} m²",
                          tolerance_m2=STUDY_AREA_TOL_M2)
    starts = list(layers["in_start"].geometry)
    warnings = [f"in_start has {len(starts)} features, expected 1"] if len(starts) != 1 else []
    return tuple(warnings + _start_warnings(starts[0], layers["in_passages"]))


def ingest_route_inputs(route_dir: Path, out_dir: Path) -> tuple[dict[str, Path], tuple[str, ...]]:
    """Read, check and write the four in_* layers; returns (layer -> path, warnings)."""
    layers = {name: read_route_input(Path(route_dir) / file, name) for name, file in ROUTE_FILES.items()}
    warnings = check_route_sanity(layers)
    for message in warnings:
        log_event(_log, EVENT_WARNING, level=logging.WARNING, message_text=message)
    written = {name: write_layer(gdf, name, Path(out_dir) / f"{name}.parquet") for name, gdf in layers.items()}
    return written, warnings
