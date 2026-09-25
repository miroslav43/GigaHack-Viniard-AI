"""Vector I/O: schema-checked GeoParquet layers and deterministic GeoJSON (EPSG:32635 / EPSG:4326)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.schemas import coerce_layer, validate_layer
from vineyard.errors import SchemaError
from vineyard.geo.ops import orient_ccw
from vineyard.geo.tiling import CRS_EPSG
from vineyard.pipeline.atomic import atomic_path, atomic_write_text

WGS84_EPSG: Final = 4326
CRS_URN_32635: Final = f"urn:ogc:def:crs:EPSG::{CRS_EPSG}"
_EPSG_IN_NAME: Final = re.compile(r"EPSG:{1,2}(\d+)$", re.IGNORECASE)


# ------------------------------------------------------------------ GeoParquet


def write_layer(gdf: gpd.GeoDataFrame, name: str, path: Path) -> Path:
    """Coerce to the contract schema, validate, and write GeoParquet atomically. Returns `path`."""
    out = coerce_layer(gdf, name)
    validate_layer(out, name)
    target = Path(path)
    with atomic_path(target) as tmp:
        out.to_parquet(tmp, index=False)
    return target


def read_layer(path: Path, name: str | None = None, *, validate: bool = True) -> gpd.GeoDataFrame:
    """Read a GeoParquet layer; validate it against `name` when given and `validate` is set."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"layer file not found: {source} (layer={name})")
    gdf = gpd.read_parquet(source)
    if name is not None and validate:
        validate_layer(gdf, name)
    return gdf


# ------------------------------------------------------------------ GeoJSON


def round_geometry(geom: BaseGeometry, decimals: int) -> BaseGeometry:
    """Round every coordinate to `decimals`; -0.0 becomes 0.0."""
    return shapely.transform(geom, lambda xy: np.round(xy, decimals) + 0.0)


def _epsg_of(gdf: gpd.GeoDataFrame) -> int | None:
    return None if gdf.crs is None else gdf.crs.to_epsg()


def _require_epsg(gdf: gpd.GeoDataFrame, expected: int, what: str) -> None:
    found = _epsg_of(gdf)
    if found != expected:
        raise SchemaError(f"{what}: expected EPSG:{expected}", path=what, got=str(gdf.crs)[:60])


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _orient_polygonal(geom: BaseGeometry) -> BaseGeometry:
    if isinstance(geom, Polygon):
        return orient_ccw(geom)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([orient_ccw(p) for p in geom.geoms])
    return geom


def _geometry_json(geom: BaseGeometry | None, decimals: int, ccw: bool) -> dict[str, Any] | None:
    if geom is None or geom.is_empty:
        return None
    shaped = _orient_polygonal(geom) if ccw else geom
    return mapping(round_geometry(shaped, decimals))


def _features(gdf: gpd.GeoDataFrame, decimals: int, id_column: str | None, ccw: bool) -> list[dict[str, Any]]:
    geom_col = gdf.geometry.name
    columns = [c for c in gdf.columns if c != geom_col]
    features: list[dict[str, Any]] = []
    for row in gdf.itertuples(index=False, name=None):
        record = dict(zip(gdf.columns, row, strict=True))
        feature: dict[str, Any] = {"type": "Feature"}
        if id_column is not None:
            feature["id"] = _json_value(record[id_column])
        feature["properties"] = {c: _json_value(record[c]) for c in columns}
        feature["geometry"] = _geometry_json(record[geom_col], decimals, ccw)
        features.append(feature)
    return features


def _check_id_column(gdf: gpd.GeoDataFrame, id_column: str | None, path: Path) -> None:
    if id_column is not None and id_column not in gdf.columns:
        raise SchemaError(f"{path.name}: id_column {id_column!r} not found", path=str(path), columns=list(gdf.columns))


def _dump_collection(path: Path, collection: dict[str, Any]) -> Path:
    text = json.dumps(collection, ensure_ascii=False, allow_nan=False) + "\n"
    return atomic_write_text(Path(path), text)


def write_geojson(
    gdf: gpd.GeoDataFrame, path: Path, *, decimals: int = 3, id_column: str | None = None
) -> Path:
    """EPSG:32635 FeatureCollection with the organizers' `crs` member; deterministic bytes."""
    target = Path(path)
    _require_epsg(gdf, CRS_EPSG, target.name)
    _check_id_column(gdf, id_column, target)
    collection = {
        "type": "FeatureCollection",
        "name": target.stem,
        "crs": {"type": "name", "properties": {"name": CRS_URN_32635}},
        "features": _features(gdf, decimals, id_column, ccw=False),
    }
    return _dump_collection(target, collection)


def write_geojson_4326(
    gdf: gpd.GeoDataFrame, path: Path, *, decimals: int = 7, id_column: str | None = None
) -> Path:
    """RFC 7946 lon/lat FeatureCollection (no `crs` member); exteriors CCW, holes CW."""
    target = Path(path)
    if gdf.crs is None:
        raise SchemaError(f"{target.name}: input frame has no CRS", path=str(target))
    _check_id_column(gdf, id_column, target)
    lonlat = gdf.to_crs(epsg=WGS84_EPSG)
    collection = {
        "type": "FeatureCollection",
        "name": target.stem,
        "features": _features(lonlat, decimals, id_column, ccw=True),
    }
    return _dump_collection(target, collection)


def _declared_epsg(doc: dict[str, Any], path: Path) -> int:
    crs = doc.get("crs")
    if crs is None:
        return WGS84_EPSG  # RFC 7946 default
    name = str(crs.get("properties", {}).get("name", ""))
    match = _EPSG_IN_NAME.search(name)
    if match is None:
        raise SchemaError(f"{path.name}: unrecognised crs member {name!r}", path=str(path))
    return int(match.group(1))


def read_geojson(path: Path, *, expect_epsg: int = CRS_EPSG) -> gpd.GeoDataFrame:
    """Read a FeatureCollection with plain json; the declared CRS must equal `expect_epsg`."""
    source = Path(path)
    doc = json.loads(source.read_bytes())
    if not isinstance(doc, dict) or doc.get("type") != "FeatureCollection":
        raise SchemaError(f"{source.name}: not a GeoJSON FeatureCollection", path=str(source))
    epsg = _declared_epsg(doc, source)
    if epsg != expect_epsg:
        raise SchemaError(f"{source.name}: declared EPSG:{epsg}, expected EPSG:{expect_epsg}", path=str(source))
    features = doc.get("features") or []
    crs = f"EPSG:{epsg}"
    if not features:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    return gpd.GeoDataFrame.from_features(features, crs=crs)
