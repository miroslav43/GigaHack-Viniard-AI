"""route.geojson (contract §5.1, web contract §6.3): writer, reader and an independent file checker.

The written line is rounded to `decimals`, consecutive duplicates created by rounding are dropped, the
first and last vertex are START exactly, and `length_m` is computed on the rounded line. The checker
re-derives everything from the file and never trusts its own properties (publish uses it).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, shape
from shapely.geometry.base import BaseGeometry

from vineyard.config import RouteConfig
from vineyard.errors import RouteValidationError
from vineyard.geo.tiling import CRS_EPSG
from vineyard.geo.vector_io import CRS_URN_32635, write_geojson
from vineyard.route.validate import outside_lengths

CRS_URN: Final = CRS_URN_32635
LENGTH_PROP: Final = "length_m"
MIN_DISTINCT_VERTICES: Final = 2


@dataclass(frozen=True)
class RouteFileLimits:
    max_outside_frac: float
    closure_max_m: float
    length_tol_m: float
    grid_size_m: float
    decimals: int

    @classmethod
    def from_route_cfg(cls, cfg: RouteConfig) -> RouteFileLimits:
        return cls(max_outside_frac=cfg.max_outside_frac_publish, closure_max_m=cfg.validate_.closure_max_m,
                   length_tol_m=cfg.validate_.length_tol_m, grid_size_m=cfg.domain.grid_size_m,
                   decimals=cfg.validate_.coord_decimals)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


def _dedupe(xy: np.ndarray) -> np.ndarray:
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.any(xy[1:] != xy[:-1], axis=1)
    return xy[keep]


def finalize_route_line(line: LineString, start_xy: tuple[float, float], decimals: int) -> LineString:
    """Rounded line with START exactly as first and last vertex and no zero-length segments."""
    start = np.round(np.asarray(start_xy, dtype=np.float64), decimals) + 0.0
    xy = np.round(shapely.get_coordinates(line), decimals) + 0.0
    if len(xy) and np.any(xy[0] != start):
        xy = np.vstack([start, xy])
    if np.any(xy[-1] != start):
        xy = np.vstack([xy, start])
    xy = _dedupe(xy)
    if len(np.unique(xy, axis=0)) < MIN_DISTINCT_VERTICES:
        raise RouteValidationError("route has fewer than 2 distinct vertices after rounding",
                                   n_vertices=len(xy), decimals=decimals)
    return LineString(xy)


def write_route_geojson(line: LineString, props: Mapping[str, Any], path: Path, *,
                        start_xy: tuple[float, float], decimals: int) -> LineString:
    """Write the single-Feature FeatureCollection; returns the exact line that was written."""
    final = finalize_route_line(line, start_xy, decimals)
    record = {LENGTH_PROP: round(final.length, decimals)} | {k: v for k, v in props.items() if k != LENGTH_PROP}
    gdf = gpd.GeoDataFrame([record], geometry=[final], crs=CRS_EPSG)
    write_geojson(gdf, Path(path), decimals=decimals)
    return final


def _load_feature(doc: Any, source: Path) -> dict[str, Any]:
    if not isinstance(doc, dict) or doc.get("type") != "FeatureCollection":
        raise RouteValidationError("route file is not a GeoJSON FeatureCollection", path=str(source))
    features = doc.get("features") or []
    if len(features) != 1:
        raise RouteValidationError("route file must hold exactly one Feature", path=str(source),
                                   n_features=len(features))
    return features[0]


def read_route_geojson(path: Path) -> tuple[BaseGeometry, dict[str, Any]]:
    """(geometry, properties) of the single Feature in `path`."""
    source = Path(path)
    feature = _load_feature(json.loads(source.read_bytes()), source)
    return shape(feature["geometry"]), dict(feature.get("properties") or {})


def _decimals_ok(coords: Any, decimals: int) -> bool:
    flat = np.asarray(coords, dtype=np.float64).ravel()
    return bool(np.all(np.round(flat, decimals) == flat))


def _geometry_checks(geom: BaseGeometry, props: Mapping[str, Any], start_xy: tuple[float, float],
                     inner: BaseGeometry, limits: RouteFileLimits) -> list[Check]:
    xy = shapely.get_coordinates(geom)
    start = np.asarray(start_xy, dtype=np.float64)
    closure = float(max(np.hypot(*(xy[0] - start)), np.hypot(*(xy[-1] - start))))
    declared = props.get(LENGTH_PROP)
    length_ok = isinstance(declared, int | float) and abs(declared - geom.length) <= limits.length_tol_m
    outside = float(outside_lengths(geom, inner, limits.grid_size_m).sum())
    frac = outside / geom.length if geom.length > 0 else 1.0
    seg_len = np.hypot(*np.diff(xy, axis=0).T)
    return [
        Check("closure", closure <= limits.closure_max_m, f"closure_m={closure:.4f}"),
        Check(LENGTH_PROP, bool(length_ok), f"declared={declared} recomputed={geom.length:.4f}"),
        Check("outside_frac", frac <= limits.max_outside_frac, f"outside_frac={frac:.5f} outside_m={outside:.3f}"),
        Check("zero_length_segments", bool(np.all(seg_len > 0)), f"n_zero={int(np.sum(seg_len <= 0))}"),
    ]


def check_route_file(path: Path, *, start_xy: tuple[float, float], inner: BaseGeometry,
                     limits: RouteFileLimits) -> tuple[Check, ...]:
    """Every publish check, recomputed from the file itself (contract §5.1)."""
    source = Path(path)
    try:
        doc = json.loads(source.read_bytes())
    except (OSError, ValueError) as exc:
        return (Check("readable", False, f"{type(exc).__name__}: {exc}"),)
    if not isinstance(doc, dict) or doc.get("type") != "FeatureCollection":
        return (Check("readable", False, "not a FeatureCollection"),)
    features = doc.get("features") or []
    crs_name = (doc.get("crs") or {}).get("properties", {}).get("name")
    checks = [Check("readable", True), Check("single_feature", len(features) == 1, f"n={len(features)}"),
              Check("crs", crs_name == CRS_URN, f"crs={crs_name}")]
    if not features:
        return (*checks, Check("linestring", False, "no feature"))
    geometry = features[0].get("geometry") or {}
    is_line = geometry.get("type") == "LineString" and len(geometry.get("coordinates") or []) >= 2
    checks.append(Check("linestring", is_line, f"type={geometry.get('type')}"))
    if not is_line:
        return tuple(checks)
    checks.append(Check("decimals", _decimals_ok(geometry["coordinates"], limits.decimals),
                        f"max {limits.decimals} decimals"))
    geom = shape(geometry)
    props = features[0].get("properties") or {}
    return (*checks, *_geometry_checks(geom, props, start_xy, inner, limits))


def failed_checks(checks: tuple[Check, ...]) -> tuple[Check, ...]:
    return tuple(c for c in checks if not c.ok)


def route_length_ok(declared: float, line: LineString, tol_m: float) -> bool:
    return math.isfinite(declared) and abs(declared - line.length) <= tol_m


__all__ = [
    "CRS_URN", "Check", "RouteFileLimits", "check_route_file", "failed_checks", "finalize_route_line",
    "read_route_geojson", "route_length_ok", "write_route_geojson",
]
