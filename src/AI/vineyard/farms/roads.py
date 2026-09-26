"""Road classes for the web map: public (village streets, county roads), field (tracks between parcels) and
internal (a farm's own tracks: field-track stretches inside its outline and the tracks across the rows).

Public highways stay public even inside a farm outline; only non-public ways are cut by the farm outlines.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from vineyard.farms.grouping import Farm, farm_of_block
from vineyard.geo.tiling import CRS_EPSG

CROSS_PATH_HIGHWAY: Final = "cross_path"
ROAD_PREFIX: Final = "D"
MIN_PIECE_M: Final = 1.0  # shorter leftovers of a cut are dropped (slivers at the outline)


class RoadClass(StrEnum):
    PUBLIC = "public"
    FIELD = "field"
    INTERNAL = "internal"


class RoadOrigin(StrEnum):
    OSM = "osm"
    DETECTED = "detected"
    CADASTRE = "cadastre"


@dataclass(frozen=True)
class Road:
    road_class: RoadClass
    highway: str
    name: str | None
    surface: str | None
    farm_id: str | None
    origin: RoadOrigin
    geometry: BaseGeometry  # LineString | MultiLineString, EPSG:32635
    cadastral: bool | None = None  # an official road parcel covers it (None: no cadastre snapshot)

    @property
    def length_m(self) -> float:
        return float(self.geometry.length)


def lineal(geom: BaseGeometry | None) -> BaseGeometry | None:
    """The line part of `geom`, merged where pieces chain; None when shorter than MIN_PIECE_M."""
    if geom is None or geom.is_empty:
        return None
    parts = [g for g in getattr(geom, "geoms", (geom,)) if isinstance(g, LineString | MultiLineString)]
    lines = [ln for p in parts for ln in getattr(p, "geoms", (p,)) if ln.length > 0]
    if not lines:
        return None
    merged = shapely.line_merge(MultiLineString(lines)) if len(lines) > 1 else lines[0]
    return merged if merged.length >= MIN_PIECE_M else None


def public_union(highways: gpd.GeoDataFrame, public: frozenset[str]) -> BaseGeometry | None:
    """Union of the public highways' lines (None when there is none)."""
    lines = [g for h, g in zip(highways["highway"], highways.geometry, strict=True) if h in public]
    return shapely.union_all(lines) if lines else None


def _split_way(geom: BaseGeometry, farms: Sequence[Farm], min_internal_m: float) -> tuple[list[tuple[str, BaseGeometry]],
                                                                                          BaseGeometry | None]:
    """(farm_id, stretch inside it) for every farm holding at least `min_internal_m` of the way; the rest."""
    inside: list[tuple[str, BaseGeometry]] = []
    rest = geom
    for farm in farms:
        if rest is None or not rest.intersects(farm.outline):
            continue
        stretch = lineal(rest.intersection(farm.outline))
        if stretch is not None and stretch.length >= min_internal_m:
            inside.append((farm.farm_id, stretch))
            rest = lineal(rest.difference(farm.outline))
    return inside, rest


def _osm_roads(row: Mapping[str, Any], farms: Sequence[Farm], public: frozenset[str],
               min_internal_m: float) -> list[Road]:
    geom = lineal(row["geometry"])
    if geom is None:
        return []
    tags = {"highway": str(row["highway"]), "name": row.get("name"), "surface": row.get("surface")}
    if tags["highway"] in public:
        return [Road(RoadClass.PUBLIC, **tags, farm_id=None, origin=RoadOrigin.OSM, geometry=geom)]
    inside, rest = _split_way(geom, farms, min_internal_m)
    roads = [Road(RoadClass.INTERNAL, **tags, farm_id=fid, origin=RoadOrigin.OSM, geometry=g) for fid, g in inside]
    if rest is not None:
        roads.append(Road(RoadClass.FIELD, **tags, farm_id=None, origin=RoadOrigin.OSM, geometry=rest))
    return roads


def cross_path_roads(lines: gpd.GeoDataFrame | None, farms: Sequence[Farm]) -> list[Road]:
    """The tracks across the rows (passable's cross_path_lines) as internal roads of their block's farm."""
    if lines is None or lines.empty:
        return []
    farm_of = farm_of_block(farms)
    roads = []
    for vid, geom in zip(lines["vineyard_id"], lines.geometry, strict=True):
        line = lineal(geom)
        if line is not None:
            roads.append(Road(RoadClass.INTERNAL, CROSS_PATH_HIGHWAY, None, None, farm_of.get(str(vid)),
                              RoadOrigin.DETECTED, line))
    return roads


def classify_roads(highways: gpd.GeoDataFrame, farms: Sequence[Farm], cross_lines: gpd.GeoDataFrame | None, *,
                   public: frozenset[str], min_internal_m: float) -> tuple[Road, ...]:
    """Every road piece of the snapshot plus the detected tracks across the rows, in a stable order."""
    rows: Iterable[Mapping[str, Any]] = highways.sort_values("osm_id", kind="stable").to_dict("records")
    roads = [r for row in rows for r in _osm_roads(row, farms, public, min_internal_m)]
    return tuple(roads + cross_path_roads(cross_lines, farms))


def road_ids(n: int) -> list[str]:
    return [f"{ROAD_PREFIX}{k + 1:0{max(4, len(str(n)))}d}" for k in range(n)]


def roads_frame(roads: Sequence[Road], provenance: Mapping[str, Any]) -> gpd.GeoDataFrame:
    """The `roads` layer (contract schema) of the given roads, numbered D0001… in order."""
    data = {"road_id": road_ids(len(roads)), "road_class": [r.road_class.value for r in roads],
            "highway": [r.highway for r in roads], "name": [r.name for r in roads],
            "surface": [r.surface for r in roads], "farm_id": [r.farm_id for r in roads],
            "origin": [r.origin.value for r in roads], "length_m": [r.length_m for r in roads],
            "cadastral": pd.array([r.cadastral for r in roads], dtype="boolean"),
            **{k: [v] * len(roads) for k, v in provenance.items()}}
    return gpd.GeoDataFrame(data, geometry=gpd.GeoSeries([r.geometry for r in roads], crs=CRS_EPSG), crs=CRS_EPSG)


def class_lengths(roads: Sequence[Road]) -> dict[str, float]:
    """Total length (m) per road class, every class present."""
    return {c.value: round(sum(r.length_m for r in roads if r.road_class == c), 2) for c in RoadClass}
