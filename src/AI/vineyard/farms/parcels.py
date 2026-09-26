"""Cadastral parcels against our blocks, farms and roads (vineyard.farms.cadastre is the snapshot).

- parcel_stats: the parcels a block / farm lies on (overlap >= min m² or >= min share of the parcel);
- cadastral: an OSM road is confirmed when enough of it lies in road parcels ("Cale de comunicație");
- cadastre_roads: centrelines of the road parcels that no OSM road follows (public roads missing from OSM);
  corridors wider than `road_max_width_m` (a national-road strip, a whole road network registered as one
  parcel) give no centreline.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.farms.roads import Road, RoadClass, RoadOrigin, lineal
from vineyard.route.skeleton import SkeletonParams, skeleton_lines

if TYPE_CHECKING:
    from vineyard.config.sections_post import FarmsConfig

CADASTRE_HIGHWAY: Final = "road_parcel"
SPUR_MIN_M: Final = 8.0  # skeleton spurs of a road parcel shorter than this are its corners, not roads
SIMPLIFY_M: Final = 0.5
GAP_SAMPLES: Final = 11
_FOLD: Final = str.maketrans({"ţ": "ț", "ş": "ș", "Ţ": "ț", "Ş": "ș"})


def normalize_landuse(text: str | None) -> str:
    """Comparable land use: cedilla ţ/ş -> comma ț/ș, case-folded, single spaces."""
    return " ".join(str(text or "").translate(_FOLD).casefold().split())


@dataclass(frozen=True)
class ParcelParams:
    min_overlap_m2: float
    min_overlap_frac: float
    road_landuses: frozenset[str]  # normalised
    road_cover_frac: float
    road_buffer_m: float
    road_min_len_m: float
    road_res_m: float
    road_max_width_m: float
    road_osm_gap_m: float

    @classmethod
    def from_config(cls, cfg: FarmsConfig) -> ParcelParams:
        return cls(min_overlap_m2=cfg.parcel_min_overlap_m2, min_overlap_frac=cfg.parcel_min_overlap_frac,
                   road_landuses=frozenset(normalize_landuse(v) for v in cfg.road_landuses),
                   road_cover_frac=cfg.road_cover_frac, road_buffer_m=cfg.road_parcel_buffer_m,
                   road_min_len_m=cfg.cadastre_road_min_len_m, road_res_m=cfg.cadastre_road_res_m,
                   road_max_width_m=cfg.cadastre_road_max_width_m, road_osm_gap_m=cfg.cadastre_road_osm_gap_m)


@dataclass(frozen=True)
class ParcelStats:
    n_parcels: int
    codes: tuple[str, ...]
    landuse_counts: tuple[tuple[str, int], ...]  # most frequent first, then by name

    @property
    def codes_text(self) -> str:
        return ",".join(self.codes)

    @property
    def landuse_json(self) -> str:
        return json.dumps(dict(self.landuse_counts), ensure_ascii=False)


class ParcelIndex:
    """The parcels with an STRtree over their geometries."""

    def __init__(self, parcels: gpd.GeoDataFrame) -> None:
        self.codes = [str(c) for c in parcels["codcadastral"]]
        self.landuses = [str(v or "") for v in parcels["landuse"]]
        self.geoms = list(parcels.geometry)
        self.tree = shapely.STRtree(self.geoms)

    def stats(self, area: BaseGeometry, params: ParcelParams) -> ParcelStats:
        hits = []
        for i in self.tree.query(area, predicate="intersects"):
            parcel = self.geoms[int(i)]
            overlap = parcel.intersection(area).area
            if overlap >= params.min_overlap_m2 or (parcel.area > 0 and overlap / parcel.area >= params.min_overlap_frac):
                hits.append(int(i))
        counts = Counter(self.landuses[i] for i in hits)
        ranked = tuple(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        return ParcelStats(len(hits), tuple(sorted(self.codes[i] for i in hits)), ranked)


def stats_by_key(areas: Mapping[str, BaseGeometry], index: ParcelIndex, params: ParcelParams) -> dict[str, ParcelStats]:
    return {key: index.stats(geom, params) for key, geom in areas.items()}


def road_parcels(parcels: gpd.GeoDataFrame, params: ParcelParams) -> gpd.GeoDataFrame:
    mask = [normalize_landuse(v) in params.road_landuses for v in parcels["landuse"]]
    return parcels.loc[mask]


def mark_cadastral(roads: Sequence[Road], road_area: BaseGeometry | None, params: ParcelParams) -> tuple[Road, ...]:
    """Every road with `cadastral` set: True when >= road_cover_frac of it lies in the (buffered) road parcels."""
    if road_area is None or road_area.is_empty:
        return tuple(replace(r, cadastral=False) for r in roads)
    zone = road_area.buffer(params.road_buffer_m) if params.road_buffer_m > 0 else road_area
    shapely.prepare(zone)
    return tuple(replace(r, cadastral=r.length_m > 0 and r.geometry.intersection(zone).length / r.length_m
                         >= params.road_cover_frac) for r in roads)


def _mean_width(poly: BaseGeometry) -> float:
    return 2.0 * poly.area / poly.length if poly.length > 0 else 0.0


def _median_gap_m(line: BaseGeometry, other: BaseGeometry) -> float:
    """Median distance to `other` of GAP_SAMPLES points evenly spread along `line`."""
    points = [line.interpolate(k / (GAP_SAMPLES - 1), normalized=True) for k in range(GAP_SAMPLES)]
    return float(np.median([p.distance(other) for p in points]))


def _pieces(geom: BaseGeometry | None) -> list[BaseGeometry]:
    line = lineal(geom)
    return [] if line is None else list(getattr(line, "geoms", (line,)))


def cadastre_roads(parcels: gpd.GeoDataFrame, osm_lines: BaseGeometry | None, params: ParcelParams) -> tuple[Road, ...]:
    """Centrelines of the narrow road parcels, minus what an OSM road already follows (within half the parcel
    width + road_buffer_m), kept from road_min_len_m on and when they do not run alongside an OSM road (median gap
    <= road_osm_gap_m: the same road, offset in the cadastre): public roads the OSM snapshot lacks."""
    skeleton = SkeletonParams(res_m=params.road_res_m, erode_m=0.0, spur_min_m=SPUR_MIN_M, simplify_tol_m=SIMPLIFY_M)
    roads: list[Road] = []
    for poly in road_parcels(parcels, params).geometry:
        width = _mean_width(poly)
        if width <= 0 or width > params.road_max_width_m:
            continue
        lines = skeleton_lines(poly, skeleton)
        if not lines:
            continue
        centre = shapely.union_all(list(lines))
        has_osm = osm_lines is not None and not osm_lines.is_empty
        if has_osm:
            centre = centre.difference(osm_lines.buffer(width / 2.0 + params.road_buffer_m))
        for piece in _pieces(centre):
            if piece.length < params.road_min_len_m or (has_osm and _median_gap_m(piece, osm_lines)
                                                        <= params.road_osm_gap_m):
                continue
            roads.append(Road(RoadClass.PUBLIC, CADASTRE_HIGHWAY, None, None, None, RoadOrigin.CADASTRE, piece,
                              cadastral=True))
    return tuple(sorted(roads, key=lambda r: (round(r.geometry.bounds[0], 2), round(r.geometry.bounds[1], 2))))
