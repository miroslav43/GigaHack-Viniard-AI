from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, box

from vineyard.geo.tiling import CRS_EPSG
from vineyard.web.bundle import (
    FARM_PROPERTIES,
    ROAD_PROPERTIES,
    farm_features,
    farm_of_block_ids,
    road_features,
)

FARMS = gpd.GeoDataFrame({"farm_id": ["F02", "F01"], "vineyard_ids": ["V03", "V01,V02"], "n_blocks": [1, 2],
                          "area_m2": [10.0, 20.0]}, geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)], crs=CRS_EPSG)
ROADS = gpd.GeoDataFrame({"road_id": ["D0002", "D0001"], "road_class": ["internal", "public"],
                          "highway": ["cross_path", "residential"], "name": [None, "Str. Viilor"],
                          "surface": [None, "asphalt"], "farm_id": ["F01", None], "origin": ["detected", "osm"],
                          "length_m": [5.0, 10.0]},
                         geometry=[LineString([(0, 0), (5, 0)]), LineString([(0, 0), (10, 0)])], crs=CRS_EPSG)


def test_farm_features_sorted_with_id_arrays() -> None:
    f = farm_features(FARMS)
    assert tuple(c for c in f.columns if c != "geometry") == FARM_PROPERTIES
    assert list(f.farm_id) == ["F01", "F02"] and list(f.vineyard_ids) == [["V01", "V02"], ["V03"]]
    assert farm_features(None) is None


def test_road_features_rename_origin_to_source() -> None:
    r = road_features(ROADS)
    assert tuple(c for c in r.columns if c != "geometry") == ROAD_PROPERTIES
    assert list(r.road_id) == ["D0001", "D0002"] and list(r.source) == ["osm", "detected"]
    assert list(r.farm_id) == [None, "F01"] and list(r.name) == ["Str. Viilor", None]
    assert road_features(None) is None


def test_farm_of_block_ids() -> None:
    assert farm_of_block_ids(FARMS) == {"V01": "F01", "V02": "F01", "V03": "F02"}
    assert farm_of_block_ids(None) == {}
