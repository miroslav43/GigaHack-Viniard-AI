from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, box

from vineyard.farms.grouping import Farm
from vineyard.farms.roads import (
    CROSS_PATH_HIGHWAY,
    RoadClass,
    class_lengths,
    classify_roads,
    farms_frame,
    lineal,
    public_union,
    road_ids,
    roads_frame,
)
from vineyard.geo.tiling import CRS_EPSG
from vineyard.geo.vector_io import write_layer

X0, Y0 = 629000.0, 5220000.0
PUBLIC = frozenset({"residential", "tertiary"})
FARM = Farm("F01", ("V01", "V02"), box(X0, Y0, X0 + 100, Y0 + 100))
PROV = {"source": "model", "run_id": "20260926T1540-post-rc7", "model_version": "pipe@0.1.0", "confidence": 1.0,
        "qa_flags": ""}


def highways(*rows: tuple[int, str, LineString, str | None]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"osm_id": [r[0] for r in rows], "highway": [r[1] for r in rows],
                             "name": [r[3] for r in rows], "surface": [None] * len(rows), "tracktype": [None] * len(rows)},
                            geometry=[r[2] for r in rows], crs=CRS_EPSG)


def across(y: float) -> LineString:
    return LineString([(X0 - 50, Y0 + y), (X0 + 150, Y0 + y)])


def test_a_track_through_a_farm_is_internal_inside_and_field_outside() -> None:
    roads = classify_roads(highways((7, "track", across(50), None)), [FARM], None, public=PUBLIC, min_internal_m=5.0)
    by_class = {r.road_class: r for r in roads}
    assert by_class[RoadClass.INTERNAL].farm_id == "F01"
    assert by_class[RoadClass.INTERNAL].length_m == pytest.approx(100.0)
    assert by_class[RoadClass.FIELD].length_m == pytest.approx(100.0)
    assert by_class[RoadClass.FIELD].farm_id is None


def test_public_roads_stay_public_inside_a_farm() -> None:
    (road,) = classify_roads(highways((3, "residential", across(50), "Str. Viilor")), [FARM], None, public=PUBLIC,
                             min_internal_m=5.0)
    assert (road.road_class, road.name, road.length_m) == (RoadClass.PUBLIC, "Str. Viilor", pytest.approx(200.0))


def test_a_short_stretch_inside_a_farm_stays_a_field_track() -> None:
    corner = LineString([(X0 - 50, Y0 - 2), (X0 + 3, Y0 + 1)])
    (road,) = classify_roads(highways((1, "track", corner, None)), [FARM], None, public=PUBLIC, min_internal_m=5.0)
    assert road.road_class == RoadClass.FIELD


def test_cross_paths_are_internal_roads_of_their_block_farm() -> None:
    lines = gpd.GeoDataFrame({"vineyard_id": ["V02", "V99"]}, geometry=[across(20), across(30)], crs=CRS_EPSG)
    roads = classify_roads(highways(), [FARM], lines, public=PUBLIC, min_internal_m=5.0)
    assert [(r.road_class, r.highway, r.farm_id, r.origin.value) for r in roads] == [
        (RoadClass.INTERNAL, CROSS_PATH_HIGHWAY, "F01", "detected"),
        (RoadClass.INTERNAL, CROSS_PATH_HIGHWAY, None, "detected"),
    ]


def test_roads_come_in_osm_id_order_then_detected() -> None:
    hw = highways((9, "tertiary", across(-30), None), (2, "residential", across(-60), None))
    assert [r.length_m for r in classify_roads(hw, [], None, public=PUBLIC, min_internal_m=5.0)] == [200.0, 200.0]
    assert [r.highway for r in classify_roads(hw, [], None, public=PUBLIC, min_internal_m=5.0)] == [
        "residential", "tertiary"]


def test_lineal_keeps_lines_merges_chains_and_drops_slivers() -> None:
    chain = MultiLineString([[(0, 0), (5, 0)], [(5, 0), (9, 0)]])
    assert lineal(chain).geom_type == "LineString"
    assert lineal(GeometryCollection([Point(0, 0), LineString([(0, 0), (0.5, 0)])])) is None
    assert lineal(None) is None and lineal(LineString()) is None


def test_public_union_and_lengths() -> None:
    hw = highways((1, "track", across(10), None), (2, "residential", across(20), None))
    assert public_union(hw, PUBLIC).length == pytest.approx(200.0)
    assert public_union(hw, frozenset({"motorway"})) is None
    roads = classify_roads(hw, [FARM], None, public=PUBLIC, min_internal_m=5.0)
    assert class_lengths(roads) == {"public": 200.0, "field": 100.0, "internal": 100.0}
    assert road_ids(3) == ["D0001", "D0002", "D0003"]


def test_frames_pass_the_layer_contract(tmp_path: Path) -> None:
    roads = classify_roads(highways((1, "track", across(50), None)), [FARM], None, public=PUBLIC, min_internal_m=5.0)
    rf = roads_frame(roads, PROV)
    assert list(rf.road_id) == ["D0001", "D0002"]
    write_layer(rf, "roads", tmp_path / "roads.parquet")
    ff = farms_frame([FARM], PROV)
    assert (ff.vineyard_ids.iloc[0], int(ff.n_blocks.iloc[0])) == ("V01,V02", 2)
    write_layer(ff, "farms", tmp_path / "farms.parquet")
    write_layer(farms_frame([], PROV), "farms", tmp_path / "empty_farms.parquet")
    write_layer(roads_frame([], PROV), "roads", tmp_path / "empty_roads.parquet")
