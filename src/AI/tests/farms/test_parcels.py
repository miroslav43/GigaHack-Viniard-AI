from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from vineyard.farms.frames import farm_blocks_frame, farms_frame
from vineyard.farms.grouping import Farm
from vineyard.farms.parcels import (
    CADASTRE_HIGHWAY,
    ParcelIndex,
    ParcelParams,
    cadastre_roads,
    mark_cadastral,
    normalize_landuse,
    road_parcels,
    stats_by_key,
)
from vineyard.farms.roads import Road, RoadClass, RoadOrigin
from vineyard.geo.tiling import CRS_EPSG
from vineyard.geo.vector_io import write_layer

X0, Y0 = 629000.0, 5220000.0
ROAD = "Cale de comunicaţie"  # cedilla, as the AGCC writes it
P = ParcelParams(min_overlap_m2=20.0, min_overlap_frac=0.5, road_landuses=frozenset({normalize_landuse("Cale de comunicație")}),
                 road_cover_frac=0.5, road_buffer_m=2.0, road_min_len_m=25.0, road_res_m=0.5, road_max_width_m=20.0,
                 road_osm_gap_m=15.0)
PROV = {"source": "model", "run_id": "20260926T1540-post-rc7", "model_version": "pipe@0.1.0", "confidence": 1.0,
        "qa_flags": ""}


def parcels(*rows: tuple[str, str, object]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"codcadastral": [r[0] for r in rows], "landuse": [r[1] for r in rows]},
                            geometry=[r[2] for r in rows], crs=CRS_EPSG)


def rect(x0: float, y0: float, x1: float, y1: float):
    return box(X0 + x0, Y0 + y0, X0 + x1, Y0 + y1)


def test_normalize_landuse_folds_cedillas_case_and_spaces() -> None:
    assert normalize_landuse("Cale de  comunicaţie") == normalize_landuse("cale de comunicație") == "cale de comunicație"
    assert normalize_landuse(None) == ""


def test_parcel_stats_use_the_overlap_thresholds() -> None:
    frame = parcels(("A", "Teren pentru grădini", rect(0, 0, 50, 50)),       # 50 x 10 = 500 m² overlap
                    ("B", "Teren pentru grădini", rect(50, 0, 60, 3)),       # 30 m², 100 % of it
                    ("C", "Pentru construcţii", rect(-100, 0, 0.5, 20)),     # 10 m² of 2010: sliver
                    ("D", "Teren pentru obținerea producției agricole", rect(0, 9, 100, 200)))  # 1 m high: 100 m²
    block = rect(0, 0, 100, 10)
    stats = stats_by_key({"V01": block}, ParcelIndex(frame), P)["V01"]
    assert (stats.n_parcels, stats.codes) == (3, ("A", "B", "D"))
    assert stats.landuse_counts == (("Teren pentru grădini", 2), ("Teren pentru obținerea producției agricole", 1))
    assert json.loads(stats.landuse_json) == {"Teren pentru grădini": 2,
                                              "Teren pentru obținerea producției agricole": 1}
    assert stats.codes_text == "A,B,D"


def test_road_parcels_and_cadastral_flag() -> None:
    frame = parcels(("R", ROAD, rect(0, -3, 200, 3)), ("G", "Teren pentru grădini", rect(0, 10, 200, 50)))
    assert list(road_parcels(frame, P).codcadastral) == ["R"]
    on = Road(RoadClass.PUBLIC, "residential", None, None, None, RoadOrigin.OSM,
              LineString([(X0, Y0 + 1), (X0 + 100, Y0 + 1)]))
    off = Road(RoadClass.FIELD, "track", None, None, None, RoadOrigin.OSM,
               LineString([(X0, Y0 + 30), (X0 + 100, Y0 + 30)]))
    marked = mark_cadastral([on, off], road_parcels(frame, P).geometry.iloc[0], P)
    assert [r.cadastral for r in marked] == [True, False]
    assert [r.cadastral for r in mark_cadastral([on], None, P)] == [False]


def test_a_road_parcel_without_osm_road_gives_a_public_centreline() -> None:
    frame = parcels(("R", ROAD, rect(0, -3, 200, 3)))
    (road,) = cadastre_roads(frame, None, P)
    assert (road.road_class, road.highway, road.origin, road.cadastral) == (
        RoadClass.PUBLIC, CADASTRE_HIGHWAY, RoadOrigin.CADASTRE, True)
    assert road.length_m == pytest.approx(200, abs=12)
    assert road.geometry.distance(LineString([(X0, Y0), (X0 + 200, Y0)])) < 1.0


@pytest.mark.parametrize("osm_y", [0.0, 10.0])
def test_road_parcels_an_osm_road_follows_give_nothing(osm_y: float) -> None:
    # osm_y = 0: the OSM road runs inside the parcel; 10: alongside it (the cadastre is offset), same road
    frame = parcels(("R", ROAD, rect(0, -3, 200, 3)))
    osm = LineString([(X0 - 10, Y0 + osm_y), (X0 + 210, Y0 + osm_y)])
    assert cadastre_roads(frame, osm, P) == ()


def test_wide_or_short_road_parcels_give_nothing() -> None:
    assert cadastre_roads(parcels(("W", ROAD, rect(0, 0, 400, 60))), None, P) == ()
    assert cadastre_roads(parcels(("S", ROAD, rect(0, 0, 20, 6))), None, P) == ()
    assert cadastre_roads(parcels(("G", "Teren pentru grădini", rect(0, -3, 200, 3))), None, P) == ()


def test_frames_with_and_without_parcel_stats(tmp_path: Path) -> None:
    farm = Farm("F01", ("V01", "V02"), rect(0, 0, 100, 10))
    blocks = {"V01": rect(0, 0, 40, 10), "V02": rect(60, 0, 100, 10), "V99": rect(500, 0, 510, 10)}
    index = ParcelIndex(parcels(("A", "Teren pentru grădini", rect(0, 0, 50, 50))))
    fstats, bstats = stats_by_key({"F01": farm.outline}, index, P), stats_by_key(blocks, index, P)
    ff = farms_frame([farm], PROV, fstats)
    assert (int(ff.n_parcels.iloc[0]), ff.cadastral_codes.iloc[0]) == (1, "A")
    fb = farm_blocks_frame(blocks, [farm], PROV, bstats)
    assert list(fb.vineyard_id) == ["V01", "V02"] and list(fb.n_parcels) == [1, 0]
    write_layer(ff, "farms", tmp_path / "farms.parquet")
    write_layer(fb, "farm_blocks", tmp_path / "farm_blocks.parquet")
    bare = farm_blocks_frame(blocks, [farm], PROV)
    assert bare.n_parcels.isna().all() and bare.cadastral_codes.isna().all()
    write_layer(bare, "farm_blocks", tmp_path / "bare.parquet")
