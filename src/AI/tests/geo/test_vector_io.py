"""GeoParquet layers (schema-checked) and deterministic GeoJSON in EPSG:32635 / EPSG:4326."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from vineyard.errors import SchemaError
from vineyard.geo.vector_io import (
    CRS_URN_32635,
    read_geojson,
    read_layer,
    round_geometry,
    write_geojson,
    write_geojson_4326,
    write_layer,
)


def _signed_area(ring: list[list[float]]) -> float:
    xy = np.asarray(ring, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.rglob("*") if ".tmp-" in p.name)


@pytest.fixture
def points_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "fid": ["b", "a"],
            "n": np.array([1, 2], dtype=np.int16),
            "x": [0.123456, float("nan")],
            "flag": [True, False],
        },
        geometry=[Point(629504.70049, 5220250.75051), Point(629500.0, 5220200.0)],
        crs="EPSG:32635",
    )


# ---------------------------------------------------------------- GeoParquet


def _tile_valid_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"tile_id": ["siret3_r006_c004"], "valid_frac": np.array([1.0], dtype=np.float64)},
        geometry=[box(629196.8, 5220864.0, 629248.0, 5220915.2)],
        crs="EPSG:32635",
    )


def test_write_read_layer_roundtrip(tmp_path: Path) -> None:
    src = _tile_valid_gdf()
    out = write_layer(src, "tile_valid", tmp_path / "layers" / "tile_valid.parquet")
    assert out.exists()
    assert _leftovers(tmp_path) == []
    back = read_layer(out, "tile_valid")
    assert back.crs.to_epsg() == 32635
    assert back["valid_frac"].dtype == np.float32  # coerced to the contract dtype
    assert back.geometry.iloc[0].equals(src.geometry.iloc[0])
    assert src["valid_frac"].dtype == np.float64  # input frame untouched


def test_write_layer_rejects_wrong_crs(tmp_path: Path) -> None:
    bad = _tile_valid_gdf().to_crs(4326)
    with pytest.raises(SchemaError):
        write_layer(bad, "tile_valid", tmp_path / "bad.parquet")
    assert not (tmp_path / "bad.parquet").exists()
    assert _leftovers(tmp_path) == []


def test_read_layer_without_name_skips_validation(tmp_path: Path) -> None:
    path = tmp_path / "raw.parquet"
    _tile_valid_gdf().to_crs(4326).to_parquet(path)
    assert read_layer(path).crs.to_epsg() == 4326
    with pytest.raises(SchemaError):
        read_layer(path, "tile_valid")
    assert len(read_layer(path, "tile_valid", validate=False)) == 1


def test_read_layer_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        read_layer(tmp_path / "missing.parquet", "tile_valid")


# ---------------------------------------------------------------- GeoJSON 32635


def test_write_geojson_crs_member_and_decimals(tmp_path: Path, points_gdf: gpd.GeoDataFrame) -> None:
    path = write_geojson(points_gdf, tmp_path / "pts.geojson", id_column="fid")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["type"] == "FeatureCollection"
    assert doc["crs"] == {"type": "name", "properties": {"name": CRS_URN_32635}}
    f0 = doc["features"][0]
    assert f0["id"] == "b"
    assert f0["geometry"] == {"type": "Point", "coordinates": [629504.7, 5220250.751]}
    assert f0["properties"] == {"fid": "b", "n": 1, "x": 0.123456, "flag": True}
    assert doc["features"][1]["properties"]["x"] is None  # NaN -> null


def test_write_geojson_route_style_two_decimals(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"name": ["route"]},
        geometry=[LineString([(629504.7049, 5220250.7549), (629510.0, 5220260.0)])],
        crs="EPSG:32635",
    )
    doc = json.loads(write_geojson(gdf, tmp_path / "route.geojson", decimals=2).read_text())
    assert doc["features"][0]["geometry"]["coordinates"][0] == [629504.7, 5220250.75]
    assert "id" not in doc["features"][0]


def test_write_geojson_deterministic_bytes(tmp_path: Path, points_gdf: gpd.GeoDataFrame) -> None:
    a = write_geojson(points_gdf, tmp_path / "a" / "x.geojson")
    b = write_geojson(points_gdf.copy(), tmp_path / "b" / "x.geojson")
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes().endswith(b"\n")
    assert _leftovers(tmp_path) == []


def test_write_geojson_rejects_other_crs(tmp_path: Path, points_gdf: gpd.GeoDataFrame) -> None:
    with pytest.raises(SchemaError, match="32635"):
        write_geojson(points_gdf.to_crs(4326), tmp_path / "x.geojson")
    with pytest.raises(SchemaError, match="id_column"):
        write_geojson(points_gdf, tmp_path / "y.geojson", id_column="nope")


def test_write_geojson_property_types(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "arr": [np.array([1, 2]), np.array([3])],
            "na": pd.array([1, None], dtype="Int16"),
            "ts": [pd.Timestamp("2026-09-26T03:10:00+03:00"), pd.NaT],
            "s": pd.array(["a", None], dtype="string"),
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:32635",
    )
    props = [f["properties"] for f in json.loads(write_geojson(gdf, tmp_path / "p.geojson").read_text())["features"]]
    assert props[0] == {"arr": [1, 2], "na": 1, "ts": "2026-09-26T03:10:00+03:00", "s": "a"}
    assert props[1] == {"arr": [3], "na": None, "ts": None, "s": None}


def test_write_geojson_4326_requires_crs(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="no CRS"):
        write_geojson_4326(gpd.GeoDataFrame({"k": [1]}, geometry=[Point(0, 0)]), tmp_path / "x.geojson")


def test_write_geojson_null_geometry(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame({"k": [1, 2]}, geometry=[None, Polygon()], crs="EPSG:32635")
    doc = json.loads(write_geojson(gdf, tmp_path / "n.geojson").read_text())
    assert [f["geometry"] for f in doc["features"]] == [None, None]


# ---------------------------------------------------------------- GeoJSON 4326


def test_write_geojson_4326_ccw_no_crs_7_decimals(tmp_path: Path) -> None:
    cw_shell = [(629200.0, 5220900.0), (629240.0, 5220900.0), (629240.0, 5220870.0), (629200.0, 5220870.0)]
    ccw_hole = [(629210.0, 5220880.0), (629220.0, 5220880.0), (629220.0, 5220890.0), (629210.0, 5220890.0)]
    poly = Polygon(cw_shell, [ccw_hole])
    multi = MultiPolygon([poly, box(629300.0, 5220800.0, 629310.0, 5220810.0, ccw=False)])
    gdf = gpd.GeoDataFrame({"k": ["p", "m"]}, geometry=[poly, multi], crs="EPSG:32635")
    path = write_geojson_4326(gdf, tmp_path / "web.geojson", id_column="k")
    doc = json.loads(path.read_text())
    assert "crs" not in doc
    rings = doc["features"][0]["geometry"]["coordinates"]
    assert _signed_area(rings[0]) > 0  # exterior CCW (RFC 7946)
    assert _signed_area(rings[1]) < 0  # hole CW
    for part in doc["features"][1]["geometry"]["coordinates"]:
        assert _signed_area(part[0]) > 0
    lon, lat = rings[0][0]
    assert 28.0 < lon < 29.5 and 46.5 < lat < 47.5
    assert all(len(repr(v).split(".")[1]) <= 7 for v in (lon, lat))
    assert gdf.crs.to_epsg() == 32635  # input untouched


def test_start_point_4326_matches_organizer_lonlat(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame({"name": ["START"]}, geometry=[Point(629504.70, 5220250.75)], crs="EPSG:32635")
    doc = json.loads(write_geojson_4326(gdf, tmp_path / "s.geojson").read_text())
    lon, lat = doc["features"][0]["geometry"]["coordinates"]
    assert lon == pytest.approx(28.7073776, abs=2e-7)
    assert lat == pytest.approx(47.1230335, abs=2e-7)


# ---------------------------------------------------------------- read_geojson


def test_read_geojson_roundtrip(tmp_path: Path, points_gdf: gpd.GeoDataFrame) -> None:
    path = write_geojson(points_gdf, tmp_path / "rt.geojson")
    back = read_geojson(path)
    assert back.crs.to_epsg() == 32635
    assert list(back["fid"]) == ["b", "a"]
    assert back.geometry.iloc[0].equals(Point(629504.7, 5220250.751))


def test_read_geojson_epsg_mismatch(tmp_path: Path, points_gdf: gpd.GeoDataFrame) -> None:
    path = write_geojson_4326(points_gdf, tmp_path / "w.geojson")
    with pytest.raises(SchemaError, match="4326"):
        read_geojson(path)
    assert read_geojson(path, expect_epsg=4326).crs.to_epsg() == 4326


def test_read_geojson_empty_and_invalid(tmp_path: Path) -> None:
    empty = tmp_path / "e.geojson"
    empty.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name",
                                 "properties": {"name": CRS_URN_32635}}, "features": []}))
    gdf = read_geojson(empty)
    assert len(gdf) == 0 and gdf.crs.to_epsg() == 32635
    bad = tmp_path / "bad.geojson"
    bad.write_text(json.dumps({"type": "Feature"}))
    with pytest.raises(SchemaError, match="FeatureCollection"):
        read_geojson(bad)
    weird = tmp_path / "weird.geojson"
    weird.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name",
                                 "properties": {"name": "not-a-crs"}}, "features": []}))
    with pytest.raises(SchemaError, match="not-a-crs"):
        read_geojson(weird)


@pytest.mark.examples
def test_read_organizer_route_files(data_root: Path) -> None:
    route = data_root / "02_route"
    start = read_geojson(route / "start.geojson")
    assert len(start) == 1
    pt = start.geometry.iloc[0]
    assert (pt.x, pt.y) == (629504.7, 5220250.75)
    for name in ("passages", "forbidden", "study_area"):
        gdf = read_geojson(route / f"{name}.geojson")
        assert len(gdf) == 1, name
        assert gdf.crs.to_epsg() == 32635
    study = read_geojson(route / "study_area.geojson")
    assert study.geometry.iloc[0].area == pytest.approx(311 * 2621.44, rel=1e-9)


# ---------------------------------------------------------------- round_geometry


def test_round_geometry_no_negative_zero() -> None:
    g = round_geometry(LineString([(-0.0004, 1.23456), (2.0, -0.00049)]), 3)
    coords = list(g.coords)
    assert coords == [(0.0, 1.235), (2.0, 0.0)]
    assert all(str(v) != "-0.0" for c in coords for v in c)
