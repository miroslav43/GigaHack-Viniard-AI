from __future__ import annotations

import json
from pathlib import Path

import pytest

from vineyard.errors import SchemaError
from vineyard.farms.osm import (
    ATTRIBUTION,
    COLUMNS,
    bbox_4326,
    highways_collection,
    overpass_query,
    read_highways,
    write_snapshot,
)

BBOX = (28.70, 47.11, 28.72, 47.13)
WAY = {"type": "way", "id": 42, "tags": {"highway": "track", "surface": "ground", "foo": "bar"},
       "geometry": [{"lat": 47.12, "lon": 28.70}, {"lat": 47.121, "lon": 28.701}]}


def test_collection_keeps_ways_with_two_points_sorted_and_only_known_tags() -> None:
    elements = [{**WAY, "id": 50}, {"type": "node", "id": 1}, {**WAY, "geometry": WAY["geometry"][:1], "id": 7}, WAY]
    doc = highways_collection(elements, BBOX, "2026-09-26T15:34:00+03:00")
    assert [f["properties"]["osm_id"] for f in doc["features"]] == [42, 50]
    assert doc["features"][0]["properties"] == {"osm_id": 42, "highway": "track", "name": None, "surface": "ground",
                                                 "tracktype": None}
    assert doc["attribution"] == ATTRIBUTION and doc["bbox_4326"] == list(BBOX)


def test_snapshot_round_trip_is_utm(tmp_path: Path) -> None:
    path = write_snapshot(highways_collection([WAY], BBOX, "t"), tmp_path / "osm.geojson")
    frame = read_highways(path)
    assert frame.crs.to_epsg() == 32635 and tuple(c for c in frame.columns if c != "geometry") == COLUMNS
    assert frame.surface.iloc[0] == "ground" and frame.name.iloc[0] is None
    assert frame.geometry.iloc[0].length == pytest.approx(135, abs=5)


def test_snapshot_without_highway_column_is_rejected(tmp_path: Path) -> None:
    doc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"osm_id": 1},
                                                       "geometry": {"type": "LineString",
                                                                    "coordinates": [[28.7, 47.1], [28.71, 47.1]]}}]}
    path = tmp_path / "bad.geojson"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SchemaError, match="lacks columns"):
        read_highways(path)


def test_bbox_is_padded_and_query_is_south_west_north_east() -> None:
    west, south, east, north = bbox_4326((629000.0, 5220000.0, 630000.0, 5221000.0), 300.0)
    assert west < 28.70 < east and south < 47.12 < north
    assert overpass_query(BBOX, 30) == '[out:json][timeout:30];way["highway"](47.11,28.7,47.13,28.72);out tags geom;'


def test_committed_snapshot_reads(project_root: Path) -> None:
    frame = read_highways(project_root / "data" / "osm" / "siret3_highways.geojson")
    assert len(frame) > 100 and {"track", "residential"} <= set(frame.highway)
