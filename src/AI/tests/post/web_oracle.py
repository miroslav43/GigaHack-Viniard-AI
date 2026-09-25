"""Independent oracle for the web data bundle (src/Web/CLAUDE.md §6.2-6.4); never imports the writer."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import pytest
from shapely.geometry import shape

CRS_MEMBER: Final = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32635"}}
CSV_HEADER: Final = ("level,vineyard_id,row_id,block_count,row_count,row_length_m,canopy_area_m2,canopy_area_ha,"
                     "interrow_area_m2,interrow_area_ha,plant_count,row_structure")
UTM_X: Final = (600_000.0, 700_000.0)
UTM_Y: Final = (5_200_000.0, 5_250_000.0)
MIN_DECIMALS: Final = 3
# §6.3: file -> (geometry types, required properties, properties that are unique non-empty string ids)
SPEC: Final = {
    "blocks.geojson": ({"Polygon", "MultiPolygon"}, ("vineyard_id", "area_m2"), ("vineyard_id",)),
    "rows.geojson": ({"LineString", "MultiLineString"},
                     ("row_id", "vineyard_id", "row_structure", "length_m", "plant_count", "max_gap_m",
                      "tile_structures"), ("row_id",)),
    "canopies.geojsonl": ({"Polygon"}, ("canopy_id", "vineyard_id", "tile", "row_id"), ("canopy_id",)),
    "interrows.geojson": ({"Polygon"}, ("interrow_id", "vineyard_id", "interrow_cover", "tile", "row_ids",
                                        "area_m2"), ("piece_id",)),
    "waste.geojson": ({"Polygon"}, ("waste_id", "vineyard_id", "tile", "confidence"), ("waste_id",)),
    "targets.geojson": ({"Point"}, ("target_id", "type", "vineyard_id", "row_id", "waste_id", "route_order",
                                    "reachable", "gap_length_m", "note"), ("target_id",)),
    "route.geojson": ({"LineString"}, ("length_m", "duration_min", "baseline_length_m", "outside_share"), ()),
}
ENUMS: Final = {"row_structure": {"regular", "disrupted", "unassessable"},
                "interrow_cover": {"bare_soil", "vegetation", "mixed", "unassessable"},
                "type": {"gap", "missing", "waste"}}
MANIFEST_KEYS: Final = ("survey_id", "name", "captured_at", "gsd_m", "crs", "source", "license", "stage",
                        "generated_at", "pipeline_version", "tiles")


def _coords(c: Any) -> list[tuple[float, float]]:
    return [(c[0], c[1])] if isinstance(c[0], int | float) else [p for sub in c for p in _coords(sub)]


def features_of(path: Path) -> list[dict[str, Any]]:
    """Features of a FeatureCollection (crs member required) or of a GeoJSONSeq file."""
    if path.suffix == ".geojsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["type"] == "FeatureCollection" and doc["crs"] == CRS_MEMBER, path.name
    return doc["features"]


def _check_layer(name: str, feats: list[dict[str, Any]]) -> None:
    geoms, required, ids = SPEC[name]
    for f in feats:
        assert f["type"] == "Feature" and f["geometry"]["type"] in geoms, name
        for x, y in _coords(f["geometry"]["coordinates"]):
            assert UTM_X[0] < x < UTM_X[1] and UTM_Y[0] < y < UTM_Y[1], (name, x, y)
            assert round(x, MIN_DECIMALS) == x and round(y, MIN_DECIMALS) == y, (name, x, y)
        assert not [k for k in required if k not in f["properties"]], (name, f["properties"])
        assert all(f["properties"][k] in ENUMS[k] for k in ENUMS if k in f["properties"]), name
    for key in ids:
        values = [f["properties"][key] for f in feats]
        assert all(isinstance(v, str) and v for v in values) and len(set(values)) == len(values), (name, key)


def _check_refs(p: Mapping[str, list[dict[str, Any]]]) -> None:
    blocks = {f["vineyard_id"] for f in p["blocks.geojson"]}
    rows = {f["row_id"] for f in p["rows.geojson"]}
    waste = {f["waste_id"] for f in p["waste.geojson"]}
    assert {f["vineyard_id"] for f in p["rows.geojson"]} <= blocks
    assert all(re.fullmatch(rf"{f['tile']}#\d+", f["canopy_id"]) for f in p["canopies.geojsonl"])
    assert {f["row_id"] for f in p["canopies.geojsonl"] if f["row_id"]} <= rows
    assert all(set(f["row_ids"]) <= rows for f in p["interrows.geojson"])
    targets = p["targets.geojson"]
    assert [f["target_id"] for f in targets] == [f"T{k:03d}" for k in range(1, len(targets) + 1)]
    orders = [f["route_order"] for f in targets]
    visited = [o for o in orders if o is not None]
    assert visited == list(range(1, len(visited) + 1)) and orders[: len(visited)] == visited
    assert all(isinstance(f["reachable"], bool) and (f["route_order"] is None or f["reachable"]) for f in targets)
    assert all(f["row_id"] is None or f["row_id"] in rows for f in targets)
    assert all(f["waste_id"] is None or f["waste_id"] in waste for f in targets)


def check_bundle(out: Path) -> dict[str, list[dict[str, Any]]]:
    """Assert the §6.2/§6.3 rules on a written bundle; returns properties (+ `_geom`) per file."""
    found: dict[str, list[dict[str, Any]]] = {}
    for name in SPEC:
        feats = features_of(out / name)
        _check_layer(name, feats)
        found[name] = [f["properties"] | {"_geom": shape(f["geometry"])} for f in feats]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert not [k for k in MANIFEST_KEYS if k not in manifest] and manifest["stage"] in {"model", "marcaj_corrected"}
    assert manifest["crs"] == "EPSG:32635" and manifest["captured_at"] == "2025-05-20"
    assert (out / "measurements.csv").read_text(encoding="utf-8").splitlines()[0] == CSV_HEADER
    (route,) = found["route.geojson"]
    assert route["length_m"] == pytest.approx(route["_geom"].length, abs=0.01)
    _check_refs(found)
    return found
