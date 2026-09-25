"""ingest.route_inputs: 02_route GeoJSON -> in_* layers with sanity checks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from vineyard.errors import IngestError
from vineyard.geo.vector_io import read_layer
from vineyard.ingest.route_inputs import (
    EXPECTED_STUDY_AREA_M2,
    ROUTE_FILES,
    check_route_sanity,
    ingest_route_inputs,
    read_route_input,
)

ROUTE_SUBDIR = Path("02_route")


@pytest.fixture
def route_dir(data_root: Path, tmp_path: Path) -> Path:
    src = data_root / ROUTE_SUBDIR
    if not src.is_dir():
        pytest.skip(f"route inputs missing under {src}")
    dst = tmp_path / "02_route"
    shutil.copytree(src, dst)
    return dst


def test_expected_study_area_is_311_tiles() -> None:
    assert pytest.approx(815267.84, abs=1e-6) == EXPECTED_STUDY_AREA_M2


def test_real_route_inputs_become_valid_layers(route_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "layers"
    written, warnings = ingest_route_inputs(route_dir, out)
    assert set(written) == set(ROUTE_FILES)
    assert warnings == ()
    start = read_layer(written["in_start"], "in_start")
    assert (start.geometry.iloc[0].x, start.geometry.iloc[0].y) == (629504.7, 5220250.75)
    assert start["name"].iloc[0] == "START"
    study = read_layer(written["in_study_area"], "in_study_area")
    assert study.area.sum() == pytest.approx(815267.84, abs=1.0)
    passages = read_layer(written["in_passages"], "in_passages")
    assert list(passages["fid"]) == [1] and passages["type"].iloc[0] == "passage"


def _rewrite(path: Path, mutate) -> None:
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc))


def test_wrong_crs_member_is_an_error(route_dir: Path, tmp_path: Path) -> None:
    _rewrite(route_dir / "passages.geojson",
             lambda d: d["crs"]["properties"].update(name="urn:ogc:def:crs:EPSG::4326"))
    with pytest.raises(IngestError, match="passages"):
        ingest_route_inputs(route_dir, tmp_path / "layers")


def test_missing_file_is_an_error(route_dir: Path, tmp_path: Path) -> None:
    (route_dir / "start.geojson").unlink()
    with pytest.raises(IngestError, match="start.geojson"):
        ingest_route_inputs(route_dir, tmp_path / "layers")


def test_empty_collection_is_an_error(route_dir: Path) -> None:
    _rewrite(route_dir / "start.geojson", lambda d: d.update(features=[]))
    with pytest.raises(IngestError, match="no features"):
        read_route_input(route_dir / "start.geojson", "in_start")


def test_wrong_geometry_type_is_an_error(route_dir: Path) -> None:
    point = {"type": "Point", "coordinates": [629504.7, 5220250.75]}
    _rewrite(route_dir / "study_area.geojson", lambda d: d["features"][0].update(geometry=point))
    with pytest.raises(IngestError, match="geometry"):
        read_route_input(route_dir / "study_area.geojson", "in_study_area")


def test_invalid_polygon_is_repaired(route_dir: Path) -> None:
    bowtie = {"type": "Polygon", "coordinates": [[[629000, 5220000], [629010, 5220010], [629010, 5220000],
                                                   [629000, 5220010], [629000, 5220000]]]}
    _rewrite(route_dir / "forbidden.geojson", lambda d: d["features"][0].update(geometry=bowtie))
    gdf = read_route_input(route_dir / "forbidden.geojson", "in_forbidden")
    assert gdf.geometry.iloc[0].is_valid and gdf.geometry.iloc[0].geom_type == "MultiPolygon"


def test_start_outside_passages_is_a_warning(route_dir: Path) -> None:
    far = {"type": "Point", "coordinates": [629100.0, 5220700.0]}
    _rewrite(route_dir / "start.geojson", lambda d: d["features"][0].update(geometry=far))
    layers = {name: read_route_input(route_dir / f, name) for name, f in ROUTE_FILES.items()}
    warnings = check_route_sanity(layers)
    assert any("passages" in w for w in warnings)
    assert any("r018_c010" in w for w in warnings)


def test_study_area_mismatch_is_an_error(route_dir: Path) -> None:
    square = {"type": "Polygon", "coordinates": [[[629000, 5220000], [629100, 5220000], [629100, 5220100],
                                                   [629000, 5220100], [629000, 5220000]]]}
    _rewrite(route_dir / "study_area.geojson", lambda d: d["features"][0].update(geometry=square))
    layers = {name: read_route_input(route_dir / f, name) for name, f in ROUTE_FILES.items()}
    with pytest.raises(IngestError, match="study_area"):
        check_route_sanity(layers)
