"""route.geojson writer, reader and file checker (contract §5.1, web contract §6.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import shapely
from shapely.geometry import LineString, box

from vineyard.errors import RouteValidationError
from vineyard.route.geojson import (
    CRS_URN,
    RouteFileLimits,
    check_route_file,
    finalize_route_line,
    read_route_geojson,
    write_route_geojson,
)

START = (629504.70, 5220250.75)
DOMAIN = box(START[0] - 1.0, START[1] - 2.0, START[0] + 100.0, START[1] + 2.0)
INNER = DOMAIN.buffer(-0.05, join_style="mitre")
LIMITS = RouteFileLimits(max_outside_frac=0.005, closure_max_m=0.01, length_tol_m=0.01, grid_size_m=0.001,
                         decimals=2)


def _route() -> LineString:
    x0, y0 = START
    return LineString([(x0, y0), (x0 + 10.123456, y0 + 0.004), (x0 + 50.001, y0 + 0.3333),
                       (x0 + 50.003, y0 + 0.3334), (x0 + 10.0, y0), (x0, y0)])


def _write(tmp_path: Path, line: LineString | None = None, props: dict | None = None) -> tuple[Path, LineString]:
    path = tmp_path / "route.geojson"
    written = write_route_geojson(line or _route(), props or {"route_id": "RT-x", "duration_min": 1.5},
                                  path, start_xy=START, decimals=2)
    return path, written


def test_finalize_rounds_dedupes_and_pins_start() -> None:
    line = finalize_route_line(_route(), START, 2)
    xy = shapely.get_coordinates(line)
    assert tuple(xy[0]) == START and tuple(xy[-1]) == START
    assert (xy == xy.round(2)).all()
    assert (shapely.length(shapely.linestrings(list(zip(xy[:-1], xy[1:], strict=True)))) > 0).all()
    assert len(xy) == 5  # the two vertices 2 mm apart collapse into one


def test_finalize_prepends_start_when_route_starts_elsewhere() -> None:
    line = LineString([(START[0] + 1.0, START[1]), (START[0] + 5.0, START[1])])
    xy = shapely.get_coordinates(finalize_route_line(line, START, 2))
    assert tuple(xy[0]) == START and tuple(xy[-1]) == START and len(xy) == 4


def test_finalize_rejects_degenerate() -> None:
    with pytest.raises(RouteValidationError, match="distinct"):
        finalize_route_line(LineString([START, (START[0] + 0.001, START[1])]), START, 2)


def test_written_file_matches_contract(tmp_path: Path) -> None:
    path, written = _write(tmp_path)
    doc = json.loads(path.read_text())
    assert doc["type"] == "FeatureCollection"
    assert doc["crs"] == {"type": "name", "properties": {"name": CRS_URN}}
    assert len(doc["features"]) == 1
    feature = doc["features"][0]
    coords = feature["geometry"]["coordinates"]
    assert feature["geometry"]["type"] == "LineString"
    assert coords[0] == [629504.7, 5220250.75] and coords[-1] == [629504.7, 5220250.75]
    assert all(round(c, 2) == c for pt in coords for c in pt)
    assert feature["properties"]["length_m"] == round(written.length, 2)
    assert feature["properties"]["route_id"] == "RT-x"
    assert feature["properties"]["duration_min"] == 1.5


def test_round_trip(tmp_path: Path) -> None:
    path, written = _write(tmp_path)
    line, props = read_route_geojson(path)
    assert line.equals_exact(written, 0.0)
    assert props["length_m"] == round(written.length, 2)


def test_check_route_file_accepts_valid(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)
    checks = check_route_file(path, start_xy=START, inner=INNER, limits=LIMITS)
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


def _mutate(path: Path, fn) -> Path:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))
    return path


def _failed(path: Path) -> set[str]:
    return {c.name for c in check_route_file(path, start_xy=START, inner=INNER, limits=LIMITS) if not c.ok}


def test_check_refuses_length_mismatch(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)
    _mutate(path, lambda d: d["features"][0]["properties"].__setitem__(
        "length_m", d["features"][0]["properties"]["length_m"] + 0.02))
    assert _failed(path) == {"length_m"}


def test_check_refuses_closure(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)
    _mutate(path, lambda d: d["features"][0]["geometry"]["coordinates"][-1].__setitem__(0, 629504.72))
    assert "closure" in _failed(path)


def test_check_refuses_outside(tmp_path: Path) -> None:
    x0, y0 = START
    line = LineString([START, (x0 + 50.0, y0), (x0 + 50.0, y0 + 2.55), (x0 + 50.0, y0), START])
    path, _ = _write(tmp_path, line)  # 0.6 m outside x2 on ~105 m = 1.1%
    assert "outside_frac" in _failed(path)


def test_check_refuses_multilinestring(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)

    def to_multi(d: dict) -> None:
        geom = d["features"][0]["geometry"]
        geom["type"], geom["coordinates"] = "MultiLineString", [geom["coordinates"]]

    _mutate(path, to_multi)
    assert "linestring" in _failed(path)


def test_check_refuses_extra_feature_crs_and_decimals(tmp_path: Path) -> None:
    path, _ = _write(tmp_path)

    def spoil(d: dict) -> None:
        d["features"].append(d["features"][0])
        d["crs"]["properties"]["name"] = "EPSG:4326"
        d["features"][0]["geometry"]["coordinates"][1][0] += 0.001

    _mutate(path, spoil)
    assert {"single_feature", "crs", "decimals"} <= _failed(path)


def test_check_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "route.geojson"
    path.write_text("{not json")
    assert _failed(path) == {"readable"}


def test_read_rejects_wrong_structure(tmp_path: Path) -> None:
    path = tmp_path / "bad.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    with pytest.raises(RouteValidationError, match="exactly one Feature"):
        read_route_geojson(path)
