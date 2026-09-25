"""Web data bundle edge cases: exact union areas, block completion, dangling ids, route export, CSV and config."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from tests.post.factories import BlockSpec, layer_frame, make_annset
from tests.post.web_oracle import CRS_MEMBER, CSV_HEADER, features_of
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.errors import ConfigError, SchemaError, StageError
from vineyard.measure.measurements import union_area
from vineyard.pipeline.context import new_run_context
from vineyard.pipeline.stages.web_bundle import find_layers_run, read_route_export
from vineyard.web.bundle import (
    WebInputs,
    block_features,
    build_web_bundle,
    decimals_of,
    measurements_bytes,
    rounded_properties,
    web_params,
    write_geojsonseq,
)
from vineyard.web.objects_export import RouteInfo, target_features
from vineyard.web.rows_export import features_frame

START: Final = (629504.70, 5220250.75)
GEN: Final = "2026-09-26T03:10:00+03:00"
X0, Y0 = 629600.0, 5220300.0


def _polys(*geoms: Any) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"tile_id": ["t"] * len(geoms)}, geometry=list(geoms), crs=32635)


def test_union_area_is_exact_with_overlaps_and_disjoint_parts() -> None:
    overlapping = (box(X0, Y0, X0 + 2, Y0 + 1), box(X0 + 1, Y0, X0 + 3, Y0 + 1))  # union 3 m²
    alone = (box(X0 + 10, Y0, X0 + 11, Y0 + 1), box(X0 + 20, Y0, X0 + 20.5, Y0 + 2))  # 1 + 1 m²
    assert union_area(_polys(*overlapping, *alone)) == pytest.approx(5.0)
    assert union_area(_polys()) == 0.0


def test_blocks_complete_derive_blocks_with_outlines() -> None:
    ann = make_annset(BlockSpec(n_rows=3), BlockSpec(vineyard_id="V02", n_rows=3, origin_xy=(629560.0, 5220270.0)))
    given = gpd.GeoDataFrame({"vineyard_id": ["V01"]}, geometry=[box(X0, Y0, X0 + 5, Y0 + 5)], crs=32635)
    blocks = block_features(WebInputs(annset=ann, blocks=given), buffer_m=2.5)
    assert list(blocks.vineyard_id) == ["V01", "V02"] and blocks.area_m2[0] == pytest.approx(25.0)
    outline = blocks.geometry.iloc[1]
    canopies = ann.canopies[ann.canopies.vineyard_id == "V02"].union_all()
    rows = ann.row_pieces[ann.row_pieces.vineyard_id == "V02"].union_all()
    # a closing covers its members up to the round-cap approximation of the buffers
    assert canopies.difference(outline).area < 1e-3 * canopies.area
    assert rows.intersection(outline).length > 0.99 * rows.length


def test_blocks_from_canopies_when_a_block_has_no_rows() -> None:
    ann = make_annset(BlockSpec(n_rows=2))
    only_canopies = ann.with_layer("row_pieces", ann.row_pieces.iloc[0:0]).with_layer(
        "interrow_pieces", ann.interrow_pieces.iloc[0:0])
    (area,) = block_features(WebInputs(annset=only_canopies), buffer_m=2.5).area_m2
    assert area > ann.canopies.area.sum()


def _targets() -> gpd.GeoDataFrame:
    rec = {"target_id": "T-GAP-0001", "kind": "row_gap", "vineyard_id": "V01", "tile_id": "siret3_r018_c011",
           "row_id": "V09-R001", "interrow_id": None, "waste_id": "W0042", "x": X0, "y": Y0, "gap_length_m": 5.4,
           "priority": 1, "reachable": False, "reach_note": "", "snap_dist_m": 0.0, "reason": "interior gap",
           "geometry": Point(X0, Y0)}
    return layer_frame("targets", [rec])


def test_targets_null_dangling_ids_and_keep_them_in_the_note() -> None:
    t = target_features(_targets(), route=None, visit_radius_m=2.0, known_rows={"V01-R001"}, known_waste=set())
    assert (t.row_id[0], t.waste_id[0], t.route_order[0], t.reachable[0]) == (None, None, None, False)
    assert t.note[0] == "row_gap; interior gap; row=V09-R001; waste=W0042"
    assert t.gap_length_m[0] == pytest.approx(5.4) and t.tile[0] == "siret3_r018_c011"
    empty = target_features(None, route=None, visit_radius_m=2.0)
    assert empty.empty and "route_order" in empty.columns


def test_route_info_rejects_bad_numbers() -> None:
    line = LineString([START, (START[0] + 5, START[1])])
    assert RouteInfo(line, baseline_length_m=10.0, outside_share=0.0).outside_share == 0.0
    with pytest.raises(SchemaError):
        RouteInfo(line, baseline_length_m=-1.0)
    with pytest.raises(SchemaError):
        RouteInfo(LineString())


def _write(path: Path, doc: Any) -> Path:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_read_route_export_accepts_the_route_stage_file(tmp_path: Path) -> None:
    geometry = {"type": "LineString", "coordinates": [list(START), [START[0] + 3, START[1]], list(START)]}
    collection = {"type": "FeatureCollection", "crs": CRS_MEMBER, "features": [
        {"type": "Feature", "properties": {"length_m": 6.0, "outside_share": 0.001}, "geometry": geometry}]}
    info = read_route_export(_write(tmp_path / "fc.geojson", collection))
    assert info is not None and info.line.length == pytest.approx(6.0)
    assert (info.baseline_length_m, info.outside_share) == (None, 0.001)
    bare = {"type": "Feature", "crs": CRS_MEMBER, "properties": {}, "geometry": geometry}
    assert read_route_export(_write(tmp_path / "bare.geojson", bare)) is not None
    point = collection | {"features": [{"type": "Feature", "properties": {},
                                        "geometry": {"type": "Point", "coordinates": list(START)}}]}
    with pytest.raises(StageError):
        read_route_export(_write(tmp_path / "point.geojson", point))


def test_find_layers_run_rejects_a_corrupt_run_json(tmp_path: Path) -> None:
    cfg = load_config(environ={"VINEYARD_WORK_DIR": str(tmp_path / "work")})
    other = cfg.paths.work_dir / "runs" / "20260926T0100-post-aaaaaa"
    other.mkdir(parents=True)
    (other / "run.json").write_text("{", encoding="utf-8")
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", annset_ref="LATEST_MARCAJ",
                          run_id="20260926T0200-post-bbbbbb")
    with pytest.raises(StageError):
        find_layers_run(ctx)


CONTRACT_EXAMPLE: Final = (f"{CSV_HEADER}\nsurvey,,,1,1,1.10,,,,,3,\nblock,V01,,,1,1.10,,,,,3,\n"
                           "row,V01,V01-R01,,,1.10,,,,,3,regular\n")


@pytest.mark.parametrize("data", [b"\xff\xfe", f"{CSV_HEADER}\nsurvey,,\n".encode(),
                                  f"{CSV_HEADER}\ntotal,,,1,1,1.00,1.00,0.0001,1.00,0.0001,1,\n".encode(), b"",
                                  CONTRACT_EXAMPLE.replace("block,V01,,,1,1.10", "block,V01,,,1,9.10").encode()])
def test_bundle_refuses_a_bad_measure_csv(data: bytes) -> None:
    params = web_params(load_config(environ={}))
    with pytest.raises(SchemaError):
        measurements_bytes(WebInputs(annset=make_annset(BlockSpec(n_rows=1)), measurements_csv=data), params)


def test_bundle_copies_a_consistent_measure_csv_verbatim() -> None:
    params = web_params(load_config(environ={}))
    data = CONTRACT_EXAMPLE.encode()
    assert measurements_bytes(WebInputs(annset=make_annset(BlockSpec(n_rows=1)), measurements_csv=data),
                              params) is data


@pytest.mark.parametrize(("step", "expected"), [(0.01, 2), (0.0001, 4), (1.0, 0)])
def test_decimals_of_power_of_ten_steps(step: float, expected: int) -> None:
    assert decimals_of(step, "k") == expected


@pytest.mark.parametrize("step", [0.03, 0.0, -0.1, 10.0])
def test_decimals_of_rejects_other_steps(step: float) -> None:
    with pytest.raises(ConfigError):
        decimals_of(step, "k")


def test_geojsonseq_is_compact_and_handles_empty_frames(tmp_path: Path) -> None:
    frame = features_frame([{"a": 1.5, "b": None, "c": [1, 2]}], [Point(X0 + 0.00049, Y0)], ("a", "b", "c"))
    path = write_geojsonseq(frame, tmp_path / "x.geojsonl", decimals=3)
    assert path.read_text(encoding="utf-8") == \
        '{"type":"Feature","properties":{"a":1.5,"b":null,"c":[1,2]},' \
        f'"geometry":{{"type":"Point","coordinates":[{X0},{Y0}]}}}}\n'
    empty = features_frame([], [], ("a",))
    assert write_geojsonseq(empty, tmp_path / "e.geojsonl", decimals=3).read_text(encoding="utf-8") == ""


def test_bundle_uses_derive_layers_when_given(tmp_path: Path) -> None:
    ann = make_annset(BlockSpec(n_rows=3))
    linked = make_annset(BlockSpec(n_rows=3), linked=True).interrow_pieces
    rows = pd.DataFrame({"row_id": ["V01-R002"], "max_gap_m": [6.5]})
    params = web_params(load_config(environ={}))
    result = build_web_bundle(WebInputs(annset=ann, rows=rows, interrows=linked), tmp_path, params,
                              generated_at=GEN, pipeline_version="x", run_id="r")
    assert result.counts["route"] == 0 and (tmp_path / "targets.geojson").is_file()
    interrows = features_of(tmp_path / "interrows.geojson")
    assert sorted({f["properties"]["interrow_id"] for f in interrows}) == ["V01-I001", "V01-I002"]
    by_row = {f["properties"]["row_id"]: f["properties"] for f in features_of(tmp_path / "rows.geojson")}
    assert by_row["V01-R002"]["max_gap_m"] == 6.5 and by_row["V01-R001"]["max_gap_m"] is None
    assert by_row["V01-R001"]["length_m"] == pytest.approx(60.0)
    assert math.isclose(sum(f["properties"]["area_m2"] for f in interrows),
                        float(linked.geometry.area.sum()), abs_tol=0.01)


def test_rounding_keeps_the_precision_of_fractions() -> None:
    frame = features_frame([{"length_m": 1.23456, "outside_share": 0.004671}], [Point(X0, Y0)],
                           ("length_m", "outside_share"))
    rounded = rounded_properties(frame, 3)
    assert (rounded.length_m[0], rounded.outside_share[0]) == (1.235, 0.00467)


def test_find_layers_run_matches_a_latest_alias_of_the_same_annset(tmp_path: Path) -> None:
    cfg = load_config(environ={"VINEYARD_WORK_DIR": str(tmp_path / "work")})
    runs = cfg.paths.work_dir / "runs"
    ref = "20260926T0100-marcaj-aaaaaa"
    (runs / ref / "annset").mkdir(parents=True)
    (runs / "LATEST_MARCAJ").symlink_to(ref, target_is_directory=True)
    post = runs / "20260926T0200-post-bbbbbb"
    (post / "exports").mkdir(parents=True)
    (post / "exports" / "route.geojson").write_text("{}", encoding="utf-8")
    (post / "run.json").write_text(json.dumps({"annset_ref": ref}), encoding="utf-8")
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", annset_ref="LATEST_MARCAJ",
                          run_id="20260926T0300-post-cccccc")
    assert find_layers_run(ctx) == post
