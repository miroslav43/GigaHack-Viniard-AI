"""Web data bundle (src/Web/CLAUDE.md §6.1-6.4): unit rules, the reference bundle against the §6.3 oracle, the stage."""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from tests.post.factories import (
    BlockSpec,
    Prov,
    build_annset,
    canopy_record,
    interrow_piece_record,
    layer_frame,
    make_annset,
    reference_annset,
    row_piece_record,
    waste_record,
)
from tests.post.web_oracle import CRS_MEMBER, CSV_HEADER, check_bundle
from vineyard.annset.io import write_annset
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.errors import SchemaError, StageError
from vineyard.geo.vector_io import read_geojson, write_layer
from vineyard.pipeline.context import ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages.web_bundle import find_layers_run, read_route_export
from vineyard.web.bundle import WebInputs, build_web_bundle, bundle_dir, measurements_bytes, web_params
from vineyard.web.manifest import build_manifest, generated_at_from_run_id, manifest_stage, survey_info
from vineyard.web.objects_export import (
    RouteInfo,
    canopy_features,
    canopy_web_ids,
    interrow_features,
    route_features,
    target_features,
    waste_features,
)
from vineyard.web.rows_export import aggregate_structure, physical_rows

GEN: Final = "2026-09-26T03:10:00+03:00"
TZ: Final = "Europe/Chisinau"
T_V01: Final = "siret3_r021_c012"
T_V02: Final = "siret3_r006_c004"
T_SYN: Final = "siret3_r018_c011"
START: Final = (629504.70, 5220250.75)
MANIFEST_KEYS: Final = ("survey_id", "name", "captured_at", "gsd_m", "crs", "source", "license", "stage",
                        "generated_at", "pipeline_version", "tiles")


# ------------------------------------------------------------------ builders


def _target(tid: str, kind: str, xy: tuple[float, float], *, vid: str = "V01", row: str | None = None,
            waste: str | None = None, gap: float = math.nan, reachable: bool = True) -> dict[str, Any]:
    return {"target_id": tid, "kind": kind, "vineyard_id": vid, "tile_id": T_V01, "row_id": row,
            "interrow_id": None, "waste_id": waste, "x": xy[0], "y": xy[1], "gap_length_m": gap, "priority": 1,
            "reachable": reachable, "reach_note": "" if reachable else "far", "snap_dist_m": 0.0,
            "geometry": Point(xy)}


def _frame(name: str, records: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    return layer_frame(name, records)


@pytest.fixture(scope="module")
def params() -> Any:
    return web_params(load_config(environ={}))


def _csv(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
    return {(r["level"], r["vineyard_id"], r["row_id"]): r for r in rows}


# ------------------------------------------------------------------ manifest


def test_manifest_has_every_contract_field() -> None:
    info = survey_info(load_config(environ={}))
    doc = build_manifest(info, stage="model", generated_at=GEN, pipeline_version="a" * 40, extras={"run_id": "r1"})
    assert {k: doc[k] for k in MANIFEST_KEYS} == {
        "survey_id": "siret3", "name": "Sireț3", "captured_at": "2025-05-20", "gsd_m": 0.025, "crs": "EPSG:32635",
        "source": "3DATA COLLECT / OpenAerialMap", "license": "CC BY 4.0", "stage": "model", "generated_at": GEN,
        "pipeline_version": "a" * 40, "tiles": 311}
    assert doc["run_id"] == "r1"


@pytest.mark.parametrize("kwargs", [{"stage": "mock"}, {"generated_at": "yesterday"}, {"extras": {"stage": "x"}}])
def test_manifest_rejects_bad_values(kwargs: dict[str, Any]) -> None:
    base = {"stage": "model", "generated_at": GEN, "pipeline_version": "x", "extras": None} | kwargs
    with pytest.raises(SchemaError):
        build_manifest(survey_info(load_config(environ={})), **base)


def test_survey_identity_comes_from_the_web_config() -> None:
    cfg = load_config(overrides=("web.survey_id=siret3-nn1", "web.survey_name=Sireț3 NN1"), environ={})
    info = survey_info(cfg)
    assert (info.survey_id, info.name) == ("siret3-nn1", "Sireț3 NN1")
    assert web_params(cfg).survey == info


@pytest.mark.parametrize(("survey_id", "name"), [("siret3_nn1", "Sireț3"), ("x", "Sireț3"), ("a" * 41, "Sireț3"),
                                                 ("siret3\n", "Sireț3"), ("siret3", "Y"), ("siret3", "N" * 161)])
def test_survey_info_rejects_what_the_web_cannot_register(survey_id: str, name: str) -> None:
    info = survey_info(load_config(environ={}))
    assert replace(info, survey_id="s3", name="N" * 160).survey_id == "s3"
    with pytest.raises(SchemaError):
        replace(info, survey_id=survey_id, name=name)


def test_stage_and_generated_at() -> None:
    assert [manifest_stage(s) for s in Source] == ["model", "marcaj_corrected", "marcaj_corrected"]
    with pytest.raises(SchemaError):
        manifest_stage("bogus")
    assert generated_at_from_run_id("20260926T0310-post-a1b2c3", TZ) == GEN
    assert generated_at_from_run_id("my-run", TZ) is None
    assert generated_at_from_run_id("20261399T9999-post-a1b2c3", TZ) is None


# ------------------------------------------------------------------ rows and measurements


@pytest.mark.parametrize(("values", "expected"), [
    (["regular", "disrupted"], "disrupted"), (["unassessable", "unassessable"], "unassessable"),
    (["regular", "unassessable"], "regular"), (["Regular "], "regular"), (["???"], "unassessable")])
def test_aggregate_structure(values: list[str], expected: str) -> None:
    assert aggregate_structure(values) == expected


def test_physical_rows_merge_tile_pieces() -> None:
    ann = make_annset(BlockSpec(n_rows=3, gaps=((2, 10.0, 17.0),)))
    pieces = ann.row_pieces.copy()
    pieces.loc[(pieces.row_id == "V01-R001") & (pieces.tile_id == T_SYN), "row_structure"] = "disrupted"
    derived = pd.DataFrame({"row_id": ["V01-R002"], "max_gap_m": [7.5]})
    rows = physical_rows(pieces, ann.canopies, rows=derived).set_index("row_id")
    assert list(rows.index) == ["V01-R001", "V01-R002", "V01-R003"]
    assert (ann.row_pieces.groupby("row_id").size() == 2).all()
    r1, r2 = rows.loc["V01-R001"], rows.loc["V01-R002"]
    assert r1.length_m == pytest.approx(60.0) and r1.geometry.length == pytest.approx(60.0)
    assert r1.row_structure == "disrupted" and r1.tile_structures == {T_SYN: "disrupted", "siret3_r018_c012": "regular"}
    assert r1.plant_count == int((ann.canopies.row_id == "V01-R001").sum()) > r2.plant_count
    assert r2.max_gap_m == 7.5 and r1.max_gap_m is None


def _tiny_annset() -> Any:
    x0, y0 = 629610.0, 5220120.0
    return build_annset(
        canopies=[canopy_record(f"{T_V01}:C0001", T_V01, "V01", box(x0, y0, x0 + 2, y0 + 1), row_id="V01-R001"),
                  canopy_record(f"{T_V01}:C0002", T_V01, "V01", box(x0 + 1, y0, x0 + 3, y0 + 1), row_id="V01-R001")],
        row_pieces=[row_piece_record("V01-R001", "V01", T_V01, LineString([(x0, y0 + 0.5), (x0 + 10, y0 + 0.5)]))],
        interrow_pieces=[interrow_piece_record(f"{T_V01}:I001", T_V01, "V01", box(x0, y0 - 2, x0 + 10, y0 - 1))])


def test_measurements_use_union_areas_and_exact_format(params: Any) -> None:
    text = measurements_bytes(WebInputs(annset=_tiny_annset()), params).decode("utf-8")
    assert text == (f"{CSV_HEADER}\n"
                    "survey,,,1,1,10.00,3.00,0.0003,10.00,0.0010,2,\n"
                    "block,V01,,,1,10.00,3.00,0.0003,10.00,0.0010,2,\n"
                    "row,V01,V01-R001,,,10.00,,,,,2,regular\n")


# ------------------------------------------------------------------ object layers


def test_canopy_web_ids_keep_numbers_or_renumber() -> None:
    assert canopy_web_ids([f"{T_V01}:C0003", f"{T_V01}:C0001"], [T_V01, T_V01]) == [f"{T_V01}#0003", f"{T_V01}#0001"]
    assert canopy_web_ids(["b", "a", f"{T_V02}:C0007"], [T_V01, T_V01, T_V02]) == \
        [f"{T_V01}#0002", f"{T_V01}#0001", f"{T_V02}#0007"]


def test_canopy_area_from_the_layer_with_geometry_fallback() -> None:
    canopies = make_annset(BlockSpec(n_rows=2)).canopies
    n = len(canopies)
    given = canopies.assign(area_m2=[1.25, math.nan] + [None] * (n - 2))
    web_ids = canopy_web_ids([str(c) for c in canopies.canopy_id], [str(t) for t in canopies.tile_id])
    expected = dict(zip(web_ids, [1.25, *canopies.geometry.area.iloc[1:]], strict=True))
    f = canopy_features(given)
    assert dict(zip(f.canopy_id, f.area_m2, strict=True)) == pytest.approx(expected)
    assert all(type(a) is float for a in f.area_m2)
    missing = canopy_features(canopies.drop(columns="area_m2"))
    assert list(missing.area_m2) == pytest.approx(list(missing.geometry.area))


def test_interrow_total_is_the_union_area_of_the_interrow_pieces() -> None:
    linked = make_annset(BlockSpec(n_rows=3), linked=True)
    f = interrow_features(linked.interrow_pieces, linked.row_pieces, link_tol_m=1.0)
    for iid, pieces in f.groupby("interrow_id"):
        assert len(pieces) == 2, iid
        assert list(pieces.interrow_total_m2) == pytest.approx([pieces.geometry.union_all().area] * 2)
    x0, y0 = 629610.0, 5220120.0
    overlapping = layer_frame("interrow_pieces", [
        interrow_piece_record(f"{T_V01}:I001", T_V01, "V01", box(x0, y0, x0 + 10, y0 + 2), interrow_id="V01-I001"),
        interrow_piece_record(f"{T_V01}:I002", T_V01, "V01", box(x0 + 5, y0, x0 + 15, y0 + 2), interrow_id="V01-I001"),
        interrow_piece_record(f"{T_V01}:I003", T_V01, "V01", box(x0, y0 + 5, x0 + 4, y0 + 6))])
    g = interrow_features(overlapping, linked.row_pieces, link_tol_m=1.0)
    assert list(g.piece_id) == [f"{T_V01}:I001", f"{T_V01}:I002", f"{T_V01}:I003"]
    assert list(g.area_m2) == pytest.approx([20.0, 20.0, 4.0])
    assert list(g.interrow_total_m2) == pytest.approx([30.0, 30.0, 4.0])


def test_interrows_linked_ids_or_piece_ids_with_geometric_row_links() -> None:
    linked = make_annset(BlockSpec(n_rows=3), linked=True)
    f = interrow_features(linked.interrow_pieces, linked.row_pieces, link_tol_m=1.0)
    assert sorted(set(f.interrow_id)) == ["V01-I001", "V01-I002"] and len(f) == 4
    assert all(ids == ["V01-R001", "V01-R002"] for i, ids in zip(f.interrow_id, f.row_ids, strict=True) if i == "V01-I001")
    plain = make_annset(BlockSpec(n_rows=3))
    g = interrow_features(plain.interrow_pieces, plain.row_pieces, link_tol_m=1.0)
    assert list(g.interrow_id) == list(g.piece_id)
    assert sorted(map(tuple, g.row_ids)) == [("V01-R001", "V01-R002")] * 2 + [("V01-R002", "V01-R003")] * 2
    assert list(g.tile) == sorted(g.tile)


def test_waste_block_rule_and_confidence() -> None:
    frame = layer_frame("waste", [waste_record("W0001", T_V01, (629630.0, 5220120.0), 0.5, vineyard_id="V01"),
                                  waste_record("W0002", T_V01, (629640.0, 5220120.0), 0.5, vineyard_id=""),
                                  waste_record("W0003", T_V01, (629650.0, 5220120.0), 0.5, vineyard_id="V01")])
    frame = frame.assign(dist_block_m=[0.0, 30.0, 12.0], confidence=[1.5, math.nan, 0.25])
    w = waste_features(frame, max_block_dist_m=10.0)
    assert list(w.waste_id) == ["W0001", "W0002", "W0003"]
    assert list(w.vineyard_id) == ["V01", None, None] and list(w.confidence) == [1.0, None, 0.25]


def _route() -> RouteInfo:
    line = LineString([START, (629520.0, START[1]), (629520.0, 5220270.0), START])
    return RouteInfo(line, baseline_length_m=120.0, outside_share=0.001)


def _four_targets() -> gpd.GeoDataFrame:
    return _frame("targets", [
        _target("T-GAP-0001", "row_gap", (629520.0, 5220270.0), row="V01-R001", gap=6.25),
        _target("T-WST-0001", "waste", (629520.0, START[1] + 1.5), waste="W0001", vid=""),
        _target("T-MSP-0001", "missing_plant", (629600.0, 5220400.0), reachable=False),
        _target("T-END-0001", "row_end_short", (629700.0, 5220400.0))])


def test_targets_route_order_types_and_reachability() -> None:
    t = target_features(_four_targets(), route=_route(), visit_radius_m=2.0)
    assert list(t.source_target_id) == ["T-WST-0001", "T-GAP-0001", "T-END-0001", "T-MSP-0001"]
    assert list(t.target_id) == ["T001", "T002", "T003", "T004"]
    assert list(t["type"]) == ["waste", "gap", "gap", "missing"] and list(t.route_order) == [1, 2, None, None]
    assert list(t.reachable) == [True, True, True, False] and list(t.vineyard_id) == [None, "V01", "V01", "V01"]
    assert t.gap_length_m[1] == 6.25 and t.gap_length_m[0] is None and t.note[3] == "missing_plant; far"
    assert list(t.skip_reason) == [None, None, "not_covered", "far"]
    assert list(t.priority) == [1, 1, 1, 1] and all(type(p) is int for p in t.priority)
    assert list(t.route_role) == [None] * 4  # neither the targets layer nor the route stage gave one


_NOTES: Final = ("", "disconnected", "too_far", "optional_detour", "truncated", "", "outside_budget")


def _noted_targets() -> gpd.GeoDataFrame:
    """T-GAP-0001..0007 far from the stub route; 0001 is the only one the route covers."""
    records = [_target(f"T-GAP-000{k}", "row_gap", (629600.0 + 10.0 * k, 5220400.0)) for k in range(1, 8)]
    return _frame("targets", records).assign(priority=[1, 2, 3, 1, 2, 3, 2],
                                             route_role=["must", "must", "optional", "optional", "optional",
                                                         "optional", "must"])


def _noted_visits(covered: tuple[bool, ...] = (True,) + (False,) * 6) -> pd.DataFrame:
    return pd.DataFrame({"target_id": [f"T-GAP-000{k}" for k in range(1, 8)], "covered": list(covered),
                         "reachable_final": [c or n == "" for c, n in zip(covered, _NOTES, strict=True)],
                         "reach_note": list(_NOTES),
                         "route_role": ["must", "optional", "optional", "optional", "optional", "optional", "must"]})


def test_targets_reachable_only_when_walkable_and_skip_reason_when_unvisited() -> None:
    t = target_features(_noted_targets(), route=_route(), visits=_noted_visits(), visit_radius_m=2.0)
    assert list(t.source_target_id) == [f"T-GAP-000{k}" for k in range(1, 8)]
    assert list(t.route_order) == [1] + [None] * 6
    assert list(t.reachable) == [True, False, False, True, True, True, True]
    assert list(t.skip_reason) == [None, "disconnected", "too_far", "optional_detour", "truncated", "not_covered",
                                   "outside_budget"]
    assert list(t.priority) == [1, 2, 3, 1, 2, 3, 2] and all(type(p) is int for p in t.priority)
    # the route stage's role wins over the targets layer's (T-GAP-0002 is must in targets, optional in visits)
    assert list(t.route_role) == ["must", "optional", "optional", "optional", "optional", "optional", "must"]
    assert t.note[1] == "row_gap; disconnected" and t.note[0] == "row_gap"


def test_targets_all_visited_have_no_skip_reason() -> None:
    covered = (True,) * 7
    t = target_features(_noted_targets(), route=_route(), visits=_noted_visits(covered), visit_radius_m=2.0)
    assert list(t.route_order) == list(range(1, 8)) and all(type(o) is int for o in t.route_order)
    assert list(t.reachable) == [True] * 7 and list(t.skip_reason) == [None] * 7


def test_targets_without_a_route_keep_the_targets_layer_verdict() -> None:
    targets = _noted_targets().assign(reachable=[True, False, True, True, True, True, False],
                                      reach_note=["", "too_far", "", "", "", "", "in_forbidden"],
                                      priority=[1, 2, 3, 1, 2, 3, 2])
    t = target_features(targets.drop(columns="route_role"), route=None, visit_radius_m=2.0)
    assert list(t.route_order) == [None] * 7
    assert list(t.reachable) == [True, False, True, True, True, True, False]
    assert list(t.skip_reason) == ["no_route", "too_far", "no_route", "no_route", "no_route", "no_route",
                                   "in_forbidden"]
    assert list(t.route_role) == [None] * 7


def test_targets_route_role_from_the_targets_layer_without_visits() -> None:
    t = target_features(_noted_targets(), route=_route(), visit_radius_m=2.0)
    assert list(t.route_role) == ["must", "must", "optional", "optional", "optional", "optional", "must"]


def test_targets_prefer_route_stops_and_visit_flags() -> None:
    stops = _frame("route_stops", [{"route_id": "R1", "seq": s, "target_id": tid, "cum_dist_m": d, "leg_m": 1.0,
                                    "geometry": Point(START)} for s, tid, d in ((1, "T-GAP-0001", 1.0),
                                                                                 (2, "T-END-0001", 3.0))])
    visits = pd.DataFrame({"target_id": ["T-END-0001"], "covered": [False], "reachable_final": [False],
                           "reach_note": ["cut off"]})
    t = target_features(_four_targets(), route=_route(), stops=stops, visits=visits, visit_radius_m=2.0)
    assert list(t.source_target_id) == ["T-GAP-0001", "T-WST-0001", "T-END-0001", "T-MSP-0001"]
    assert list(t.route_order) == [1, 2, None, None] and t.reachable[2] is False


def test_route_feature_length_on_written_coordinates() -> None:
    r = route_features(RouteInfo(LineString([START, (START[0] + 1000.0004, START[1]), START])), speed_kmh=4.0,
                       decimals=3)
    assert r.length_m[0] == pytest.approx(2000.0) and r.duration_min[0] == pytest.approx(30.0)
    assert r.speed_kmh[0] == 4.0 and type(r.speed_kmh[0]) is float
    assert r.baseline_length_m[0] is None and r.outside_share[0] is None
    with pytest.raises(SchemaError):
        RouteInfo(LineString([START, (START[0] + 1, START[1])]), outside_share=1.5)
    with pytest.raises(SchemaError):
        RouteInfo(Polygon([START, (START[0] + 1, START[1]), (START[0], START[1] + 1)]))  # type: ignore[arg-type]


# ------------------------------------------------------------------ acceptance: AnnSet(reference) + a stub route


def _reference_inputs(examples_xml: bytes, data_root: Path) -> WebInputs:
    ann = reference_annset(examples_xml)
    ann = ann.with_layer("waste", layer_frame("waste", [
        waste_record("W0001", T_V01, (629630.0, 5220120.0), 0.6, vineyard_id="V01"),
        waste_record("W0002", T_V02, (629220.0, 5220890.0), 0.4, vineyard_id="")]))
    start = read_geojson(data_root / "02_route" / "start.geojson").geometry.iloc[0]
    line_of = dict(zip(ann.row_pieces.row_id, ann.row_pieces.geometry, strict=True))
    gap_pt = line_of["V01-R05"].interpolate(0.5, normalized=True)
    msp_pt = line_of["V02-R03"].interpolate(0.5, normalized=True)
    targets = _frame("targets", [
        _target("T-GAP-0001", "row_gap", (gap_pt.x, gap_pt.y), row="V01-R05", gap=5.4),
        _target("T-WST-0001", "waste", (629630.0, 5220120.0), waste="W0001"),
        _target("T-MSP-0001", "missing_plant", (msp_pt.x, msp_pt.y), vid="V02", row="V02-R03", gap=3.1),
        _target("T-END-0001", "row_end_short", tuple(line_of["V02-R10"].coords[0]), vid="V02", row="V02-R10")])
    route = LineString([start, gap_pt, (629630.0, 5220120.0), msp_pt, start])
    return WebInputs(annset=ann, targets=targets, route=RouteInfo(route, baseline_length_m=5000.0, outside_share=0.0))


@pytest.mark.examples
def test_reference_bundle_passes_the_web_contract(tmp_path: Path, examples_xml: bytes, data_root: Path,
                                                  params: Any) -> None:
    inputs = _reference_inputs(examples_xml, data_root)
    result = build_web_bundle(inputs, tmp_path / "a", params, generated_at=GEN, pipeline_version="abc", run_id="r")
    found = check_bundle(tmp_path / "a")
    assert result.counts == {"blocks": 2, "rows": 51, "canopies": 650, "interrows": 49, "waste": 2, "targets": 4,
                             "route": 1}
    assert [(f["type"], f["route_order"]) for f in found["targets.geojson"]] == \
        [("gap", 1), ("waste", 2), ("missing", 3), ("gap", None)]
    assert sum(f["row_structure"] == "disrupted" for f in found["rows.geojson"]) == 5
    assert [f["vineyard_id"] for f in found["waste.geojson"]] == ["V01", None]
    assert all(f["area_m2"] == pytest.approx(f["_geom"].area, abs=0.01) for f in found["canopies.geojsonl"])
    assert all(f["interrow_total_m2"] >= f["area_m2"] for f in found["interrows.geojson"])
    assert [f["priority"] for f in found["targets.geojson"]] == [1, 1, 1, 1]
    assert found["route.geojson"][0]["speed_kmh"] == params.speed_kmh
    table = _csv(tmp_path / "a" / "measurements.csv")
    facts = {"V01": (25, 910.1, 237.1, 2068.0, 399), "V02": (26, 1031.5, 299.1, 1996.4, 251)}
    for vid, (n_rows, length, canopy, interrow, plants) in facts.items():
        row = table[("block", vid, "")]
        assert (int(row["row_count"]), int(row["plant_count"])) == (n_rows, plants)
        assert float(row["row_length_m"]) == pytest.approx(length, abs=0.06)
        assert float(row["canopy_area_m2"]) == pytest.approx(canopy, abs=0.06)
        assert float(row["interrow_area_m2"]) == pytest.approx(interrow, abs=0.06)
    assert table[("survey", "", "")]["row_count"] == "51"
    again = build_web_bundle(inputs, tmp_path / "b", params, generated_at=GEN, pipeline_version="abc", run_id="r")
    assert [p.name for p in again.written] == [p.name for p in result.written]
    assert all((tmp_path / "a" / p.name).read_bytes() == p.read_bytes() for p in again.written)


def test_bundle_without_route_removes_stale_files_and_checks_csv(tmp_path: Path, params: Any) -> None:
    out = bundle_dir(tmp_path, "siret3")
    out.mkdir(parents=True)
    (out / "route.geojson").write_text("{}")
    (out / "notes.txt").write_text("keep")
    result = build_web_bundle(WebInputs(annset=_tiny_annset()), out, params, generated_at=GEN,
                              pipeline_version="x", run_id="r")
    assert out == tmp_path / "surveys" / "siret3" / "pipeline" and (out / "notes.txt").is_file()
    assert [p.name for p in result.removed] == ["route.geojson"] and result.counts["route"] == 0
    with pytest.raises(SchemaError):
        build_web_bundle(WebInputs(annset=_tiny_annset(), measurements_csv=b"level,n_rows\n"), out, params,
                         generated_at=GEN, pipeline_version="x", run_id="r")


# ------------------------------------------------------------------ stage


@dataclass(frozen=True)
class _FakeStageResult:
    stage: str
    n_items: int
    n_cached: int
    n_failed: int
    failed: tuple[str, ...] = ()
    outputs: tuple[Path, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)


@pytest.fixture
def post_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    runner = types.ModuleType("vineyard.pipeline.runner")
    runner.StageResult = _FakeStageResult  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vineyard.pipeline.runner", runner)
    cfg = load_config(overrides=(f"web.out_dir={tmp_path / 'web'}",),
                      environ={"VINEYARD_WORK_DIR": str(tmp_path / "work")})
    ref = "20260926T0100-marcaj-aaaaaa"
    ann = make_annset(BlockSpec(n_rows=3), prov=Prov(Source.MARCAJ, ref, "m"),
                      waste=[(T_SYN, (629565.0, 5220295.0), 0.5, "V01")])
    write_annset(ann, cfg.paths.work_dir / "runs" / ref / "annset")
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", annset_ref=ref, run_id="20260926T0310-post-abcdef")
    ensure_run_dirs(ctx.paths)
    line = ann.row_pieces.geometry.iloc[0]
    target = _target("T-GAP-0001", "row_gap", tuple(line.coords[-1]), row="V01-R001", gap=5.2)
    write_layer(_frame("targets", [target]), "targets", ctx.paths.layers_dir / "targets.parquet")
    route = {"type": "FeatureCollection", "crs": CRS_MEMBER, "features": [{"type": "Feature", "properties": {
        "length_m": 1.0, "baseline_length_m": 99.5, "outside_share": 0.002}, "geometry": {
        "type": "LineString", "coordinates": [list(START), list(line.coords[-1]), list(START)]}}]}
    (ctx.paths.exports_dir / "route.geojson").write_text(json.dumps(route))
    return ctx


def test_stage_builds_the_bundle_from_the_post_run(post_run: Any) -> None:
    csv_bytes = (f"{CSV_HEADER}\nsurvey,,,1,3,180.00,1.00,0.0001,2.00,0.0002,3,\n"
                 "block,V01,,,3,180.00,1.00,0.0001,2.00,0.0002,3,\n"
                 + "".join(f"row,V01,V01-R00{k},,,60.00,,,,,1,regular\n" for k in (1, 2, 3))).encode()
    (post_run.paths.exports_dir / "measurements.csv").write_bytes(csv_bytes)
    spec = load_stage("web_bundle")
    result = spec.run(post_run)
    out = bundle_dir(post_run.cfg.web.out_dir, "siret3")
    assert (spec.name, spec.scope, result.stage, result.n_failed) == ("web_bundle", "global", "web_bundle", 0)
    assert (out / "measurements.csv").read_bytes() == csv_bytes and result.n_items == len(result.outputs) == 9
    found = check_bundle(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["stage"], manifest["generated_at"], manifest["pipeline_version"]) == \
        ("marcaj_corrected", GEN, post_run.git_sha)
    (route,) = found["route.geojson"]
    assert (route["baseline_length_m"], route["outside_share"]) == (99.5, 0.002)
    assert route["speed_kmh"] == post_run.cfg.route.walking_speed_kmh
    (target,) = found["targets.geojson"]
    assert type(target["route_order"]) is int and target["route_order"] == 1 and target["skip_reason"] is None
    assert manifest["model_version"] == "m" and manifest["tiles_with_objects"] == 2
    bbox = manifest["bbox_32635"]
    assert len(bbox) == 4 and all(round(v, 3) == v for v in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]
    assert (post_run.paths.metrics_dir / "web_bundle.json").is_file()


def test_stage_writes_the_configured_survey(post_run: Any, tmp_path: Path) -> None:
    cfg = load_config(overrides=(f"web.out_dir={tmp_path / 'web'}", "web.survey_id=siret3-nn1",
                                 "web.survey_name=Sireț3 NN1"),
                      environ={"VINEYARD_WORK_DIR": str(tmp_path / "work")})
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", annset_ref=post_run.annset_ref,
                          run_id=post_run.run_id)
    load_stage("web_bundle").run(ctx)
    manifest = json.loads((bundle_dir(tmp_path / "web", "siret3-nn1") / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["survey_id"], manifest["name"]) == ("siret3-nn1", "Sireț3 NN1")
    assert not bundle_dir(tmp_path / "web", "siret3").exists()


def test_stage_finds_the_post_run_of_the_same_annset(post_run: Any) -> None:
    (post_run.paths.run_dir / "run.json").write_text(json.dumps({"annset_ref": post_run.annset_ref}))
    fresh = new_run_context(post_run.cfg, source=Source.MARCAJ, kind="post", annset_ref=post_run.annset_ref,
                            run_id="20260926T0400-post-bbbbbb")
    assert find_layers_run(fresh) == post_run.paths.run_dir
    other = new_run_context(post_run.cfg, source=Source.MARCAJ, kind="post", annset_ref="LATEST_MODEL",
                            run_id="20260926T0400-post-cccccc")
    assert find_layers_run(other) == other.paths.run_dir
    with pytest.raises(StageError):
        load_stage("web_bundle").run(new_run_context(post_run.cfg, source=Source.MARCAJ, kind="post",
                                                     run_id="20260926T0400-post-dddddd"))


def test_read_route_export_rejects_bad_files(tmp_path: Path) -> None:
    assert read_route_export(tmp_path / "missing.geojson") is None
    feat = {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [START, [1, 2]]}}
    bad = {"two": {"type": "FeatureCollection", "features": [feat, feat]},
           "crs": {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                   "features": [feat]}, "json": None}
    for name, doc in bad.items():
        path = tmp_path / f"{name}.geojson"
        path.write_text("{" if doc is None else json.dumps(doc))
        with pytest.raises(StageError):
            read_route_export(path)
