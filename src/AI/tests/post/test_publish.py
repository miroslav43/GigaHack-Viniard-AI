"""publish: validated copy of route.geojson + measurements.csv to the repo root (contract §5, web §6.4)."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, box

from tests.post.factories import BlockSpec, Prov, layer_frame, make_annset
from tests.post.tgt_helpers import tile_valid_frame
from vineyard.annset.io import write_annset
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.errors import StageError
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.geo.vector_io import write_geojson, write_layer
from vineyard.pipeline.context import ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages import _post_io
from vineyard.pipeline.stages.publish import (
    PublishLimits,
    PublishRefused,
    PublishSources,
    evaluate_publish,
    prepared_route_bytes,
    write_published,
)
from vineyard.route.geojson import RouteFileLimits

START = (629504.70, 5220250.75)
CRS = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32635"}}
HEADER = ("level,vineyard_id,row_id,block_count,row_count,row_length_m,canopy_area_m2,canopy_area_ha,"
          "interrow_area_m2,interrow_area_ha,plant_count,row_structure")
CSV_OK = "\n".join((HEADER, "survey,,,1,1,50.00,1.00,0.0001,2.00,0.0002,3,",
                    "block,V01,,,1,50.00,1.00,0.0001,2.00,0.0002,3,", "row,V01,V01-R001,,,50.00,,,,,3,regular")) + "\n"
# The block line counts 1 m² of inter-row that the survey union has once (two blocks sharing ground).
CSV_DOUBLED = CSV_OK.replace("block,V01,,,1,50.00,1.00,0.0001,2.00,0.0002,3,",
                             "block,V01,,,1,50.00,1.00,0.0001,3.00,0.0003,3,")
# 4 m wide corridor east of START; inner = buffer(-0.05) -> |dy| <= 1.95.
DOMAIN = box(START[0] - 2.0, START[1] - 2.0, START[0] + 100.0, START[1] + 2.0)
INNER = DOMAIN.buffer(-0.05)
LIMITS = PublishLimits(route=RouteFileLimits(max_outside_frac=0.005, closure_max_m=0.01, length_tol_m=0.01,
                                             grid_size_m=0.001, decimals=2), sum_tol_m=0.05, sum_tol_m2=0.05,
                        require_source=None)


def _pt(dx: float, dy: float = 0.0) -> list[float]:
    return [round(START[0] + dx, 2), round(START[1] + dy, 2)]


def _route_doc(coords, *, length: float | None = None, geom_type: str = "LineString", crs=CRS) -> dict:
    line_len = LineString(coords).length if geom_type == "LineString" else 0.0
    props = {"length_m": round(line_len if length is None else length, 2), "duration_min": 1.5,
             "baseline_length_m": 200.0, "outside_share": 0.0}
    geometry = {"type": geom_type, "coordinates": coords if geom_type == "LineString" else [coords, coords]}
    doc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": props, "geometry": geometry}]}
    return doc | ({"crs": crs} if crs else {})


OUT_AND_BACK = [list(START), _pt(50.0), list(START)]


def _write(tmp_path: Path, doc: dict, csv_text: str = CSV_OK) -> PublishSources:
    route, table = tmp_path / "src" / "route.geojson", tmp_path / "src" / "measurements.csv"
    route.parent.mkdir(parents=True, exist_ok=True)
    route.write_text(json.dumps(doc), encoding="utf-8")
    table.write_text(csv_text, encoding="utf-8")
    return PublishSources(route=route, csv=table)


def _evaluate(tmp_path: Path, doc: dict, csv_text: str = CSV_OK, *, limits: PublishLimits = LIMITS,
              source: str = "marcaj", inner=INNER):
    return evaluate_publish(_write(tmp_path, doc, csv_text), tmp_path / "staging", inner=inner, start_xy=START,
                            limits=limits, source=source)


def _failed(report) -> set[str]:
    return {c.name for c in report.checks if not c.ok}


# ------------------------------------------------------------------ accept


def test_valid_out_and_back_route_is_published_byte_identically(tmp_path):
    report = _evaluate(tmp_path, _route_doc(OUT_AND_BACK))
    assert report.passed and _failed(report) == set()
    out = write_published(report, tmp_path / "root")
    assert [p.name for p in out] == ["route.geojson", "measurements.csv"]
    assert (tmp_path / "root" / "route.geojson").read_bytes() == (tmp_path / "src" / "route.geojson").read_bytes()
    assert (tmp_path / "root" / "measurements.csv").read_bytes() == CSV_OK.encode()
    doc = json.loads((tmp_path / "root" / "route.geojson").read_text(encoding="utf-8"))
    coords = doc["features"][0]["geometry"]["coordinates"]
    assert coords[0] == coords[-1] == [629504.7, 5220250.75] and doc["crs"] == CRS
    assert doc["features"][0]["properties"]["length_m"] == 100.0


def test_endpoints_within_closure_tolerance_are_snapped_to_start(tmp_path):
    near = [[629504.71, 5220250.75], _pt(50.0), [629504.70, 5220250.74]]
    report = _evaluate(tmp_path, _route_doc(near, length=LineString(near).length))
    assert report.passed
    write_published(report, tmp_path / "root")
    doc = json.loads((tmp_path / "root" / "route.geojson").read_text(encoding="utf-8"))
    coords = doc["features"][0]["geometry"]["coordinates"]
    assert coords[0] == coords[-1] == [629504.7, 5220250.75]
    assert doc["features"][0]["properties"]["length_m"] == 100.0
    assert doc["features"][0]["properties"]["baseline_length_m"] == 200.0


def test_prepared_route_bytes_leaves_unusable_documents_alone():
    assert prepared_route_bytes(b"{", START, LIMITS.route) == b"{"
    raw = json.dumps(_route_doc(OUT_AND_BACK, geom_type="MultiLineString")).encode()
    assert prepared_route_bytes(raw, START, LIMITS.route) == raw
    far = json.dumps(_route_doc([_pt(1.0), _pt(50.0), _pt(1.0)])).encode()
    assert prepared_route_bytes(far, START, LIMITS.route) == far


# ------------------------------------------------------------------ refuse


def test_three_percent_outside_is_refused(tmp_path):
    detour = [list(START), _pt(50.0), _pt(50.0, 3.55), _pt(50.0), list(START)]
    report = _evaluate(tmp_path, _route_doc(detour))
    assert _failed(report) == {"outside_frac"}
    outside = next(c for c in report.checks if c.name == "outside_frac")
    assert "outside_frac=0.029" in outside.detail or "outside_frac=0.030" in outside.detail
    with pytest.raises(PublishRefused, match="outside_frac"):
        write_published(report, tmp_path / "root")
    assert not (tmp_path / "root").exists()


def _detour(dy: float) -> list[list[float]]:
    return [list(START), _pt(50.0), _pt(50.0, dy), _pt(50.0), list(START)]


@pytest.mark.parametrize(("dy", "ok"), [(2.45, True), (2.85, False)])
def test_publish_limit_is_the_configured_1_5_percent(tmp_path, dy, ok):
    """0.95 % outside is published (the old 0.5 % limit refused it); 1.70 % is refused (official: 2 %)."""
    route_cfg = load_config(environ={}).route
    limits = PublishLimits(route=RouteFileLimits.from_route_cfg(route_cfg), sum_tol_m=0.05, sum_tol_m2=0.05,
                           require_source=None)
    assert limits.route.max_outside_frac == pytest.approx(0.015)
    assert limits.route.max_outside_frac < route_cfg.max_outside_frac_official
    report = _evaluate(tmp_path, _route_doc(_detour(dy)), limits=limits)
    assert report.passed is ok and _failed(report) == (set() if ok else {"outside_frac"})
    assert not _evaluate(tmp_path, _route_doc(_detour(dy))).passed  # 0.5 % test limits refuse both


@pytest.mark.parametrize(("doc", "name"), [
    (_route_doc([[629504.72, 5220250.75], _pt(50.0), [629504.72, 5220250.75]]), "closure"),
    (_route_doc(OUT_AND_BACK, geom_type="MultiLineString"), "linestring"),
    (_route_doc(OUT_AND_BACK, length=100.02), "length_m"),
    (_route_doc(OUT_AND_BACK, crs=None), "crs"),
    (_route_doc([list(START), [629554.701, 5220250.75], list(START)]), "decimals"),
])
def test_route_refusals(tmp_path, doc, name):
    assert name in _failed(_evaluate(tmp_path, doc))


def test_bad_csv_is_refused(tmp_path):
    assert _failed(_evaluate(tmp_path, _route_doc(OUT_AND_BACK), "level,n_rows\nsurvey,1\n")) == {"header"}


def test_double_counted_interrow_area_is_refused(tmp_path):
    assert _failed(_evaluate(tmp_path, _route_doc(OUT_AND_BACK), CSV_DOUBLED)) == {"block_interrow_sum"}


def test_require_source(tmp_path):
    strict = PublishLimits(route=LIMITS.route, sum_tol_m=0.05, sum_tol_m2=0.05, require_source="marcaj")
    assert _failed(_evaluate(tmp_path, _route_doc(OUT_AND_BACK), limits=strict, source="reference")) == \
        {"require_source"}
    assert _evaluate(tmp_path, _route_doc(OUT_AND_BACK), limits=strict, source="marcaj").passed


def test_missing_domain_is_refused(tmp_path):
    report = _evaluate(tmp_path, _route_doc(OUT_AND_BACK), inner=None)
    assert "passable_domain" in _failed(report) and not report.passed


def test_unreadable_sources_raise(tmp_path):
    with pytest.raises(PublishRefused, match="cannot read"):
        evaluate_publish(PublishSources(tmp_path / "no.geojson", tmp_path / "no.csv"), tmp_path / "s",
                         inner=INNER, start_xy=START, limits=LIMITS, source="marcaj")


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


REF = "20260926T0100-marcaj-aaaaaa"


def _static_start(ctx) -> None:
    frame = gpd.GeoDataFrame({"fid": [0]}, geometry=[Point(START)], crs="EPSG:32635")
    write_layer(frame, "in_start", ctx.paths.static_layers_dir / "in_start.parquet")


@pytest.fixture
def post_run(tmp_path, monkeypatch):
    runner = types.ModuleType("vineyard.pipeline.runner")
    runner.StageResult = _FakeStageResult  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vineyard.pipeline.runner", runner)
    cfg = load_config(overrides=(f"paths.publish_dir={tmp_path / 'root'}",),
                      environ={"VINEYARD_WORK_DIR": str(tmp_path / "work")})
    write_annset(make_annset(BlockSpec(n_rows=2), prov=Prov(Source.MARCAJ, REF, "m")),
                 cfg.paths.work_dir / "runs" / REF / "annset")
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", annset_ref=REF, run_id="20260926T0310-post-abcdef")
    ensure_run_dirs(ctx.paths)
    _static_start(ctx)
    domain = layer_frame("passable_domain", [{"domain_id": "D1", "area_m2": DOMAIN.area, "erosion_m": 0.0,
                                              "n_components": 1, "geometry": MultiPolygon([DOMAIN])}],
                         Prov(Source.MARCAJ))
    write_layer(domain, "passable_domain", ctx.paths.layers_dir / "passable_domain.parquet")
    (ctx.paths.exports_dir / "measurements.csv").write_text(CSV_OK, encoding="utf-8")
    return ctx


def test_stage_publishes_to_the_configured_root(post_run):
    (post_run.paths.exports_dir / "route.geojson").write_text(json.dumps(_route_doc(OUT_AND_BACK)))
    spec = load_stage("publish")
    result = spec.run(post_run)
    root = post_run.cfg.paths.publish_dir
    assert (spec.name, spec.scope, result.stage, result.n_items) == ("publish", "global", "publish", 2)
    assert (root / "route.geojson").is_file() and (root / "measurements.csv").read_text() == CSV_OK
    report = json.loads((post_run.paths.metrics_dir / "publish_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True and report["source"] == "marcaj"
    assert result.metrics["passed"] == 1.0


def test_stage_refuses_and_keeps_the_root_untouched(post_run):
    detour = [list(START), _pt(50.0), _pt(50.0, 3.55), _pt(50.0), list(START)]
    (post_run.paths.exports_dir / "route.geojson").write_text(json.dumps(_route_doc(detour)))
    with pytest.raises(PublishRefused) as info:
        load_stage("publish").run(post_run)
    assert info.value.exit_code != 0
    assert not (post_run.cfg.paths.publish_dir / "route.geojson").exists()
    report = json.loads((post_run.paths.metrics_dir / "publish_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and "outside_frac" in report["failed"]


def test_stage_require_source_from_config(post_run):
    (post_run.paths.exports_dir / "route.geojson").write_text(json.dumps(_route_doc(OUT_AND_BACK)))
    cfg = load_config(overrides=(f"paths.publish_dir={post_run.cfg.paths.publish_dir}",
                                 "publish.require_source=model"),
                      environ={"VINEYARD_WORK_DIR": str(post_run.cfg.paths.work_dir)})
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", annset_ref=REF, run_id=post_run.run_id)
    with pytest.raises(PublishRefused, match="require_source"):
        load_stage("publish").run(ctx)


def test_stage_refuses_a_double_counted_interrow_area(post_run):
    (post_run.paths.exports_dir / "route.geojson").write_text(json.dumps(_route_doc(OUT_AND_BACK)))
    (post_run.paths.exports_dir / "measurements.csv").write_text(CSV_DOUBLED, encoding="utf-8")
    with pytest.raises(PublishRefused, match="block_interrow_sum"):
        load_stage("publish").run(post_run)
    assert not (post_run.cfg.paths.publish_dir / "measurements.csv").exists()


def test_stage_refuses_without_a_passable_domain(post_run):
    (post_run.paths.exports_dir / "route.geojson").write_text(json.dumps(_route_doc(OUT_AND_BACK)))
    (post_run.paths.layers_dir / "passable_domain.parquet").unlink()
    with pytest.raises(PublishRefused, match="passable_domain"):
        load_stage("publish").run(post_run)


# ------------------------------------------------------------------ shared post-stage I/O (_post_io)


def _bare_ctx(tmp_path: Path, *, data_root: Path | None = None, annset_ref: str | None = REF, run_id: str = "x1"):
    env = {"VINEYARD_WORK_DIR": str(tmp_path / "work")} | ({"VINEYARD_DATA_ROOT": str(data_root)} if data_root else {})
    ctx = new_run_context(load_config(environ=env), source=Source.MARCAJ, kind="post", annset_ref=annset_ref,
                          run_id=run_id)
    ensure_run_dirs(ctx.paths)
    return ctx


def test_static_layers_fall_back_to_the_organizers_geojson(tmp_path):
    zone = box(629500, 5220200, 629510, 5220210)
    route_dir = tmp_path / "data" / "02_route"
    route_dir.mkdir(parents=True)
    write_geojson(gpd.GeoDataFrame({"fid": [1]}, geometry=[zone], crs="EPSG:32635"), route_dir / "forbidden.geojson")
    ctx = _bare_ctx(tmp_path, data_root=tmp_path / "data")
    assert _post_io.static_geometry(ctx, "in_forbidden").equals(zone)
    assert _post_io.static_geometry(ctx, "in_passages") is None
    with pytest.raises(StageError, match="unknown static"):
        _post_io.static_geometry(ctx, "in_rivers")
    with pytest.raises(StageError, match="START"):
        _post_io.start_xy(ctx, "publish")


def test_coverage_uses_tile_valid_and_full_squares_for_missing_tiles(tmp_path):
    ctx = _bare_ctx(tmp_path)
    tiles = ["siret3_r018_c011", "siret3_r018_c012"]
    hole = box(629560, 5220280, 629570, 5220290)
    assert _post_io.coverage(ctx, tiles)[1] is False
    write_layer(tile_valid_frame(tiles[:1], {tiles[0]: hole}), "tile_valid", ctx.paths.tile_valid)
    covered, used = _post_io.coverage(ctx, tiles)
    assert used and not covered.intersects(hole.buffer(-0.01))
    assert covered.area == pytest.approx(2 * tile_box(tile_ref(tiles[0])).area - hole.area)


def test_walking_domain_of_an_already_eroded_layer(tmp_path):
    ctx = _bare_ctx(tmp_path)
    assert _post_io.walking_domain(ctx.paths, ctx) is None
    frame = layer_frame("passable_domain", [{"domain_id": "D1", "area_m2": DOMAIN.area, "erosion_m": 0.05,
                                             "n_components": 1, "geometry": MultiPolygon([DOMAIN])}], Prov())
    write_layer(frame, "passable_domain", ctx.paths.layers_dir / "passable_domain.parquet")
    dom = _post_io.walking_domain(ctx.paths, ctx)
    assert dom.inner.area == pytest.approx(DOMAIN.area) and dom.eroded.area < dom.inner.area


def test_find_post_run_prefers_the_newest_run_of_the_same_annset(tmp_path):
    ctx = _bare_ctx(tmp_path, run_id="20260926T0500-post-cccccc")
    for run_id in ("20260926T0100-post-aaaaaa", "20260926T0200-post-bbbbbb"):
        run = ctx.paths.runs_dir / run_id
        (run / "layers").mkdir(parents=True)
        (run / "layers" / "rows.parquet").write_bytes(b"x")
        (run / "run.json").write_text(json.dumps({"annset_ref": REF}), encoding="utf-8")
    assert _post_io.find_post_run(ctx, ("layers/rows.parquet",), "targets").run_dir.name == \
        "20260926T0200-post-bbbbbb"
    (ctx.paths.runs_dir / "20260926T0200-post-bbbbbb" / "run.json").write_text("{", encoding="utf-8")
    with pytest.raises(StageError, match="unreadable run record"):
        _post_io.find_post_run(ctx, ("layers/rows.parquet",), "targets")


def test_annset_lookups_fail_with_context(tmp_path):
    with pytest.raises(StageError, match="needs --annset"):
        _post_io.annset_dir(_bare_ctx(tmp_path, annset_ref=None), "measure")
    ctx = _bare_ctx(tmp_path)
    (ctx.paths.runs_dir / REF / "annset").mkdir(parents=True)
    (ctx.paths.runs_dir / REF / "annset" / "annset.json").write_text("{", encoding="utf-8")
    with pytest.raises(StageError, match="annset.json"):
        _post_io.load_annset_meta(ctx, "publish")


def test_find_post_run_matches_a_latest_alias_of_the_same_annset(tmp_path):
    ctx = _bare_ctx(tmp_path, run_id="20260926T0500-post-cccccc")
    (ctx.paths.runs_dir / REF / "annset").mkdir(parents=True)
    (ctx.paths.runs_dir / "LATEST_MARCAJ").symlink_to(REF, target_is_directory=True)
    run = ctx.paths.runs_dir / "20260926T0100-post-aaaaaa"
    (run / "exports").mkdir(parents=True)
    (run / "exports" / "route.geojson").write_text("{}", encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"annset_ref": "LATEST_MARCAJ"}), encoding="utf-8")
    assert _post_io.find_post_run(ctx, ("exports/route.geojson",), "publish").run_dir == run
    (run / "run.json").write_text(json.dumps({"annset_ref": "20260101T0000-marcaj-gone00"}), encoding="utf-8")
    with pytest.raises(StageError, match="no post run"):
        _post_io.find_post_run(ctx, ("exports/route.geojson",), "publish")
    assert _post_io.find_optional_post_run(ctx, ("exports/route.geojson",)) is None
