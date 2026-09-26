"""Stage `route` on a synthetic post run: layers in, route.geojson + layers + metrics out."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point

from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.errors import StageError
from vineyard.geo.tiling import CRS_EPSG
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.pipeline.context import ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages import route as route_stage
from vineyard.route import graph_io
from vineyard.route.geojson import CRS_URN
from vineyard.route.outputs import created_at, json_safe

from .route_factories import mini_route

REF = "20260926T0200-marcaj-abcdef"
RUN_ID = "20260926T0310-post-abcdef"
PROV = {"source": "marcaj", "run_id": RUN_ID, "model_version": "marcaj-export@12345678", "confidence": 1.0,
        "qa_flags": ""}


@dataclass(frozen=True)
class _FakeStageResult:
    stage: str
    n_items: int
    n_cached: int
    n_failed: int
    failed: tuple[str, ...] = ()
    outputs: tuple[Path, ...] = ()
    metrics: dict = field(default_factory=dict)


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = types.ModuleType("vineyard.pipeline.runner")
    runner.StageResult = _FakeStageResult  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vineyard.pipeline.runner", runner)


def _prov(n: int) -> dict[str, list]:
    return {k: [v] * n for k, v in PROV.items()}


def _walk_layers(g) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    return graph_io.graph_to_layers(g, graph_io.LayerProvenance(PROV["source"], RUN_ID, PROV["model_version"]))


def _targets(m) -> gpd.GeoDataFrame:
    pts = [(m.x0 + 12.0, m.row_ys[1]), (m.x0 + 44.0, m.row_ys[3]), (m.x0 + 20.0, m.row_ys[-1] + 40.0)]
    n = len(pts)
    data = {"target_id": [f"T-GAP-{k:04d}" for k in range(1, n + 1)], "kind": ["row_gap"] * n,
            "vineyard_id": ["V01"] * n, "tile_id": ["siret3_r010_c010"] * n, "row_id": ["V01-R002"] * n,
            "interrow_id": [None] * n, "waste_id": [None] * n, "x": [p[0] for p in pts], "y": [p[1] for p in pts],
            "gap_length_m": [6.0] * n, "priority": [2] * n, "reachable": [True] * n, "reach_note": [""] * n,
            "snap_dist_m": [1.3] * n, "route_role": ["must"] * n}
    return gpd.GeoDataFrame(data | _prov(n), geometry=[Point(p) for p in pts], crs=CRS_EPSG)


def _static(name: str, geom) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"fid": [1], "type": [None], "name": [name], "source": [None]}, geometry=[geom],
                            crs=CRS_EPSG)


@pytest.fixture
def post_ctx(tmp_path: Path, fake_runner: None):
    cfg = load_config(overrides=("route.solver.time_limit_s=1", "route.solver.fallback_time_s=1"),
                      environ={"VINEYARD_WORK_DIR": str(tmp_path / "work")})
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", annset_ref=REF, run_id=RUN_ID)
    ensure_run_dirs(ctx.paths)
    m = mini_route(n_rows=6)
    nodes, edges = _walk_layers(m.graph)
    layers = ctx.paths.layers_dir
    write_layer(nodes, "walk_nodes", layers / "walk_nodes.parquet")
    write_layer(edges, "walk_edges", layers / "walk_edges.parquet")
    domain = gpd.GeoDataFrame({"domain_id": ["D1"], "area_m2": [m.raw.area], "erosion_m": [0.0],
                               "n_components": [len(m.raw.geoms)]} | _prov(1), geometry=[m.raw], crs=CRS_EPSG)
    write_layer(domain, "passable_domain", layers / "passable_domain.parquet")
    write_layer(_targets(m), "targets", layers / "targets.parquet")
    static = ctx.paths.static_layers_dir
    write_layer(_static("start", Point(m.start_xy)), "in_start", static / "in_start.parquet")
    west = shapely.box(m.x0 - 4.0, m.row_ys[0] - 2.6, m.x0, m.row_ys[-1] + 2.6)
    write_layer(_static("passages", west), "in_passages", static / "in_passages.parquet")
    return ctx, m


def test_stage_is_registered() -> None:
    spec = load_stage("route")
    assert spec.name == "route" and spec.scope == "global" and set(spec.requires) == {"passable", "targets"}


def test_stage_writes_every_output(post_ctx) -> None:
    ctx, m = post_ctx
    result = route_stage.run(ctx)
    assert result.n_failed == 0, result.failed
    paths = ctx.paths
    doc = json.loads((paths.exports_dir / "route.geojson").read_text())
    assert doc["crs"]["properties"]["name"] == CRS_URN
    coords = doc["features"][0]["geometry"]["coordinates"]
    assert coords[0] == list(m.start_xy) and coords[-1] == list(m.start_xy)
    props = doc["features"][0]["properties"]
    assert props["length_m"] == pytest.approx(LineString(coords).length, abs=0.01)
    assert props["route_id"] == f"RT-{RUN_ID}" and props["duration_min"] > 0 and props["baseline_length_m"] > 0
    assert props["created_at"] == "2026-09-26T03:10:00+03:00"
    route = read_layer(paths.layers_dir / "route.parquet", "route")
    assert len(route) == 1 and route["n_targets"].iloc[0] == 3 and route["n_visited_est"].iloc[0] == 2
    stops = read_layer(paths.layers_dir / "route_stops.parquet", "route_stops")
    assert stops["seq"].tolist() == list(range(len(stops))) and stops["target_id"].isna().sum() == 2
    visits = read_layer(paths.layers_dir / "target_visits.parquet", "target_visits")
    assert visits.set_index("target_id")["reach_note"]["T-GAP-0003"] == "too_far"
    val = json.loads((paths.metrics_dir / "route_validation.json").read_text())
    assert val["passed"] is True and val["policy"] == "penalty" and val["headland"]["n_interrow_ends"] == 10
    assert val["policy_accepted"] is True and val["plan_outside_limit"] == pytest.approx(0.014)
    assert val["policies"][0]["acceptable"] is True and val["policies"][0]["n_interrows_reachable"] > 0
    assert val["headland"]["share_interrow_ends_near_passage"] == pytest.approx(0.5)
    base = json.loads((paths.metrics_dir / "route_baseline.json").read_text())
    assert base["serpentine_est"]["saved_pct"] > 0


def test_stage_reports_a_failing_route(post_ctx, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _ = post_ctx
    real = route_stage.check_route_file

    def strict(path, **kw):
        kw["limits"] = type(kw["limits"])(**{**kw["limits"].__dict__, "closure_max_m": -1.0})
        return real(path, **kw)

    monkeypatch.setattr(route_stage, "check_route_file", strict)
    result = route_stage.run(ctx)
    assert result.n_failed == 1 and result.failed == ("route_validation:closure",)
    assert json.loads((ctx.paths.metrics_dir / "route_validation.json").read_text())["passed"] is False


def test_inconsistent_walk_layers_are_a_stage_error(post_ctx) -> None:
    ctx, m = post_ctx
    nodes, _ = _walk_layers(m.graph)
    write_layer(nodes.assign(kind="passage"), "walk_nodes", ctx.paths.layers_dir / "walk_nodes.parquet")
    with pytest.raises(StageError, match="walk graph layers"):
        route_stage.run(ctx)


def test_walk_graph_round_trip(post_ctx) -> None:
    ctx, m = post_ctx
    g = route_stage.load_walk_graph(ctx.paths)
    assert g.n_nodes == m.graph.n_nodes and g.n_edges == m.graph.n_edges and g.start_node == m.graph.start_node
    assert np.allclose(g.edge_cost, m.graph.edge_cost) and np.allclose(g.node_xy, m.graph.node_xy)


def test_empty_domain_is_an_error(post_ctx) -> None:
    ctx, m = post_ctx
    path = ctx.paths.layers_dir / "passable_domain.parquet"
    empty = gpd.GeoDataFrame({"domain_id": ["D1"], "area_m2": [0.0], "erosion_m": [0.0], "n_components": [0]}
                             | _prov(1),
                             geometry=[shapely.box(0, 0, 1, 1)], crs=CRS_EPSG).iloc[:0]
    write_layer(empty, "passable_domain", path)
    with pytest.raises(StageError, match="passable_domain"):
        route_stage.run(ctx)


def test_json_safe_and_created_at() -> None:
    assert json_safe({"a": float("nan"), "b": (1, np.float32(2.5)), "c": float("inf")}) == {
        "a": None, "b": [1, 2.5], "c": None}
    assert created_at("not-a-run-id", "Europe/Bucharest") is None
