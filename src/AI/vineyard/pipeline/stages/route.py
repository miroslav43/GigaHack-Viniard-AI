"""Stage `route`: walk graph + passable domain + targets -> the inspection route (arch §4.11).

Reads layers/{walk_nodes,walk_edges,passable_domain,targets}.parquet of this post run (or of the newest
post run of the same annset) and the static START / passages. Writes, in the current post run:
exports/route.geojson (contract §5.1), layers/{route,route_stops,target_visits}.parquet,
metrics/route_validation.json and metrics/route_baseline.json. The written GeoJSON is read back and
validated again; a failing route is reported as a failed item (publish refuses it independently).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import numpy as np
from shapely.geometry.base import BaseGeometry

from vineyard.errors import StageError
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.atomic import atomic_write_json
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.stages._post_io import (
    find_post_run,
    layer_path,
    stage_result,
    start_xy,
    static_geometry,
    walking_domain,
)
from vineyard.route import graph_io
from vineyard.route.baseline import compute_baseline
from vineyard.route.candidates import ROLE_MUST, TargetPoint, target_points
from vineyard.route.domain import DomainSet
from vineyard.route.geojson import (
    RouteFileLimits,
    check_route_file,
    failed_checks,
    read_route_geojson,
    write_route_geojson,
)
from vineyard.route.graph_types import WalkGraph
from vineyard.route.outputs import (
    RouteProvenance,
    json_safe,
    route_frame,
    route_props,
    stops_frame,
    validation_doc,
    visits_frame,
)
from vineyard.route.planner import PlanParams, RoutePlan, plan_route
from vineyard.route.policies import headland_report
from vineyard.route.validate import RouteValidation, validate_route

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext, RunPaths

STAGE_NAME: Final = "route"
STAGE_VERSION: Final = "2"
CFG_KEYS: Final = ("route", "logging.tz")
INPUT_LAYERS: Final = ("walk_nodes", "walk_edges", "passable_domain", "targets")
ROUTE_GEOJSON: Final = "route.geojson"
VALIDATION_JSON: Final = "route_validation.json"
BASELINE_JSON: Final = "route_baseline.json"

_log = get_logger("pipeline.stages.route")


@dataclass(frozen=True, eq=False)
class RouteInputs:
    graph: WalkGraph
    domain: DomainSet
    targets: tuple[TargetPoint, ...]
    start_xy: tuple[float, float]
    passages: BaseGeometry | None
    prov: RouteProvenance


def load_walk_graph(paths: RunPaths) -> WalkGraph:
    """WalkGraph from the run's `walk_nodes` / `walk_edges` layers (passable's graph_io format)."""
    nodes = read_layer(layer_path(paths, graph_io.NODES_LAYER), graph_io.NODES_LAYER)
    edges = read_layer(layer_path(paths, graph_io.EDGES_LAYER), graph_io.EDGES_LAYER)
    try:
        return graph_io.graph_from_layers(nodes, edges)
    except ValueError as exc:
        raise StageError("walk graph layers are inconsistent", stage=STAGE_NAME, run=paths.run_dir.name,
                         error=str(exc)) from exc


def _provenance(ctx: RunContext, targets: gpd.GeoDataFrame) -> RouteProvenance:
    source = str(targets["source"].iloc[0]) if len(targets) else str(ctx.source)
    version = str(targets["model_version"].iloc[0]) if len(targets) else ctx.model_version()
    return RouteProvenance(source=source, run_id=ctx.run_id, model_version=version)


def load_inputs(ctx: RunContext) -> RouteInputs:
    paths: RunPaths = find_post_run(ctx, tuple(f"layers/{n}.parquet" for n in INPUT_LAYERS), STAGE_NAME)
    domain = walking_domain(paths, ctx)
    if domain is None or domain.raw.is_empty:
        raise StageError("passable_domain is empty", stage=STAGE_NAME, run=paths.run_dir.name)
    graph = load_walk_graph(paths)
    targets = read_layer(layer_path(paths, "targets"), "targets")
    return RouteInputs(graph=graph, domain=domain, targets=target_points(targets), start_xy=start_xy(ctx, STAGE_NAME),
                       passages=static_geometry(ctx, "in_passages"), prov=_provenance(ctx, targets))


def _revalidate(path: Path, plan: RoutePlan, inputs: RouteInputs, params: PlanParams) -> RouteValidation:
    """Validation of the geometry read back from the written file (arch §4.11.6)."""
    geom, _ = read_route_geojson(path)
    must = np.array([[t.x, t.y] for t in plan.visits if t.route_role == ROLE_MUST and t.reachable_final],
                    dtype=np.float64).reshape(-1, 2)
    every = np.array([[t.x, t.y] for t in inputs.targets], dtype=np.float64).reshape(-1, 2)
    return validate_route(geom, inputs.domain.inner, start_xy=inputs.start_xy, required_xy=must, all_xy=every,
                          params=params.validate, anchor_idx=plan.anchors)


def _write(ctx: RunContext, inputs: RouteInputs, plan: RoutePlan, params: PlanParams) -> tuple[list[Path], Any]:
    cfg = ctx.cfg.route
    base = compute_baseline(plan.graph, plan.sets, plan.validation.length_m, inputs.start_xy,
                            chunk=cfg.solver.dijkstra_chunk, walking_speed_kmh=cfg.walking_speed_kmh)
    props = route_props(plan, plan.validation, base, inputs.prov, speed_kmh=cfg.walking_speed_kmh,
                        tz=ctx.cfg.logging.tz)
    geo_path = ctx.paths.exports_dir / ROUTE_GEOJSON
    written = write_route_geojson(plan.line, props, geo_path, start_xy=inputs.start_xy,
                                  decimals=cfg.validate_.coord_decimals)
    v = _revalidate(geo_path, plan, inputs, params)
    checks = check_route_file(geo_path, start_xy=inputs.start_xy, inner=inputs.domain.inner,
                              limits=RouteFileLimits.from_route_cfg(cfg))
    bad_checks = failed_checks(checks)
    doc = validation_doc(plan, v, checks, headland_report(inputs.graph, inputs.passages))
    doc["passed"] = bool(v.passed and not bad_checks)
    outputs = [
        geo_path,
        write_layer(route_frame(written, v, plan, inputs.prov, cfg.walking_speed_kmh), "route",
                    layer_path(ctx.paths, "route")),
        write_layer(stops_frame(plan.stops, inputs.prov), "route_stops", layer_path(ctx.paths, "route_stops")),
        write_layer(visits_frame(plan.visits, inputs.prov), "target_visits", layer_path(ctx.paths, "target_visits")),
        atomic_write_json(ctx.paths.metrics_dir / VALIDATION_JSON, doc),
        atomic_write_json(ctx.paths.metrics_dir / BASELINE_JSON, json_safe(base.to_json())),
    ]
    return outputs, (v, bad_checks)


def run(ctx: RunContext) -> Any:
    inputs = load_inputs(ctx)
    params = PlanParams.from_route_cfg(ctx.cfg.route, inputs.start_xy)
    plan = plan_route(inputs.graph, inputs.targets, inputs.domain.inner, inputs.domain.eroded, params)
    outputs, (v, bad_checks) = _write(ctx, inputs, plan, params)
    failures = (*v.failures, *(c.name for c in bad_checks))
    metrics = {"length_m": round(v.length_m, 2), "outside_frac": v.outside_frac, "coverage_est": v.coverage_est,
               "coverage_required": v.coverage_required, "n_targets": float(v.n_targets),
               "solve_time_s": plan.solve_time_s, "passed": float(not failures)}
    log_event(_log, "route.written", stage=STAGE_NAME, policy=plan.policy, failures=list(failures), **metrics)
    if failures:
        log_event(_log, "route.validation_failed", stage=STAGE_NAME, failures=list(failures))
        return _failed_result(outputs, metrics, failures)
    return stage_result(STAGE_NAME, n_items=v.n_targets, outputs=outputs, metrics=metrics)


def _failed_result(outputs: list[Path], metrics: dict[str, float], failures: tuple[str, ...]) -> Any:
    from vineyard.pipeline.runner import StageResult

    return StageResult(stage=STAGE_NAME, n_items=1, n_cached=0, n_failed=1,
                       failed=tuple(f"route_validation:{f}" for f in failures), outputs=tuple(outputs),
                       metrics=metrics)


STAGE: Final = StageSpec(
    name=STAGE_NAME, version=STAGE_VERSION, scope="global", cfg_keys=CFG_KEYS, requires=("passable", "targets"),
    run=run, description="GTSP inspection route from START through every reachable target (route.geojson)",
)
