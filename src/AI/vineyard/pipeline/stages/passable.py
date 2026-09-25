"""Stage `passable` (post run): walking domain + walk graph (arch §4.11.1-2, contract §2.5.12).

Reads the AnnSet (`--annset` run), derive's `rows` (+ `interrow_pieces_linked` when present) and the
static route inputs; writes layers passable_domain / passable_parts / walk_nodes / walk_edges,
qa/issues_passable.parquet and metrics/passable_report.json (headland metrics of all strategies).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import Severity
from vineyard.contracts.qa import QaIssue, issues_to_gdf
from vineyard.errors import SchemaError, StageError
from vineyard.geo.tiling import tile_of_point
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.atomic import atomic_write_json
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.stages._post_io import load_annset, static_frame
from vineyard.route import graph_io
from vineyard.route.connectors import HeadlandStrategy
from vineyard.route.walk_build import (
    PassableInputs,
    PassableParams,
    PassableResult,
    graph_for,
    prepare_lines,
    restrict_result,
)
from vineyard.route.walk_report import choose_strategy, compare_strategies

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext
    from vineyard.pipeline.runner import StageResult

NAME: Final = "passable"
VERSION: Final = "1"
CFG_KEYS: Final = ("route.domain", "route.graph", "route.start_file", "route.max_outside_frac_publish")
REQUIRES: Final = ("derive",)
ROWS_FILE: Final = "rows.parquet"
LINKED_FILE: Final = "interrow_pieces_linked.parquet"
REPORT_FILE: Final = "passable_report.json"
QA_FILE: Final = "issues_passable.parquet"
EVENT_DONE: Final = "passable.done"

_log = get_logger("pipeline.stages.passable")


def _static(ctx: RunContext, name: str) -> gpd.GeoDataFrame:
    """work/layers/<name>.parquet (written by ingest), else the organizers' GeoJSON in paths.route_dir."""
    return static_frame(ctx, name, NAME)


def _union(frame: gpd.GeoDataFrame) -> BaseGeometry | None:
    return shapely.union_all(list(frame.geometry)) if len(frame) else None


def _start_xy(frame: gpd.GeoDataFrame) -> tuple[float, float]:
    if len(frame) != 1 or frame.geometry.iloc[0].geom_type != "Point":
        raise StageError("in_start must hold exactly one Point", stage=NAME, n=len(frame))
    point = frame.geometry.iloc[0]
    return float(point.x), float(point.y)


def load_inputs(ctx: RunContext) -> PassableInputs:
    rows_path = ctx.paths.layers_dir / ROWS_FILE
    if not rows_path.is_file():
        raise StageError("rows layer missing (run derive first)", stage=NAME, path=str(rows_path))
    annset = load_annset(ctx, NAME)  # --annset is required, like every post stage
    linked = ctx.paths.layers_dir / LINKED_FILE
    pieces = read_layer(linked, "interrow_pieces_linked") if linked.is_file() else annset.layer("interrow_pieces")
    passages = _union(_static(ctx, "in_passages"))
    if passages is None:
        raise StageError("in_passages is empty", stage=NAME)
    return PassableInputs(rows=read_layer(rows_path, "rows"), pieces=pieces,
                          canopies=tuple(annset.layer("canopies").geometry), passages=passages,
                          forbidden=_union(_static(ctx, "in_forbidden")), start_xy=_start_xy(_static(ctx, "in_start")))


def _tile_id(x: float, y: float) -> str:
    try:
        return tile_of_point(x, y).tile_id
    except SchemaError:   # off the 311-tile grid: the issue has no tile
        return ""


def qa_issues(result: PassableResult) -> tuple[QaIssue, ...]:
    """domain_disconnected per component START cannot reach; connector_outside per dropped connector."""
    issues = [QaIssue(Severity.WARNING, "domain_disconnected", _tile_id(c.x, c.y), f"component{c.label}",
                      f"componentă fără START: {c.length_m:.1f} m, {len(c.interrow_ids)} inter-rânduri", c.x, c.y)
              for c in result.components if not c.has_start]
    issues += [QaIssue(Severity.INFO, "connector_outside", _tile_id(*c.from_xy), c.ref,
                       f"conector respins ({result.strategy}): {c.outside_m:.2f} m în afara domeniului", *c.from_xy)
               for c in result.plan.dropped]
    return tuple(issues)


def _report(ctx: RunContext, params: PassableParams, base: PassableResult, chosen: PassableResult,
            passages: BaseGeometry) -> dict[str, Any]:
    limit = ctx.cfg.route.max_outside_frac_publish
    compared = compare_strategies(base, params, passages)
    return {"strategy": str(chosen.strategy), "recommended_strategy": str(choose_strategy([m for m, _ in compared], limit)),
            "max_outside_share": limit, "strategies": [m.to_json() for m, _ in compared],
            "domain": {"area_m2": chosen.lines.domain.raw.area, "n_components": chosen.lines.domain.n_components},
            "graph": {"n_nodes": chosen.graph.n_nodes, "n_edges": chosen.graph.n_edges,
                      "n_components": len(chosen.components)}}


def _write(ctx: RunContext, result: PassableResult, report: dict[str, Any]) -> tuple[Path, ...]:
    prov = graph_io.LayerProvenance(ctx.source.value, ctx.run_id, ctx.model_version())
    nodes, edges = graph_io.graph_to_layers(result.graph, prov)
    layers = {graph_io.DOMAIN_LAYER: graph_io.domain_to_layer(result.lines.domain, prov),
              graph_io.PARTS_LAYER: graph_io.parts_to_layer(result.lines.parts, prov),
              graph_io.NODES_LAYER: nodes, graph_io.EDGES_LAYER: edges}
    out = [write_layer(frame, name, ctx.paths.layers_dir / f"{name}.parquet") for name, frame in layers.items()]
    issues = issues_to_gdf(qa_issues(result), source=ctx.source, run_id=ctx.run_id, model_version=ctx.model_version())
    out.append(write_layer(issues, "qa_issues", ctx.paths.qa_dir / QA_FILE))
    out.append(atomic_write_json(ctx.paths.metrics_dir / REPORT_FILE, report))
    return tuple(out)


def run(ctx: RunContext) -> StageResult:
    from vineyard.pipeline.runner import StageResult

    inputs = load_inputs(ctx)
    params = PassableParams.from_config(ctx.cfg.route)
    try:
        base = graph_for(prepare_lines(inputs, params), params.with_strategy(HeadlandStrategy.PENALTY).connector,
                         params.graph)
    except ValueError as exc:
        raise StageError("walk graph could not be built", stage=NAME, reason=str(exc)) from exc
    chosen = base if params.connector.strategy == HeadlandStrategy.PENALTY else restrict_result(base, params.connector)
    report = _report(ctx, params, base, chosen, inputs.passages)
    outputs = _write(ctx, chosen, report)
    log_event(_log, EVENT_DONE, stage=NAME, nodes=chosen.graph.n_nodes, edges=chosen.graph.n_edges,
              components=len(chosen.components), strategy=str(chosen.strategy))
    return StageResult(stage=NAME, n_items=chosen.graph.n_edges, n_cached=0, n_failed=0, outputs=outputs,
                       metrics={"n_nodes": float(chosen.graph.n_nodes), "n_edges": float(chosen.graph.n_edges),
                                "n_components": float(len(chosen.components)),
                                "domain_area_m2": float(chosen.lines.domain.raw.area)})


STAGE: Final = StageSpec(name=NAME, version=VERSION, scope="global", cfg_keys=CFG_KEYS, requires=REQUIRES, run=run,
                         description="walking domain, passage skeleton, interrow centerlines, connectors, walk graph")
