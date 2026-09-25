"""Stage `passable` on a synthetic post run (runner API stubbed: pipeline/runner.py is built elsewhere)."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry.base import BaseGeometry

from tests.post.factories import BlockSpec, headland_passages, layer_frame, make_annset, start_point
from vineyard.annset.io import write_annset
from vineyard.contracts.enums import Source
from vineyard.contracts.schemas import coerce_layer, empty_layer
from vineyard.errors import StageError
from vineyard.geo.tiling import CRS_EPSG
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.pipeline.context import ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import StageSpec
from vineyard.route import graph_io
from vineyard.route.graph_types import NodeKind

REF_RUN = "20260926T0000-reference-000000"
SPEC = BlockSpec()


@dataclass(frozen=True)
class _StageResult:
    stage: str
    n_items: int
    n_cached: int
    n_failed: int
    failed: tuple[str, ...] = ()
    outputs: tuple[Path, ...] = ()
    metrics: dict = field(default_factory=dict)


@pytest.fixture
def runner_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    if "vineyard.pipeline.runner" not in sys.modules:
        module = types.ModuleType("vineyard.pipeline.runner")
        module.StageResult = _StageResult
        monkeypatch.setitem(sys.modules, "vineyard.pipeline.runner", module)


def _static(name: str, geoms: list[BaseGeometry]) -> gpd.GeoDataFrame:
    if not geoms:
        return empty_layer(name)
    return coerce_layer(gpd.GeoDataFrame({"fid": list(range(len(geoms)))}, geometry=geoms, crs=CRS_EPSG), name)


def _rows_layer() -> gpd.GeoDataFrame:
    records = [{"row_id": SPEC.row_id(k), "vineyard_id": SPEC.vineyard_id, "row_index": k, "length_m": 60.0,
                "extent_m": 60.0, "n_pieces": 2, "tile_ids": "", "angle_deg": 0.0, "spacing_prev_m": 2.5,
                "spacing_next_m": 2.5, "max_gap_m": 0.0, "geometry": SPEC.axis(k)} for k in range(1, SPEC.n_rows + 1)]
    return layer_frame("rows", records)


def _context(work: Path, *, with_rows: bool = True, with_start: bool = True):
    from vineyard.config import load_config

    write_annset(make_annset(SPEC, linked=True), work / "runs" / REF_RUN / "annset")
    ctx = new_run_context(load_config(), source=Source.REFERENCE, kind="post", annset_ref=REF_RUN)
    ensure_run_dirs(ctx.paths)
    static = ctx.paths.static_layers_dir
    write_layer(_static("in_passages", [headland_passages(SPEC)]), "in_passages", static / "in_passages.parquet")
    write_layer(_static("in_forbidden", []), "in_forbidden", static / "in_forbidden.parquet")
    write_layer(_static("in_start", [start_point(SPEC)] if with_start else []), "in_start", static / "in_start.parquet")
    if with_rows:
        write_layer(_rows_layer(), "rows", ctx.paths.layers_dir / "rows.parquet")
    return ctx


def test_stage_spec() -> None:
    from vineyard.pipeline.stages.passable import STAGE

    assert isinstance(STAGE, StageSpec) and STAGE.name == "passable" and STAGE.scope == "global"
    assert STAGE.requires == ("derive",)


def test_stage_writes_layers_report_and_qa(tmp_work: Path, runner_stub: None) -> None:
    from vineyard.pipeline.stages.passable import STAGE

    ctx = _context(tmp_work)
    result = STAGE.run(ctx)
    assert result.stage == "passable" and result.n_failed == 0 and result.n_items > 0
    layers = ctx.paths.layers_dir
    nodes = read_layer(layers / "walk_nodes.parquet", "walk_nodes")
    edges = read_layer(layers / "walk_edges.parquet", "walk_edges")
    graph = graph_io.graph_from_layers(nodes, edges)
    assert graph.node_kind[graph.start_node] == NodeKind.START
    assert set(graph.component.tolist()) == {graph.start_component}
    domain = read_layer(layers / "passable_domain.parquet", "passable_domain")
    assert domain["n_components"].tolist() == [1] and set(nodes["source"]) == {"reference"}
    assert len(read_layer(layers / "passable_parts.parquet", "passable_parts")) == 3 * 2 + 1   # pieces split per tile
    report = json.loads((ctx.paths.metrics_dir / "passable_report.json").read_text())
    assert report["strategy"] == "penalty"
    assert [s["strategy"] for s in report["strategies"]] == ["penalty", "limit", "inside_only"]
    assert report["strategies"][0]["n_interrows_reachable"] == 3
    assert (ctx.paths.qa_dir / "issues_passable.parquet").is_file()
    assert result.metrics["n_components"] == 1.0


def test_stage_requires_rows(tmp_work: Path, runner_stub: None) -> None:
    from vineyard.pipeline.stages.passable import STAGE

    with pytest.raises(StageError, match="rows layer missing"):
        STAGE.run(_context(tmp_work, with_rows=False))


def test_stage_requires_a_single_start(tmp_work: Path, runner_stub: None) -> None:
    from vineyard.pipeline.stages.passable import STAGE

    with pytest.raises(StageError, match="in_start"):
        STAGE.run(_context(tmp_work, with_start=False))


def test_stage_requires_annset(tmp_work: Path, runner_stub: None) -> None:
    from dataclasses import replace

    from vineyard.pipeline.stages.passable import STAGE

    with pytest.raises(StageError, match="--annset"):
        STAGE.run(replace(_context(tmp_work), annset_ref=None))
