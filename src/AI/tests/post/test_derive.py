"""annset.derive orchestration: layers, gap statistics, coverage, acceptance on AnnSet(reference)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry

from tests.post.factories import BlockSpec, build_annset, default_config, make_annset, reference_annset
from vineyard.annset.derive import (
    DERIVED_LAYERS,
    DeriveInputs,
    DeriveParams,
    LayerProv,
    build_coverage,
    derive,
    total_length_by_block,
)
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.contracts.schemas import validate_layer
from vineyard.errors import SchemaError, StageError
from vineyard.geo.tiling import CRS_EPSG, tile_box, tile_ref
from vineyard.route.target_gaps import RowGap

T11, T12 = "siret3_r018_c011", "siret3_r018_c012"
RUN_ID = "20260926T0100-post-abcdef"


@pytest.fixture(scope="module")
def params() -> DeriveParams:
    return DeriveParams.from_config(default_config())


class FakeGaps:
    """GapFn returning fixed along-axis gaps and recording what it was called with."""

    def __init__(self, gaps: Sequence[RowGap]) -> None:
        self.gaps = tuple(gaps)
        self.calls: list[tuple[LineString, int, BaseGeometry | None, float]] = []

    def __call__(self, axis: LineString, canopies: Sequence[Polygon], unknown: BaseGeometry | None,
                 min_length_m: float) -> tuple[RowGap, ...]:
        self.calls.append((axis, len(canopies), unknown, min_length_m))
        return self.gaps


def test_params_from_config(params) -> None:
    cfg = default_config()
    assert params.merge.join_lateral_max_m == cfg.derive.join_lateral_max_m
    assert params.blocks.outline_buffer_m == cfg.blocks.outline_buffer_m == 2.5
    assert params.blocks.min_rows_per_block == 3
    assert params.link.max_offset_m == 3.8 and params.link.seam_close_m == cfg.route.domain.seam_close_m
    assert params.gap_ge_m == 5.0 and params.corridor_half_m == pytest.approx(0.3)


def test_synthetic_derive_layers_validate(params) -> None:
    annset = make_annset(BlockSpec(n_rows=4))
    result = derive(DeriveInputs(annset, build_coverage(annset.meta.tile_ids)), params, run_id=RUN_ID)
    for name in DERIVED_LAYERS:
        validate_layer(result.layer(name), name)
    assert result.issues == ()
    assert list(result.rows["row_id"]) == [f"V01-R00{k}" for k in range(1, 5)]
    assert list(result.rows["row_index"]) == [1, 2, 3, 4]
    assert set(result.rows["run_id"]) == {RUN_ID} and set(result.rows["source"]) == {"reference"}
    assert result.rows["max_gap_m"].isna().all() and result.rows["n_gaps_ge5"].isna().all()
    per_row = annset.canopies.groupby("row_id").size().to_dict()  # canopies cut by the tile edge count twice
    assert dict(zip(result.rows["row_id"], result.rows["plant_count"], strict=True)) == per_row
    assert list(result.interrows["interrow_id"]) == ["V01-I001", "V01-I002", "V01-I003"]
    assert list(result.interrow_pieces_linked["piece_id"]) == sorted(result.interrow_pieces_linked["piece_id"])
    assert result.blocks["vineyard_id"].tolist() == ["V01"]
    metrics = result.metrics()
    assert metrics["n_rows"] == 4 and metrics["n_interrow_pieces_linked"] == 6 and metrics["n_interrows"] == 3
    assert metrics["row_length_m"] == pytest.approx(240.0)


def test_layer_rejects_unknown_names(params) -> None:
    result = derive(DeriveInputs(make_annset(BlockSpec(n_rows=3))), params)
    with pytest.raises(SchemaError):
        result.layer("targets")


def test_run_id_defaults_to_the_annset_run(params) -> None:
    annset = make_annset(BlockSpec(n_rows=3))
    result = derive(DeriveInputs(annset), params)
    assert set(result.blocks["run_id"]) == {annset.meta.run_id}


def test_gap_function_fills_interior_gap_stats(params) -> None:
    annset = make_annset(BlockSpec(n_rows=3, length_m=40.0))
    gaps = FakeGaps([RowGap(0.0, 7.0, "head"), RowGap(10.0, 16.0, "interior"), RowGap(20.0, 22.0, "interior"),
                     RowGap(25.0, 30.5, "interior", censored=True)])
    result = derive(DeriveInputs(annset, build_coverage(annset.meta.tile_ids)), params, gap_fn=gaps)
    assert result.rows["max_gap_m"].tolist() == pytest.approx([6.0] * 3)
    assert result.rows["n_gaps_ge5"].tolist() == [2, 2, 2]
    axis, n_canopies, unknown, min_len = gaps.calls[0]
    assert axis.length == pytest.approx(40.0) and n_canopies == 40 and min_len == 0.0
    assert unknown is not None and unknown.intersection(axis).length == 0.0  # the row is fully imaged


def test_unknown_marks_rows_outside_coverage(params) -> None:
    annset = make_annset(BlockSpec(n_rows=2, length_m=60.0))
    gaps = FakeGaps([])
    derive(DeriveInputs(annset, build_coverage([T11])), params, gap_fn=gaps)
    axis, _, unknown, _ = gaps.calls[0]
    assert unknown.intersection(axis).length == pytest.approx(60.0 - (629606.4 - 629560.0), abs=1e-6)


def test_gap_function_without_coverage_gets_no_unknown(params) -> None:
    gaps = FakeGaps([])
    result = derive(DeriveInputs(make_annset(BlockSpec(n_rows=3, length_m=30.0))), params, gap_fn=gaps)
    assert all(call[2] is None for call in gaps.calls)
    assert result.rows["max_gap_m"].tolist() == [0.0] * 3 and result.rows["n_gaps_ge5"].tolist() == [0] * 3


def test_bad_gap_function_output_raises(params) -> None:
    bad = FakeGaps([RowGap(10.0, 99.0, "interior")])
    with pytest.raises(StageError):
        derive(DeriveInputs(make_annset(BlockSpec(n_rows=3, length_m=30.0))), params, gap_fn=bad)


def test_issues_merge_every_source(params) -> None:
    annset = make_annset(BlockSpec(n_rows=2, length_m=30.0))
    pieces = annset.row_pieces
    annset = annset.with_layer("row_pieces", pieces.assign(row_structure=["Regular"] + list(pieces["row_structure"][1:])))
    result = derive(DeriveInputs(annset), params)
    assert sorted(i.code for i in result.issues) == ["block_too_few_rows", "enum_normalized"]
    qa = LayerProv.of(annset, RUN_ID).qa_layer(result.issues)
    validate_layer(qa, "qa_issues")
    assert list(qa["issue_id"]) == ["Q00001", "Q00002"] and set(qa["run_id"]) == {RUN_ID}


def test_empty_annset_gives_empty_layers(params) -> None:
    result = derive(DeriveInputs(build_annset(tile_ids=[T11])), params, gap_fn=FakeGaps([]))
    assert all(len(result.layer(name)) == 0 for name in DERIVED_LAYERS)
    assert result.issues == ()
    assert result.metrics()["n_rows"] == 0


def test_build_coverage_boxes_and_tile_valid() -> None:
    boxes = build_coverage([T12, T11, T11])
    assert list(boxes) == [T11, T12] and boxes[T11].equals(tile_box(tile_ref(T11)))
    half = box(*tile_box(tile_ref(T11)).bounds[:2], 629580.0, 5220300.8)
    valid = gpd.GeoDataFrame({"tile_id": [T11, T12], "valid_frac": [0.5, 1.0]},
                             geometry=[half, tile_box(tile_ref(T12))], crs=CRS_EPSG)
    cov = build_coverage([T11, T12], valid)
    assert cov[T11].area == pytest.approx(half.area) and cov[T12].area == pytest.approx(51.2 * 51.2)
    with pytest.raises(SchemaError):
        build_coverage(["siret3_r018_c013"], valid)


def test_input_annset_is_not_mutated(params) -> None:
    annset = make_annset(BlockSpec(n_rows=3))
    before = {name: annset.layer(name).copy() for name in ("canopies", "row_pieces", "interrow_pieces")}
    derive(DeriveInputs(annset), params)
    assert all(annset.layer(name).equals(frame) for name, frame in before.items())


def test_model_source_ids_validate_strictly(params) -> None:
    from tests.post.factories import Prov

    annset = make_annset(BlockSpec(n_rows=3), prov=Prov(Source.MODEL, "20260926T0000-model-000000", "pipe@x"))
    result = derive(DeriveInputs(annset), params, run_id=RUN_ID)
    for name in DERIVED_LAYERS:
        validate_layer(result.layer(name), name, strict_ids=True)


@pytest.mark.examples
def test_reference_acceptance(params, examples_xml: bytes) -> None:
    annset = reference_annset(examples_xml)
    result = derive(DeriveInputs(annset, build_coverage(annset.meta.tile_ids)), params, run_id=RUN_ID)
    assert result.issues == ()
    assert len(result.rows) == 51 and len(result.blocks) == 2
    linked = result.interrow_pieces_linked
    assert len(linked) == 49 and linked["interrow_id"].notna().all()
    assert len(result.interrows) == 49
    by_block = total_length_by_block(result.rows)
    assert by_block["V01"] == pytest.approx(910.10, abs=0.005)
    assert by_block["V02"] == pytest.approx(1031.45, abs=0.005)
    assert result.rows["length_m"].sum() == pytest.approx(1941.55, abs=0.01)
    counts = dict(zip(result.rows["row_id"], result.rows["plant_count"], strict=True))
    assert sum(counts.values()) == 650
    assert all(not math.isnan(v) for v in result.rows["spacing_next_m"].iloc[:24])


# ------------------------------------------------------------------ stage `derive`

SRC_RUN = "20260926T0000-reference-000000"


@pytest.fixture
def runner_module(monkeypatch: pytest.MonkeyPatch):
    """The real pipeline.runner when present, else a stub exposing StageResult (docs/design/01 §2.4)."""
    import dataclasses
    import importlib
    import sys
    import types

    try:
        return importlib.import_module("vineyard.pipeline.runner")
    except ImportError:
        stub = types.ModuleType("vineyard.pipeline.runner")

        @dataclasses.dataclass(frozen=True)
        class StageResult:
            stage: str
            n_items: int
            n_cached: int
            n_failed: int
            failed: tuple[str, ...] = ()
            outputs: tuple = ()
            metrics: dict = dataclasses.field(default_factory=dict)

        stub.StageResult = StageResult
        monkeypatch.setitem(sys.modules, "vineyard.pipeline.runner", stub)
        return stub


def _input_layer(geom: BaseGeometry) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"fid": [1], "type": [None], "name": [None], "source": [None]}, geometry=[geom],
                            crs=CRS_EPSG)


def _work(tmp_path, *, route_dir=None, static_route: bool = True, tile_valid: bool = True):
    from vineyard.annset.io import write_annset
    from vineyard.geo.vector_io import write_layer
    from vineyard.pipeline.context import make_run_paths

    work = tmp_path / "work"
    sets = [f"paths.work_dir={work}"] + ([f"paths.route_dir={route_dir}"] if route_dir else [])
    cfg = load_config(overrides=sets)
    annset = make_annset(BlockSpec(n_rows=3))
    src = make_run_paths(cfg, SRC_RUN)
    src.annset_dir.mkdir(parents=True)
    write_annset(annset, src.annset_dir)
    layers = work / "layers"
    layers.mkdir(parents=True)
    if tile_valid:
        ids = list(annset.meta.tile_ids)
        valid = gpd.GeoDataFrame({"tile_id": ids, "valid_frac": [1.0] * len(ids)},
                                 geometry=[tile_box(tile_ref(t)) for t in ids], crs=CRS_EPSG)
        write_layer(valid, "tile_valid", layers / "tile_valid.parquet")
    if static_route:
        write_layer(_input_layer(box(629500, 5220250, 629510, 5220260)), "in_passages", layers / "in_passages.parquet")
        write_layer(_input_layer(box(629400, 5220250, 629410, 5220260)), "in_forbidden", layers / "in_forbidden.parquet")
    return cfg


def _ctx(cfg, *, run_id: str = RUN_ID, annset_ref: str | None = SRC_RUN):
    from vineyard.pipeline.context import new_run_context

    return new_run_context(cfg, source=Source.REFERENCE, kind="post", run_id=run_id, annset_ref=annset_ref,
                           workers=1)


def test_stage_spec_is_registered() -> None:
    from vineyard.config import cfg_hash
    from vineyard.pipeline.registry import load_stage

    spec = load_stage("derive")
    assert spec.name == "derive" and spec.scope == "global" and spec.requires == ()
    assert len(cfg_hash(default_config(), spec.cfg_keys)) == 40


def test_stage_run_writes_the_post_run(tmp_path, runner_module) -> None:
    from vineyard.geo.vector_io import read_layer
    from vineyard.pipeline.stages.derive import STAGE

    ctx = _ctx(_work(tmp_path))
    result = STAGE.run(ctx)
    assert isinstance(result, runner_module.StageResult)
    assert result.stage == "derive" and result.n_items == 3 and result.n_failed == 0
    assert result.metrics["n_interrow_pieces_linked"] == 4
    for name in DERIVED_LAYERS:
        path = ctx.paths.layers_dir / f"{name}.parquet"
        assert path in result.outputs and len(read_layer(path, name)) > 0
    qa = read_layer(ctx.paths.qa_dir / "issues_derive.parquet", "qa_issues")
    assert len(qa) == 0
    doc = json.loads((ctx.paths.metrics_dir / "derive.json").read_text(encoding="utf-8"))
    assert doc["annset_ref"] == SRC_RUN and doc["row_length_m_by_block"]["V01"] == pytest.approx(180.0)
    assert not (ctx.paths.runs_dir / SRC_RUN / "layers").exists()


def test_stage_refuses_the_annset_run(tmp_path, runner_module) -> None:
    from vineyard.pipeline.stages.derive import STAGE

    with pytest.raises(StageError, match="never write"):
        STAGE.run(_ctx(_work(tmp_path), run_id=SRC_RUN))


def test_stage_needs_annset_and_tile_valid(tmp_path, runner_module) -> None:
    from vineyard.pipeline.stages.derive import STAGE

    cfg = _work(tmp_path, tile_valid=False)
    with pytest.raises(StageError, match="--annset"):
        STAGE.run(_ctx(cfg, annset_ref=None))
    with pytest.raises(StageError, match="tile_valid"):
        STAGE.run(_ctx(cfg))


def test_route_inputs_fall_back_to_geojson(tmp_path) -> None:
    from vineyard.geo.vector_io import write_geojson
    from vineyard.pipeline.stages.derive import read_route_input

    route = tmp_path / "route"
    route.mkdir()
    write_geojson(_input_layer(box(0, 0, 2, 2)), route / "passages.geojson")
    ctx = _ctx(_work(tmp_path, route_dir=route, static_route=False))
    assert read_route_input(ctx, "passages").area == pytest.approx(4.0)
    with pytest.raises(StageError, match="route input missing"):
        read_route_input(ctx, "forbidden")


def test_gap_function_needs_the_engine(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vineyard.pipeline.stages import derive as stage

    ctx = _ctx(_work(tmp_path))
    monkeypatch.setattr(stage, "module_available", lambda name: False)
    assert stage.gap_function(ctx) is None
    monkeypatch.setattr(stage, "module_available", lambda name: True)
    assert callable(stage.gap_function(ctx))
