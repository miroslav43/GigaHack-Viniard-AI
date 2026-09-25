"""build_targets: reference acceptance (8 GAP / 0 END), synthetic long gaps, waste reachability, nodata, ids."""

from __future__ import annotations

import json
import math
import re
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, box

from tests.post.factories import BlockSpec, Prov, layer_frame, make_annset, reference_annset
from tests.post.tgt_helpers import reference_gap_fn, rows_from_pieces, tile_coverage
from vineyard.annset.io import write_annset
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.contracts.schemas import validate_layer
from vineyard.errors import StageError
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.pipeline.context import ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.route.targets import (
    TargetInputs,
    TargetProvenance,
    TargetSettings,
    build_targets,
    reachability,
)

PROV = TargetProvenance(Source.REFERENCE, "20260926T1200-post-abcdef", "reference-examples@test")


def _settings(*overrides: str) -> TargetSettings:
    return TargetSettings.from_config(load_config(overrides=overrides, environ={}))


def _inputs(ann, **extra) -> TargetInputs:
    return TargetInputs(rows=rows_from_pieces(ann), canopies=ann.canopies, waste=ann.waste,
                        coverage=extra.pop("coverage", tile_coverage(ann)), **extra)


@pytest.fixture(scope="module")
def reference(examples_xml):
    return reference_annset(examples_xml)


@pytest.fixture(scope="module")
def reference_targets(reference):
    settings = _settings("targets.include_missing=false", "targets.include_sparse=false")
    return build_targets(_inputs(reference), settings, PROV, reference_gap_fn)


# ------------------------------------------------------------------ reference acceptance


@pytest.mark.examples
def test_reference_gives_exactly_the_8_gap_targets(reference_targets):
    t = reference_targets.targets
    gaps = t[t["kind"] == "row_gap"]
    assert len(gaps) == 8
    assert gaps.groupby("row_id").size().to_dict() == {"V02-R06": 1, "V02-R08": 3, "V02-R09": 3, "V02-R15": 1}
    assert set(t["kind"]) == {"row_gap"}  # 0 END, 0 MRW, no waste on the examples
    assert sorted(np.round(gaps["gap_length_m"].astype(float), 3)) == sorted(
        [5.054, 7.09, 9.25, 7.745, 12.683, 7.812, 12.903, 5.009])


@pytest.mark.examples
def test_reference_gap_targets_lie_on_their_axis(reference, reference_targets):
    rows = rows_from_pieces(reference).set_index("row_id")
    for tgt in reference_targets.targets.itertuples():
        assert rows.loc[tgt.row_id, "geometry"].distance(Point(tgt.x, tgt.y)) <= 0.05
        assert tgt.geometry.equals(Point(tgt.x, tgt.y))


@pytest.mark.examples
def test_reference_priorities_ids_and_order(reference_targets):
    t = reference_targets.targets
    assert sorted(t.loc[t["priority"] == 1, "gap_length_m"].astype(float).round(2)) == [12.68, 12.9]
    assert all(re.fullmatch(r"T-GAP-\d{4}", tid) for tid in t["target_id"])
    assert list(t["target_id"]) == [f"T-GAP-{k:04d}" for k in range(1, 9)]
    keys = list(zip(t["vineyard_id"], t["row_id"], t["along_m"], strict=True))
    assert keys == sorted(keys)
    assert set(t["route_role"]) == {"must"}


@pytest.mark.examples
def test_reference_layers_are_contract_valid(reference_targets):
    validate_layer(reference_targets.targets, "targets")
    validate_layer(reference_targets.extents, "target_extents")
    ext = reference_targets.extents.set_index("target_id")
    assert set(ext.index) == set(reference_targets.targets["target_id"])
    assert reference_targets.counts["row_gap"] == 8


@pytest.mark.examples
def test_reference_missing_plants(reference):
    result = build_targets(_inputs(reference), _settings("targets.include_sparse=false"), PROV, reference_gap_fn)
    msp = result.targets[result.targets["kind"] == "missing_plant"]
    per_tile = msp.groupby("tile_id").size().to_dict()
    assert abs(per_tile["siret3_r021_c012"] - 22) <= 2
    assert abs(per_tile["siret3_r006_c004"] - 23) <= 2
    assert set(msp["route_role"]) == {"optional"}
    assert set(msp["priority"]) == {3}


# ------------------------------------------------------------------ synthetic blocks


def _synthetic(gaps, **spec):
    return make_annset(BlockSpec(gaps=gaps, **spec))


def test_synthetic_long_gap_samples_and_extents():
    ann = _synthetic(((2, 10.0, 50.0),), length_m=80.0)
    result = build_targets(_inputs(ann), _settings("targets.include_sparse=false"), PROV, reference_gap_fn)
    gaps = result.targets[result.targets["kind"] == "row_gap"]
    assert len(gaps) == 3
    lengths = result.extents.set_index("target_id").loc[list(gaps["target_id"]), "length_m"]
    assert math.isclose(lengths.sum(), float(gaps["gap_length_m"].iloc[0]), abs_tol=1e-4)
    assert set(gaps["row_id"]) == {"V01-R002"}


def test_row_ends_on_the_coverage_boundary_give_no_end_targets():
    spec = BlockSpec(n_rows=3, length_m=(60.0, 40.0, 60.0))
    ann = make_annset(spec)
    rows_end_on_coverage = build_targets(_inputs(ann), _settings(), PROV, reference_gap_fn)
    assert "row_end_short" in set(rows_end_on_coverage.targets["kind"])
    inner_coverage = shapely.union_all([tile_box(tile_ref(t)) for t in sorted(ann.tile_ids())])
    shrunk = inner_coverage.intersection(box(*spec.point(1, -1.0), *spec.point(3, 41.0)).envelope)
    none = build_targets(_inputs(ann, coverage=shrunk.buffer(0)), _settings(), PROV, reference_gap_fn)
    assert "row_end_short" not in set(none.targets["kind"])


def test_missing_row_targets_on_a_skipped_row():
    ann = make_annset(BlockSpec(n_rows=5, skip_rows=(3,), length_m=40.0), linked=True)
    result = build_targets(_inputs(ann), _settings(), PROV, reference_gap_fn)
    mrw = result.targets[result.targets["kind"] == "missing_row"]
    assert len(mrw) >= 1
    spec = BlockSpec(n_rows=5, skip_rows=(3,), length_m=40.0)
    assert all(spec.axis(3).distance(Point(x, y)) < 0.05 for x, y in zip(mrw["x"], mrw["y"], strict=True))
    assert mrw["row_id"].isna().all()


def test_no_target_in_a_nodata_hole():
    ann = _synthetic(((2, 20.0, 32.0),), length_m=60.0)
    spec = BlockSpec()
    hole = box(*spec.point(4, 24.0, -2.0), *spec.point(1, 28.0, 2.0)).envelope
    coverage = tile_coverage(ann).difference(hole)
    result = build_targets(_inputs(ann, coverage=coverage), _settings("targets.include_sparse=false"), PROV,
                           reference_gap_fn)
    pts = shapely.points(result.targets["x"], result.targets["y"])
    assert not shapely.intersects(pts, hole).any()
    assert not (result.targets["kind"] == "row_gap").any()  # 4 m + 4 m of known gap: below 5 m


def test_censored_gap_is_kept_when_known_part_is_long_enough():
    ann = _synthetic(((2, 10.0, 40.0),), length_m=60.0)
    spec = BlockSpec()
    hole = box(*spec.point(4, 24.0, -2.0), *spec.point(1, 26.0, 2.0)).envelope
    coverage = tile_coverage(ann).difference(hole)
    result = build_targets(_inputs(ann, coverage=coverage), _settings("targets.include_sparse=false"), PROV,
                           reference_gap_fn)
    gaps = result.targets[result.targets["kind"] == "row_gap"]
    assert len(gaps) == 2
    assert all("censored" in r for r in gaps["reason"])
    assert not shapely.intersects(shapely.points(gaps["x"], gaps["y"]), hole).any()


# ------------------------------------------------------------------ waste, reachability, dedupe


def test_waste_reachability_forbidden_and_too_far():
    spec = BlockSpec()
    tile = "siret3_r018_c011"
    inside = tuple(spec.point(1, 10.0, -1.25))
    in_forbidden = tuple(spec.point(1, 20.0, -1.25))
    far = tuple(spec.point(1, 30.0, 6.0))
    ann = make_annset(spec, waste=[(tile, inside, 0.5, "V01"), (tile, in_forbidden, 0.5, "V01"),
                                   (tile, far, 0.5, "")])
    domain = shapely.union_all(list(ann.interrow_pieces.geometry)).buffer(-0.05)
    forbidden = Point(in_forbidden).buffer(1.0)
    result = build_targets(_inputs(ann, reach_domain=domain, forbidden=forbidden), _settings(), PROV,
                           reference_gap_fn)
    wst = result.targets[result.targets["kind"] == "waste"].set_index("waste_id")
    assert wst.loc["W0001", ["reachable", "reach_note"]].tolist() == [True, ""]
    assert wst.loc["W0002", ["reachable", "reach_note"]].tolist() == [False, "in_forbidden"]
    assert wst.loc["W0003", ["reachable", "reach_note"]].tolist() == [False, "too_far"]
    assert wst.loc["W0003", "snap_dist_m"] > 3.0
    assert wst.loc["W0001", "snap_dist_m"] == 0.0
    assert {i.code for i in result.issues} == {"target_unreachable"}
    assert sorted(i.object_id for i in result.issues) == sorted(wst[~wst["reachable"]]["target_id"])


def test_without_domain_reachability_is_unchecked():
    ann = _synthetic(((2, 20.0, 30.0),))
    result = build_targets(_inputs(ann), _settings(), PROV, reference_gap_fn)
    assert set(result.targets["reach_note"]) == {"unchecked"}
    assert result.targets["reachable"].all()
    assert result.targets["snap_dist_m"].isna().all()


def _kinds(result) -> dict[str, int]:
    return {k: int(n) for k, n in result.targets["kind"].value_counts().items()}


def test_dedupe_keeps_the_best_priority():
    # R002 gap 19.9-30.1 (GAP at 25.0, priority 1); R003 gap 21.9-26.1 (MSP at 24.0): 2.69 m apart.
    ann = _synthetic(((2, 20.0, 30.0), (3, 22.0, 26.0)))
    loose = build_targets(_inputs(ann), _settings("targets.dedupe_m=1.0"), PROV, reference_gap_fn)
    tight = build_targets(_inputs(ann), _settings("targets.dedupe_m=3.0"), PROV, reference_gap_fn)
    assert _kinds(loose) == {"row_gap": 1, "missing_plant": 1} and loose.counts["deduped"] == 0
    assert _kinds(tight) == {"row_gap": 1} and tight.counts["deduped"] == 1
    assert list(tight.targets["target_id"]) == ["T-GAP-0001"]


def test_dedupe_never_touches_waste():
    spec = BlockSpec(gaps=((2, 20.0, 30.0),))
    centre = tuple(spec.point(2, 25.0, 0.3))
    with_waste = make_annset(spec, waste=[("siret3_r018_c011", centre, 0.4, "V01")])
    both = build_targets(_inputs(with_waste), _settings("targets.dedupe_m=3.0"), PROV, reference_gap_fn)
    assert _kinds(both) == {"row_gap": 1, "waste": 1} and both.counts["deduped"] == 0


def test_build_targets_is_deterministic():
    ann = _synthetic(((2, 20.0, 30.0), (3, 5.0, 8.0)))
    one = build_targets(_inputs(ann), _settings(), PROV, reference_gap_fn)
    two = build_targets(_inputs(ann), _settings(), PROV, reference_gap_fn)
    assert one.targets.to_wkb().equals(two.targets.to_wkb())
    assert one.extents.to_wkb().equals(two.extents.to_wkb())


def test_empty_inputs_give_empty_valid_layers():
    ann = make_annset(BlockSpec(n_rows=2, length_m=0.1 + 0.9))
    empty_rows = rows_from_pieces(ann).iloc[0:0]
    inputs = TargetInputs(rows=empty_rows, canopies=ann.canopies.iloc[0:0], waste=ann.waste,
                          coverage=tile_coverage(ann))
    result = build_targets(inputs, _settings(), PROV, reference_gap_fn)
    assert len(result.targets) == 0 and len(result.extents) == 0
    validate_layer(result.targets, "targets")
    validate_layer(result.extents, "target_extents")


def test_rows_layer_must_have_the_needed_columns():
    ann = _synthetic(())
    rows = rows_from_pieces(ann).drop(columns=["row_index"])
    with pytest.raises(Exception, match="row_index"):
        build_targets(TargetInputs(rows=rows, canopies=ann.canopies, waste=ann.waste, coverage=tile_coverage(ann)),
                      _settings(), PROV, reference_gap_fn)


def test_reachability_without_forbidden_and_empty_domain():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    reach = reachability(xy, box(-1, -1, 1, 1), None, 1.9)
    assert list(reach.reachable) == [True, False] and reach.note == ("", "too_far")
    empty = reachability(xy, Polygon(), Point(10, 0).buffer(1), 1.9)
    assert list(empty.reachable) == [True, False] and empty.note == ("unchecked", "in_forbidden")


def test_settings_reject_non_positive_values():
    cfg = load_config(environ={})
    with pytest.raises(ValueError):
        TargetSettings(cfg.targets, corridor_half_m=0.0, reach_radius_m=1.9)
    no_sparse = _settings("targets.include_sparse=false", "targets.include_missing=false")
    assert no_sparse.engine_min_length_m == 5.0 and _settings().engine_min_length_m == 0.0


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


@pytest.fixture
def stage_ctx(tmp_path, monkeypatch):
    runner = types.ModuleType("vineyard.pipeline.runner")
    runner.StageResult = _FakeStageResult  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vineyard.pipeline.runner", runner)
    spec = load_stage("targets")
    monkeypatch.setattr(sys.modules[spec.run.__module__], "make_gap_fn", lambda ctx: reference_gap_fn)
    cfg = load_config(environ={"VINEYARD_WORK_DIR": str(tmp_path / "work")})
    ann = make_annset(BlockSpec(gaps=((2, 20.0, 30.0),)), prov=Prov(Source.MARCAJ, REF, "m"),
                      waste=[("siret3_r018_c011", tuple(BlockSpec().point(1, 10.0, -1.25)), 0.5, "V01")])
    write_annset(ann, cfg.paths.work_dir / "runs" / REF / "annset")
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", annset_ref=REF, run_id="20260926T0310-post-abcdef")
    ensure_run_dirs(ctx.paths)
    write_layer(rows_from_pieces(ann), "rows", ctx.paths.layers_dir / "rows.parquet")
    return spec, ctx, ann


def _domain_layer(ann):
    union = shapely.union_all(list(ann.interrow_pieces.geometry))
    raw = union if isinstance(union, MultiPolygon) else MultiPolygon([union])
    return layer_frame("passable_domain", [{"domain_id": "D1", "area_m2": raw.area, "erosion_m": 0.0,
                                             "n_components": len(raw.geoms), "geometry": raw}], Prov(Source.MARCAJ))


def test_stage_writes_layers_issues_and_metrics(stage_ctx):
    spec, ctx, _ = stage_ctx
    result = spec.run(ctx)
    assert (spec.name, spec.scope, result.stage, result.n_failed) == ("targets", "global", "targets", 0)
    targets = read_layer(ctx.paths.layers_dir / "targets.parquet", "targets")
    read_layer(ctx.paths.layers_dir / "target_extents.parquet", "target_extents")
    assert set(targets["kind"]) == {"row_gap", "waste"} and result.n_items == len(targets)
    assert set(targets["reach_note"]) == {"unchecked"} and set(targets["source"]) == {"marcaj"}
    assert set(targets["run_id"]) == {ctx.run_id}
    assert result.metrics["n_row_gap"] == 1.0 and result.metrics["reach_checked"] == 0.0
    doc = json.loads((ctx.paths.metrics_dir / "targets.json").read_text(encoding="utf-8"))
    assert doc["rows_run"] == ctx.run_id and doc["tile_valid_used"] is False
    assert (ctx.paths.qa_dir / "issues_targets.parquet").is_file()


def test_stage_checks_reachability_against_the_passable_domain(stage_ctx):
    spec, ctx, ann = stage_ctx
    write_layer(_domain_layer(ann), "passable_domain", ctx.paths.layers_dir / "passable_domain.parquet")
    spec.run(ctx)
    targets = read_layer(ctx.paths.layers_dir / "targets.parquet", "targets")
    assert targets["reachable"].all() and set(targets["reach_note"]) == {""}


def test_standalone_stage_reads_rows_of_the_previous_post_run(stage_ctx):
    spec, ctx, _ = stage_ctx
    (ctx.paths.run_json).write_text(json.dumps({"annset_ref": REF}), encoding="utf-8")
    fresh = new_run_context(ctx.cfg, source=Source.MARCAJ, kind="post", annset_ref=REF,
                            run_id="20260926T0400-post-bbbbbb")
    ensure_run_dirs(fresh.paths)
    spec.run(fresh)
    assert (fresh.paths.layers_dir / "targets.parquet").is_file()


def test_standalone_stage_ignores_post_runs_of_other_annsets(stage_ctx):
    spec, ctx, _ = stage_ctx
    (ctx.paths.run_json).write_text(json.dumps({"annset_ref": "20260926T0000-marcaj-bbbbbb"}), encoding="utf-8")
    orphan = new_run_context(ctx.cfg, source=Source.MARCAJ, kind="post", annset_ref=REF,
                             run_id="20260926T0400-post-cccccc")
    with pytest.raises(StageError, match="no post run"):
        spec.run(orphan)


def test_model_source_ids_validate_strictly():
    ann = make_annset(BlockSpec(gaps=((2, 20.0, 30.0),)), prov=Prov(Source.MODEL))
    prov = TargetProvenance(Source.MODEL, "20260926T1200-post-abcdef", "pipe@test")
    result = build_targets(_inputs(ann), _settings(), prov, reference_gap_fn)
    validate_layer(result.targets, "targets")
    assert set(result.targets["source"]) == {"model"}
