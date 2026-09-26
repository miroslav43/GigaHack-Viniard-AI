"""measurements.csv / .json in the web contract format (src/Web/CLAUDE.md §6.4) and the `measure` stage."""

from __future__ import annotations

import csv
import io
import json
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from shapely.geometry import box

from tests.post.factories import (
    DEFAULT_ORIGIN,
    BlockSpec,
    Prov,
    build_annset,
    canopy_record,
    make_annset,
    reference_annset,
)
from tests.post.tgt_helpers import rows_from_pieces
from vineyard.annset.io import write_annset
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.measure.csv_format import (
    HEADER_LINE,
    MeasurementRecord,
    check_measurements_bytes,
    check_measurements_csv,
    decimals_for,
    describe_failed,
    format_measurements_csv,
    parse_measurements_csv,
)
from vineyard.measure.measurements import (
    MeasureInputs,
    aggregate_structure,
    compute_measurements,
    measurements_json,
    union_area,
)
from vineyard.pipeline.context import ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage

EXPECTED_HEADER = ("level,vineyard_id,row_id,block_count,row_count,row_length_m,canopy_area_m2,canopy_area_ha,"
                   "interrow_area_m2,interrow_area_ha,plant_count,row_structure")


def _inputs(ann, **extra) -> MeasureInputs:
    return MeasureInputs(canopies=ann.canopies, row_pieces=ann.row_pieces, interrow_pieces=ann.interrow_pieces,
                         waste=ann.waste, **extra)


def _table(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


@pytest.fixture(scope="module")
def reference(examples_xml):
    return reference_annset(examples_xml)


@pytest.fixture(scope="module")
def reference_measurements(reference):
    return compute_measurements(_inputs(reference, rows=rows_from_pieces(reference)))


# ------------------------------------------------------------------ reference acceptance


@pytest.mark.examples
def test_reference_survey_row(reference_measurements):
    s = reference_measurements.survey
    assert (s.level, s.block_count, s.row_count, s.plant_count) == ("survey", 2, 51, 650)
    assert s.row_length_m == pytest.approx(1941.55, abs=0.01)
    assert s.canopy_area_m2 == pytest.approx(237.12 + 299.06, abs=0.02)
    assert s.interrow_area_m2 == pytest.approx(2068.03 + 1996.36, abs=0.05)


@pytest.mark.examples
def test_reference_block_rows(reference_measurements):
    blocks = {b.vineyard_id: b for b in reference_measurements.blocks}
    assert list(blocks) == ["V01", "V02"]
    v01, v02 = blocks["V01"], blocks["V02"]
    assert (v01.row_count, v01.plant_count, v02.row_count, v02.plant_count) == (25, 399, 26, 251)
    assert v01.row_length_m == pytest.approx(910.10, abs=0.01)
    assert v02.row_length_m == pytest.approx(1031.45, abs=0.01)
    assert v01.canopy_area_m2 == pytest.approx(237.12, abs=0.01)
    assert v02.canopy_area_m2 == pytest.approx(299.06, abs=0.01)
    assert v01.interrow_area_m2 == pytest.approx(2068.03, abs=0.01)
    assert v02.interrow_area_m2 == pytest.approx(1996.36, abs=0.01)
    assert all(b.block_count is None and b.row_structure is None for b in blocks.values())


@pytest.mark.examples
def test_reference_row_records(reference_measurements):
    rows = {r.row_id: r for r in reference_measurements.rows}
    assert len(rows) == 51
    assert [r.row_id for r in reference_measurements.rows][:3] == ["V01-R01", "V01-R02", "V01-R03"]
    assert rows["V02-R09"].row_structure == "disrupted"
    assert sum(r.row_structure == "disrupted" for r in rows.values()) == 5
    assert (rows["V02-R01"].plant_count, rows["V02-R17"].plant_count) == (2, 1)
    assert all(r.canopy_area_m2 is None and r.interrow_area_m2 is None for r in rows.values())
    assert reference_measurements.extras["row"]["V02-R09"]["max_gap_m"] is None  # no gap data on the pieces


@pytest.mark.examples
def test_reference_csv_text_is_exact(reference_measurements):
    text = format_measurements_csv(reference_measurements.records(), m_decimals=2, ha_decimals=4)
    lines = text.splitlines()
    assert lines[0] == EXPECTED_HEADER == HEADER_LINE
    assert len(lines) == 1 + 1 + 2 + 51 and text.endswith("\n")
    assert lines[1:4] == ["survey,,,2,51,1941.55,536.18,0.0536,4064.39,0.4064,650,",
                          "block,V01,,,25,910.10,237.12,0.0237,2068.03,0.2068,399,",
                          "block,V02,,,26,1031.45,299.06,0.0299,1996.36,0.1996,251,"]
    assert lines[4] == "row,V01,V01-R01,,,1.10,,,,,1,regular"
    assert check_measurements_csv(text, sum_tol_m=0.05, sum_tol_m2=0.05) == ()


def test_row_max_gap_prefers_the_global_row_value():
    ann = make_annset(BlockSpec(n_rows=2))
    pieces = ann.row_pieces.assign(max_gap_m=[3.0] * len(ann.row_pieces))
    rows = rows_from_pieces(ann).assign(max_gap_m=[7.5, float("nan")])
    m = compute_measurements(_inputs(ann.with_layer("row_pieces", pieces), rows=rows))
    assert m.extras["row"]["V01-R001"]["max_gap_m"] == pytest.approx(7.5)
    assert m.extras["row"]["V01-R002"]["max_gap_m"] == pytest.approx(3.0)


@pytest.mark.examples
def test_reference_json_carries_the_same_values(reference_measurements):
    doc = measurements_json(reference_measurements, {"run_id": "r1", "source": "reference"}, m_decimals=2,
                            ha_decimals=4)
    assert doc["header"] == EXPECTED_HEADER.split(",")
    assert doc["total"]["row_length_m"] == 1941.55 and doc["total"]["block_count"] == 2
    assert [b["vineyard_id"] for b in doc["blocks"]] == ["V01", "V02"] and len(doc["rows"]) == 51
    assert doc["total"]["interrow_area_sum_m2"] == pytest.approx(doc["total"]["interrow_area_m2"], abs=0.01)
    assert doc["meta"] == {"run_id": "r1", "source": "reference"}
    json.dumps(doc, allow_nan=False)


# ------------------------------------------------------------------ semantics on synthetic data


def test_union_area_removes_overlaps_and_crosses_tiles():
    a = canopy_record("siret3_r018_c011:C0001", "siret3_r018_c011", "V01", box(629600, 5220280, 629604, 5220281))
    b = canopy_record("siret3_r018_c011:C0002", "siret3_r018_c011", "V01", box(629602, 5220280, 629606, 5220281))
    # C0001 of the next tile reaches back over the tile edge (x = 629606.4) onto C0002.
    c = canopy_record("siret3_r018_c012:C0001", "siret3_r018_c012", "V01", box(629605, 5220280, 629608, 5220281))
    ann = build_annset([a, b, c])
    assert union_area(ann.canopies.iloc[:2]) == pytest.approx(6.0)
    assert union_area(ann.canopies) == pytest.approx(8.0)
    assert union_area(ann.canopies.iloc[0:0]) == 0.0


def test_overlapping_interrows_raise_a_union_sum_warning():
    ann = make_annset(BlockSpec(n_rows=3, length_m=40.0))
    dup = ann.interrow_pieces.iloc[[0]].assign(piece_id=["siret3_r018_c011:I099"])
    import pandas as pd
    doubled = ann.with_layer("interrow_pieces", pd.concat([ann.interrow_pieces, dup], ignore_index=True))
    m = compute_measurements(_inputs(doubled))
    assert m.survey.interrow_area_m2 == pytest.approx(compute_measurements(_inputs(ann)).survey.interrow_area_m2)
    assert {i.code for i in m.issues} == {"area_union_sum_mismatch"}
    assert compute_measurements(_inputs(ann)).issues == ()


def test_row_structure_aggregation_and_majority_block():
    assert aggregate_structure(["regular", "disrupted"]) == "disrupted"
    assert aggregate_structure(["unassessable", "unassessable"]) == "unassessable"
    assert aggregate_structure(["unassessable", "regular"]) == "regular"
    ann = make_annset(BlockSpec(n_rows=2, length_m=80.0))
    pieces = ann.row_pieces.copy()
    first = pieces.index[pieces["row_id"] == "V01-R001"][0]
    pieces.loc[first, "row_structure"] = "disrupted"
    m = compute_measurements(_inputs(ann.with_layer("row_pieces", pieces)))
    rows = {r.row_id: r for r in m.rows}
    assert rows["V01-R001"].row_structure == "disrupted" and rows["V01-R002"].row_structure == "regular"
    assert m.survey.row_length_m == pytest.approx(160.0) and m.blocks[0].row_length_m == pytest.approx(160.0)


def test_targets_and_waste_counts_go_to_json_extras():
    ann = make_annset(BlockSpec(n_rows=2), waste=[("siret3_r018_c011", (629565.0, 5220295.0), 0.5, "V01")])
    targets = _targets_frame(["V01-R001", None])
    m = compute_measurements(_inputs(ann, targets=targets))
    assert m.extras["survey"]["n_targets"] == 2 and m.extras["survey"]["n_waste"] == 1
    assert m.extras["block"]["V01"]["n_targets"] == 2 and m.extras["row"]["V01-R001"]["n_targets"] == 1


def _targets_frame(row_ids):
    import geopandas as gpd
    from shapely.geometry import Point
    return gpd.GeoDataFrame({"target_id": [f"T-GAP-{k:04d}" for k in range(1, len(row_ids) + 1)],
                             "vineyard_id": ["V01"] * len(row_ids), "row_id": row_ids},
                            geometry=[Point(629570.0, 5220290.0)] * len(row_ids), crs="EPSG:32635")


def test_empty_annset_gives_a_single_zero_survey_row():
    ann = build_annset(tile_ids=["siret3_r018_c011"])
    m = compute_measurements(_inputs(ann))
    text = format_measurements_csv(m.records(), m_decimals=2, ha_decimals=4)
    assert text.splitlines()[1:] == ["survey,,,0,0,0.00,0.00,0.0000,0.00,0.0000,0,"]


# ------------------------------------------------------------------ format helpers


@pytest.mark.parametrize(("step", "decimals"), [(0.01, 2), (0.0001, 4), (1.0, 0)])
def test_decimals_for(step, decimals):
    assert decimals_for(step) == decimals


@pytest.mark.parametrize("step", [0.03, 0.0, -0.01, 2.0])
def test_decimals_for_rejects_non_power_of_ten(step):
    with pytest.raises(ValueError):
        decimals_for(step)


def _csv(*lines: str) -> str:
    return "\n".join((EXPECTED_HEADER, *lines)) + "\n"


GOOD = ("survey,,,1,2,20.00,1.00,0.0001,2.00,0.0002,3,", "block,V01,,,2,20.00,1.00,0.0001,2.00,0.0002,3,",
        "row,V01,V01-R001,,,12.00,,,,,2,regular", "row,V01,V01-R002,,,8.00,,,,,1,disrupted")


def test_check_measurements_csv_accepts_a_consistent_table():
    assert check_measurements_csv(_csv(*GOOD), sum_tol_m=0.05, sum_tol_m2=0.05) == ()
    assert len(parse_measurements_csv(_csv(*GOOD))) == 4


@pytest.mark.parametrize(("text", "name"), [
    ("level,n_rows\nsurvey,1\n", "header"),
    (_csv(*GOOD, GOOD[0]), "survey_rows"),
    (_csv(GOOD[0], "block,V01,,,2,25.00,1.00,0.0001,2.00,0.0002,3,", *GOOD[2:]), "block_sum"),
    (_csv(GOOD[0], GOOD[1], GOOD[2], "row,V01,V01-R002,,,9.00,,,,,1,disrupted"), "row_sum"),
    (_csv(GOOD[0], "block,V01,,,2,20.00,1.00,0.0001,3.00,0.0003,3,", *GOOD[2:]), "block_interrow_sum"),
    (_csv("survey,,,1,2,abc,1.00,0.0001,2.00,0.0002,3,", *GOOD[1:]), "numbers"),
    (_csv(*GOOD, "total,,,,,,,,,,,"), "levels"),
    (_csv(*GOOD, "row,V01,V01-R003,,,0.00"), "cells"),
    (_csv(*GOOD, "row,V01,V01-R003,,,0.00,,,,,0,regular,extra"), "cells"),
    ("", "header"),
])
def test_check_measurements_csv_rejects(text, name):
    assert name in {c.name for c in check_measurements_csv(text, sum_tol_m=0.05, sum_tol_m2=0.05)}


def test_block_interrow_sum_allows_the_rounding_of_the_written_values():
    survey = "survey,,,2,2,20.00,1.00,0.0001,2.00,0.0002,3,"
    blocks = ("block,V01,,,1,12.00,0.50,0.0001,1.00,0.0001,2,", "block,V02,,,1,8.00,0.50,0.0001,1.01,0.0001,1,")
    rows = ("row,V01,V01-R001,,,12.00,,,,,2,regular", "row,V02,V02-R001,,,8.00,,,,,1,regular")
    assert check_measurements_csv(_csv(survey, *blocks, *rows), sum_tol_m=0.0, sum_tol_m2=0.0) == ()
    (failed,) = check_measurements_csv(_csv(survey, blocks[0], blocks[1].replace("1.01", "1.02"), *rows),
                                       sum_tol_m=0.0, sum_tol_m2=0.0)
    assert failed.name == "block_interrow_sum" and "sum(block.interrow_area_m2)=2.02" in failed.detail


def test_check_measurements_bytes_needs_utf8():
    assert check_measurements_bytes(_csv(*GOOD).encode("utf-8"), sum_tol_m=0.05, sum_tol_m2=0.05) == ()
    (failed,) = check_measurements_bytes(b"\xff\xfe" + _csv(*GOOD).encode("utf-8"), sum_tol_m=0.05, sum_tol_m2=0.05)
    assert failed.name == "utf8" and describe_failed([failed]).startswith("utf8: ")


def test_format_writes_empty_cells_for_none():
    rec = MeasurementRecord(level="row", vineyard_id="V01", row_id="V01-R001", row_length_m=1.005, plant_count=3,
                            row_structure="regular")
    assert format_measurements_csv([rec], m_decimals=2, ha_decimals=4).splitlines()[1] == \
        "row,V01,V01-R001,,,1.00,,,,,3,regular"


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
def fake_runner(monkeypatch):
    runner = types.ModuleType("vineyard.pipeline.runner")
    runner.StageResult = _FakeStageResult  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vineyard.pipeline.runner", runner)
    return runner


def _post_ctx(tmp_path: Path, ann, ref: str = "20260926T0100-marcaj-aaaaaa"):
    cfg = load_config(environ={"VINEYARD_WORK_DIR": str(tmp_path / "work")})
    write_annset(ann, cfg.paths.work_dir / "runs" / ref / "annset")
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", annset_ref=ref,
                          run_id="20260926T0310-post-abcdef")
    ensure_run_dirs(ctx.paths)
    return ctx


def test_measure_stage_writes_csv_and_json(tmp_path, fake_runner):
    ann = make_annset(BlockSpec(n_rows=3), prov=Prov(Source.MARCAJ, "20260926T0100-marcaj-aaaaaa", "m"))
    ctx = _post_ctx(tmp_path, ann)
    write_layer(rows_from_pieces(ann), "rows", ctx.paths.layers_dir / "rows.parquet")
    spec = load_stage("measure")
    result = spec.run(ctx)
    assert (spec.name, spec.scope, result.stage, result.n_failed, result.n_items) == ("measure", "global",
                                                                                    "measure", 0, 5)
    text = (ctx.paths.exports_dir / "measurements.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0] == EXPECTED_HEADER
    assert check_measurements_csv(text, sum_tol_m=0.05, sum_tol_m2=0.05) == ()
    doc = json.loads((ctx.paths.exports_dir / "measurements.json").read_text(encoding="utf-8"))
    assert doc["meta"]["run_id"] == ctx.run_id and doc["meta"]["source"] == "marcaj"
    assert doc["meta"]["annset_ref"] == ctx.annset_ref and doc["total"]["row_count"] == 3
    assert result.metrics["row_length_m"] == pytest.approx(180.0)
    assert result.metrics["interrow_overlap_removed_here"] == 1.0 and doc["meta"]["derive_run_id"] is None
    assert (ctx.paths.metrics_dir / "measure.json").is_file()


def test_measure_stage_reads_derive_interrows(tmp_path, fake_runner):
    ann = make_annset(BlockSpec(n_rows=3), prov=Prov(Source.MARCAJ, "20260926T0100-marcaj-aaaaaa", "m"))
    ctx = _post_ctx(tmp_path, ann)
    write_layer(rows_from_pieces(ann), "rows", ctx.paths.layers_dir / "rows.parquet")
    linked = ann.interrow_pieces.iloc[:1]  # derive's pieces (overlap-free), not the AnnSet's
    write_layer(linked, "interrow_pieces_linked", ctx.paths.layers_dir / "interrow_pieces_linked.parquet")
    result = load_stage("measure").run(ctx)
    assert result.metrics["interrow_area_m2"] == pytest.approx(linked.geometry.iloc[0].area)
    assert result.metrics["interrow_area_m2"] < union_area(ann.interrow_pieces)
    assert result.metrics["interrow_overlap_removed_here"] == 0.0
    assert _json_doc(ctx)["meta"]["derive_run_id"] == ctx.run_id


def _json_doc(ctx) -> dict:
    return json.loads((ctx.paths.exports_dir / "measurements.json").read_text(encoding="utf-8"))


def _earlier_post_run(ctx, annset_ref: str, linked, rows, run_id: str = "20260926T0200-post-bbbbbb") -> str:
    """A post run (older than ctx's) holding derive's layers, recorded for `annset_ref`."""
    run = ctx.paths.runs_dir / run_id
    write_layer(linked, "interrow_pieces_linked", run / "layers" / "interrow_pieces_linked.parquet")
    write_layer(rows, "rows", run / "layers" / "rows.parquet")
    (run / "run.json").write_text(json.dumps({"annset_ref": annset_ref}), encoding="utf-8")
    return run_id


def test_standalone_measure_reads_derive_layers_of_an_earlier_post_run(tmp_path, fake_runner):
    """`vineyard measure --annset X` runs in a new post run: derive's layers come from the newest post run
    of the same AnnSet, rows (whole-row max_gap_m) included."""
    ann = make_annset(BlockSpec(n_rows=2), prov=Prov(Source.MARCAJ, "20260926T0100-marcaj-aaaaaa", "m"))
    ctx = _post_ctx(tmp_path, ann)
    linked = ann.interrow_pieces.iloc[:1]
    derive_run = _earlier_post_run(ctx, ctx.annset_ref, linked, rows_from_pieces(ann).assign(max_gap_m=[7.5, 1.0]))
    result = load_stage("measure").run(ctx)
    assert result.metrics["interrow_area_m2"] == pytest.approx(linked.geometry.iloc[0].area)
    assert result.metrics["interrow_overlap_removed_here"] == 0.0
    doc = _json_doc(ctx)
    assert doc["meta"]["derive_run_id"] == derive_run and doc["rows"][0]["max_gap_m"] == pytest.approx(7.5)


def test_derive_layers_of_another_annset_are_not_used(tmp_path, fake_runner):
    ann = make_annset(BlockSpec(n_rows=2), prov=Prov(Source.MARCAJ, "20260926T0100-marcaj-aaaaaa", "m"))
    ctx = _post_ctx(tmp_path, ann)
    _earlier_post_run(ctx, "20260926T0000-marcaj-zzzzzz", ann.interrow_pieces.iloc[:1], rows_from_pieces(ann))
    result = load_stage("measure").run(ctx)
    assert result.metrics["interrow_overlap_removed_here"] == 1.0
    assert result.metrics["interrow_area_m2"] == pytest.approx(union_area(ann.interrow_pieces))


def _crossing_blocks(prov: Prov):
    """V02 rows at 8° over V01's: their inter-row bands share ground (the r009_c002 shape)."""
    crossing = BlockSpec(vineyard_id="V02", n_rows=4, angle_deg=8.0,
                         origin_xy=(DEFAULT_ORIGIN[0] + 10.0, DEFAULT_ORIGIN[1] + 1.0))
    return make_annset(BlockSpec(n_rows=4), crossing, prov=prov)


def test_measure_without_derive_removes_the_cross_block_overlap(tmp_path, fake_runner):
    ann = _crossing_blocks(Prov(Source.MARCAJ, "20260926T0100-marcaj-aaaaaa", "m"))
    raw = compute_measurements(_inputs(ann))
    raw_text = format_measurements_csv(raw.records(), m_decimals=2, ha_decimals=4)
    assert "block_interrow_sum" in {c.name for c in check_measurements_csv(raw_text, sum_tol_m=0.05,
                                                                              sum_tol_m2=0.05)}
    ctx = _post_ctx(tmp_path, ann)
    result = load_stage("measure").run(ctx)  # no derive layer in any post run of this AnnSet
    assert result.metrics["interrow_overlap_removed_here"] == 1.0
    lines = _table((ctx.paths.exports_dir / "measurements.csv").read_text(encoding="utf-8"))
    survey = float(next(r for r in lines if r["level"] == "survey")["interrow_area_m2"])
    blocks = sum(float(r["interrow_area_m2"]) for r in lines if r["level"] == "block")
    assert blocks == pytest.approx(survey, abs=0.02) and survey <= raw.survey.interrow_area_m2 + 1e-6
    issues = read_layer(ctx.paths.qa_dir / "issues_measure.parquet", "qa_issues")
    assert "interrow_block_overlap" in set(issues["code"])


def test_measure_stage_without_annset_ref_fails(tmp_path, fake_runner):
    cfg = load_config(environ={"VINEYARD_WORK_DIR": str(tmp_path / "work")})
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", run_id="20260926T0310-post-abcdef")
    from vineyard.errors import StageError
    with pytest.raises(StageError, match="annset"):
        load_stage("measure").run(ctx)
