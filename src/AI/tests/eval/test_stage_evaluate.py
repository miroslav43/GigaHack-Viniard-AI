"""Stage `evaluate` (eval-examples): run resolution, report files, baseline write/compare, gate enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import LineString, box

from tests.eval.conftest import EXAMPLE_TILES, make_annset, tile_origin
from vineyard.annset.io import update_latest_link, write_annset
from vineyard.annset.model import AnnSet
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.errors import StageError
from vineyard.pipeline.context import RunContext, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages.evaluate import REPORT_NAME, STAGE, run

REF_RUN, PRED_RUN = "20260926T0100-reference-aaaaaa", "20260926T0200-model-bbbbbb"


def _layers(shift: float = 0.0) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"canopies": [], "row_pieces": [], "interrow_pieces": []}
    for n, tile in enumerate(EXAMPLE_TILES, 1):
        x0, y0 = tile_origin(tile)
        vid = f"V0{n}"
        for k in range(2):
            y = y0 + 10 + 2.5 * k + shift
            rid = f"{vid}-R{k + 1:03d}"
            out["row_pieces"].append({"piece_id": f"{rid}@{tile}", "row_id": rid, "vineyard_id": vid,
                                      "tile_id": tile, "row_structure": "regular",
                                      "geometry": LineString([(x0, y), (x0 + 51.2, y)])})
            out["canopies"].append({"canopy_id": f"{tile}:C{k + 1:04d}", "tile_id": tile, "vineyard_id": vid,
                                    "geometry": box(x0 + 5, y - 0.3, x0 + 7, y + 0.3)})
        out["interrow_pieces"].append({"piece_id": f"{vid}-I001@{tile}", "tile_id": tile, "vineyard_id": vid,
                                       "interrow_cover": "bare_soil",
                                       "geometry": box(x0, y0 + 10.3 + shift, x0 + 51.2, y0 + 12.2 + shift)})
    return out


def _write_run(work: Path, run_id: str, annset: AnnSet) -> Path:
    run_dir = work / "runs" / run_id
    write_annset(annset, run_dir / "annset")
    return run_dir


@pytest.fixture
def runs(tmp_work: Path) -> Path:
    ref_dir = _write_run(tmp_work, REF_RUN, make_annset(EXAMPLE_TILES, run_id=REF_RUN, **_layers()))
    update_latest_link(tmp_work, Source.REFERENCE, ref_dir)
    return tmp_work


def _ctx(work: Path, annset_ref: str | None, *overrides: str) -> RunContext:
    cfg = load_config(overrides=overrides, environ={"VINEYARD_WORK_DIR": str(work)})
    return new_run_context(cfg, source=Source.MODEL, kind="post", annset_ref=annset_ref, workers=1)


def test_stage_spec_is_registered() -> None:
    spec = load_stage("evaluate")
    assert spec is STAGE and spec.scope == "global" and spec.cfg_keys == ("eval",)


def test_reference_vs_reference_writes_reports(runs: Path) -> None:
    result = run(_ctx(runs, "LATEST_REFERENCE", "eval.enforce_gates=true"))
    metrics_dir = runs / "runs" / REF_RUN / "metrics"
    assert result.outputs == (metrics_dir / f"{REPORT_NAME}.json", metrics_dir / f"{REPORT_NAME}.md")
    doc = json.loads((metrics_dir / f"{REPORT_NAME}.json").read_text(encoding="utf-8"))
    assert doc["gates_passed"] is True and doc["mean"]["canopy.score"] == 1.0
    assert result.metrics["mean.rows.f1"] == 1.0 and result.metrics["gates_passed"] == 1.0
    assert result.n_items == 2 and result.n_failed == 0


def test_gates_enforced_after_reports_written(runs: Path) -> None:
    _write_run(runs, PRED_RUN, make_annset(EXAMPLE_TILES, run_id=PRED_RUN, **_layers(shift=0.6)))
    run(_ctx(runs, PRED_RUN))  # not enforced: reports only
    with pytest.raises(StageError, match="gates"):
        run(_ctx(runs, PRED_RUN, "eval.enforce_gates=true"))
    doc = json.loads((runs / "runs" / PRED_RUN / "metrics" / f"{REPORT_NAME}.json").read_text(encoding="utf-8"))
    assert doc["mean"]["rows.f1"] == 0.0 and doc["gates_passed"] is False


def test_write_baseline_then_regression(runs: Path) -> None:
    base = runs / "eval_baseline.json"
    result = run(_ctx(runs, "LATEST_REFERENCE", "eval.write_baseline=true"))
    assert base in result.outputs and json.loads(base.read_text(encoding="utf-8"))["mean"]["canopy.score"] == 1.0
    _write_run(runs, PRED_RUN, make_annset(EXAMPLE_TILES, run_id=PRED_RUN, **_layers(shift=0.6)))
    baseline_set = f"eval.baseline_file={json.dumps(str(base))}"
    result = run(_ctx(runs, PRED_RUN, baseline_set))
    assert result.metrics["regressions"] > 0
    with pytest.raises(StageError, match="gates"):
        run(_ctx(runs, PRED_RUN, baseline_set, "eval.enforce_gates=true", "eval.gates.row_f1=0.0",
                 "eval.gates.canopy_score=0.0", "eval.gates.interrow_iou=0.0", "eval.gates.attributes=0.0"))


def test_first_baseline_write_to_a_new_file_skips_comparison(runs: Path) -> None:
    base = runs / "new_baseline.json"
    overrides = (f"eval.baseline_file={json.dumps(str(base))}", "eval.write_baseline=true")
    result = run(_ctx(runs, "LATEST_REFERENCE", *overrides))
    assert base in result.outputs and result.metrics["regressions"] == 0.0
    with pytest.raises(StageError, match="baseline not found"):
        run(_ctx(runs, "LATEST_REFERENCE", f"eval.baseline_file={json.dumps(str(runs / 'absent.json'))}"))


def test_missing_reference_is_a_clear_error(tmp_work: Path) -> None:
    _write_run(tmp_work, PRED_RUN, make_annset(EXAMPLE_TILES, run_id=PRED_RUN, **_layers()))
    with pytest.raises(StageError, match="reference AnnSet not found"):
        run(_ctx(tmp_work, PRED_RUN))
    with pytest.raises(StageError, match="evaluated AnnSet not found"):
        run(_ctx(tmp_work, "no-such-run"))


@pytest.mark.examples
def test_imported_reference_matches_eval_oracle(examples_xml: bytes, reference_annset: AnnSet) -> None:
    """Integration with P1-CVIO: the imported reference equals the test oracle on every metric."""
    importer = pytest.importorskip("vineyard.pipeline.stages.import_reference")
    from vineyard.eval.report import evaluate_annsets

    cfg = load_config(environ={})
    imported, _ = importer.build_reference_annset(cfg, examples_xml, run_id=REF_RUN, source_name="examples")
    rep = evaluate_annsets(imported, reference_annset, EXAMPLE_TILES, cfg.eval)
    assert all(v == 1.0 for k, v in rep.pooled.items() if k.endswith((".score", ".f1", ".iou")) and v is not None)
