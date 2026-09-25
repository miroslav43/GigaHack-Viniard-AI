"""eval.report on synthetic AnnSets: extraction, per tile / mean / pooled, gates, baseline, writers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import LineString, box

from tests.eval.conftest import make_annset, tile_origin
from vineyard.config import EvalConfig
from vineyard.errors import StageError
from vineyard.eval.report import (
    EvalParams,
    compare_to_baseline,
    evaluate_annsets,
    load_baseline,
    render_markdown,
    report_to_dict,
    tile_objects,
    write_report_json,
    write_report_markdown,
)

TA, TB = "siret3_r006_c004", "siret3_r021_c012"
HEADLINE = ("canopy.score", "canopy.iou", "canopy.f1", "rows.f1", "interrow.iou", "interrow.f1",
            "attributes.score", "grouping.f1", "counts.score", "waste.f1")


def _tile_layers(tile: str, vid: str, *, dx: float = 0.0, n_rows: int = 3, structure: str = "regular",
                 cover: str = "bare_soil", waste: bool = False) -> dict[str, list[dict]]:
    x0, y0 = tile_origin(tile)
    rows, canopies, interrows = [], [], []
    for k in range(n_rows):
        y = y0 + 5 + 2.5 * k
        rid = f"{vid}-R{k + 1:03d}"
        rows.append({"piece_id": f"{rid}@{tile}", "row_id": rid, "vineyard_id": vid, "tile_id": tile,
                     "row_structure": structure, "geometry": LineString([(x0 + dx, y), (x0 + 51.2 + dx, y)])})
        canopies += [{"canopy_id": f"{tile}:C{10 * k + c + 1:04d}", "tile_id": tile, "vineyard_id": vid,
                      "geometry": box(x0 + 2 + 4 * c + dx, y - 0.3, x0 + 3.5 + 4 * c + dx, y + 0.3)}
                     for c in range(5)]
        if k + 1 < n_rows:
            interrows.append({"piece_id": f"{vid}-I{k + 1:03d}@{tile}", "tile_id": tile, "vineyard_id": vid,
                              "interrow_cover": cover,
                              "geometry": box(x0 + dx, y + 0.3, x0 + 51.2 + dx, y + 2.2)})
    out = {"canopies": canopies, "row_pieces": rows, "interrow_pieces": interrows}
    if waste:
        out["waste"] = [{"waste_id": "W0001", "tile_id": tile, "vineyard_id": vid,
                         "geometry": box(x0 + 30, y0 + 1, x0 + 30.5, y0 + 1.4)}]
    return out


def _annset(*parts: tuple[str, dict[str, list[dict]]], tiles: tuple[str, ...] = (TA, TB)):
    merged: dict[str, list[dict]] = {}
    for _, layers in parts:
        for name, recs in layers.items():
            merged.setdefault(name, []).extend(recs)
    return make_annset(tiles, **merged)


@pytest.fixture
def ref():
    return _annset((TA, _tile_layers(TA, "V01", waste=True)), (TB, _tile_layers(TB, "V02", n_rows=4)))


def test_params_from_config(eval_cfg: EvalConfig) -> None:
    p = EvalParams.from_config(eval_cfg)
    assert (p.canopy_match_iou, p.row_tol_m, p.waste_match_iou, p.length_tol) == (0.5, 0.4, 0.3, 0.1)


def test_identical_annsets_score_exactly_one(ref, eval_cfg: EvalConfig) -> None:
    rep = evaluate_annsets(ref, ref, (TA, TB), eval_cfg)
    for scope in (rep.per_tile[TA], rep.per_tile[TB], rep.mean, rep.pooled):
        for key in HEADLINE:
            assert scope[key] == 1.0, key
    assert rep.gates_passed and rep.per_tile[TB]["rows.tp"] == 4
    assert rep.pooled["counts.blocks.ref"] == 2 and rep.pooled["counts.rows.ref"] == 7


def test_missing_prediction_tile_scores_zero(ref, eval_cfg: EvalConfig) -> None:
    pred = _annset((TA, _tile_layers(TA, "V01", waste=True)), tiles=(TA,))
    rep = evaluate_annsets(pred, ref, (TA, TB), eval_cfg)
    b = rep.per_tile[TB]
    assert (b["canopy.score"], b["rows.f1"], b["interrow.iou"]) == (0.0, 0.0, 0.0)
    assert b["attributes.row_structure.accuracy"] == 0.0 and b["counts.score"] == 0.0
    assert rep.per_tile[TA]["canopy.score"] == 1.0
    assert rep.mean["canopy.score"] == pytest.approx(0.5)
    assert not rep.gates_passed
    gate = {g.name: g for g in rep.gates}["row_f1"]
    assert (gate.scope, gate.value, gate.passed) == ("min_tile", 0.0, False)


def test_shifted_prediction_mean_vs_pooled(ref, eval_cfg: EvalConfig) -> None:
    pred = _annset((TA, _tile_layers(TA, "V01", dx=0.6, waste=True)), (TB, _tile_layers(TB, "V02", n_rows=4)))
    rep = evaluate_annsets(pred, ref, (TA, TB), eval_cfg)
    a = rep.per_tile[TA]
    assert a["canopy.iou"] == pytest.approx(0.9 / 2.1, abs=1e-6)  # 1.5 m boxes shifted 0.6 m
    assert a["canopy.f1"] == 0.0 and a["rows.f1"] == 1.0  # shift along the rows keeps the axes
    pooled_iou = (0.9 * 15 * 0.6 + 1.5 * 20 * 0.6) / (2.1 * 15 * 0.6 + 1.5 * 20 * 0.6)
    assert rep.pooled["canopy.iou"] == pytest.approx(pooled_iou, abs=1e-6)
    assert rep.pooled["canopy.f1"] == pytest.approx(2 * 20 / 70)
    assert rep.mean["canopy.iou"] == pytest.approx((a["canopy.iou"] + 1.0) / 2)


def test_attribute_and_grouping_errors(ref, eval_cfg: EvalConfig) -> None:
    pred = _annset((TA, _tile_layers(TA, "V09", structure="disrupted", cover="mixed", waste=True)),
                   (TB, _tile_layers(TB, "V09", n_rows=4)))
    rep = evaluate_annsets(pred, ref, (TA, TB), eval_cfg)
    a = rep.per_tile[TA]
    assert a["attributes.row_structure.accuracy"] == 0.0 and a["attributes.interrow_cover.accuracy"] == 0.0
    assert a["grouping.f1"] == 1.0 and rep.per_tile[TB]["grouping.f1"] == 1.0
    assert rep.pooled["grouping.f1"] < 1.0  # one id for two reference blocks
    assert rep.pooled["counts.blocks.pred"] == 1 and rep.pooled["counts.blocks.score"] == 0.0
    assert rep.pooled["attributes.row_structure.n_ref"] == 7


def test_reference_without_vineyard_uses_fp_penalty(eval_cfg: EvalConfig) -> None:
    x0, y0 = tile_origin(TA)
    ref = make_annset((TA,))
    fp = [{"canopy_id": f"{TA}:C0001", "tile_id": TA, "vineyard_id": "V01",
           "geometry": box(x0, y0, x0 + 10, y0 + 26.2144)}]
    rep = evaluate_annsets(make_annset((TA,), canopies=fp), ref, (TA,), eval_cfg)
    t = rep.per_tile[TA]
    assert t["canopy.score"] is None and t["canopy.fp_penalty"] == pytest.approx(0.05)
    assert rep.pooled["canopy.fp_penalty"] == pytest.approx(0.05)
    assert rep.mean["canopy.score"] is None
    assert not {g.name: g for g in rep.gates}["canopy_score"].passed


def test_tile_objects_clip_and_missing_columns(eval_cfg: EvalConfig) -> None:
    x0, y0 = tile_origin(TA)
    canopies = [{"canopy_id": f"{TA}:C0001", "tile_id": TA, "vineyard_id": "V01",
                 "geometry": box(x0 - 1, y0 + 1, x0 + 1, y0 + 2)},
                {"canopy_id": f"{TA}:C0002", "tile_id": TA, "vineyard_id": "V01",
                 "geometry": box(x0 - 3, y0 + 1, x0 - 2, y0 + 2)},
                {"canopy_id": f"{TB}:C0001", "tile_id": TB, "vineyard_id": "V01",
                 "geometry": box(0, 0, 1, 1)}]
    ann = make_annset((TA, TB), canopies=canopies)
    objs = tile_objects(ann, TA)
    assert len(objs.canopies.geoms) == 1 and objs.canopies.geoms[0].area == pytest.approx(1.0)
    assert len(tile_objects(ann, TA, clip=False).canopies.geoms) == 2
    no_vid = ann.with_layer("canopies", ann.canopies.drop(columns=["vineyard_id"]))
    assert tile_objects(no_vid, TA).canopies.vineyard_ids == (None,)


def test_tiles_validation(ref, eval_cfg: EvalConfig) -> None:
    with pytest.raises(StageError, match="no tiles"):
        evaluate_annsets(ref, ref, (), eval_cfg)
    with pytest.raises(StageError, match="reference"):
        evaluate_annsets(ref, ref, ("siret3_r000_c000",), eval_cfg)


def test_compare_to_baseline(ref, eval_cfg: EvalConfig) -> None:
    rep = evaluate_annsets(ref, ref, (TA, TB), eval_cfg)
    base = report_to_dict(rep)
    assert compare_to_baseline(rep, base, max_drop=0.01) == ()
    worse = {**base, "mean": {**base["mean"], "canopy.score": 1.5}}
    msgs = compare_to_baseline(rep, worse, max_drop=0.01)
    assert len(msgs) == 1 and "canopy.score" in msgs[0] and "mean" in msgs[0]
    tile_worse = {**base, "per_tile": {TA: {"rows.f1": 1.0105}}}
    assert len(compare_to_baseline(rep, tile_worse, max_drop=0.01)) == 1
    assert compare_to_baseline(rep, {"per_tile": {TA: {"rows.f1": 1.009}}}, max_drop=0.01) == ()


def test_writers_and_baseline_roundtrip(ref, eval_cfg: EvalConfig, tmp_path: Path) -> None:
    rep = evaluate_annsets(ref, make_annset((TA, TB)), (TA, TB), eval_cfg)  # reference without vineyard
    js = write_report_json(rep, tmp_path / "m" / "eval_examples.json")
    doc = json.loads(js.read_text(encoding="utf-8"))
    assert doc["tiles"] == [TA, TB] and doc["gates_passed"] is False
    assert doc["per_tile"][TA]["counts.blocks.rel_err"] is None  # inf is not JSON
    assert load_baseline(js)["mean"]["canopy.fp_penalty"] > 0
    md = write_report_markdown(rep, tmp_path / "m" / "eval_examples.md").read_text(encoding="utf-8")
    assert "| canopy.score |" in md and "FAIL" in md and TA in md
    assert render_markdown(rep.with_regressions(("x dropped",))).count("x dropped") == 1


def test_load_baseline_errors(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="not found"):
        load_baseline(tmp_path / "none.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(StageError, match="JSON"):
        load_baseline(bad)
