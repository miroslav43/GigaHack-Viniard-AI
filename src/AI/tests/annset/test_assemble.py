"""assemble: AnnSet + tile_status + qa_issues from stage outputs (02 §3.10, plan S5/S8/S9)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.qa import annset_factory as af
from vineyard.annset import assemble as asm
from vineyard.annset.io import read_annset
from vineyard.annset.model import make_meta
from vineyard.config import load_config
from vineyard.contracts.enums import Severity, Source
from vineyard.contracts.qa import QaIssue
from vineyard.contracts.schemas import validate_layer
from vineyard.errors import SchemaError
from vineyard.qa import review

A, B = af.TILE_A, af.TILE_B
C = "siret3_r020_c012"
SETTINGS = asm.AssembleSettings.from_config(load_config(environ={}))


def _meta(tiles, source=Source.MODEL):
    return make_meta(source, "run1", "mv", tiles, created_at="2026-09-26T00:00:00+03:00")


def _model_inputs(**kw) -> asm.AssembleInputs:
    rows = af.row_pieces([(A, "V01-R001", af.hline(A, 10.0)), (A, "V01-R002", af.hline(A, 12.5))])
    irs = af.interrow_pieces([(A, "V01-I001", af.rect(A, 0, 10.3, 51.2, 12.2))])
    can = af.canopies([(A, "V01", af.rect(A, 5, 9.8, 6, 10.2))])
    base = {"canopies": can, "row_pieces": rows, "interrow_pieces": irs}
    return asm.AssembleInputs(**(base | kw))


def _codes(result: asm.AssembleResult, tile: str) -> list[str]:
    return sorted(i.code for i in result.issues if i.tile_id == tile)


def _status(result: asm.AssembleResult, tile: str) -> pd.Series:
    ts = result.tile_status
    return ts[ts["tile_id"] == tile].iloc[0]


def test_clean_model_tile() -> None:
    res = asm.assemble_annset(_model_inputs(), SETTINGS, _meta([A, B]))
    validate_layer(res.tile_status, "tile_status")
    validate_layer(res.qa_issues, "qa_issues")
    a = _status(res, A)
    assert (a["status"], bool(a["has_vineyard"]), a["review_priority"]) == ("ok", True, 3)
    assert (a["n_canopies"], a["n_row_pieces"], a["n_interrow_pieces"], a["n_waste"]) == (1, 2, 1, 0)
    assert a["image_id"] == -1 and pd.isna(a["upload_zip"])
    b = _status(res, B)
    assert (b["status"], b["review_priority"], b["issues"]) == ("no_vineyard", 1, "empty_tile_confirm")
    assert res.annset.meta.counts == {"canopies": 1, "row_pieces": 2, "interrow_pieces": 1, "waste": 0}
    assert res.n_errors == 0


def test_missing_waste_layer_gives_empty_waste() -> None:
    res = asm.assemble_annset(_model_inputs(waste=None), SETTINGS, _meta([A]))
    assert len(res.annset.waste) == 0
    validate_layer(res.annset.waste, "waste")


def test_waste_layer_is_kept() -> None:
    wst = af.waste([(A, "V01", af.rect(A, 20, 20, 21, 21))])
    res = asm.assemble_annset(_model_inputs(waste=wst), SETTINGS, _meta([A]))
    assert list(res.annset.waste["waste_id"]) == ["W0001"]
    assert _status(res, A)["n_waste"] == 1


def test_force_empty_tile() -> None:
    res = asm.assemble_annset(_model_inputs(force_empty_tiles=(A,)), SETTINGS, _meta([A]))
    assert res.annset.counts() == {"canopies": 0, "row_pieces": 0, "interrow_pieces": 0, "waste": 0}
    a = _status(res, A)
    assert (a["status"], a["review_priority"]) == ("no_vineyard", 1)
    assert _codes(res, A) == ["empty_tile_confirm", "override_applied"]


def test_large_overlap_subtracted_small_kept() -> None:
    can = af.canopies([(A, "V01", af.rect(A, 0, 9.8, 0.8, 10.4))])
    res = asm.assemble_annset(_model_inputs(canopies=can), SETTINGS, _meta([A]))
    assert res.fixed_overlap_tiles == (A,)
    assert res.n_errors == 0
    small = af.canopies([(A, "V01", af.rect(A, 0, 9.8, 0.2, 10.4))])
    res2 = asm.assemble_annset(_model_inputs(canopies=small), SETTINGS, _meta([A]))
    assert res2.fixed_overlap_tiles == ()
    assert res2.annset.interrow_pieces.geometry.iloc[0].equals(_model_inputs().interrow_pieces.geometry.iloc[0])


def test_failed_and_nodata_tiles() -> None:
    ev = {B: asm.TileEvidence(empty_nodata=True)}
    fail = QaIssue(Severity.ERROR, "tile_failed", C, "canopy", "boom")
    res = asm.assemble_annset(_model_inputs(evidence=ev, failed_tiles=(C,), upstream_issues=(fail,)), SETTINGS,
                              _meta([A, B, C]))
    assert _status(res, B)["status"] == "empty_nodata"
    c = _status(res, C)
    assert (c["status"], c["review_priority"]) == ("failed", 1)
    assert "tile_failed" in c["issues"]
    assert res.n_errors == 1


def test_canopies_without_vine_evidence_flagged() -> None:
    ev = {A: asm.TileEvidence(veg_frac=0.3, snr=1.5)}
    res = asm.assemble_annset(_model_inputs(evidence=ev), SETTINGS, _meta([A]))
    assert _codes(res, A) == ["canopy_on_non_vineyard_tile", "low_snr"]
    a = _status(res, A)
    assert a["review_priority"] == 1 and a["veg_frac"] == pytest.approx(0.3)
    ok = asm.assemble_annset(_model_inputs(evidence={A: asm.TileEvidence(snr=12.0)}), SETTINGS, _meta([A]))
    assert _codes(ok, A) == []


def test_interpolated_rows_with_canopies_flagged() -> None:
    inputs = _model_inputs()
    rows = inputs.row_pieces.assign(qa_flags=["row_interpolated", ""])
    res = asm.assemble_annset(_model_inputs(row_pieces=rows), SETTINGS, _meta([A]))
    assert _codes(res, A) == ["canopy_on_non_vineyard_tile", "row_interpolated"]


def test_object_flags_become_issues() -> None:
    rows = _model_inputs().row_pieces.assign(qa_flags=["structure_borderline", "override:A001"])
    res = asm.assemble_annset(_model_inputs(row_pieces=rows), SETTINGS, _meta([A]))
    assert _codes(res, A) == ["override_applied", "structure_borderline"]
    borderline = next(i for i in res.issues if i.code == "structure_borderline")
    assert borderline.severity == Severity.WARNING and borderline.object_id == f"V01-R001@{A}"
    assert borderline.x is not None


def test_invariant_errors_reported() -> None:
    wst = af.waste([(A, "V07", af.rect(A, 20, 20, 21, 21))])
    res = asm.assemble_annset(_model_inputs(waste=wst), SETTINGS, _meta([A]))
    assert _codes(res, A) == ["waste_block_unknown"]
    assert res.n_errors == 1 and _status(res, A)["review_priority"] == 1


def test_schema_violation_stops_assemble() -> None:
    bad = _model_inputs().canopies.assign(canopy_id=["not-an-id"])
    with pytest.raises(SchemaError):
        asm.assemble_annset(_model_inputs(canopies=bad), SETTINGS, _meta([A]))


def test_deterministic_outputs() -> None:
    wst = af.waste([(A, "V07", af.rect(A, 20, 20, 21, 21))])
    ev = {A: asm.TileEvidence(snr=2.0)}
    r1 = asm.assemble_annset(_model_inputs(waste=wst, evidence=ev), SETTINGS, _meta([B, A]))
    r2 = asm.assemble_annset(_model_inputs(waste=wst, evidence=ev), SETTINGS, _meta([A, B]))
    pd.testing.assert_frame_equal(r1.tile_status.drop(columns="geometry"), r2.tile_status.drop(columns="geometry"))
    pd.testing.assert_frame_equal(r1.qa_issues.drop(columns="geometry"), r2.qa_issues.drop(columns="geometry"))
    assert list(r1.qa_issues["issue_id"]) == [f"Q{k:05d}" for k in range(1, len(r1.qa_issues) + 1)]


def test_write_outputs(tmp_path: Path) -> None:
    res = asm.assemble_annset(_model_inputs(), SETTINGS, _meta([A, B]))
    written = asm.write_assemble_outputs(res, annset_dir=tmp_path / "annset", layers_dir=tmp_path / "layers",
                                         qa_dir=tmp_path / "qa")
    assert all(p.is_file() for p in written)
    back = read_annset(tmp_path / "annset")
    assert back.counts() == res.annset.counts()
    doc = json.loads((tmp_path / "annset" / "annset.json").read_text())
    assert set(doc) == {"contract_version", "source", "run_id", "model_version", "created_at", "n_tiles", "counts",
                        "inputs", "tile_ids"}
    queue = pd.read_csv(tmp_path / "qa" / "review_queue.csv")
    assert list(queue.columns) == list(review.QUEUE_COLUMNS) and list(queue["tile_id"]) == [B, A]
    assert (tmp_path / "qa" / "tile_status.csv").is_file()


# ------------------------------------------------------------------ reference acceptance


@pytest.mark.examples
def test_reference_annset_assembles_with_zero_errors(examples_dir: Path, tmp_path: Path) -> None:
    layers = af.reference_layers(examples_dir / "annotations.xml")
    inputs = asm.AssembleInputs(canopies=layers["canopies"], row_pieces=layers["row_pieces"],
                                interrow_pieces=layers["interrow_pieces"], waste=layers["waste"])
    res = asm.assemble_annset(inputs, SETTINGS, _meta([A, B], Source.REFERENCE))
    assert res.n_errors == 0, [i for i in res.issues if i.severity == Severity.ERROR]
    assert res.fixed_overlap_tiles == ()
    per_tile = {t: _status(res, t) for t in (A, B)}
    assert (per_tile[A]["n_canopies"], per_tile[B]["n_canopies"]) == (399, 251)
    assert (per_tile[A]["n_row_pieces"], per_tile[B]["n_row_pieces"]) == (25, 26)
    assert (per_tile[A]["n_interrow_pieces"], per_tile[B]["n_interrow_pieces"]) == (24, 25)
    assert all(per_tile[t]["status"] == "ok" for t in (A, B))
    asm.write_assemble_outputs(res, annset_dir=tmp_path / "annset", layers_dir=tmp_path / "layers",
                               qa_dir=tmp_path / "qa")
    assert read_annset(tmp_path / "annset").counts() == {"canopies": 650, "row_pieces": 51, "interrow_pieces": 49,
                                                          "waste": 0}


@pytest.mark.examples
def test_imported_reference_annset_assembles_with_zero_errors(examples_xml: bytes) -> None:
    from vineyard.pipeline.stages.import_reference import build_reference_annset

    cfg = load_config(environ={})
    ref, _ = build_reference_annset(cfg, examples_xml, run_id="ref", source_name="annotations.xml")
    inputs = asm.AssembleInputs(canopies=ref.canopies, row_pieces=ref.row_pieces,
                                interrow_pieces=ref.interrow_pieces, waste=ref.waste)
    res = asm.assemble_annset(inputs, SETTINGS, _meta(ref.meta.tile_ids, Source.REFERENCE))
    assert res.n_errors == 0 and res.fixed_overlap_tiles == ()
    assert res.annset.counts() == {"canopies": 650, "row_pieces": 51, "interrow_pieces": 49, "waste": 0}
    assert list(res.tile_status["review_priority"]) == [3, 3]
