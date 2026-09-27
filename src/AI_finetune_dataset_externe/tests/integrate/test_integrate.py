"""Tests for the integration helpers: run copy safety, cover CSV join, gates, CLI env."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import box

from fte.integrate.annset_edit import apply_cover_csv
from fte.integrate.gates import Gate, metric_gates, render_md, waste_invariants
from fte.integrate.run_copy import FTE_MARKER, RunCopyError, copy_run, update_marker
from fte.integrate.vineyard_cli import MAKE_ENV, vineyard_env

TILE = "siret3_r006_c004"


def _fake_run(root: Path, name: str) -> Path:
    run = root / name
    (run / "annset").mkdir(parents=True)
    (run / "annset" / "annset.json").write_text("{}")
    (run / "layers").mkdir()
    (run / "layers" / "x.parquet").write_text("x")
    (run / "exports").mkdir()
    (run / "exports" / "big.zip").write_text("zip")
    (run / "qa" / "previews").mkdir(parents=True)
    (run / "qa" / "previews" / "p.jpg").write_text("p")
    (run / "qa" / "qa_issues.parquet").write_text("q")
    (run / "run.json").write_text("{}")
    return run


def test_copy_run_skips_exports_and_refuses_frozen(tmp_path: Path) -> None:
    _fake_run(tmp_path, "base")
    out = copy_run("base", "out1", runs_dir=tmp_path)
    assert (out / "annset" / "annset.json").is_file()
    assert (out / "layers" / "x.parquet").is_file()
    assert (out / "qa" / "qa_issues.parquet").is_file()
    assert not (out / "exports").exists()
    assert not (out / "qa" / "previews").exists()
    assert (out / FTE_MARKER).is_file()
    with pytest.raises(RunCopyError):
        copy_run("base", "complete-v4", runs_dir=tmp_path)
    with pytest.raises(RunCopyError):
        copy_run("base", "out1", runs_dir=tmp_path)
    assert copy_run("base", "out1", runs_dir=tmp_path, overwrite=True) == out
    doc = update_marker(out, edits={"n": 1})
    assert doc["edits"] == {"n": 1} and json.loads((out / FTE_MARKER).read_text())["base_run"] == "base"


def test_copy_run_never_replaces_foreign_run(tmp_path: Path) -> None:
    _fake_run(tmp_path, "base")
    _fake_run(tmp_path, "foreign")
    with pytest.raises(RunCopyError):
        copy_run("base", "foreign", runs_dir=tmp_path, overwrite=True)
    assert (tmp_path / "foreign" / "exports" / "big.zip").is_file()


def _pieces() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"piece_id": [f"V01-I001@{TILE}", f"V01-I002@{TILE}"], "tile_id": [TILE, TILE],
                             "interrow_cover": ["bare_soil", "mixed"]},
                            geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)], crs="EPSG:32635")


def test_apply_cover_csv_joins_and_validates(tmp_path: Path) -> None:
    pieces = _pieces()
    csv = tmp_path / "c.csv"
    csv.write_text(f"piece_id,interrow_cover,veg_frac,veg_frac2,changed\nV01-I001@{TILE},mixed,0.2,0.4,True\n")
    out, n = apply_cover_csv(pieces, csv)
    assert n == 1 and list(out["interrow_cover"]) == ["mixed", "mixed"]
    assert list(pieces["interrow_cover"]) == ["bare_soil", "mixed"]
    bad = tmp_path / "bad.csv"
    bad.write_text(f"piece_id,interrow_cover\nV01-I001@{TILE},grass\n")
    with pytest.raises(ValueError):
        apply_cover_csv(pieces, bad)
    unknown = tmp_path / "unknown.csv"
    unknown.write_text(f"piece_id,interrow_cover\nV09-I001@{TILE},mixed\n")
    with pytest.raises(ValueError):
        apply_cover_csv(pieces, unknown)


def _eval(r006: float, r021: float, attr: float = 1.0) -> dict:
    tiles = {"siret3_r006_c004": {"canopy.score": r006}, "siret3_r021_c012": {"canopy.score": r021}}
    mean = {"canopy.score": (r006 + r021) / 2, "attributes.score": attr, "rows.f1": 1.0, "interrow.iou": 0.96,
            "interrow.f1": 1.0, "grouping.f1": 1.0}
    return {"per_tile": tiles, "mean": mean}


def test_metric_gates() -> None:
    base = _eval(0.8941, 0.8455)
    assert all(g.passed for g in metric_gates(base, _eval(0.9037, 0.8516)))
    failed = {g.name for g in metric_gates(base, _eval(0.95, 0.84, attr=0.9)) if not g.passed}
    assert failed == {"canopy.tile.siret3_r021_c012", "not_worse.attributes.score"}


def test_waste_invariants_and_render() -> None:
    frame = gpd.GeoDataFrame({"tile_id": ["siret3_r006_c004"] + ["siret3_r001_c001"] * 6},
                             geometry=[box(0, 0, 1, 1)] * 7, crs="EPSG:32635")
    gates = {g.name: g.passed for g in waste_invariants(SimpleNamespace(waste=frame))}
    assert gates == {"waste.examples_zero": False, "waste.per_tile_max": False, "waste.total_max": True}
    md = render_md("r", "b", [Gate("x", True, 1, 1), Gate("y", False, 2, 1)])
    assert "FAIL" in md and "**Overall: FAIL**" in md


def test_vineyard_env_unsets_proj_and_sets_threads() -> None:
    env = vineyard_env({"PROJ_LIB": "/bad", "CONDA_PREFIX": "/c", "HOME": "/h"})
    assert "PROJ_LIB" not in env and "CONDA_PREFIX" not in env and env["HOME"] == "/h"
    assert all(env[k] == v for k, v in MAKE_ENV.items())
