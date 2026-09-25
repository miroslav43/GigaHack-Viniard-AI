import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from vineyard.annset.io import (
    ANNSET_JSON,
    LATEST_LINKS,
    read_annset,
    resolve_run_dir,
    update_latest_link,
    write_annset,
)
from vineyard.annset.model import ANNSET_LAYERS, AnnSet, empty_annset, make_meta
from vineyard.contracts.enums import Source
from vineyard.contracts.schemas import coerce_layer
from vineyard.errors import SchemaError, StageError

T1 = "siret3_r006_c004"
T2 = "siret3_r021_c012"
PROV = {"source": "model", "run_id": "run1", "model_version": "pipe@0.1", "confidence": 0.8, "qa_flags": ""}


def _annset() -> AnnSet:
    meta = make_meta(Source.MODEL, "run1", "pipe@0.1", (T1, T2), inputs=("tiles",), created_at="2026-09-26T03:10:00+03:00")
    canopies = coerce_layer(gpd.GeoDataFrame(
        [{"canopy_id": f"{T2}:C0001", "tile_id": T2, "vineyard_id": "V01", "row_id": "V01-R001", "area_m2": 1.0,
          "n_vertices": 4, "along_m": 1.0, "is_clump": False, "touches_edge": False, **PROV}],
        geometry=[box(0, 0, 1, 1)], crs=32635), "canopies")
    rows = coerce_layer(gpd.GeoDataFrame(
        [{"piece_id": f"V01-R001@{T2}", "row_id": "V01-R001", "vineyard_id": "V01", "tile_id": T2,
          "row_structure": "regular", "length_m": 10.0, "max_gap_m": 1.0, "n_vertices": 2, **PROV}],
        geometry=[LineString([(0, 0), (10, 0)])], crs=32635), "row_pieces")
    return empty_annset(meta).with_layer("canopies", canopies).with_layer("row_pieces", rows)


def test_write_read_round_trip(tmp_path: Path) -> None:
    ann = _annset()
    out = write_annset(ann, tmp_path / "annset")
    assert out == tmp_path / "annset"
    assert sorted(p.name for p in out.iterdir()) == sorted([*(f"{n}.parquet" for n in ANNSET_LAYERS), ANNSET_JSON])
    back = read_annset(out)
    assert back.meta == ann.meta
    for name in ANNSET_LAYERS:
        assert back.layer(name).drop(columns="geometry").equals(ann.layer(name).drop(columns="geometry")), name
        assert back.layer(name).geometry.geom_equals_exact(ann.layer(name).geometry, 0).all()
    doc = json.loads((out / ANNSET_JSON).read_text())
    assert doc["counts"] == {"canopies": 1, "row_pieces": 1, "interrow_pieces": 0, "waste": 0}
    assert doc["tile_ids"] == [T1, T2]


def test_write_is_deterministic(tmp_path: Path) -> None:
    a = write_annset(_annset(), tmp_path / "a")
    b = write_annset(_annset(), tmp_path / "b")
    for name in (*(f"{n}.parquet" for n in ANNSET_LAYERS), ANNSET_JSON):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_write_rejects_invalid_layer(tmp_path: Path) -> None:
    ann = _annset()
    bad = ann.with_layer("canopies", ann.canopies.assign(vineyard_id=["v1"]))
    with pytest.raises(SchemaError):
        write_annset(bad, tmp_path / "x")
    assert not (tmp_path / "x" / ANNSET_JSON).exists()


def test_write_rejects_objects_outside_meta_tiles(tmp_path: Path) -> None:
    ann = _annset()
    narrow = AnnSet(meta=make_meta(Source.MODEL, "run1", "m", (T1,)), canopies=ann.canopies,
                    row_pieces=ann.row_pieces, interrow_pieces=ann.interrow_pieces, waste=ann.waste)
    with pytest.raises(SchemaError, match="not in meta.tile_ids"):
        write_annset(narrow, tmp_path / "x")


def test_read_errors(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="annset.json"):
        read_annset(tmp_path)
    out = write_annset(_annset(), tmp_path / "ok")
    (out / "waste.parquet").unlink()
    with pytest.raises(SchemaError, match="waste"):
        read_annset(out)
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / ANNSET_JSON).write_text("{not json")
    with pytest.raises(SchemaError):
        read_annset(tmp_path / "bad")


def test_read_without_validation(tmp_path: Path) -> None:
    out = write_annset(_annset(), tmp_path / "ok")
    assert len(read_annset(out, validate=False).canopies) == 1


def test_latest_links_and_resolve(tmp_path: Path) -> None:
    work = tmp_path / "work"
    run = work / "runs" / "20260926T0310-model-a1b2c3"
    write_annset(_annset(), run / "annset")
    update_latest_link(work, Source.MODEL, run)
    link = work / "runs" / LATEST_LINKS[Source.MODEL]
    assert link.is_symlink()
    assert not Path(link.readlink()).is_absolute()
    assert resolve_run_dir(work, "LATEST_MODEL") == run.resolve()
    assert resolve_run_dir(work, run.name) == run.resolve()
    assert resolve_run_dir(work, str(run)) == run.resolve()
    assert resolve_run_dir(work, str(run / "annset")) == run.resolve()
    other = work / "runs" / "second"
    other.mkdir(parents=True)
    update_latest_link(work, Source.MODEL, other)
    assert resolve_run_dir(work, "LATEST_MODEL") == other.resolve()


def test_latest_link_outside_runs_is_absolute(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    update_latest_link(tmp_path / "work", Source.MARCAJ, elsewhere)
    link = tmp_path / "work" / "runs" / "LATEST_MARCAJ"
    assert Path(link.readlink()).is_absolute()
    assert resolve_run_dir(tmp_path / "work", "LATEST_MARCAJ") == elsewhere.resolve()


def test_resolve_errors(tmp_path: Path) -> None:
    with pytest.raises(StageError):
        resolve_run_dir(tmp_path, "LATEST_REFERENCE")
    with pytest.raises(StageError):
        resolve_run_dir(tmp_path, "no-such-run")
    with pytest.raises(StageError):
        resolve_run_dir(tmp_path, "")
    with pytest.raises(StageError):
        update_latest_link(tmp_path, Source.MODEL, tmp_path / "missing")
