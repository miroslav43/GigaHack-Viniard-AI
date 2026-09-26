"""probe_workflow + the `vineyard waste probe-data | probe-train | sam3-check` commands (fakes, no network)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.waste.synth_waste import make_candidate
from tests.waste.test_verify import FakeSam
from tests.waste.test_verify_setup import IdEmbedder
from vineyard.config import AppConfig, load_config
from vineyard.perception.waste import probe_workflow as pw
from vineyard.perception.waste.cli import app
from vineyard.perception.waste.crop_store import CropRecord, write_crop_store
from vineyard.perception.waste.probe import load_probe
from vineyard.perception.waste.types import RejectReason, candidate_to_record

TILE = "siret3_r006_c004"
OTHER = "siret3_r010_c010"
CROP_PX = 32
BLUE = (20, 60, 230)
SOIL = (120, 100, 80)
runner = CliRunner()


def _cfg(work: Path, sam: bool = True) -> AppConfig:
    return load_config(overrides=(
        f"paths.work_dir={json.dumps(str(work))}",
        f"paths.models_dir={json.dumps(str(work / 'models'))}",
        f"waste.sam3.enabled={str(sam).lower()}",
        "waste.probe.cv_folds=2",
    ))


def _pool() -> tuple:
    live = make_candidate(tile_id=TILE, centroid=(600.0, 600.0))
    tubes = tuple(
        replace(make_candidate(tile_id=TILE, centroid=(400.0 + 40 * k, 300.0), is_white=True),
                reject_reason=RejectReason.NEAR_AXIS)
        for k in range(4)
    )
    soil = replace(make_candidate(tile_id=TILE, centroid=(1500.0, 1500.0)), reject_reason=RejectReason.BRIGHT_SOIL)
    return (live, *tubes, soil)


def _write_cache(work: Path, tile: str, cands: tuple) -> None:
    path = work / "cache" / "waste" / f"{tile}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([candidate_to_record(c) for c in cands]).to_parquet(path)


def _image(_tile: str) -> np.ndarray:
    img = np.empty((2048, 2048, 3), np.uint8)
    img[...] = SOIL
    img[590:610, 590:610] = BLUE
    return img


def test_check_candidates_survivors_first_then_tubes() -> None:
    pool = _pool()
    chosen = pw.check_candidates(pool, 3)
    assert len(chosen) == 3 and not chosen[0].rejected
    assert all(c.reject_reason is RejectReason.NEAR_AXIS for c in chosen[1:])
    assert len(pw.check_candidates(pool, 10)) == 5  # soil rejects are never chosen


def test_sam3_check_times_the_crops(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    (tmp_path / "models" / "hf" / "sam3").mkdir(parents=True)
    _write_cache(tmp_path, TILE, _pool())
    check = pw.sam3_check(cfg, (TILE,), 2, loader=lambda _s, _p: FakeSam(), lister=lambda _s: ("model.safetensors",),
                          read=_image)
    assert check.status.available and len(check.results) == 2
    by_key = dict(check.results)
    bag = by_key[f"{TILE}@0600_0600"]
    assert bag is not None and bag.score == pytest.approx(0.7)  # the blue bag, not the tube
    assert len(check.seconds) == 2 and check.steady_s_per_crop >= 0.0


def test_sam3_check_without_sam_reports_the_status(tmp_path: Path) -> None:
    check = pw.sam3_check(_cfg(tmp_path, sam=False), (TILE,), 2)
    assert not check.status.available and check.results == () and np.isnan(check.steady_s_per_crop)


def test_sam3_check_needs_the_waste_cache(tmp_path: Path) -> None:
    (tmp_path / "models" / "hf" / "sam3").mkdir(parents=True)
    with pytest.raises(pw.WasteWorkflowError, match="waste tile cache missing"):
        pw.sam3_check(_cfg(tmp_path), (TILE,), 1, loader=lambda _s, _p: FakeSam(),
                      lister=lambda _s: ("model.safetensors",))


def _store(folder: Path, label: int, colour: tuple[int, int, int], tile: str, n: int) -> Path:
    crops = np.empty((n, CROP_PX, CROP_PX, 3), np.uint8)
    crops[...] = colour
    records = [
        CropRecord(key=f"{folder.name}:{i}", source="uavvaste" if label else "example_candidate",
                   group=f"uav:batch_{i % 4}" if label else tile, label=label,
                   meta={} if label else {"is_example": True, "survives": i % 2 == 0, "tile_id": tile})
        for i in range(n)
    ]
    return write_crop_store(folder, iter(crops), records, CROP_PX, {"kind": "test"})


def test_train_probe_from_stores_saves_a_loadable_probe(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    pos = _store(tmp_path / "pos", 1, BLUE, "", 24)
    neg_a = _store(tmp_path / "neg_a", 0, SOIL, TILE, 24)
    neg_b = _store(tmp_path / "neg_b", 0, SOIL, OTHER, 24)
    model, cv = pw.train_probe_from_stores(cfg, (pos, neg_a, neg_b), "v9", train_cmd="pytest",
                                           embedder=IdEmbedder())
    assert (cv.n_pos, cv.n_neg) == (24, 48) and cv.fp_example_at_threshold == 0
    loaded = load_probe(tmp_path / "models", cfg.waste.probe.name, "v9")
    assert loaded.sha256 == model.sha256 and loaded.embed_model == IdEmbedder().model_id


def test_train_probe_needs_a_model_id(tmp_path: Path) -> None:
    from tests.waste.test_verify import FakeEmbedder

    with pytest.raises(pw.WasteWorkflowError, match="model_id"):
        pw.train_probe_from_stores(_cfg(tmp_path), (), "v9", train_cmd="pytest", embedder=FakeEmbedder())


def test_build_probe_stores_checks_its_inputs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(pw.WasteWorkflowError, match="model run not found"):
        pw.build_probe_stores(cfg, "nope")
    run = tmp_path / "runs" / "r1" / "layers"
    run.mkdir(parents=True)
    pd.DataFrame({"tile_id": [TILE, OTHER]}).to_parquet(run / "tile_status.parquet")
    _write_cache(tmp_path, TILE, _pool())
    with pytest.raises(pw.WasteWorkflowError, match="waste tile cache missing"):
        pw.build_probe_stores(cfg, "r1")


def test_build_probe_stores_wires_positives_and_negatives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    run = tmp_path / "runs" / "r1" / "layers"
    run.mkdir(parents=True)
    pd.DataFrame({"tile_id": [TILE]}).to_parquet(run / "tile_status.parquet")
    _write_cache(tmp_path, TILE, _pool())
    seen: dict[str, object] = {}
    monkeypatch.setattr(pw, "load_dataset", lambda root: seen.setdefault("root", root))
    monkeypatch.setattr(pw, "background_sampler", lambda tiles, src, p: "bg")
    monkeypatch.setattr(pw, "build_positive_store",
                        lambda ds, p, out, background: _store(out, 1, BLUE, "", 3) if background == "bg" else None)
    monkeypatch.setattr(pw, "build_negative_store", lambda tiles, src, p, out: _store(out, 0, SOIL, tiles[0], 5))
    stores = pw.build_probe_stores(cfg, "r1")
    assert (stores.n_pos, stores.n_neg, stores.n_tiles) == (3, 5, 1)
    assert stores.pos == tmp_path / "probe" / "pos" and seen["root"] == pw.uavvaste_root(cfg)


def test_cli_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vineyard.perception.waste.probe import CvReport, ProbeModel

    work = tmp_path / "work"
    (work / "runs" / "r1").mkdir(parents=True)
    monkeypatch.setenv("VINEYARD_WORK_DIR", str(work))
    monkeypatch.setattr(pw, "build_probe_stores", lambda cfg, run_id: pw.ProbeStores(
        work / "p", work / "n", 7, 9, 2))
    res = runner.invoke(app, ["probe-data", "--run", "r1"])
    assert res.exit_code == 0, res.output
    assert "pozitive: 7" in res.output and "negative: 9" in res.output
    model = ProbeModel(np.zeros(2), 0.0, 0.9, 0.5, 0.4, True, "m", "waste-probe", "v1", "ab" * 32)
    cv = CvReport(7, 9, 4, 5, 0.97, 0.9, 0.5, 0.3, 0.5, 0.9, 0.4, 0, 0, {}, {}, 0.8, 0.01, True, "ok")
    monkeypatch.setattr(pw, "train_probe_from_stores", lambda cfg, stores, version, train_cmd: (model, cv))
    res = runner.invoke(app, ["probe-train"])
    assert res.exit_code == 0, res.output
    assert "waste-probe@v1+abababab" in res.output and "AUC 0.970" in res.output
    from vineyard.perception.waste.sam3_adapter import SamResult
    from vineyard.perception.waste.types import Category

    done = (SamResult(0.6, "trash", Category.DEBRIS, 0.5, (), s, 2) for s in (40.0, 12.0))
    check = pw.Sam3Check(
        status=pw.Sam3Status(True, "hf/sam3", "ok", (), "cpu", 1.5),
        results=(*(("k" + str(i), r) for i, r in enumerate(done)), ("late", None)),
    )
    monkeypatch.setattr(pw, "sam3_check", lambda cfg, tiles, n: check)
    res = runner.invoke(app, ["sam3-check", "--n", "3"])
    assert res.exit_code == 0, res.output
    assert "încărcare 1.5 s" in res.output and "fără timp" in res.output
    assert "s/crop (fără primul): 12.00" in res.output and "≈ 150 crop-uri" in res.output
