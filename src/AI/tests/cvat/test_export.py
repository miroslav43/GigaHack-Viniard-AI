"""cvat.export.export_upload: staging -> validate -> self-check -> atomic publish, manifests, failures."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from tests.cvat.test_to_cvat import canopy_row, interrow_row, make_annset, make_layer, px_square, row_row
from vineyard.annset.io import write_annset
from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import Source
from vineyard.contracts.schemas import coerce_layer, validate_layer
from vineyard.cvat import export as export_mod
from vineyard.cvat import to_cvat as to_cvat_mod
from vineyard.cvat.export import export_upload, qa_error_count, require_no_qa_errors, tile_sources
from vineyard.cvat.report import EMPTY_TILES_COLUMNS, MANIFEST_COLUMNS
from vineyard.errors import ExportBlocked, StageError
from vineyard.geo.tiling import CRS_EPSG, TILE_M, tile_box, tile_ref
from vineyard.geo.vector_io import write_layer
from vineyard.pipeline.context import POST_KIND, RunContext, ensure_run_dirs, new_run_context
from vineyard.pipeline.runner import run_stages
from vineyard.pipeline.stages import export_cvat as export_stage
from vineyard.pipeline.stages.export_cvat import EMPTY_REF

TILES = ("siret3_r006_c004", "siret3_r018_c010", "siret3_r021_c012")
RUN = "20260926T0200-post-abcdef"


def with_cvat(cfg: AppConfig, **update: object) -> AppConfig:
    cvat = cfg.export.cvat.model_copy(update=update)
    return cfg.model_copy(update={"export": cfg.export.model_copy(update={"cvat": cvat})})


@pytest.fixture(scope="module")
def base_cfg() -> AppConfig:
    return load_config()


def make_tile_index(tmp_path: Path, tiles: tuple[str, ...] = TILES, size: int = 5000) -> gpd.GeoDataFrame:
    rows = []
    for k, tile_id in enumerate(tiles):
        data = bytes([k + 1]) * (size + 100 * k)
        path = tmp_path / "tiles" / f"{tile_id}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        t = tile_ref(tile_id)
        rows.append({"tile_id": tile_id, "file_name": path.name, "grid_row": t.grid_row, "grid_col": t.grid_col,
                     "x0": t.x0, "y0": t.y0, "x1": t.x0 + TILE_M, "y1": t.y0 - TILE_M, "gsd_m": 0.025,
                     "width_px": 2048, "height_px": 2048, "src_zip": "part1.zip", "path": str(path),
                     "sha256": hashlib.sha256(data).hexdigest(), "file_size": len(data),
                     "nodata_frac": float("nan"), "valid_area_m2": float("nan"), "geometry": tile_box(t)})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_EPSG)
    out = coerce_layer(gdf, "tile_index")
    validate_layer(out, "tile_index")
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_empty_annset_export(tmp_path: Path, base_cfg: AppConfig) -> None:
    index = make_tile_index(tmp_path)
    out = tmp_path / "exports" / "marcaj_upload"
    res = export_upload(make_annset(TILES), index, base_cfg, out, run_id=RUN)
    assert res.out_dir == out and out.is_dir()
    assert [p.name for p in res.zips] == ["siret3_upload_01of01.zip"]
    assert res.report.ok and res.failed_tiles == ()
    manifest = _read_csv(res.manifest)
    assert tuple(manifest[0]) == MANIFEST_COLUMNS
    assert [r["tile_id"] for r in manifest] == list(TILES) and [r["image_id"] for r in manifest] == ["0", "1", "2"]
    empty = _read_csv(res.empty_tiles)
    assert tuple(empty[0]) == EMPTY_TILES_COLUMNS
    assert [(r["tile_id"], r["reason"]) for r in empty] == [(t, "no_objects") for t in TILES]
    registry = json.loads(res.id_registry.read_text())
    assert set(registry) >= {"contract_version", "run_id", "last_vineyard", "blocks", "manual_row_start"}
    assert registry["last_vineyard"] is None and registry["manual_row_start"] == 900
    summary = json.loads(res.summary.read_text())
    assert summary["n_tiles"] == 3 and summary["n_zips"] == 1 and summary["n_empty_tiles"] == 3
    assert json.loads(res.validation_report.read_text())["ok"] is True
    assert sorted(p.name for p in out.parent.iterdir()) == ["marcaj_upload"]


def test_export_with_objects_multi_part_and_deterministic(tmp_path: Path, base_cfg: AppConfig) -> None:
    index = make_tile_index(tmp_path, size=20000)
    t = TILES[2]
    annset = make_annset(
        TILES,
        canopies=[canopy_row(t, 1, px_square(10, 10, 20)), canopy_row(t, 2, px_square(60, 10, 25))],
        row_pieces=[row_row(t, 1, LineString([(0, 100), (2048, 120)])),
                    row_row(t, 2, LineString([(0, 200), (2048, 220)]))],
        interrow_pieces=[interrow_row(t, 1, px_square(500, 500, 100))],
    )
    cfg = with_cvat(base_cfg, max_zip_bytes=45000)
    status = gpd.GeoDataFrame({"tile_id": [TILES[0]], "status": ["empty_nodata"]}, geometry=[tile_box(tile_ref(TILES[0]))],
                              crs=CRS_EPSG)
    res1 = export_upload(annset, index, cfg, tmp_path / "a" / "marcaj_upload", tile_status=status, run_id=RUN)
    res2 = export_upload(annset, index, cfg, tmp_path / "b" / "marcaj_upload", tile_status=status, run_id=RUN)
    assert len(res1.zips) >= 2 and all(p.stat().st_size <= 45000 for p in res1.zips)
    assert [p.read_bytes() for p in res1.zips] == [p.read_bytes() for p in res2.zips]
    registry = json.loads(res1.id_registry.read_text())
    assert registry["last_vineyard"] == "V01" and registry["blocks"]["V01"] == {"last_row": 2, "last_interrow": 1}
    empty = {r["tile_id"]: r["reason"] for r in _read_csv(res1.empty_tiles)}
    assert empty == {TILES[0]: "empty_nodata", TILES[1]: "no_objects"}
    manifest = {r["tile_id"]: r for r in _read_csv(res1.manifest)}
    assert manifest[t]["n_vineyard"] == "2" and manifest[t]["n_row"] == "2" and manifest[t]["n_interrow_area"] == "1"
    with zipfile.ZipFile(res1.zips[-1]) as zf:
        assert zf.namelist()[0] == "annotations.xml"


def test_republish_replaces_previous_export(tmp_path: Path, base_cfg: AppConfig) -> None:
    index = make_tile_index(tmp_path)
    out = tmp_path / "marcaj_upload"
    export_upload(make_annset(TILES), index, base_cfg, out)
    (out / "stale.txt").write_text("x")
    export_upload(make_annset(TILES), index, base_cfg, out)
    assert not (out / "stale.txt").exists()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("marcaj")) == ["marcaj_upload"]


def _overlapping(t: str):
    return make_annset(TILES, canopies=[canopy_row(t, 1, px_square(542, 500, 12))],
                       interrow_pieces=[interrow_row(t, 1, px_square(500, 500, 50))])


def test_invalid_object_blocks_and_leaves_failed_dir(tmp_path: Path, base_cfg: AppConfig) -> None:
    index = make_tile_index(tmp_path)
    out = tmp_path / "marcaj_upload"
    with pytest.raises(ExportBlocked) as excinfo:
        export_upload(_overlapping(TILES[1]), index, base_cfg, out)
    assert excinfo.value.exit_code == 2
    failed = [p for p in tmp_path.iterdir() if ".FAILED-" in p.name]
    assert len(failed) == 1 and not out.exists()
    report = json.loads((failed[0] / "validation_report.json").read_text())
    assert report["by_code"]["canopy_interrow_overlap"] == 1


def test_invalid_tile_policy_empty(tmp_path: Path, base_cfg: AppConfig) -> None:
    index = make_tile_index(tmp_path)
    res = export_upload(_overlapping(TILES[1]), index, base_cfg, tmp_path / "marcaj_upload",
                        invalid_tile_policy="empty")
    empty = {r["tile_id"]: r["reason"] for r in _read_csv(res.empty_tiles)}
    assert empty[TILES[1]] == "invalid"
    assert res.report.has_code("tile_emptied")


def test_sha_mismatch_blocks(tmp_path: Path, base_cfg: AppConfig) -> None:
    index = make_tile_index(tmp_path)
    bad = index.assign(sha256=["0" * 64] + list(index["sha256"][1:]))
    with pytest.raises(ExportBlocked):
        export_upload(make_annset(TILES), bad, base_cfg, tmp_path / "marcaj_upload")
    assert any(".FAILED-" in p.name for p in tmp_path.iterdir())


def test_failed_tile_is_exported_empty(tmp_path: Path, base_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    real = to_cvat_mod.tile_to_image

    def flaky(frames, t, image_id, cfg):
        if t.tile_id == TILES[0]:
            raise ValueError("boom")
        return real(frames, t, image_id, cfg)

    monkeypatch.setattr(to_cvat_mod, "tile_to_image", flaky)
    index = make_tile_index(tmp_path)
    res = export_upload(make_annset(TILES), index, base_cfg, tmp_path / "marcaj_upload")
    assert res.failed_tiles == (TILES[0],)
    empty = {r["tile_id"]: r["reason"] for r in _read_csv(res.empty_tiles)}
    assert empty[TILES[0]] == "failed"
    assert json.loads(res.summary.read_text())["failed_tiles"] == [TILES[0]]


def test_underestimated_plan_is_repacked(tmp_path: Path, base_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_mod, "estimate_tile_bytes", lambda *a: 1)
    index = make_tile_index(tmp_path, size=20000)
    cfg = with_cvat(base_cfg, max_zip_bytes=45000)
    res = export_upload(make_annset(TILES), index, cfg, tmp_path / "marcaj_upload")
    assert len(res.zips) == 2 and all(p.stat().st_size <= 45000 for p in res.zips)


def test_repack_gives_up(tmp_path: Path, base_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_mod, "estimate_tile_bytes", lambda *a: 1)
    index = make_tile_index(tmp_path, size=20000)
    # every single-tile ZIP (>= 20000 B STORED + headers + XML) is over the limit even after 2 repacks
    cfg = with_cvat(base_cfg, max_zip_bytes=20050)
    with pytest.raises(ExportBlocked, match="packing"):
        export_upload(make_annset(TILES), index, cfg, tmp_path / "marcaj_upload")
    failed = [p for p in tmp_path.iterdir() if ".FAILED-" in p.name]
    assert len(failed) == 1 and not any(".staging-" in p.name for p in tmp_path.iterdir())
    report = json.loads((failed[0] / "validation_report.json").read_text())
    assert report["by_code"] == {"packing_failed": 1}
    assert "after repacking" in report["issues"][0]["message"]


def test_tile_sources_errors(tmp_path: Path) -> None:
    index = make_tile_index(tmp_path)
    assert [s.tile_id for s in tile_sources(index, [TILES[1]])] == [TILES[1]]
    with pytest.raises(ExportBlocked):
        tile_sources(index, ["siret3_r030_c030"])
    Path(index["path"][0]).unlink()
    with pytest.raises(ExportBlocked):
        tile_sources(index)


# ---------------------------------------------------------------- stage export_cvat


def _stage_ctx(work: Path, annset_ref: str | None, *, allow_qa: bool = False, run_id: str = "exp-run") -> RunContext:
    cfg = with_cvat(load_config(), allow_qa_errors=allow_qa)
    assert cfg.paths.work_dir == work
    return new_run_context(cfg, source=Source.MODEL, run_id=run_id, workers=1, annset_ref=annset_ref,
                           kind=POST_KIND)


def _stage_work(tmp_work: Path) -> Path:
    index = make_tile_index(tmp_work)
    write_layer(index, "tile_index", tmp_work / "tile_index.parquet")
    return tmp_work


def test_stage_empty_ref_exports_every_tile(tmp_work: Path) -> None:
    work = _stage_work(tmp_work)
    ctx = _stage_ctx(work, EMPTY_REF)
    ensure_run_dirs(ctx.paths)
    result = export_stage.run(ctx)
    out = ctx.paths.exports_dir / "marcaj_upload"
    assert result.n_items == 3 and result.n_failed == 0 and result.metrics["n_zips"] == 1.0
    assert (out / "siret3_upload_01of01.zip").is_file() and out / "upload_manifest.csv" in result.outputs
    assert [r["tile_id"] for r in _read_csv(out / "empty_tiles.csv")] == list(TILES)


def test_stage_tile_filter(tmp_work: Path) -> None:
    work = _stage_work(tmp_work)
    ctx = replace(_stage_ctx(work, EMPTY_REF), tile_filter=("*_r021_*",))
    result = export_stage.run(ctx)
    assert result.n_items == 1 and result.metrics["n_tiles"] == 1.0
    with pytest.raises(StageError):
        export_stage.run(replace(ctx, tile_filter=("nothing*",)))


def _model_run(work: Path, qa_severity: str | None) -> Path:
    run_dir = work / "runs" / RUN
    t = TILES[2]
    annset = make_annset(TILES, canopies=[canopy_row(t, 1, px_square(10, 10, 20))])
    write_annset(annset, run_dir / "annset")
    if qa_severity is not None:
        qa = make_layer("qa_issues", [{"issue_id": "Q00001", "severity": qa_severity, "code": "x", "tile_id": t,
                                       "geometry": None}])
        write_layer(qa, "qa_issues", run_dir / "qa" / "qa_issues.parquet")
    return run_dir


def test_stage_exports_into_the_annset_run(tmp_work: Path) -> None:
    work = _stage_work(tmp_work)
    run_dir = _model_run(work, "warning")
    result = export_stage.run(_stage_ctx(work, RUN))
    out = run_dir / "exports" / "marcaj_upload"
    manifest = {r["tile_id"]: r["n_vineyard"] for r in _read_csv(out / "upload_manifest.csv")}
    assert manifest[TILES[2]] == "1" and result.outputs[0].parent == out


def test_stage_qa_errors_block_unless_allowed(tmp_work: Path) -> None:
    work = _stage_work(tmp_work)
    _model_run(work, "error")
    with pytest.raises(ExportBlocked):
        export_stage.run(_stage_ctx(work, RUN))
    assert export_stage.run(_stage_ctx(work, RUN, allow_qa=True)).n_failed == 0


def test_stage_missing_annset(tmp_work: Path) -> None:
    work = _stage_work(tmp_work)
    (work / "runs" / "empty-run").mkdir(parents=True)
    with pytest.raises(StageError):
        export_stage.run(_stage_ctx(work, "empty-run"))
    with pytest.raises(StageError):
        export_stage.run(_stage_ctx(work, None))


def test_stage_through_the_runner_exit_codes(tmp_work: Path) -> None:
    work = _stage_work(tmp_work)
    assert export_stage.STAGE.name == "export_cvat" and export_stage.STAGE.scope == "global"
    assert run_stages(_stage_ctx(work, EMPTY_REF), ("export_cvat",)).exit_code == 0
    _model_run(work, "error")
    assert run_stages(_stage_ctx(work, RUN, run_id="exp-run2"), ("export_cvat",)).exit_code == 2


def test_qa_error_gate() -> None:
    qa = gpd.GeoDataFrame({"severity": ["error", "warning"], "code": ["missing_attr", "low_snr"]},
                          geometry=[None, None], crs=CRS_EPSG)
    assert qa_error_count(qa) == 1 and qa_error_count(None) == 0
    with pytest.raises(ExportBlocked):
        require_no_qa_errors(qa, allow=False)
    require_no_qa_errors(qa, allow=True)
    require_no_qa_errors(None, allow=False)
