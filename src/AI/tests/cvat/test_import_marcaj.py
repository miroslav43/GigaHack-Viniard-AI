"""Stage import_marcaj: Marcaj CVAT 1.1 export(s) -> AnnSet(marcaj) + qa_issues + LATEST_MARCAJ."""

import json
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from vineyard.annset.io import read_annset, resolve_run_dir
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.contracts.ids import tile_grid_ids
from vineyard.errors import StageError
from vineyard.geo.vector_io import read_layer
from vineyard.pipeline.context import ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages.import_marcaj import (
    STAGE,
    build_marcaj_annset,
    read_export,
    tile_id_of,
)

RUN_A = "20260927T1200-marcaj-aaaaaa"
PARTIAL = "import.require_all_tiles=false"
FIXTURE_XML = Path(__file__).resolve().parents[1] / "fixtures" / "cvat" / "marcaj_export.xml"
GRID = tile_grid_ids()
T1, T2, T3 = GRID[0], GRID[1], GRID[2]

ROW = """<polyline label="row" source="manual" occluded="0" points="100.00,1000.00;1900.00,1000.00" z_order="0">
      <attribute name="vineyard_id">V01</attribute><attribute name="row_id">V01-R01</attribute>
      <attribute name="row_structure">regular</attribute></polyline>"""
INTERROW = """<polygon label="interrow_area" source="manual" occluded="0"
      points="100.00,1020.00;1900.00,1020.00;1900.00,1100.00;100.00,1100.00" z_order="0">
      <attribute name="vineyard_id">V01</attribute><attribute name="interrow_cover">vegetation</attribute></polygon>"""


def _canopy(x: float) -> str:
    pts = f"{x:.2f},990.00;{x + 20:.2f},990.00;{x + 20:.2f},1010.00;{x:.2f},1010.00"
    return (f'<polygon label="vineyard" source="manual" occluded="0" points="{pts}" z_order="0">'
            '<attribute name="vineyard_id">V01</attribute></polygon>')


def _image(k: int, tile_id: str, n_canopies: int = 0, *, rows: bool = False, prefix: str = "") -> str:
    shapes = ([ROW, INTERROW] if rows else []) + [_canopy(200.0 + 50 * i) for i in range(n_canopies)]
    return (f'<image id="{k}" name="{prefix}{tile_id}.tif" subset="default" task_id="7" width="2048" '
            f'height="2048">{"".join(shapes)}</image>')


def _xml(images: Sequence[str]) -> bytes:
    return ('<?xml version="1.0" encoding="utf-8"?>\n<annotations><version>1.1</version>'
            f'<meta><dumped>2026-09-27</dumped></meta>{"".join(images)}</annotations>').encode()


def _write_xml(path: Path, images: Sequence[str]) -> Path:
    path.write_bytes(_xml(images))
    return path


def _write_zip(path: Path, images: Sequence[str], *, inner: str = "annotations.xml") -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(inner, _xml(images))
        zf.writestr("images/ignored.tif", b"not a raster")
    return path


def _ctx(files: Sequence[Path], *overrides: str):
    cfg = load_config(overrides=(f"import.files={json.dumps([str(f) for f in files])}", *overrides))
    ctx = new_run_context(cfg, source=Source.MARCAJ, run_id=RUN_A, workers=1)
    ensure_run_dirs(ctx.paths)
    return ctx


def _qa_codes(ctx) -> list[str]:
    return list(read_layer(ctx.paths.qa_dir / "qa_issues.parquet", "qa_issues")["code"])


def test_stage_spec() -> None:
    assert load_stage("import_marcaj") is STAGE
    assert (STAGE.name, STAGE.scope) == ("import_marcaj", "global")
    assert "import" in STAGE.cfg_keys and "canopy" in STAGE.cfg_keys


def test_tile_id_of_strips_prefix_and_raster_suffix() -> None:
    assert tile_id_of(f"images/{T1}.tif") == T1
    assert tile_id_of(f"{T1}.JPG") == T1
    assert tile_id_of(T1) == T1


def test_multi_file_merge(tmp_work: Path, tmp_path: Path) -> None:
    a = _write_xml(tmp_path / "a.xml", [_image(0, T1, 2, rows=True)])
    b = _write_zip(tmp_path / "b.zip", [_image(0, T2, 3, rows=True), _image(1, T3)])
    ctx = _ctx([a, b], PARTIAL)
    result = STAGE.run(ctx)
    annset = read_annset(ctx.paths.annset_dir)
    assert annset.meta.source == Source.MARCAJ
    assert annset.meta.tile_ids == (T1, T2, T3)
    assert annset.counts() == {"canopies": 5, "row_pieces": 2, "interrow_pieces": 2, "waste": 0}
    assert annset.meta.model_version.startswith("marcaj-export@") and len(annset.meta.model_version) == 22
    assert (result.n_items, result.metrics["canopies"], result.metrics["files"]) == (3, 5.0, 2.0)
    assert resolve_run_dir(tmp_work, "LATEST_MARCAJ") == ctx.paths.run_dir.resolve()
    assert _qa_codes(ctx) == ["images_missing"]


def test_duplicate_tile_last_wins(tmp_work: Path, tmp_path: Path) -> None:
    first = _write_xml(tmp_path / "first.xml", [_image(0, T1, 1, rows=True)])
    last = _write_xml(tmp_path / "last.xml", [_image(0, T1, 4, rows=True, prefix="images/")])
    ctx = _ctx([first, last], PARTIAL)
    STAGE.run(ctx)
    annset = read_annset(ctx.paths.annset_dir)
    assert annset.meta.tile_ids == (T1,)
    assert annset.counts()["canopies"] == 4
    assert "schema_violation" in _qa_codes(ctx)


def test_duplicate_policy_error_raises(tmp_work: Path, tmp_path: Path) -> None:
    a = _write_xml(tmp_path / "a.xml", [_image(0, T1)])
    b = _write_xml(tmp_path / "b.xml", [_image(0, T1)])
    with pytest.raises(StageError, match="duplicate"):
        STAGE.run(_ctx([a, b], PARTIAL, "import.duplicate_policy=error"))


def test_require_all_tiles_errors_on_missing(tmp_work: Path, tmp_path: Path) -> None:
    a = _write_xml(tmp_path / "a.xml", [_image(0, T1, 1, rows=True)])
    ctx = _ctx([a])
    with pytest.raises(StageError, match=f"1 of {len(GRID)} tiles, {len(GRID) - 1} missing") as exc:
        STAGE.run(ctx)
    assert T2 in str(exc.value)
    assert not (ctx.paths.annset_dir / "annset.json").exists()


def test_all_tiles_across_files_pass(tmp_work: Path, tmp_path: Path) -> None:
    half = len(GRID) // 2
    a = _write_zip(tmp_path / "a.zip", [_image(k, t) for k, t in enumerate(GRID[:half])])
    b = _write_zip(tmp_path / "b.zip", [_image(k, t) for k, t in enumerate(GRID[half:])])
    ctx = _ctx([a, b])
    assert STAGE.run(ctx).n_items == len(GRID)
    assert read_annset(ctx.paths.annset_dir).meta.tile_ids == GRID
    assert "images_missing" not in _qa_codes(ctx)


def test_zip_with_annotations_in_subfolder(tmp_path: Path) -> None:
    z = _write_zip(tmp_path / "job.zip", [_image(0, T1, 2, rows=True)], inner="task_7/job_12/annotations.xml")
    annset, _ = build_marcaj_annset(load_config(overrides=(PARTIAL,)), [read_export(z)], run_id=RUN_A)
    assert annset.meta.tile_ids == (T1,)
    assert annset.counts()["canopies"] == 2


def test_unknown_image_is_a_qa_error(tmp_path: Path) -> None:
    a = _write_xml(tmp_path / "a.xml", [_image(0, T1, 1, rows=True), _image(1, "siret3_r999_c999")])
    annset, qa = build_marcaj_annset(load_config(overrides=(PARTIAL,)), [read_export(a)], run_id=RUN_A)
    assert annset.meta.tile_ids == (T1,)
    unknown = qa[qa["object_id"] == "siret3_r999_c999.tif"]
    assert list(unknown["severity"]) == ["error"]


def test_missing_or_bad_files_raise(tmp_work: Path, tmp_path: Path) -> None:
    empty_zip = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w") as zf:
        zf.writestr("images/a.tif", b"x")
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_bytes(b"<annotations><image")
    for path in (tmp_path / "nope.zip", empty_zip, bad_xml):
        with pytest.raises(StageError):
            STAGE.run(_ctx([path], PARTIAL))
    with pytest.raises(StageError, match="import.files is empty"):
        STAGE.run(_ctx([], PARTIAL))


def test_marcaj_export_fixture(tmp_path: Path) -> None:
    """The Marcaj-style export (images/ prefix, subset/task_id, masks, tags, raw enum spellings)."""
    annset, qa = build_marcaj_annset(load_config(overrides=(PARTIAL,)), [read_export(FIXTURE_XML)], run_id=RUN_A)
    assert annset.meta.tile_ids == ("siret3_r005_c004", "siret3_r006_c004", "siret3_r021_c012")
    counts = annset.counts()
    assert counts["row_pieces"] == 1 and counts["interrow_pieces"] == 1 and counts["waste"] == 1
    assert counts["canopies"] >= 1
    codes = set(qa["code"])
    assert {"images_missing", "enum_normalized", "id_whitespace"} <= codes
