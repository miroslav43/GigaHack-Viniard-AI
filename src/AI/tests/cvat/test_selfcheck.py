"""cvat.selfcheck: written ZIPs re-read with the tolerant reader and compared to what we meant to write."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import pytest
from shapely.geometry import LineString

from tests.cvat.test_to_cvat import (
    canopy_row,
    interrow_row,
    make_annset,
    px_square,
    row_row,
    waste_row,
)
from vineyard.config import load_config
from vineyard.config.sections_io import CvatExportConfig
from vineyard.cvat.model import CvatDocument
from vineyard.cvat.packer import write_upload_zip
from vineyard.cvat.selfcheck import compare_documents, selfcheck_upload
from vineyard.cvat.to_cvat import annset_to_images
from vineyard.cvat.writer import serialize_document
from vineyard.geo.tiling import tile_ref

T1, T2 = "siret3_r006_c004", "siret3_r021_c012"
PX = 2048


@pytest.fixture(scope="module")
def cfg() -> CvatExportConfig:
    return load_config().export.cvat


def _annset():
    return make_annset(
        [T1, T2],
        canopies=[canopy_row(T2, 1, px_square(10, 10, 20)), canopy_row(T2, 2, px_square(60, 10, 25))],
        row_pieces=[row_row(T2, 1, LineString([(0, 100), (2048, 120)]))],
        interrow_pieces=[interrow_row(T2, 1, px_square(500, 500, 100))],
        waste=[waste_row(T1, 1, px_square(40, 40, 10))],
    )


def _write(tmp_path: Path, doc: CvatDocument, name: str = "p1.zip") -> tuple[Path, dict[str, str]]:
    images = []
    for img in doc.images:
        src = tmp_path / "src" / img.name
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(f"tif {img.name}".encode())
        images.append((img.name, src))
    out = tmp_path / name
    shas = write_upload_zip(out, serialize_document(doc), images, deflate_level=9)
    return out, {n[:-4]: s for n, s in shas.items()}


def _doc(cfg: CvatExportConfig) -> CvatDocument:
    images, _ = annset_to_images(_annset(), [tile_ref(T1), tile_ref(T2)], cfg)
    return CvatDocument(images)


def test_clean_upload_passes(tmp_path: Path, cfg: CvatExportConfig) -> None:
    doc = _doc(cfg)
    path, shas = _write(tmp_path, doc)
    rep = selfcheck_upload([(path, doc)], expected_sha256=shas, expected_tiles=frozenset({T1, T2}), cfg=cfg,
                           annset=_annset())
    assert rep.ok, rep.summary_lines()


def test_missing_tile_and_sha_mismatch(tmp_path: Path, cfg: CvatExportConfig) -> None:
    doc = _doc(cfg)
    path, shas = _write(tmp_path, doc)
    rep = selfcheck_upload([(path, doc)], expected_sha256=shas | {T1: "0" * 64},
                           expected_tiles=frozenset({T1, T2, "siret3_r018_c010"}), cfg=cfg)
    codes = {i.code for i in rep.errors}
    assert {"selfcheck_sha256", "selfcheck_tiles"} <= codes


def test_duplicate_tile_across_parts(tmp_path: Path, cfg: CvatExportConfig) -> None:
    doc = _doc(cfg)
    a, shas = _write(tmp_path / "a", doc, "p1.zip")
    b, _ = _write(tmp_path / "b", doc, "p2.zip")
    rep = selfcheck_upload([(a, doc), (b, doc)], expected_sha256=shas, expected_tiles=frozenset({T1, T2}), cfg=cfg)
    assert "selfcheck_tiles" in {i.code for i in rep.errors}


def test_tampered_xml_is_detected(tmp_path: Path, cfg: CvatExportConfig) -> None:
    doc = _doc(cfg)
    path, shas = _write(tmp_path, doc)
    with zipfile.ZipFile(path) as zf:
        entries = {n: zf.read(n) for n in zf.namelist()}
    xml = re.sub(rb'(<polygon label="vineyard"[^>]*points=")(\d+\.\d)',
                 lambda m: m.group(1) + f"{float(m.group(2)) + 0.2:.1f}".encode(),
                 entries["annotations.xml"].replace(b">regular<", b">disrupted<"), count=1)
    tampered = tmp_path / "t.zip"
    with zipfile.ZipFile(tampered, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, xml if name == "annotations.xml" else data)
    rep = selfcheck_upload([(tampered, doc)], expected_sha256=shas, expected_tiles=frozenset({T1, T2}), cfg=cfg)
    codes = {i.code for i in rep.errors}
    assert {"selfcheck_attributes", "selfcheck_vertices"} <= codes


def test_unreadable_zip(tmp_path: Path, cfg: CvatExportConfig) -> None:
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"nope")
    rep = selfcheck_upload([(junk, _doc(cfg))], expected_sha256={}, expected_tiles=frozenset({T1, T2}), cfg=cfg)
    assert "selfcheck_unreadable" in {i.code for i in rep.errors}


def test_compare_documents_counts_and_names(cfg: CvatExportConfig) -> None:
    doc = _doc(cfg)
    fewer = CvatDocument((doc.images[0], doc.images[1].with_shapes(doc.images[1].shapes[:-1])))
    assert "selfcheck_count" in {i.code for i in compare_documents(doc, fewer, max_dev_px=0.06, zip_name="z")}
    renamed = CvatDocument((doc.images[0],))
    assert "selfcheck_images" in {i.code for i in compare_documents(doc, renamed, max_dev_px=0.06, zip_name="z")}
    assert compare_documents(doc, doc, max_dev_px=0.06, zip_name="z") == []


def test_geometry_mismatch_against_annset_is_detected(tmp_path: Path, cfg: CvatExportConfig) -> None:
    doc = _doc(cfg)
    path, shas = _write(tmp_path, doc)
    moved = make_annset(
        [T1, T2],
        canopies=[canopy_row(T2, 1, px_square(900, 900, 20)), canopy_row(T2, 2, px_square(60, 10, 25))],
        row_pieces=[row_row(T2, 1, LineString([(0, 100), (2048, 120)]))],
        interrow_pieces=[interrow_row(T2, 1, px_square(500, 500, 100))],
        waste=[waste_row(T1, 1, px_square(40, 40, 10))],
    )
    rep = selfcheck_upload([(path, doc)], expected_sha256=shas, expected_tiles=frozenset({T1, T2}), cfg=cfg,
                           annset=moved)
    assert [i.code for i in rep.errors] == ["selfcheck_union_iou"]
    assert hashlib.sha256(b"x").hexdigest()


def test_deliberately_emptied_tiles_skip_the_geometry_check(tmp_path: Path, cfg: CvatExportConfig) -> None:
    doc = _doc(cfg)
    emptied = CvatDocument((doc.images[0], doc.images[1].with_shapes(())))
    path, shas = _write(tmp_path, emptied)
    kwargs = {"expected_sha256": shas, "expected_tiles": frozenset({T1, T2}), "cfg": cfg, "annset": _annset()}
    assert not selfcheck_upload([(path, emptied)], **kwargs).ok
    assert selfcheck_upload([(path, emptied)], **kwargs, skip_geometry=frozenset({T2})).ok
