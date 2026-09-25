"""cvat.validator_zip: whitelist, name bijection, image ids, sha256, size, meta, and the upload set."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from vineyard.config import load_config
from vineyard.config.sections_io import CvatExportConfig
from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.packer import write_upload_zip
from vineyard.cvat.template import META_BLOCK
from vineyard.cvat.validator_zip import parse_annotations, validate_upload_set, validate_zip
from vineyard.cvat.writer import serialize_document

PX = 2048
T1, T2, T3 = "siret3_r006_c004", "siret3_r021_c012", "siret3_r018_c010"


@pytest.fixture(scope="module")
def cfg() -> CvatExportConfig:
    return load_config().export.cvat


def _row() -> CvatShape:
    attrs = (("vineyard_id", "V01"), ("row_id", "V01-R001"), ("row_structure", "regular"))
    return CvatShape("polyline", "row", ((0.0, 5.0), (300.0, 5.0)), attrs)


def _doc(tiles: list[str], shapes: dict[str, tuple[CvatShape, ...]] | None = None) -> CvatDocument:
    shapes = shapes or {}
    return CvatDocument(tuple(
        CvatImage(k, f"{t}.tif", PX, PX, shapes.get(t, ())) for k, t in enumerate(sorted(tiles))
    ))


def _tifs(tmp_path: Path, tiles: list[str]) -> tuple[list[tuple[str, Path]], dict[str, str]]:
    images, shas = [], {}
    for t in tiles:
        data = f"fake tif {t}".encode()
        path = tmp_path / "src" / f"{t}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        images.append((f"{t}.tif", path))
        shas[t] = hashlib.sha256(data).hexdigest()
    return images, shas


def _good_zip(tmp_path: Path, tiles: list[str], name: str = "part.zip") -> tuple[Path, dict[str, str]]:
    images, shas = _tifs(tmp_path, tiles)
    xml = serialize_document(_doc(tiles, {tiles[0]: (_row(),)}))
    out = tmp_path / name
    write_upload_zip(out, xml, images, deflate_level=9)
    return out, shas


def _codes(path: Path, shas: dict[str, str], cfg: CvatExportConfig) -> list[str]:
    return sorted(i.code for i in validate_zip(path, expected_sha256=shas, cfg=cfg, tile_px=PX).errors)


def _rezip(src: Path, dst: Path, *, drop: tuple[str, ...] = (), add: dict[str, bytes] | None = None,
           replace: dict[str, bytes] | None = None) -> Path:
    replace = replace or {}
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            if info.filename in drop:
                continue
            zout.writestr(info.filename, replace.get(info.filename, zin.read(info.filename)))
        for name, data in (add or {}).items():
            zout.writestr(name, data)
    return dst


def test_good_zip_passes(tmp_path: Path, cfg: CvatExportConfig) -> None:
    path, shas = _good_zip(tmp_path, [T1, T2])
    rep = validate_zip(path, expected_sha256=shas, cfg=cfg, tile_px=PX)
    assert rep.ok, rep.summary_lines()


@pytest.mark.parametrize(
    ("add", "code"),
    [
        ({".DS_Store": b"x"}, "zip_forbidden_entry"),
        ({"images/.DS_Store": b"x"}, "zip_forbidden_entry"),
        ({"__MACOSX/._annotations.xml": b"x"}, "zip_forbidden_entry"),
        ({"debug.xml": b"<a/>"}, "zip_extra_xml"),
        ({"images/": b""}, "zip_dir_entry"),
        ({"readme.txt": b"x"}, "zip_extra_entry"),
        ({"images/sub/siret3_r018_c010.tif": b"x"}, "zip_extra_entry"),
    ],
)
def test_whitelist(tmp_path: Path, cfg: CvatExportConfig, add: dict[str, bytes], code: str) -> None:
    path, shas = _good_zip(tmp_path, [T1, T2])
    bad = _rezip(path, tmp_path / "bad.zip", add=add)
    assert code in _codes(bad, shas, cfg)


def test_orphan_image_file_and_orphan_xml_name(tmp_path: Path, cfg: CvatExportConfig) -> None:
    path, shas = _good_zip(tmp_path, [T1, T2])
    orphan_file = _rezip(path, tmp_path / "o1.zip", add={f"images/{T3}.tif": b"x"})
    assert "file_without_image" in _codes(orphan_file, shas | {T3: hashlib.sha256(b"x").hexdigest()}, cfg)
    orphan_name = _rezip(path, tmp_path / "o2.zip", drop=(f"images/{T2}.tif",))
    assert "image_without_file" in _codes(orphan_name, shas, cfg)


@pytest.mark.parametrize(
    ("entry", "code"),
    [
        ("images/SIRET3_R021_C012.tif", "image_name_invalid"),
        ("images/siret3_r021_c012.TIF", "image_bad_ext"),
    ],
)
def test_case_mismatch_is_an_error(tmp_path: Path, cfg: CvatExportConfig, entry: str, code: str) -> None:
    path, shas = _good_zip(tmp_path, [T1, T2])
    with zipfile.ZipFile(path) as zin:
        data = zin.read(f"images/{T2}.tif")
    bad = _rezip(path, tmp_path / "case.zip", drop=(f"images/{T2}.tif",), add={entry: data})
    codes = _codes(bad, shas, cfg)
    assert code in codes and "image_without_file" in codes


def test_nfd_image_name_is_an_error(tmp_path: Path, cfg: CvatExportConfig) -> None:
    path, shas = _good_zip(tmp_path, [T1, T2])
    bad = _rezip(path, tmp_path / "nfd.zip", add={"images/siret3_r021_ç012.tif": b"x"})  # c + cedilla
    assert "image_name_not_nfc" in _codes(bad, shas, cfg)


def test_sha_mismatch_and_unknown(tmp_path: Path, cfg: CvatExportConfig) -> None:
    path, shas = _good_zip(tmp_path, [T1, T2])
    assert "sha256_mismatch" in _codes(path, shas | {T1: "0" * 64}, cfg)
    assert "sha256_unknown" in _codes(path, {T1: shas[T1]}, cfg)


def test_zip_over_the_size_limit(tmp_path: Path, cfg: CvatExportConfig) -> None:
    path, shas = _good_zip(tmp_path, [T1, T2])
    small = cfg.model_copy(update={"max_zip_bytes": 100})
    rep = validate_zip(path, expected_sha256=shas, cfg=small, tile_px=PX)
    assert "zip_too_large" in [i.code for i in rep.errors]


def test_image_ids_must_follow_sorted_names(tmp_path: Path, cfg: CvatExportConfig) -> None:
    images, shas = _tifs(tmp_path, [T1, T2])
    unsorted = CvatDocument((CvatImage(0, f"{T2}.tif", PX, PX, ()), CvatImage(1, f"{T1}.tif", PX, PX, ())))
    out = tmp_path / "ids.zip"
    write_upload_zip(out, serialize_document(unsorted), images, deflate_level=9)
    assert "image_id_sequence" in _codes(out, shas, cfg)
    gap = CvatDocument((CvatImage(0, f"{T1}.tif", PX, PX, ()), CvatImage(2, f"{T2}.tif", PX, PX, ())))
    write_upload_zip(out, serialize_document(gap), images, deflate_level=9)
    assert "image_id_sequence" in _codes(out, shas, cfg)


def test_meta_must_be_verbatim(tmp_path: Path, cfg: CvatExportConfig) -> None:
    images, shas = _tifs(tmp_path, [T1])
    doc = CvatDocument(_doc([T1]).images, meta_xml=META_BLOCK.replace("regular\n", "Regular\n"))
    out = tmp_path / "meta.zip"
    write_upload_zip(out, serialize_document(doc), images, deflate_level=9)
    assert "meta_mismatch" in _codes(out, shas, cfg)


def test_missing_xml_corrupt_zip_and_missing_file(tmp_path: Path, cfg: CvatExportConfig) -> None:
    path, shas = _good_zip(tmp_path, [T1])
    no_xml = _rezip(path, tmp_path / "noxml.zip", drop=("annotations.xml",))
    assert "zip_missing_xml" in _codes(no_xml, shas, cfg)
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not a zip")
    assert _codes(junk, shas, cfg) == ["zip_corrupt"]
    assert _codes(tmp_path / "missing.zip", shas, cfg) == ["zip_missing"]


def test_unparseable_xml_and_bad_shapes(tmp_path: Path, cfg: CvatExportConfig) -> None:
    path, shas = _good_zip(tmp_path, [T1])
    broken = _rezip(path, tmp_path / "broken.zip", replace={"annotations.xml": b"<annotations><image"})
    assert "xml_unparseable" in _codes(broken, shas, cfg)


def test_document_errors_surface_through_the_zip(tmp_path: Path, cfg: CvatExportConfig) -> None:
    images, shas = _tifs(tmp_path, [T1])
    attrs = (("vineyard_id", "V01"), ("row_id", "V01-R001"), ("row_structure", "Regular"))
    bad_row = CvatShape("polyline", "row", ((0.0, 5.0), (300.0, 5.0)), attrs)
    out = tmp_path / "doc.zip"
    write_upload_zip(out, serialize_document(_doc([T1], {T1: (bad_row,)})), images, deflate_level=9)
    assert "bad_enum" in _codes(out, shas, cfg)


def test_parse_annotations_strict() -> None:
    xml = serialize_document(_doc([T1], {T1: (_row(),)}))
    doc, issues = parse_annotations(xml, zip_name="p.zip")
    assert issues == [] and doc is not None
    assert doc.images[0].shapes[0].attributes == _row().attributes
    track = xml.replace(b"</image>", b'<track id="0" label="row"></track>\n</image>')
    _, issues = parse_annotations(track, zip_name="p.zip")
    assert [i.code for i in issues] == ["unsupported_element"]
    badpts = xml.replace(b'points="0.0,5.0;300.0,5.0"', b'points="0.0,5.0;abc"')
    _, issues = parse_annotations(badpts, zip_name="p.zip")
    assert [i.code for i in issues] == ["shape_unparseable"]


def test_upload_set_union_duplicates_and_missing(tmp_path: Path, cfg: CvatExportConfig) -> None:
    a, sha_a = _good_zip(tmp_path / "a", [T1, T2], "p1.zip")
    b, sha_b = _good_zip(tmp_path / "b", [T3], "p2.zip")
    shas = sha_a | sha_b
    ok = validate_upload_set([a, b], expected_tiles=frozenset({T1, T2, T3}), expected_sha256=shas, cfg=cfg,
                             tile_px=PX)
    assert ok.ok, ok.summary_lines()
    missing = validate_upload_set([a], expected_tiles=frozenset({T1, T2, T3}), expected_sha256=shas, cfg=cfg,
                                  tile_px=PX)
    assert [i.code for i in missing.errors] == ["missing_tile"]
    c, _ = _good_zip(tmp_path / "c", [T2], "p3.zip")
    dup = validate_upload_set([a, b, c], expected_tiles=frozenset({T1, T2, T3}), expected_sha256=shas, cfg=cfg,
                              tile_px=PX)
    assert "duplicate_tile" in [i.code for i in dup.errors]
    extra = validate_upload_set([a, b], expected_tiles=frozenset({T1, T3}), expected_sha256=shas, cfg=cfg,
                                tile_px=PX)
    assert "unexpected_tile" in [i.code for i in extra.errors]
    same_name = validate_upload_set([a, a], expected_tiles=frozenset({T1, T2}), expected_sha256=shas, cfg=cfg,
                                    tile_px=PX)
    assert "duplicate_zip" in [i.code for i in same_name.errors]
