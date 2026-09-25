"""cvat.reader: tolerant CVAT 1.1 parser (ZIP or XML, Marcaj quirks, multi-file merge)."""

from pathlib import Path

import pytest
from shapely.geometry import Polygon

from vineyard.contracts.enums import Severity
from vineyard.cvat.reader import (
    image_key,
    load_annotation_bytes,
    parse_xml,
    read_cvat_file,
    read_cvat_files,
)
from vineyard.errors import CvatFormatError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cvat"
MARCAJ = FIXTURES / "marcaj_export.xml"
TRACK = FIXTURES / "marcaj_track.xml"

MINI = (
    b'<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n<version>1.1</version>\n'
    b'<image id="0" name="{name}" width="2048" height="2048">\n'
    b'<polygon label="vineyard" source="manual" occluded="0" points="{pts}" z_order="0">'
    b'<attribute name="vineyard_id">{vid}</attribute></polygon>\n</image>\n</annotations>\n'
)


def _mini(name: str = "siret3_r021_c012.tif", pts: str = "0,0;10,0;10,10", vid: str = "V01") -> bytes:
    return MINI.replace(b"{name}", name.encode()).replace(b"{pts}", pts.encode()).replace(b"{vid}", vid.encode())


def _codes(issues: tuple) -> list[tuple[str, str]]:
    return [(i.severity.value, i.code) for i in issues]


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("siret3_r021_c012.tif", "siret3_r021_c012.tif"),
        ("images/siret3_r021_c012.tif", "siret3_r021_c012.tif"),
        ("a\\b\\siret3_r021_c012.tif", "siret3_r021_c012.tif"),
        ("siret3_r021_c012", "siret3_r021_c012.tif"),
        (" images/siret3_r021_c012.tif ", "siret3_r021_c012.tif"),
    ],
)
def test_image_key(name: str, key: str) -> None:
    assert image_key(name) == key


def test_image_key_is_nfc() -> None:
    assert image_key("café.tif") == "café.tif"


@pytest.mark.examples
def test_example_counts(examples_xml: bytes) -> None:
    doc, issues = parse_xml(examples_xml, source_name="annotations.xml")
    assert issues == ()
    r021, r006 = doc.images
    assert (r021.count("row"), r021.count("interrow_area"), r021.count("vineyard"), r021.count("waste")) == (
        25, 24, 399, 0)
    assert (r006.count("row"), r006.count("interrow_area"), r006.count("vineyard"), r006.count("waste")) == (
        26, 25, 251, 0)
    assert {s.source for img in doc.images for s in img.shapes} == {"manual"}
    assert r021.shapes[0].points == ((2048.0, 32.1), (2017.8, 0.0))


@pytest.mark.examples
def test_example_nested_in_zip_parses_identically(examples_xml: bytes, tmp_path: Path, make_zip) -> None:
    z = make_zip(tmp_path / "exp.zip", {"export/job_1/annotations.xml": examples_xml, "export/readme.txt": b"x"})
    assert load_annotation_bytes(z) == examples_xml
    assert read_cvat_file(z)[0] == parse_xml(examples_xml, source_name="x")[0]


def test_marcaj_export_variant() -> None:
    doc, issues = read_cvat_file(MARCAJ)
    assert [i.name for i in doc.images] == [
        "images/siret3_r021_c012.tif", "images/siret3_r006_c004.tif", "images/siret3_r005_c004.tif"]
    assert [i.id for i in doc.images] == [57, 58, 59]
    r021 = doc.images[0]
    assert [s.label for s in r021.shapes] == ["row", "interrow_area", "vineyard", "waste", "vineyard"]
    row = r021.shapes[0]
    assert row.points == ((2048.0, 32.1), (2017.8, 0.0))
    assert row.source == "file"
    assert row.attributes == (("vineyard_id", "V01"), ("row_id", "V01-R01"), ("row_structure", "Regular "))
    assert r021.shapes[2].occluded == 1 and r021.shapes[2].z_order == 2
    assert r021.shapes[2].attr("vineyard_id") == " V01 "  # raw values are kept
    assert r021.shapes[3].box_xyxy == (500.0, 600.0, 540.0, 630.0)
    assert r021.shapes[3].attr("vineyard_id") == ""
    assert doc.images[1].shapes == () and doc.images[2].shapes == ()
    assert "<job>" in (doc.meta_xml or "")
    codes = _codes(issues)
    assert ("warning", "geom_repaired") in codes  # mask -> polygon
    assert codes.count(("info", "geom_dropped")) == 2  # tag + points


def test_mask_becomes_pixel_exact_polygon() -> None:
    doc, _ = read_cvat_file(MARCAJ)
    mask_poly = doc.images[0].shapes[4]
    assert mask_poly.tag == "polygon" and mask_poly.source == "semi-auto"
    assert Polygon(mask_poly.points).area == pytest.approx(8.0)
    assert mask_poly.attr("vineyard_id") == "V01"


def test_masks_rejected_when_not_accepted() -> None:
    doc, issues = parse_xml(MARCAJ.read_bytes(), source_name="m", accept_masks=False)
    assert doc.images[0].count("vineyard") == 1
    assert any(i.severity == Severity.ERROR and "mask" in i.message for i in issues)


def test_track_and_bad_shapes_are_error_issues() -> None:
    doc, issues = read_cvat_file(TRACK)
    assert len(doc.images[0].shapes) == 1
    errors = [i for i in issues if i.severity == Severity.ERROR]
    assert len(errors) == 3  # track, unparseable points, ellipse
    assert any("track" in i.message for i in errors)
    assert all(i.code == "geom_dropped" for i in errors)
    assert {i.tile_id for i in errors} == {"", "siret3_r006_c004"}


def test_duplicate_image_last_wins_with_warning(tmp_path: Path) -> None:
    a = tmp_path / "a.xml"
    b = tmp_path / "b.xml"
    a.write_bytes(_mini(vid="OLD"))
    b.write_bytes(_mini(name="images/siret3_r021_c012.tif", vid="NEW"))
    doc, issues = read_cvat_files([a, b], duplicate_policy="last_wins")
    assert len(doc.images) == 1
    assert doc.images[0].shapes[0].attr("vineyard_id") == "NEW"
    assert ("warning", "schema_violation") in _codes(issues)
    with pytest.raises(CvatFormatError):
        read_cvat_files([a, b], duplicate_policy="error")


def test_multi_file_merge_sorts_and_renumbers(tmp_path: Path) -> None:
    a = tmp_path / "a.xml"
    a.write_bytes(_mini(name="siret3_r021_c012.tif"))
    doc, issues = read_cvat_files([MARCAJ, a], duplicate_policy="last_wins")
    assert [image_key(i.name) for i in doc.images] == [
        "siret3_r005_c004.tif", "siret3_r006_c004.tif", "siret3_r021_c012.tif"]
    assert [i.id for i in doc.images] == [0, 1, 2]
    assert doc.images[2].shapes[0].attr("vineyard_id") == "V01"


def test_single_file_is_returned_unchanged() -> None:
    assert read_cvat_files([MARCAJ], duplicate_policy="error")[0] == read_cvat_file(MARCAJ)[0]


def test_read_cvat_files_needs_paths() -> None:
    with pytest.raises(CvatFormatError):
        read_cvat_files([], duplicate_policy="last_wins")


@pytest.mark.parametrize(
    "data",
    [
        b"<annotations><image",
        b"<root/>",
        b'<annotations><image id="x" name="a.tif" width="2048" height="2048"/></annotations>',
        b'<annotations><image id="0" name="a.tif" width="0" height="2048"/></annotations>',
        b'<annotations><image id="0" width="2048" height="2048"/></annotations>',
    ],
)
def test_structural_errors_raise(data: bytes) -> None:
    with pytest.raises(CvatFormatError):
        parse_xml(data, source_name="bad")


def test_two_decimal_and_whitespace_points() -> None:
    doc, issues = parse_xml(_mini(pts=" 1.25, 2.50 ; 3.00,4.00;5,6 "), source_name="m")
    assert doc.images[0].shapes[0].points == ((1.25, 2.5), (3.0, 4.0), (5.0, 6.0))
    assert issues == ()


def test_zip_errors(tmp_path: Path, make_zip) -> None:
    none = make_zip(tmp_path / "none.zip", {"images/x.tif": b"x"})
    two = make_zip(tmp_path / "two.zip", {"a/annotations.xml": b"<a/>", "b/annotations.xml": b"<a/>"})
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    other = tmp_path / "x.json"
    other.write_text("{}")
    for path in (none, two, bad, other, tmp_path / "missing.xml"):
        with pytest.raises(CvatFormatError):
            load_annotation_bytes(path)


def test_macosx_copies_are_ignored(tmp_path: Path, make_zip) -> None:
    z = make_zip(tmp_path / "m.zip", {"__MACOSX/._annotations.xml": b"junk", "annotations.xml": _mini()})
    assert load_annotation_bytes(z) == _mini()
