"""cvat.testzip: Marcaj format-test ZIP (2 example tiles + 1 waste box + 1 empty tile)."""

import hashlib
import zipfile
from pathlib import Path

import pytest

from vineyard.config import load_config
from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.reader import parse_xml
from vineyard.cvat.testzip import (
    EMPTY_TILE,
    WASTE_BOX_PX,
    WASTE_TILE,
    build_test_document,
    default_test_zip_path,
    locate_tile,
    make_test_zip,
    nearest_vineyard_id,
)
from vineyard.errors import CvatFormatError

R006, R021 = "siret3_r006_c004", "siret3_r021_c012"


def _examples_doc() -> CvatDocument:
    row = CvatShape("polyline", "row", ((0.0, 1.0), (10.0, 1.0)),
                    (("vineyard_id", "V01"), ("row_id", "V01-R01"), ("row_structure", "regular")))
    return CvatDocument((CvatImage(0, f"{R021}.tif", 2048, 2048, (row,)),
                         CvatImage(1, f"{R006}.tif", 2048, 2048, ())))


def test_build_test_document_layout() -> None:
    doc = build_test_document(_examples_doc(), shape_source="manual", waste_vineyard_id="")
    assert [(i.id, i.name) for i in doc.images] == [
        (0, f"{EMPTY_TILE}.tif"), (1, f"{R006}.tif"), (2, f"{R021}.tif")]
    assert doc.images[0].shapes == ()
    r021 = doc.image(f"{WASTE_TILE}.tif")
    waste = r021.shapes[-1]
    assert (waste.tag, waste.label, waste.box_xyxy) == ("box", "waste", WASTE_BOX_PX)
    assert waste.attributes == (("vineyard_id", ""),)
    assert r021.shapes[0].label == "row"
    assert doc.meta_xml is None


def test_waste_vineyard_id_defaults_to_the_nearest_annotated_block() -> None:
    doc = build_test_document(_examples_doc(), shape_source="manual", waste_vineyard_id=None)
    assert doc.image(f"{WASTE_TILE}.tif").shapes[-1].attributes == (("vineyard_id", "V01"),)


def test_nearest_vineyard_id_picks_closest_shape_then_order() -> None:
    xtl, ytl, xbr, ybr = WASTE_BOX_PX
    near = CvatShape("polyline", "row", ((xtl, ybr + 5.0), (xbr, ybr + 5.0)), (("vineyard_id", "V02"),))
    far = CvatShape("polygon", "vineyard", ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)), (("vineyard_id", "V01"),))
    tie = CvatShape("polyline", "row", ((xtl, ytl - 5.0), (xbr, ytl - 5.0)), (("vineyard_id", "V03"),))
    blank = CvatShape("polyline", "row", ((xtl, ytl), (xbr, ybr)), (("vineyard_id", " "),))
    image = CvatImage(0, f"{WASTE_TILE}.tif", 2048, 2048, (far, near, tie, blank))
    assert nearest_vineyard_id(image, WASTE_BOX_PX) == "V02"
    assert nearest_vineyard_id(CvatImage(0, f"{WASTE_TILE}.tif", 2048, 2048, (blank,)), WASTE_BOX_PX) == ""
    around = CvatShape("polygon", "interrow_area", ((xtl - 50, ytl - 50), (xbr + 50, ytl - 50), (xbr + 50, ybr + 50),
                                                    (xtl - 50, ybr + 50)), (("vineyard_id", "V04"),))
    inside = CvatImage(0, f"{WASTE_TILE}.tif", 2048, 2048, (near, around))
    assert nearest_vineyard_id(inside, WASTE_BOX_PX) == "V04"


def test_build_test_document_needs_the_waste_tile() -> None:
    doc = CvatDocument((CvatImage(0, f"{R006}.tif", 2048, 2048, ()),))
    with pytest.raises(CvatFormatError):
        build_test_document(doc, shape_source="manual", waste_vineyard_id="")


def test_locate_tile_search_order(tmp_path: Path, make_zip) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    second.mkdir()
    (second / "siret3_r005_c004.tif").write_bytes(b"from-dir")
    part = make_zip(tmp_path / "part1of5.zip", {"siret3_r005_c004.tif": b"from-zip", "x.tif": b"x"})
    scratch = tmp_path / "scratch"
    assert locate_tile(EMPTY_TILE, search_dirs=(first, second), zip_paths=(part,), scratch=scratch).read_bytes() \
        == b"from-dir"
    got = locate_tile(EMPTY_TILE, search_dirs=(first,), zip_paths=(part,), scratch=scratch)
    assert got.parent == scratch and got.read_bytes() == b"from-zip"
    with pytest.raises(CvatFormatError):
        locate_tile("siret3_r039_c033", search_dirs=(first,), zip_paths=(part,), scratch=scratch)


def test_default_path_is_under_work() -> None:
    cfg = load_config()
    assert default_test_zip_path(cfg).is_relative_to(cfg.paths.work_dir)


@pytest.mark.examples
@pytest.mark.needs_tiles
def test_make_test_zip(tmp_path: Path, examples_dir: Path) -> None:
    from vineyard.cvat.validator_zip import validate_zip

    cfg = load_config()
    out = make_test_zip(tmp_path / "test.zip", cfg)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert names == ["annotations.xml", f"images/{EMPTY_TILE}.tif", f"images/{R006}.tif", f"images/{R021}.tif"]
        assert zf.getinfo("annotations.xml").compress_type == zipfile.ZIP_DEFLATED
        assert zf.getinfo(f"images/{R006}.tif").compress_type == zipfile.ZIP_STORED
        xml = zf.read("annotations.xml")
        shas = {n[len("images/"):-len(".tif")]: hashlib.sha256(zf.read(n)).hexdigest() for n in names[1:]}
    doc, issues = parse_xml(xml, source_name="test.zip")
    assert issues == ()
    assert [img.count("waste") for img in doc.images] == [0, 0, 1]
    assert [len(img.shapes) for img in doc.images] == [0, 302, 449]
    assert doc.images[2].shapes[-1].attributes == (("vineyard_id", "V01"),)
    assert shas[R006] == hashlib.sha256((examples_dir / "images" / f"{R006}.tif").read_bytes()).hexdigest()
    report = validate_zip(out, expected_sha256=shas, cfg=cfg.export.cvat, tile_px=2048)
    assert report.ok, report.summary_lines()
