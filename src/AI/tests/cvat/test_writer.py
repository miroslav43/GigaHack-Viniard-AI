"""cvat.writer: byte-exact CVAT 1.1 serializer (contract §4.1, design 01 §3.7)."""

import pytest

from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.reader import parse_xml
from vineyard.cvat.template import META_BLOCK, XML_FOOTER, XML_HEADER
from vineyard.cvat.writer import (
    escape_attr,
    escape_text,
    format_coord,
    format_points,
    serialize_document,
    serialize_image,
    serialize_shape,
)
from vineyard.errors import CvatFormatError

TILE = "siret3_r021_c012.tif"


def _box(vid: str = "") -> CvatShape:
    return CvatShape("box", "waste", ((10.0, 20.5), (30.25, 40.0)), (("vineyard_id", vid),))


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (-0.04, 1, "0.0"),
        (-0.0, 1, "0.0"),
        (0.0, 1, "0.0"),
        (2048.0, 1, "2048.0"),
        (32.14, 1, "32.1"),
        (-1.26, 1, "-1.3"),
        (1.25, 2, "1.25"),
        (-0.004, 2, "0.00"),
        (3.4, 0, "3"),
    ],
)
def test_format_coord(value: float, decimals: int, expected: str) -> None:
    assert format_coord(value, decimals) == expected


def test_format_coord_rejects_bad_decimals() -> None:
    with pytest.raises(CvatFormatError):
        format_coord(1.0, -1)


def test_format_points() -> None:
    assert format_points(((2048.0, 32.1), (2017.8, -0.01)), 1) == "2048.0,32.1;2017.8,0.0"


def test_escaping() -> None:
    assert escape_text('A&B"<>') == 'A&amp;B"&lt;&gt;'
    assert escape_attr('A&B"<>') == "A&amp;B&quot;&lt;&gt;"


def test_polyline_line_exact() -> None:
    shape = CvatShape(
        "polyline", "row", ((2048.0, 32.1), (2017.8, 0.0)),
        (("vineyard_id", "V01"), ("row_id", "V01-R01"), ("row_structure", "regular")),
    )
    assert serialize_shape(shape, decimals=1) == (
        '<polyline label="row" source="manual" occluded="0" points="2048.0,32.1;2017.8,0.0" z_order="0">'
        '<attribute name="vineyard_id">V01</attribute><attribute name="row_id">V01-R01</attribute>'
        '<attribute name="row_structure">regular</attribute></polyline>'
    )


def test_box_line_and_empty_attribute_exact() -> None:
    assert serialize_shape(_box(), decimals=1) == (
        '<box label="waste" source="manual" occluded="0" xtl="10.0" ytl="20.5" xbr="30.2" ybr="40.0" '
        'z_order="0"><attribute name="vineyard_id"></attribute></box>'
    )


def test_shape_source_occluded_zorder_are_written() -> None:
    shape = CvatShape("polygon", "vineyard", ((0, 0), (1, 0), (1, 1)), (("vineyard_id", "V1"),),
                      source="auto", occluded=1, z_order=3)
    line = serialize_shape(shape, decimals=1)
    assert line.startswith('<polygon label="vineyard" source="auto" occluded="1" points="0.0,0.0;1.0,0.0;1.0,1.0"')
    assert 'z_order="3">' in line


def test_empty_image_exact() -> None:
    img = CvatImage(4, TILE, 2048, 2048, ())
    assert serialize_image(img, decimals=1) == f'<image id="4" name="{TILE}" width="2048" height="2048">\n</image>\n'


def test_image_one_shape_per_line() -> None:
    img = CvatImage(0, TILE, 2048, 2048, (_box("V01"), _box()))
    lines = serialize_image(img, decimals=1).split("\n")
    assert lines[0].startswith('<image id="0"')
    assert lines[1].startswith("<box ") and lines[2].startswith("<box ")
    assert lines[3] == "</image>" and lines[4] == ""


def test_document_frame_uses_meta_block_by_default() -> None:
    doc = CvatDocument((CvatImage(0, TILE, 2048, 2048, ()),))
    data = serialize_document(doc)
    assert data.startswith((XML_HEADER + META_BLOCK + "\n").encode("utf-8"))
    assert data.endswith(XML_FOOTER.encode("utf-8"))


def test_document_keeps_custom_meta() -> None:
    doc = CvatDocument((), meta_xml="<meta><x/></meta>")
    assert serialize_document(doc) == (XML_HEADER + "<meta><x/></meta>\n" + XML_FOOTER).encode("utf-8")


def test_escaped_values_round_trip() -> None:
    shape = CvatShape("polygon", "vineyard", ((0, 0), (5, 0), (5, 5)), (("vineyard_id", 'A&B"<'),))
    doc = CvatDocument((CvatImage(0, TILE, 2048, 2048, (shape,)),))
    parsed, issues = parse_xml(serialize_document(doc), source_name="mem")
    assert parsed.images[0].shapes[0].attr("vineyard_id") == 'A&B"<'
    assert issues == ()


@pytest.mark.examples
def test_example_round_trip_is_byte_identical(examples_xml: bytes) -> None:
    doc, issues = parse_xml(examples_xml, source_name="annotations.xml")
    assert issues == ()
    assert [(i.id, i.name) for i in doc.images] == [(0, "siret3_r021_c012.tif"), (1, "siret3_r006_c004.tif")]
    assert serialize_document(doc) == examples_xml
