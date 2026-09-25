import pytest
from lxml import etree

from vineyard.contracts.enums import LABEL_ATTRIBUTES, LABEL_GEOMETRY, InterrowCover, Label, RowStructure
from vineyard.cvat.template import (
    CVAT_TYPE_TO_SHAPE_TAG,
    LABEL_SPECS,
    META_BLOCK,
    XML_FOOTER,
    XML_HEADER,
    AttrSpec,
    parse_label_specs,
)
from vineyard.errors import CvatFormatError


@pytest.mark.examples
def test_meta_block_is_byte_identical(examples_xml: bytes) -> None:
    start = examples_xml.index(b"<meta>")
    end = examples_xml.index(b"</meta>") + len(b"</meta>")
    assert META_BLOCK.encode("utf-8") == examples_xml[start:end]


@pytest.mark.examples
def test_header_and_footer_match_example(examples_xml: bytes) -> None:
    assert examples_xml.startswith(XML_HEADER.encode("utf-8") + META_BLOCK.encode("utf-8") + b"\n")
    assert examples_xml.endswith(XML_FOOTER.encode("utf-8"))


def test_label_specs_shapes() -> None:
    assert list(LABEL_SPECS) == ["vineyard", "waste", "row", "interrow_area"]
    assert (LABEL_SPECS["vineyard"].cvat_type, LABEL_SPECS["vineyard"].shape_tag) == ("polygon", "polygon")
    assert (LABEL_SPECS["waste"].cvat_type, LABEL_SPECS["waste"].shape_tag) == ("rectangle", "box")
    assert (LABEL_SPECS["row"].cvat_type, LABEL_SPECS["row"].shape_tag) == ("polyline", "polyline")
    assert (LABEL_SPECS["interrow_area"].cvat_type, LABEL_SPECS["interrow_area"].shape_tag) == ("polygon", "polygon")
    for label in Label:
        assert LABEL_SPECS[label].shape_tag == CVAT_TYPE_TO_SHAPE_TAG[LABEL_SPECS[label].cvat_type]
        assert LABEL_SPECS[label].shape_tag == {"polygon": "polygon", "box": "box", "polyline": "polyline"}[
            LABEL_GEOMETRY[label].value
        ]


def test_label_attributes_match_enums_table() -> None:
    for label in Label:
        names = tuple(a.name for a in LABEL_SPECS[label].attributes)
        assert names == LABEL_ATTRIBUTES[label]


def test_select_attributes() -> None:
    row_structure = LABEL_SPECS["row"].attributes[2]
    assert row_structure == AttrSpec("row_structure", "select", ("regular", "disrupted", "unassessable"), "regular")
    assert row_structure.values == tuple(m.value for m in RowStructure)
    cover = LABEL_SPECS["interrow_area"].attributes[1]
    assert cover.input_type == "select"
    assert cover.default == "bare_soil"
    assert cover.values == tuple(m.value for m in InterrowCover)
    vid = LABEL_SPECS["waste"].attributes[0]
    assert vid == AttrSpec("vineyard_id", "text", (), "")


def test_label_spec_attribute_lookup() -> None:
    assert LABEL_SPECS["row"].attribute("row_id").input_type == "text"
    with pytest.raises(KeyError):
        LABEL_SPECS["row"].attribute("nope")


def test_meta_parses_as_xml() -> None:
    root = etree.fromstring(META_BLOCK.encode("utf-8"))
    assert root.tag == "meta"
    assert root.findtext("task/name") == "Vineyard AI Field Challenge — examples"


def test_parse_label_specs_rejects_unknown_type() -> None:
    bad = META_BLOCK.replace("<type>rectangle</type>", "<type>ellipse</type>")
    with pytest.raises(CvatFormatError):
        parse_label_specs(bad)
    with pytest.raises(CvatFormatError):
        parse_label_specs("<meta><task>")
