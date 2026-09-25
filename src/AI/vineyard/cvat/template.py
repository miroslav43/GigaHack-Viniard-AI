"""CVAT 1.1 XML frame copied verbatim from the organizers' example (contract §4.1).

META_BLOCK is byte-identical to lines 4-21 of `05_examples/.../annotations.xml`; Marcaj is known to
accept exactly this block. LABEL_SPECS is parsed from it at import so the two can never drift.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, cast

from lxml import etree

from vineyard.errors import CvatFormatError

CvatType = Literal["polygon", "rectangle", "polyline"]
ShapeTag = Literal["polygon", "box", "polyline"]

XML_HEADER: Final[str] = '<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n<version>1.1</version>\n'
XML_FOOTER: Final[str] = "</annotations>\n"

_TEXT_ATTR = (
    "<mutable>False</mutable><input_type>text</input_type><default_value></default_value><values></values>"
    "</attribute>"
)
_VID_ATTR = f"  <attribute><name>vineyard_id</name>{_TEXT_ATTR}"
_CLOSE_LABEL = "</attributes></label>"

META_BLOCK: Final[str] = "\n".join(
    (
        "<meta><task><name>Vineyard AI Field Challenge — examples</name><labels>",
        "<label><name>vineyard</name><type>polygon</type><attributes>",
        f"{_VID_ATTR}{_CLOSE_LABEL}",
        "<label><name>waste</name><type>rectangle</type><attributes>",
        f"{_VID_ATTR}{_CLOSE_LABEL}",
        "<label><name>row</name><type>polyline</type><attributes>",
        _VID_ATTR,
        f"  <attribute><name>row_id</name>{_TEXT_ATTR}",
        "  <attribute><name>row_structure</name><mutable>False</mutable><input_type>select</input_type>"
        "<default_value>regular</default_value><values>regular",
        "disrupted",
        f"unassessable</values></attribute>{_CLOSE_LABEL}",
        "<label><name>interrow_area</name><type>polygon</type><attributes>",
        _VID_ATTR,
        "  <attribute><name>interrow_cover</name><mutable>False</mutable><input_type>select</input_type>"
        "<default_value>bare_soil</default_value><values>bare_soil",
        "vegetation",
        "mixed",
        f"unassessable</values></attribute>{_CLOSE_LABEL}",
        "</labels></task></meta>",
    )
)

CVAT_TYPE_TO_SHAPE_TAG: Final[Mapping[CvatType, ShapeTag]] = MappingProxyType(
    {"polygon": "polygon", "rectangle": "box", "polyline": "polyline"}
)


@dataclass(frozen=True)
class AttrSpec:
    name: str
    input_type: Literal["text", "select"]
    values: tuple[str, ...]
    default: str


@dataclass(frozen=True)
class LabelSpec:
    name: str
    cvat_type: CvatType
    shape_tag: ShapeTag
    attributes: tuple[AttrSpec, ...]

    def attribute(self, name: str) -> AttrSpec:
        for attr in self.attributes:
            if attr.name == name:
                return attr
        raise KeyError(f"label {self.name!r} has no attribute {name!r}")


def _attr_spec(node: etree._Element) -> AttrSpec:
    input_type = node.findtext("input_type") or ""
    if input_type not in ("text", "select"):
        raise CvatFormatError("unsupported attribute input_type", input_type=input_type)
    raw_values = node.findtext("values") or ""
    values = tuple(v for v in raw_values.split("\n") if v)
    default = node.findtext("default_value") or ""
    return AttrSpec(node.findtext("name") or "", cast(Literal["text", "select"], input_type), values, default)


def _label_spec(node: etree._Element) -> LabelSpec:
    name = node.findtext("name") or ""
    cvat_type = node.findtext("type") or ""
    if cvat_type not in CVAT_TYPE_TO_SHAPE_TAG:
        raise CvatFormatError("unsupported label type", label=name, type=cvat_type)
    attrs = tuple(_attr_spec(a) for a in node.iterfind("attributes/attribute"))
    typed = cast(CvatType, cvat_type)
    return LabelSpec(name, typed, CVAT_TYPE_TO_SHAPE_TAG[typed], attrs)


def parse_label_specs(meta_xml: str) -> Mapping[str, LabelSpec]:
    """Label table from a `<meta>` block, in document order."""
    try:
        root = etree.fromstring(meta_xml.encode("utf-8"), parser=etree.XMLParser(resolve_entities=False))
    except etree.XMLSyntaxError as exc:
        raise CvatFormatError("unparseable <meta> block", error=str(exc)) from exc
    labels = [_label_spec(node) for node in root.iterfind("task/labels/label")]
    return MappingProxyType({spec.name: spec for spec in labels})


LABEL_SPECS: Final[Mapping[str, LabelSpec]] = parse_label_specs(META_BLOCK)
