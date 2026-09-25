"""Byte-exact CVAT 1.1 serializer mirroring the organizers' example (contract §4.1, design 01 §3.7).

Pure formatting: geometry cleaning (clip, validity, orientation) happens before, in `cvat.to_cvat`.
Frame = XML_HEADER + <meta> + "\\n" + one `<image>` block per image + XML_FOOTER; one shape per line.
"""

from collections.abc import Iterable, Sequence
from typing import Final

from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.template import META_BLOCK, XML_FOOTER, XML_HEADER
from vineyard.errors import CvatFormatError

DEFAULT_DECIMALS: Final = 1
MAX_DECIMALS: Final = 6
ENCODING: Final = "utf-8"
POINT_SEP: Final = ";"
COORD_SEP: Final = ","
BOX_KEYS: Final = ("xtl", "ytl", "xbr", "ybr")

_TEXT_ESCAPES: Final = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"))
_ATTR_ESCAPES: Final = (*_TEXT_ESCAPES, ('"', "&quot;"))


def _escape(value: str, table: Sequence[tuple[str, str]]) -> str:
    out = value
    for raw, entity in table:
        out = out.replace(raw, entity)
    return out


def escape_text(value: str) -> str:
    """Element text: &, <, > escaped (same as xml.sax.saxutils.escape)."""
    return _escape(value, _TEXT_ESCAPES)


def escape_attr(value: str) -> str:
    """Attribute value in double quotes: text escaping plus `"`."""
    return _escape(value, _ATTR_ESCAPES)


def _check_decimals(decimals: int) -> None:
    if not (0 <= decimals <= MAX_DECIMALS):
        raise CvatFormatError("coordinate decimals out of range", decimals=decimals, max=MAX_DECIMALS)


def format_coord(v: float, decimals: int) -> str:
    """'%.<decimals>f' that never yields a negative zero ("-0.0" -> "0.0")."""
    _check_decimals(decimals)
    text = f"{float(v):.{decimals}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def format_points(points: Iterable[tuple[float, float]], decimals: int) -> str:
    """'u,v;u,v;...' in CVAT point-list syntax."""
    return POINT_SEP.join(f"{format_coord(u, decimals)}{COORD_SEP}{format_coord(v, decimals)}" for u, v in points)


def _attributes_xml(shape: CvatShape) -> str:
    return "".join(
        f'<attribute name="{escape_attr(name)}">{escape_text(value)}</attribute>' for name, value in shape.attributes
    )


def _geometry_attrs(shape: CvatShape, decimals: int) -> str:
    if shape.tag == "box":
        values = (format_coord(c, decimals) for c in shape.box_xyxy)
        return " ".join(f'{key}="{val}"' for key, val in zip(BOX_KEYS, values, strict=True))
    return f'points="{format_points(shape.points, decimals)}"'


def serialize_shape(shape: CvatShape, *, decimals: int = DEFAULT_DECIMALS) -> str:
    """One shape on one line, attribute order label, source, occluded, geometry, z_order (as the example)."""
    head = (
        f'<{shape.tag} label="{escape_attr(shape.label)}" source="{escape_attr(shape.source)}" '
        f'occluded="{shape.occluded}" {_geometry_attrs(shape, decimals)} z_order="{shape.z_order}">'
    )
    return f"{head}{_attributes_xml(shape)}</{shape.tag}>"


def serialize_image(image: CvatImage, *, decimals: int = DEFAULT_DECIMALS) -> str:
    """`<image ...>` line, one line per shape, `</image>` line; an empty image is `<image ...>\\n</image>\\n`."""
    head = (
        f'<image id="{image.id}" name="{escape_attr(image.name)}" width="{image.width}" '
        f'height="{image.height}">\n'
    )
    body = "".join(f"{serialize_shape(s, decimals=decimals)}\n" for s in image.shapes)
    return f"{head}{body}</image>\n"


def serialize_document(doc: CvatDocument, *, decimals: int = DEFAULT_DECIMALS) -> bytes:
    """Whole annotations.xml as UTF-8 bytes; images are written in document order (ids as given)."""
    _check_decimals(decimals)
    meta = META_BLOCK if doc.meta_xml is None else doc.meta_xml
    images = "".join(serialize_image(img, decimals=decimals) for img in doc.images)
    return f"{XML_HEADER}{meta}\n{images}{XML_FOOTER}".encode(ENCODING)
