"""Tolerant CVAT for images 1.1 reader (contract §4.3, arch §4.13.8, design 01 §3.11).

The reader is faithful: names, attribute values and ids stay raw (normalization happens in
`cvat.to_annset`). Tolerated Marcaj quirks: 2-decimal coordinates, `subset`/`task_id`/`group_id`
attributes, `images/` prefixes, `<tag>`/`<points>` (ignored, info), empty `<image/>`, `<mask>`
(RLE -> polygon, warning). `<track>` and unknown shape kinds are error issues and are skipped.
Structural problems (unparseable XML, bad image header) raise CvatFormatError.
"""

import unicodedata
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from lxml import etree

from vineyard.contracts.enums import Severity
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.contracts.qa import QaIssue
from vineyard.cvat.model import DEFAULT_SHAPE_SOURCE, SHAPE_TAGS, CvatDocument, CvatImage, CvatShape
from vineyard.cvat.rle import rle_to_rings
from vineyard.errors import CvatFormatError

DuplicatePolicy = Literal["last_wins", "error"]
ANNOTATIONS_XML: Final = "annotations.xml"
TILE_SUFFIX: Final = ".tif"
IGNORED_ZIP_PREFIX: Final = "__MACOSX/"
ROOT_TAG: Final = "annotations"
DEFAULT_VERSION: Final = "1.1"
IGNORED_SHAPES: Final = frozenset({"tag", "points"})
CODE_DROPPED: Final = "geom_dropped"
CODE_MASK: Final = "geom_repaired"
CODE_DUPLICATE: Final = "schema_violation"
_META_OPEN: Final = b"<meta>"
_META_CLOSE: Final = b"</meta>"
_BOX_KEYS: Final = ("xtl", "ytl", "xbr", "ybr")
_MASK_KEYS: Final = ("left", "top", "width", "height")


def image_key(name: str) -> str:
    """Matching key of an image name: NFC basename (any `/` or `\\` prefix dropped), `.tif` added if absent."""
    base = PurePosixPath(unicodedata.normalize("NFC", name.strip()).replace("\\", "/")).name
    return base if PurePosixPath(base).suffix else f"{base}{TILE_SUFFIX}"


def _tile_of(name: str) -> str:
    stem = image_key(name).removesuffix(TILE_SUFFIX)
    return stem if is_valid_id(IdKind.TILE, stem) else ""


def _issue(severity: Severity, code: str, tile_id: str, object_id: str, message: str) -> QaIssue:
    return QaIssue(severity, code, tile_id, object_id, message)


def _parse_points(text: str | None) -> tuple[tuple[float, float], ...]:
    pairs = [p for p in (text or "").split(";") if p.strip()]
    if not pairs:
        raise ValueError("no points")
    out = []
    for pair in pairs:
        u, v = pair.split(",")
        out.append((float(u), float(v)))
    return tuple(out)


def _attributes(el: etree._Element) -> tuple[tuple[str, str], ...]:
    return tuple((a.get("name", ""), a.text or "") for a in el.iterfind("attribute"))


def _int_attr(el: etree._Element, key: str, default: int) -> int:
    try:
        return int(el.get(key, default))
    except ValueError:
        return default


def _shape(el: etree._Element, points: tuple[tuple[float, float], ...], tag: str) -> CvatShape:
    return CvatShape(
        tag=tag,  # type: ignore[arg-type]
        label=el.get("label", ""),
        points=points,
        attributes=_attributes(el),
        source=el.get("source", DEFAULT_SHAPE_SOURCE),
        occluded=1 if el.get("occluded", "0").strip() == "1" else 0,
        z_order=_int_attr(el, "z_order", 0),
    )


def _geometry_shape(el: etree._Element) -> CvatShape:
    if el.tag == "box":
        xtl, ytl, xbr, ybr = (float(el.get(k)) for k in _BOX_KEYS)
        return _shape(el, ((xtl, ytl), (xbr, ybr)), "box")
    return _shape(el, _parse_points(el.get("points")), el.tag)


def _mask_shapes(el: etree._Element) -> tuple[CvatShape, ...]:
    left, top, width, height = (int(float(el.get(k))) for k in _MASK_KEYS)
    rings = rle_to_rings(el.get("rle", ""), left=left, top=top, width=width, height=height)
    return tuple(_shape(el, ring, "polygon") for ring in rings)


def _child_shapes(el: etree._Element, tile_id: str, object_id: str, accept_masks: bool,
                  issues: list[QaIssue]) -> tuple[CvatShape, ...]:
    """Shapes for one child element of <image>; problems are appended to `issues`."""
    if el.tag in IGNORED_SHAPES:
        issues.append(_issue(Severity.INFO, CODE_DROPPED, tile_id, object_id, f"<{el.tag}> ignored"))
        return ()
    if el.tag == "mask":
        if not accept_masks:
            issues.append(_issue(Severity.ERROR, CODE_DROPPED, tile_id, object_id, "mask rejected (accept_masks off)"))
            return ()
        shapes = _mask_shapes(el)
        issues.append(_issue(Severity.WARNING, CODE_MASK, tile_id, object_id,
                             f"mask converted to {len(shapes)} polygon(s); redraw it as a polygon"))
        return shapes
    if el.tag not in SHAPE_TAGS:
        issues.append(_issue(Severity.ERROR, CODE_DROPPED, tile_id, object_id, f"unsupported shape <{el.tag}>"))
        return ()
    return (_geometry_shape(el),)


def _image_header(el: etree._Element, source_name: str) -> tuple[int, str, int, int]:
    name = el.get("name")
    if not name:
        raise CvatFormatError("<image> without a name", source=source_name)
    try:
        header = (int(el.get("id", "")), name, int(el.get("width", "")), int(el.get("height", "")))
    except ValueError as exc:
        raise CvatFormatError("<image> id/width/height must be integers", source=source_name, name=name) from exc
    return header


def _parse_image(el: etree._Element, source_name: str, accept_masks: bool,
                 issues: list[QaIssue]) -> CvatImage:
    image_id, name, width, height = _image_header(el, source_name)
    tile_id = _tile_of(name)
    shapes: list[CvatShape] = []
    for k, child in enumerate(c for c in el if isinstance(c.tag, str)):
        object_id = f"{child.get('label', '')}[{k}]"
        try:
            shapes.extend(_child_shapes(child, tile_id, object_id, accept_masks, issues))
        except (ValueError, TypeError, CvatFormatError) as exc:
            issues.append(_issue(Severity.ERROR, CODE_DROPPED, tile_id, object_id,
                                 f"unreadable <{child.tag}> skipped: {exc}"))
    return CvatImage(image_id, name, width, height, tuple(shapes))


def _meta_xml(data: bytes, root: etree._Element) -> str | None:
    if root.find("meta") is None:
        return None
    start, end = data.find(_META_OPEN), data.find(_META_CLOSE)
    if start < 0 or end < start:
        return None
    return data[start : end + len(_META_CLOSE)].decode("utf-8")


def _root(data: bytes, source_name: str) -> etree._Element:
    parser = etree.XMLParser(resolve_entities=False, huge_tree=True, no_network=True)
    try:
        root = etree.fromstring(data, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise CvatFormatError("unparseable CVAT XML", source=source_name, error=str(exc)) from exc
    if root.tag != ROOT_TAG:
        raise CvatFormatError("root element is not <annotations>", source=source_name, root=str(root.tag))
    return root


def parse_xml(data: bytes, *, source_name: str,
              accept_masks: bool = True) -> tuple[CvatDocument, tuple[QaIssue, ...]]:
    """Parse one annotations.xml; per-object problems come back as QaIssues (object skipped)."""
    root = _root(data, source_name)
    issues: list[QaIssue] = []
    images: list[CvatImage] = []
    for el in root:
        if el.tag == "image":
            images.append(_parse_image(el, source_name, accept_masks, issues))
        elif el.tag == "track":
            issues.append(_issue(Severity.ERROR, CODE_DROPPED, "", f"track[{el.get('id', '')}]",
                                 f"<track> (video annotation) is not supported, skipped ({source_name})"))
    version = (root.findtext("version") or DEFAULT_VERSION).strip()
    return CvatDocument(tuple(images), _meta_xml(data, root), version), tuple(issues)


def _zip_annotations(path: Path) -> bytes:
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist()
                     if PurePosixPath(n).name == ANNOTATIONS_XML and not n.startswith(IGNORED_ZIP_PREFIX)]
            if len(names) != 1:
                raise CvatFormatError("ZIP must contain exactly one annotations.xml", path=str(path),
                                      found=names[:5])
            return zf.read(names[0])
    except zipfile.BadZipFile as exc:
        raise CvatFormatError("not a readable ZIP", path=str(path), error=str(exc)) from exc


def load_annotation_bytes(path: Path) -> bytes:
    """annotations.xml bytes from a `.xml` file or a `.zip` (annotations.xml at any depth)."""
    source = Path(path)
    if not source.is_file():
        raise CvatFormatError("annotation file not found", path=str(source))
    suffix = source.suffix.lower()
    if suffix == ".zip":
        return _zip_annotations(source)
    if suffix == ".xml":
        return source.read_bytes()
    raise CvatFormatError("expected a .zip or .xml CVAT export", path=str(source))


def read_cvat_file(path: Path, *, accept_masks: bool = True) -> tuple[CvatDocument, tuple[QaIssue, ...]]:
    return parse_xml(load_annotation_bytes(path), source_name=str(path), accept_masks=accept_masks)


def _merge(parsed: Sequence[tuple[str, CvatDocument]],
           policy: DuplicatePolicy) -> tuple[CvatDocument, tuple[QaIssue, ...]]:
    winners: dict[str, CvatImage] = {}
    issues: list[QaIssue] = []
    duplicates: list[str] = []
    for source_name, doc in parsed:
        for img in doc.images:
            key = image_key(img.name)
            if key in winners:
                duplicates.append(key)
                issues.append(_issue(Severity.WARNING, CODE_DUPLICATE, _tile_of(img.name), key,
                                     f"image {key} appears more than once; the copy from {source_name} wins"))
            winners[key] = img
    if duplicates and policy == "error":
        raise CvatFormatError("duplicate images across CVAT files", duplicates=sorted(set(duplicates))[:10])
    if len(parsed) == 1 and not duplicates:
        return parsed[0][1], ()
    images = tuple(winners[k].with_id(i) for i, k in enumerate(sorted(winners)))
    return CvatDocument(images, parsed[0][1].meta_xml, parsed[0][1].version), tuple(issues)


def read_cvat_files(paths: Sequence[Path], *, duplicate_policy: DuplicatePolicy = "last_wins",
                    accept_masks: bool = True) -> tuple[CvatDocument, tuple[QaIssue, ...]]:
    """Read and merge several exports. One clean file is returned as is; otherwise images are keyed by
    `image_key`, the last copy wins (warning) or duplicates raise, and ids are renumbered 0..n-1 by key.
    """
    if not paths:
        raise CvatFormatError("no CVAT files given")
    parsed: list[tuple[str, CvatDocument]] = []
    issues: list[QaIssue] = []
    for path in paths:
        doc, file_issues = read_cvat_file(Path(path), accept_masks=accept_masks)
        parsed.append((str(path), doc))
        issues.extend(file_issues)
    merged, merge_issues = _merge(parsed, duplicate_policy)
    return merged, (*issues, *merge_issues)
