"""ZIP-level validator (arch §4.13 steps 2 and 4): exact layout, name bijection, ids, sha256, size, meta.

CVAT loads every **/*.xml of an archive and falls back to the image id when a name does not match,
silently attaching annotations to another tile; so anything outside the whitelist is an error.
The XML is parsed strictly here (no normalisation), independently of the tolerant reader.
"""

from __future__ import annotations

import hashlib
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from lxml import etree

from vineyard.config.sections_io import CvatExportConfig
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.packer import ANNOTATIONS_NAME, IMAGE_EXT, IMAGES_DIR
from vineyard.cvat.report import Issue, ValidationReport, error, warning
from vineyard.cvat.template import META_BLOCK
from vineyard.cvat.validator_doc import WASTE_EMPTY_VID_MIN_DIST_M, validate_document
from vineyard.errors import CvatFormatError

CVAT_VERSION_TAG: Final = b"<version>1.1</version>"
META_OPEN: Final = b"<meta>"
META_CLOSE: Final = b"</meta>"
FORBIDDEN_BASENAMES: Final = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
FORBIDDEN_PREFIXES: Final = ("__MACOSX/", "._")
SHAPE_TAGS: Final = ("polygon", "polyline", "box")
BOX_KEYS: Final = ("xtl", "ytl", "xbr", "ybr")
_IMAGES_PREFIX: Final = f"{IMAGES_DIR}/"
_PARSER: Final = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)


@dataclass(frozen=True)
class _ZipScan:
    report: ValidationReport
    image_files: tuple[str, ...]  # file names (basename) of well-formed images/<name>.tif entries


# ---------------------------------------------------------------- entries


def _entry_issue(name: str, where: dict[str, str]) -> Issue | None:
    base = name.rsplit("/", 1)[-1]
    at = where | {"object_ref": name}
    if name.endswith("/"):
        return error("zip_dir_entry", f"directory entry {name!r}", **at)
    if base in FORBIDDEN_BASENAMES or any(name.startswith(p) or base.startswith(p) for p in FORBIDDEN_PREFIXES):
        return error("zip_forbidden_entry", f"forbidden entry {name!r}", **at)
    if name == ANNOTATIONS_NAME:
        return None
    if name.lower().endswith(".xml"):
        return error("zip_extra_xml", f"extra XML {name!r} (CVAT loads every *.xml)", **at)
    if not name.startswith(_IMAGES_PREFIX) or "/" in name[len(_IMAGES_PREFIX):]:
        return error("zip_extra_entry", f"entry {name!r} is not annotations.xml or images/<name>.tif", **at)
    return None


def _image_name_issues(file_name: str, at: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    if unicodedata.normalize("NFC", file_name) != file_name:
        issues.append(error("image_name_not_nfc", f"{file_name!r} is not NFC", **at))
    if not file_name.endswith(IMAGE_EXT):
        return [*issues, error("image_bad_ext", f"{file_name!r} must end with {IMAGE_EXT}", **at)]
    if not is_valid_id(IdKind.TILE, file_name[: -len(IMAGE_EXT)]):
        issues.append(error("image_name_invalid", f"{file_name!r} is not <tile_id>{IMAGE_EXT}", **at))
    return issues


def _scan_entries(infos: Sequence[zipfile.ZipInfo], zip_name: str) -> tuple[list[Issue], list[zipfile.ZipInfo]]:
    where = {"zip_name": zip_name}
    names = [i.filename for i in infos]
    issues = [error("zip_duplicate_entry", f"entry {n!r} repeated", **where, object_ref=n)
              for n, c in sorted(Counter(names).items()) if c > 1]
    if ANNOTATIONS_NAME not in names:
        issues.append(error("zip_missing_xml", "annotations.xml missing", **where))
    elif names[0] != ANNOTATIONS_NAME:
        issues.append(warning("xml_not_first", "annotations.xml is not the first entry", **where))
    images: list[zipfile.ZipInfo] = []
    for info in infos:
        found = _entry_issue(info.filename, where)
        if found is not None:
            issues.append(found)
        elif info.filename != ANNOTATIONS_NAME:
            name_issues = _image_name_issues(info.filename[len(_IMAGES_PREFIX):], where | {"object_ref": info.filename})
            issues += name_issues
            images += [] if name_issues else [info]
    return issues, images


def _sha_issues(zf: zipfile.ZipFile, images: Sequence[zipfile.ZipInfo], expected: Mapping[str, str],
                zip_name: str) -> list[Issue]:
    issues: list[Issue] = []
    for info in images:
        file_name = info.filename[len(_IMAGES_PREFIX):]
        tile_id = file_name[: -len(IMAGE_EXT)]
        at = {"zip_name": zip_name, "tile_id": tile_id, "object_ref": info.filename}
        if info.compress_type != zipfile.ZIP_STORED:
            issues.append(warning("image_not_stored", "image entry is compressed", **at))
        digest = hashlib.sha256(zf.read(info)).hexdigest()
        want = expected.get(tile_id)
        if want is None:
            issues.append(error("sha256_unknown", "no expected sha256 for this tile", **at))
        elif digest != want:
            issues.append(error("sha256_mismatch", f"sha256 {digest[:12]}.. != {want[:12]}..", **at))
    return issues


# ---------------------------------------------------------------- annotations.xml


def _meta_issues(xml: bytes, zip_name: str) -> list[Issue]:
    issues: list[Issue] = []
    if CVAT_VERSION_TAG not in xml[: len(META_BLOCK)]:
        issues.append(error("version_mismatch", "<version>1.1</version> missing", zip_name=zip_name))
    start, stop = xml.find(META_OPEN), xml.find(META_CLOSE)
    meta = xml[start: stop + len(META_CLOSE)] if start >= 0 and stop > start else b""
    if meta != META_BLOCK.encode("utf-8"):
        issues.append(error("meta_mismatch", "<meta> block differs from the organizers' template", zip_name=zip_name))
    return issues


def _points(el: etree._Element) -> tuple[tuple[float, float], ...]:
    if el.tag == "box":
        xtl, ytl, xbr, ybr = (float(el.get(k, "nan")) for k in BOX_KEYS)
        return ((xtl, ytl), (xbr, ybr))
    pairs = (el.get("points") or "").split(";")
    return tuple((float(u), float(v)) for u, v in (p.split(",") for p in pairs))


def _shape(el: etree._Element) -> CvatShape:
    attrs = tuple((a.get("name") or "", a.text or "") for a in el.iterfind("attribute"))
    try:
        return CvatShape(el.tag, el.get("label") or "", _points(el), attrs, source=el.get("source") or "",
                         occluded=int(el.get("occluded", "0")), z_order=int(el.get("z_order", "0")))
    except (ValueError, TypeError) as exc:
        raise CvatFormatError("unparseable shape", tag=el.tag, error=str(exc)) from exc


def _image(el: etree._Element, zip_name: str) -> tuple[CvatImage | None, list[Issue]]:
    name = el.get("name") or ""
    issues: list[Issue] = []
    shapes: list[CvatShape] = []
    for k, child in enumerate(el):
        at = {"zip_name": zip_name, "object_ref": f"{name}#{k}"}
        if child.tag not in SHAPE_TAGS:
            issues.append(error("unsupported_element", f"<{child.tag}> is not polygon/polyline/box", **at))
            continue
        try:
            shapes.append(_shape(child))
        except CvatFormatError as exc:
            issues.append(error("shape_unparseable", str(exc), **at))
    try:
        image = CvatImage(int(el.get("id", "-1")), name, int(el.get("width", "0")), int(el.get("height", "0")),
                          tuple(shapes))
    except (ValueError, CvatFormatError) as exc:
        return None, [*issues, error("image_unparseable", str(exc), zip_name=zip_name, object_ref=name)]
    return image, issues


def parse_annotations(xml: bytes, *, zip_name: str) -> tuple[CvatDocument | None, list[Issue]]:
    """Strict parse (raw values kept) of an annotations.xml; None when it is not well-formed."""
    try:
        root = etree.fromstring(xml, parser=_PARSER)
    except etree.XMLSyntaxError as exc:
        return None, [error("xml_unparseable", str(exc), zip_name=zip_name)]
    images: list[CvatImage] = []
    issues: list[Issue] = []
    for el in root.iterfind("image"):
        image, found = _image(el, zip_name)
        issues += found
        images += [] if image is None else [image]
    return CvatDocument(tuple(images)), issues


def _sequence_issues(doc: CvatDocument, zip_name: str) -> list[Issue]:
    ids = [img.id for img in doc.images]
    names = list(doc.image_names)
    if ids == list(range(len(ids))) and names == sorted(names):
        return []
    return [error("image_id_sequence", "image ids must be 0..n-1 in sorted name order", zip_name=zip_name)]


def _bijection_issues(doc: CvatDocument, files: Sequence[str], zip_name: str) -> list[Issue]:
    in_xml, in_zip = set(doc.image_names), set(files)
    issues = [error("image_without_file", f"<image name={n!r}> has no images/ entry", zip_name=zip_name,
                    object_ref=n) for n in sorted(in_xml - in_zip)]
    issues += [error("file_without_image", f"images/{n} has no <image>", zip_name=zip_name, object_ref=n)
               for n in sorted(in_zip - in_xml)]
    return issues


# ---------------------------------------------------------------- public API


def _scan_zip(path: Path, expected_sha256: Mapping[str, str], cfg: CvatExportConfig, tile_px: int,
              waste_min_dist_m: float) -> _ZipScan:
    zip_name = Path(path).name
    if not Path(path).is_file():
        return _ZipScan(ValidationReport((error("zip_missing", f"{path} not found", zip_name=zip_name),)), ())
    issues: list[Issue] = []
    size = Path(path).stat().st_size
    if size > cfg.max_zip_bytes:
        issues.append(error("zip_too_large", f"{size} B > {cfg.max_zip_bytes} B", zip_name=zip_name))
    try:
        with zipfile.ZipFile(path) as zf:
            entry_issues, images = _scan_entries(zf.infolist(), zip_name)
            issues += entry_issues + _sha_issues(zf, images, expected_sha256, zip_name)
            xml = zf.read(ANNOTATIONS_NAME) if ANNOTATIONS_NAME in zf.namelist() else None
    except (zipfile.BadZipFile, OSError) as exc:
        return _ZipScan(ValidationReport((*issues, error("zip_corrupt", str(exc), zip_name=zip_name))), ())
    files = tuple(sorted(i.filename[len(_IMAGES_PREFIX):] for i in images))
    if xml is None:
        return _ZipScan(ValidationReport(tuple(issues)), files)
    issues += _meta_issues(xml, zip_name)
    doc, parse_issues = parse_annotations(xml, zip_name=zip_name)
    issues += parse_issues
    report = ValidationReport(tuple(issues))
    if doc is not None:
        issues_doc = _sequence_issues(doc, zip_name) + _bijection_issues(doc, files, zip_name)
        report = report.with_issues(*issues_doc).merge(validate_document(
            doc, cfg=cfg, tile_px=tile_px, zip_name=zip_name, waste_min_dist_m=waste_min_dist_m))
    return _ZipScan(report, files)


def validate_zip(
    path: Path,
    *,
    expected_sha256: Mapping[str, str],
    cfg: CvatExportConfig,
    tile_px: int,
    waste_min_dist_m: float = WASTE_EMPTY_VID_MIN_DIST_M,
) -> ValidationReport:
    return _scan_zip(Path(path), expected_sha256, cfg, tile_px, waste_min_dist_m).report


def _set_issues(files_per_zip: Sequence[tuple[str, tuple[str, ...]]], expected: frozenset[str]) -> list[Issue]:
    counts = Counter(f[: -len(IMAGE_EXT)] for _, files in files_per_zip for f in files)
    issues = [error("duplicate_tile", f"{t} is in {c} ZIPs", tile_id=t) for t, c in sorted(counts.items()) if c > 1]
    issues += [error("missing_tile", f"{t} is in no ZIP", tile_id=t) for t in sorted(expected - set(counts))]
    issues += [error("unexpected_tile", f"{t} is not an expected tile", tile_id=t)
               for t in sorted(set(counts) - expected)]
    return issues


def validate_upload_set(
    zip_paths: Sequence[Path],
    *,
    expected_tiles: frozenset[str],
    expected_sha256: Mapping[str, str],
    cfg: CvatExportConfig,
    tile_px: int,
    waste_min_dist_m: float = WASTE_EMPTY_VID_MIN_DIST_M,
) -> ValidationReport:
    """Every ZIP valid, and together they hold each expected tile exactly once."""
    names = [Path(p).name for p in zip_paths]
    report = ValidationReport(tuple(error("duplicate_zip", f"ZIP name {n!r} repeated", zip_name=n)
                                    for n, c in sorted(Counter(names).items()) if c > 1))
    scanned = []
    for path in zip_paths:
        scan = _scan_zip(Path(path), expected_sha256, cfg, tile_px, waste_min_dist_m)
        report = report.merge(scan.report)
        scanned.append((Path(path).name, scan.image_files))
    return report.with_issues(*_set_issues(scanned, frozenset(expected_tiles)))
