"""Post-write self-check of the upload ZIPs (contract §4.1, 01 §3.10 step 10).

Each written ZIP is re-read with the tolerant reader (cvat.reader) and compared with the in-memory
document: images, per-label counts, attributes, vertices within roundtrip_max_dev_px; the image
sha256 are recomputed; the union of all ZIPs holds every expected tile exactly once. When the
source AnnSet is given, per tile the union IoU (UTM) of written vs source polygons/boxes must reach
selfcheck_min_union_iou, and written row polylines must stay within selfcheck_max_row_dev_m (Hausdorff)
of the source rows (catches axis flips and offsets; a thin-band IoU flagged mm-level simplification).
"""

from __future__ import annotations

import hashlib
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry

from vineyard.annset.model import TILE_COLUMN, AnnSet
from vineyard.config.sections_io import CvatExportConfig
from vineyard.contracts.enums import ANNSET_LAYER_OF_LABEL, Label, Severity
from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.packer import IMAGE_EXT, IMAGES_DIR
from vineyard.cvat.reader import read_cvat_file
from vineyard.cvat.report import Issue, ValidationReport, error
from vineyard.errors import CvatFormatError
from vineyard.geo.tiling import px_to_utm, tile_box, tile_ref

MIN_DIFF_M2: Final = 0.05  # IoU misses whose symmetric difference is below this are rounding noise
_IMAGES_PREFIX: Final = f"{IMAGES_DIR}/"


# ---------------------------------------------------------------- (a) document round-trip


def _shape_issues(exp: CvatShape, act: CvatShape, max_dev_px: float, at: dict[str, str]) -> list[Issue]:
    if (exp.tag, exp.label) != (act.tag, act.label):
        return [error("selfcheck_count", f"shape {exp.tag}/{exp.label} read back as {act.tag}/{act.label}", **at)]
    issues: list[Issue] = []
    if exp.attributes != act.attributes:
        issues.append(error("selfcheck_attributes", f"{exp.attributes} read back as {act.attributes}", **at))
    a, b = np.asarray(exp.points), np.asarray(act.points)
    if a.shape != b.shape:
        issues.append(error("selfcheck_vertices", f"{len(a)} vertices read back as {len(b)}", **at))
    elif len(a) and float(np.abs(a - b).max()) > max_dev_px:
        issues.append(error("selfcheck_vertices", f"vertex deviation {np.abs(a - b).max():.3f} px", **at))
    return issues


def _image_issues(exp: CvatImage, act: CvatImage, max_dev_px: float, zip_name: str) -> list[Issue]:
    tile_id = exp.name[: -len(IMAGE_EXT)]
    where = {"zip_name": zip_name, "tile_id": tile_id}
    if exp.id != act.id:
        return [error("selfcheck_images", f"image id {exp.id} read back as {act.id}", **where)]
    counts_exp = Counter(s.label for s in exp.shapes)
    counts_act = Counter(s.label for s in act.shapes)
    if counts_exp != counts_act:
        return [error("selfcheck_count", f"counts {dict(counts_exp)} read back as {dict(counts_act)}", **where)]
    issues: list[Issue] = []
    for k, (a, b) in enumerate(zip(exp.shapes, act.shapes, strict=True)):
        issues += _shape_issues(a, b, max_dev_px, where | {"object_ref": a.ref_id or f"{a.label}#{k}"})
    return issues


def compare_documents(expected: CvatDocument, actual: CvatDocument, *, max_dev_px: float,
                      zip_name: str) -> list[Issue]:
    if expected.image_names != actual.image_names:
        return [error("selfcheck_images", f"{len(expected.images)} images written, {len(actual.images)} read back",
                      zip_name=zip_name)]
    return [i for exp, act in zip(expected.images, actual.images, strict=True)
            for i in _image_issues(exp, act, max_dev_px, zip_name)]


def _read_back(path: Path, zip_name: str) -> tuple[CvatDocument | None, list[Issue]]:
    try:
        doc, qa = read_cvat_file(path)
    except CvatFormatError as exc:
        return None, [error("selfcheck_unreadable", str(exc), zip_name=zip_name)]
    issues = [error("selfcheck_reader", f"{q.code}: {q.message}", zip_name=zip_name, tile_id=q.tile_id,
                    object_ref=q.object_id) for q in qa if q.severity is Severity.ERROR]
    return doc, issues


# ---------------------------------------------------------------- (c) sha256, (d) tile set


def _sha_issues(path: Path, expected: Mapping[str, str], zip_name: str) -> tuple[list[str], list[Issue]]:
    names: list[str] = []
    issues: list[Issue] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.startswith(_IMAGES_PREFIX):
                continue
            tile_id = info.filename[len(_IMAGES_PREFIX): -len(IMAGE_EXT)]
            names.append(tile_id)
            if hashlib.sha256(zf.read(info)).hexdigest() != expected.get(tile_id):
                issues.append(error("selfcheck_sha256", "image bytes differ from the tile index", zip_name=zip_name,
                                    tile_id=tile_id))
    return names, issues


def _tile_set_issues(tiles: Sequence[str], expected: frozenset[str]) -> list[Issue]:
    counts = Counter(tiles)
    dup = sorted(t for t, c in counts.items() if c > 1)
    missing = sorted(expected - set(counts))
    extra = sorted(set(counts) - expected)
    if not (dup or missing or extra):
        return []
    msg = f"{len(counts)} unique tiles written, expected {len(expected)}: dup={dup[:5]} missing={missing[:5]} " \
          f"extra={extra[:5]}"
    return [error("selfcheck_tiles", msg)]


# ---------------------------------------------------------------- (b) geometry vs the source AnnSet


def _shape_utm(shape: CvatShape, tile_id: str) -> BaseGeometry:
    pts = px_to_utm(tile_ref(tile_id), np.asarray(shape.points))
    if shape.tag == "box":
        (x0, y0), (x1, y1) = pts
        return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if shape.tag == "polyline":
        return LineString(pts)
    return shapely.make_valid(Polygon(pts))


def _source_union(annset: AnnSet, label: Label, tile_id: str) -> BaseGeometry:
    frame = annset.layer(ANNSET_LAYER_OF_LABEL[label])
    geoms = frame.geometry[frame[TILE_COLUMN] == tile_id].to_numpy()
    return shapely.intersection(shapely.union_all(geoms), tile_box(tile_ref(tile_id)))


def _iou_issue(annset: AnnSet, img: CvatImage, label: Label, min_iou: float, zip_name: str) -> list[Issue]:
    tile_id = img.name[: -len(IMAGE_EXT)]
    written = shapely.union_all([_shape_utm(s, tile_id) for s in img.shapes if s.label == label.value])
    source = _source_union(annset, label, tile_id)
    union_area = shapely.union(written, source).area
    if union_area == 0.0:
        return []
    inter = shapely.intersection(written, source).area
    iou = inter / union_area
    if iou >= min_iou or union_area - inter <= MIN_DIFF_M2:
        return []
    return [error("selfcheck_union_iou", f"{label.value} union IoU {iou:.4f} < {min_iou}", zip_name=zip_name,
                  tile_id=tile_id)]


def _row_dev_issue(annset: AnnSet, img: CvatImage, max_dev_m: float, zip_name: str) -> list[Issue]:
    tile_id = img.name[: -len(IMAGE_EXT)]
    written = shapely.union_all([_shape_utm(s, tile_id) for s in img.shapes if s.label == Label.ROW.value])
    source = _source_union(annset, Label.ROW, tile_id)
    if written.is_empty and source.is_empty:
        return []
    dev = float("inf") if written.is_empty or source.is_empty else shapely.hausdorff_distance(written, source)
    if dev <= max_dev_m:
        return []
    return [error("selfcheck_row_deviation", f"row max deviation {dev:.4f} m > {max_dev_m} m", zip_name=zip_name,
                  tile_id=tile_id)]


def _geometry_issues(annset: AnnSet, doc: CvatDocument, cfg: CvatExportConfig, zip_name: str,
                     skip: frozenset[str]) -> list[Issue]:
    kept = [img for img in doc.images if img.name[: -len(IMAGE_EXT)] not in skip]
    areal = [i for img in kept for label in Label if label != Label.ROW
             for i in _iou_issue(annset, img, label, cfg.selfcheck_min_union_iou, zip_name)]
    return areal + [i for img in kept for i in _row_dev_issue(annset, img, cfg.selfcheck_max_row_dev_m, zip_name)]


# ---------------------------------------------------------------- public API


def _check_part(path: Path, doc: CvatDocument, expected_sha256: Mapping[str, str], cfg: CvatExportConfig,
                annset: AnnSet | None, skip: frozenset[str]) -> tuple[list[str], list[Issue]]:
    zip_name = path.name
    read, issues = _read_back(path, zip_name)
    if read is None:
        return [], issues
    issues += compare_documents(doc, read, max_dev_px=cfg.roundtrip_max_dev_px, zip_name=zip_name)
    names, sha_issues = _sha_issues(path, expected_sha256, zip_name)
    issues += sha_issues
    if annset is not None and cfg.verify_roundtrip:
        issues += _geometry_issues(annset, read, cfg, zip_name, skip)
    return names, issues


def selfcheck_upload(
    parts: Sequence[tuple[Path, CvatDocument]],
    *,
    expected_sha256: Mapping[str, str],
    expected_tiles: frozenset[str],
    cfg: CvatExportConfig,
    annset: AnnSet | None = None,
    skip_geometry: frozenset[str] = frozenset(),
) -> ValidationReport:
    """(a) re-read vs in-memory, (b) geometry vs AnnSet, (c) sha256, (d) every tile exactly once.

    `skip_geometry` names tiles the export emptied on purpose (failed / invalid): (b) is skipped there.
    """
    tiles: list[str] = []
    issues: list[Issue] = []
    for path, doc in parts:
        names, found = _check_part(Path(path), doc, expected_sha256, cfg, annset, frozenset(skip_geometry))
        tiles += names
        issues += found
    issues += _tile_set_issues(tiles, frozenset(expected_tiles))
    return ValidationReport(tuple(issues))
