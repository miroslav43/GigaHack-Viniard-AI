"""Document-level CVAT validator (arch §4.13 step 2): the only safety net, CVAT import checks nothing.

Checks labels and geometry kinds, the exact attribute sets and enum values, ids, geometry validity
and minimum sizes, coordinates in [0, tile_px], waste without a block, and the canopy ∩ interrow
overlap per tile measured on the px geometry actually written.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.validation import explain_validity

from vineyard.config.sections_io import CvatExportConfig
from vineyard.contracts.enums import Label
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.report import Issue, ValidationReport, error, warning
from vineyard.cvat.template import LABEL_SPECS, AttrSpec, LabelSpec
from vineyard.geo.ops import drop_consecutive_duplicates
from vineyard.geo.tiling import GSD_M

WASTE_EMPTY_VID_MIN_DIST_M: Final = 10.0  # mirrors waste.block_assign_max_m; callers pass the config value
VINEYARD_ID_ATTR: Final = "vineyard_id"
IMAGE_EXT: Final = ".tif"
MIN_POLYGON_VERTICES: Final = 3
MIN_POLYLINE_VERTICES: Final = 2
PX_AREA_M2: Final = GSD_M * GSD_M


@dataclass(frozen=True)
class _Ctx:
    cfg: CvatExportConfig
    tile_px: int
    zip_name: str
    tile_id: str
    waste_dist: Mapping[str, float] | None
    waste_min_dist_m: float

    def at(self, ref: str) -> dict[str, str]:
        return {"zip_name": self.zip_name, "tile_id": self.tile_id, "object_ref": ref}


def _shape_ref(shape: CvatShape, k: int) -> str:
    return f"{shape.label}#{k}" + (f":{shape.ref_id}" if shape.ref_id else "")


def _image_issues(img: CvatImage, tile_px: int, zip_name: str) -> tuple[str, list[Issue]]:
    where = {"zip_name": zip_name, "object_ref": img.name}
    issues: list[Issue] = []
    if img.width != tile_px or img.height != tile_px:
        issues.append(error("image_size", f"{img.width}x{img.height} != {tile_px}x{tile_px}", **where))
    base = img.name[: -len(IMAGE_EXT)] if img.name.endswith(IMAGE_EXT) else ""
    nfc = unicodedata.normalize("NFC", img.name) == img.name
    if not (base and nfc and is_valid_id(IdKind.TILE, base)):
        issues.append(error("image_name_invalid", f"image name {img.name!r} is not <tile_id>.tif", **where))
        return "", issues
    return base, issues


# ---------------------------------------------------------------- attributes


def _check_names(shape: CvatShape, spec: LabelSpec, at: dict[str, str]) -> list[Issue]:
    names = shape.attribute_names
    expected = tuple(a.name for a in spec.attributes)
    dup = sorted(n for n, c in Counter(names).items() if c > 1)
    missing = [n for n in expected if n not in names]
    extra = sorted(set(names) - set(expected))
    issues = [error("duplicate_attr", f"attribute {n!r} repeated", **at) for n in dup]
    issues += [error("missing_attr", f"attribute {n!r} missing on {shape.label}", **at) for n in missing]
    issues += [error("extra_attr", f"unknown attribute {n!r} on {shape.label}", **at) for n in extra]
    if not issues and names != expected:
        issues.append(warning("attr_order", f"attribute order {names} != {expected}", **at))
    return issues


def _check_waste_vid(shape: CvatShape, ctx: _Ctx, at: dict[str, str]) -> list[Issue]:
    dist = None if ctx.waste_dist is None or shape.ref_id is None else ctx.waste_dist.get(shape.ref_id)
    if dist is None:
        return [warning("waste_block_unknown", "empty waste vineyard_id with unknown block distance", **at)]
    if dist <= ctx.waste_min_dist_m:
        msg = f"empty waste vineyard_id but block at {dist:.2f} m <= {ctx.waste_min_dist_m} m"
        return [error("waste_vid_empty_near_block", msg, **at)]
    return []


def _check_value(shape: CvatShape, attr: AttrSpec, value: str, ctx: _Ctx, at: dict[str, str]) -> list[Issue]:
    if attr.input_type == "select":
        ok = value in attr.values
        return [] if ok else [error("bad_enum", f"{attr.name}={value!r} not in {attr.values}", **at)]
    if value == "":
        if shape.label == Label.WASTE and attr.name == VINEYARD_ID_ATTR:
            return _check_waste_vid(shape, ctx, at)
        return [error("empty_id", f"{attr.name} is empty", **at)]
    if value != value.strip():
        return [error("id_whitespace", f"{attr.name}={value!r} has surrounding whitespace", **at)]
    return []


def _check_attributes(shape: CvatShape, spec: LabelSpec, ctx: _Ctx, at: dict[str, str]) -> list[Issue]:
    issues = _check_names(shape, spec, at)
    known = {a.name: a for a in spec.attributes}
    for name, value in shape.attributes:
        if name in known:
            issues += _check_value(shape, known[name], value, ctx, at)
    return issues


# ---------------------------------------------------------------- geometry


def _shoelace(pts: np.ndarray) -> float:
    u, v = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(u, np.roll(v, -1)) - np.dot(np.roll(u, -1), v))


def _n_distinct(pts: np.ndarray) -> int:
    return len(np.unique(pts, axis=0))


def _check_polygon(pts: np.ndarray, ctx: _Ctx, at: dict[str, str]) -> tuple[list[Issue], Polygon | None]:
    issues: list[Issue] = []
    if len(pts) >= 2 and np.array_equal(pts[0], pts[-1]):
        issues.append(error("closing_vertex", "polygon repeats its first vertex", **at))
    ring = drop_consecutive_duplicates(pts, closed=True)
    if _n_distinct(ring) < MIN_POLYGON_VERTICES:
        return [*issues, error("too_few_vertices", f"{_n_distinct(ring)} distinct vertices", **at)], None
    if len(ring) < len(pts) - (1 if issues else 0):
        issues.append(warning("duplicate_vertex", "consecutive duplicate vertices", **at))
    poly = Polygon(ring)
    if not poly.is_valid:
        return [*issues, error("invalid_geometry", explain_validity(poly), **at)], None
    if poly.area < ctx.cfg.min_polygon_px2:
        issues.append(error("area_too_small", f"area {poly.area:.2f} px2 < {ctx.cfg.min_polygon_px2}", **at))
    if _shoelace(ring) >= 0.0:
        issues.append(error("wrong_orientation", "polygon is not CCW in UTM (px shoelace >= 0)", **at))
    return issues, poly


def _check_polyline(pts: np.ndarray, ctx: _Ctx, at: dict[str, str]) -> list[Issue]:
    if _n_distinct(pts) < MIN_POLYLINE_VERTICES:
        return [error("too_few_vertices", f"{_n_distinct(pts)} distinct points", **at)]
    length = LineString(pts).length
    if length < ctx.cfg.min_polyline_px:
        return [error("polyline_too_short", f"length {length:.2f} px < {ctx.cfg.min_polyline_px}", **at)]
    return []


def _check_box(shape: CvatShape, at: dict[str, str]) -> list[Issue]:
    xtl, ytl, xbr, ybr = shape.box_xyxy
    if xtl < xbr and ytl < ybr:
        return []
    return [error("box_degenerate", f"box ({xtl}, {ytl}, {xbr}, {ybr}) needs xtl<xbr and ytl<ybr", **at)]


def _check_geometry(shape: CvatShape, ctx: _Ctx, at: dict[str, str]) -> tuple[list[Issue], Polygon | None]:
    pts = np.asarray(shape.points, dtype=np.float64).reshape(-1, 2)
    issues: list[Issue] = []
    if len(pts) and (pts.min() < 0.0 or pts.max() > ctx.tile_px):
        issues.append(error("coord_out_of_range", f"coordinates outside [0, {ctx.tile_px}]", **at))
    if shape.tag == "polygon":
        found, poly = _check_polygon(pts, ctx, at)
        return issues + found, poly
    if shape.tag == "polyline":
        return issues + _check_polyline(pts, ctx, at), None
    return issues + _check_box(shape, at), None


# ---------------------------------------------------------------- shapes and images


def _check_shape(shape: CvatShape, k: int, ctx: _Ctx) -> tuple[list[Issue], Polygon | None]:
    at = ctx.at(_shape_ref(shape, k))
    spec = LABEL_SPECS.get(shape.label)
    if spec is None:
        return [error("unknown_label", f"label {shape.label!r} is not in the task", **at)], None
    if shape.tag != spec.shape_tag:
        return [error("wrong_shape", f"{shape.label} must be a {spec.shape_tag}, got {shape.tag}", **at)], None
    issues = _check_attributes(shape, spec, ctx, at)
    if shape.source != ctx.cfg.shape_source:
        issues.append(warning("shape_source", f"source={shape.source!r} != {ctx.cfg.shape_source!r}", **at))
    found, poly = _check_geometry(shape, ctx, at)
    return issues + found, poly


def _overlap_issue(canopies: list[Polygon], interrows: list[Polygon], ctx: _Ctx) -> list[Issue]:
    if not canopies or not interrows:
        return []
    inter = shapely.intersection(shapely.union_all(canopies), shapely.union_all(interrows))
    area_m2 = float(inter.area) * PX_AREA_M2
    if area_m2 <= ctx.cfg.max_canopy_interrow_overlap_m2:
        return []
    msg = f"canopy ∩ interrow = {area_m2:.4f} m2 > {ctx.cfg.max_canopy_interrow_overlap_m2} m2"
    return [error("canopy_interrow_overlap", msg, **ctx.at("tile"))]


def validate_image(
    img: CvatImage,
    *,
    cfg: CvatExportConfig,
    tile_px: int,
    waste_block_dist_m: Mapping[str, float] | None = None,
    waste_min_dist_m: float = WASTE_EMPTY_VID_MIN_DIST_M,
    zip_name: str = "",
) -> ValidationReport:
    tile_id, issues = _image_issues(img, tile_px, zip_name)
    ctx = _Ctx(cfg, tile_px, zip_name, tile_id, waste_block_dist_m, waste_min_dist_m)
    polys: dict[str, list[Polygon]] = {Label.VINEYARD: [], Label.INTERROW_AREA: []}
    for k, shape in enumerate(img.shapes):
        found, poly = _check_shape(shape, k, ctx)
        issues += found
        if poly is not None and shape.label in polys:
            polys[shape.label].append(poly)
    issues += _overlap_issue(polys[Label.VINEYARD], polys[Label.INTERROW_AREA], ctx)
    return ValidationReport(tuple(issues))


def validate_document(
    doc: CvatDocument,
    *,
    cfg: CvatExportConfig,
    tile_px: int,
    waste_block_dist_m: Mapping[str, float] | None = None,
    waste_min_dist_m: float = WASTE_EMPTY_VID_MIN_DIST_M,
    zip_name: str = "",
) -> ValidationReport:
    dups = sorted(n for n, c in Counter(doc.image_names).items() if c > 1)
    report = ValidationReport(tuple(
        error("duplicate_image", f"image {n!r} appears more than once", zip_name=zip_name, object_ref=n)
        for n in dups
    ))
    for img in doc.images:
        report = report.merge(validate_image(img, cfg=cfg, tile_px=tile_px, waste_block_dist_m=waste_block_dist_m,
                                             waste_min_dist_m=waste_min_dist_m, zip_name=zip_name))
    return report
