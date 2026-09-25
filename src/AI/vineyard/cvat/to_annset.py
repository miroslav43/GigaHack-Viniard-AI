"""CvatDocument -> AnnSet + qa_issues (contract §4.3-§4.4, design 01 §3.12, resolutions I2/X8).

Geometry: px -> UTM (contract px_to_utm) -> make_valid (polygonal parts exploded) -> CCW. IDs:
canopies `<tile>:C0001` and interrow pieces `<tile>:I001` in XML order, row pieces
`<row_id>@<tile>[#k]`, waste `W0001` sorted by (tile, ytl, xtl). Raw ids stay raw (relaxed
validation for marcaj/reference). A canopy's row_id is recomputed: nearest row piece of its tile
within `import.canopy_row_assign_max_m`, ties by centroid perpendicular distance, then piece_id.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import geopandas as gpd
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry

from vineyard.annset.model import AnnSet, empty_annset, make_meta
from vineyard.config.sections_io import ImportConfig
from vineyard.config.sections_perception import CanopyConfig
from vineyard.contracts.enums import Label, Severity, Source
from vineyard.contracts.ids import (
    format_canopy_id,
    format_marcaj_interrow_piece_id,
    format_row_piece_id,
    format_waste_id,
)
from vineyard.contracts.qa import QaIssue, issues_to_gdf
from vineyard.contracts.schemas import coerce_layer, empty_layer, validate_layer
from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.normalize import case_collision_issues, normalize_shape_attributes
from vineyard.cvat.reader import image_key
from vineyard.cvat.template import LABEL_SPECS
from vineyard.errors import CvatFormatError
from vineyard.geo.ops import drop_consecutive_duplicates, make_valid_polygonal, orient_ccw
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, TileRef, px_to_utm, utm_to_px

IMPORT_CONFIDENCE: Final = 1.0
EDGE_TOUCH_PX: Final = 1.0  # index-convention canopies end at 2047, one pixel short of the edge
WASTE_CATEGORY: Final = "unknown"
WASTE_DETECTOR: Final = "manual"
TIE_DECIMALS: Final = 9  # distances equal to 1e-9 m are ties
MIN_RING_POINTS: Final = 3
MIN_LINE_POINTS: Final = 2
CODE_LABEL: Final = "schema_violation"
CODE_DROPPED: Final = "geom_dropped"
CODE_REPAIRED: Final = "geom_repaired"
_NO_BOX: Final = (float("nan"),) * 4

RowRef = tuple[str, str, LineString]  # (row_id, piece_id, axis)
BoxPx = tuple[float, float, float, float]


@dataclass(frozen=True)
class ImportedObject:
    """One accepted CVAT object (a polygon may yield several after make_valid)."""

    label: Label
    tile_id: str
    geom: BaseGeometry
    attrs: Mapping[str, str]
    box_px: BoxPx = _NO_BOX


@dataclass(frozen=True)
class _Ctx:
    tref: TileRef
    object_id: str

    def issue(self, severity: Severity, code: str, message: str) -> QaIssue:
        return QaIssue(severity, code, self.tref.tile_id, self.object_id, message)


# ---------------------------------------------------------------- shape geometry


def _polygons(shape: CvatShape, ctx: _Ctx) -> tuple[list[BaseGeometry], list[QaIssue]]:
    pts = drop_consecutive_duplicates(np.asarray(shape.points, dtype=np.float64), closed=True)
    if len(np.unique(pts, axis=0)) < MIN_RING_POINTS:
        return [], [ctx.issue(Severity.ERROR, CODE_DROPPED, "polygon has fewer than 3 distinct points")]
    poly = Polygon(px_to_utm(ctx.tref, pts))
    parts = make_valid_polygonal(poly)
    if not parts:
        return [], [ctx.issue(Severity.ERROR, CODE_DROPPED, "polygon has no area after make_valid")]
    issues = []
    if len(parts) > 1 or not poly.is_valid:
        issues.append(ctx.issue(Severity.INFO, CODE_REPAIRED, f"invalid polygon repaired into {len(parts)} part(s)"))
    return [orient_ccw(p) for p in parts], issues


def _polyline(shape: CvatShape, ctx: _Ctx, min_length_m: float) -> tuple[list[BaseGeometry], list[QaIssue]]:
    pts = drop_consecutive_duplicates(np.asarray(shape.points, dtype=np.float64), closed=False)
    line = LineString(px_to_utm(ctx.tref, pts)) if len(pts) >= MIN_LINE_POINTS else None
    if line is None or line.length < min_length_m:
        return [], [ctx.issue(Severity.ERROR, CODE_DROPPED, f"polyline shorter than {min_length_m} m")]
    return [line], []


def _box(shape: CvatShape, ctx: _Ctx) -> tuple[list[BaseGeometry], list[QaIssue], BoxPx]:
    xtl, ytl, xbr, ybr = shape.box_xyxy
    (u0, u1), (v0, v1) = sorted((xtl, xbr)), sorted((ytl, ybr))
    if u1 <= u0 or v1 <= v0:
        return [], [ctx.issue(Severity.ERROR, CODE_DROPPED, "box has no area")], _NO_BOX
    issues = []
    if (u0, v0) != (xtl, ytl):
        issues.append(ctx.issue(Severity.WARNING, CODE_REPAIRED, "box corners were swapped"))
    (x0, y1), (x1, y0) = px_to_utm(ctx.tref, np.array([[u0, v0], [u1, v1]]))
    return [box(x0, y0, x1, y1)], issues, (u0, v0, u1, v1)


def _label(shape: CvatShape, ctx: _Ctx) -> tuple[Label | None, list[QaIssue]]:
    spec = LABEL_SPECS.get(shape.label)
    if spec is None or spec.shape_tag != shape.tag:
        message = f"label {shape.label!r} as <{shape.tag}> is not allowed; object skipped"
        return None, [ctx.issue(Severity.ERROR, CODE_LABEL, message)]
    return Label(shape.label), []


def convert_shape(shape: CvatShape, tref: TileRef, k: int,
                  cfg: ImportConfig) -> tuple[list[ImportedObject], list[QaIssue]]:
    """Objects (UTM, CCW) and qa issues for the k-th shape of a tile's image."""
    ctx = _Ctx(tref, f"{shape.label}[{k}]")
    label, issues = _label(shape, ctx)
    if label is None:
        return [], issues
    attrs, attr_issues, usable = normalize_shape_attributes(label, shape.attributes, cfg=cfg,
                                                            tile_id=tref.tile_id, object_id=ctx.object_id)
    issues.extend(attr_issues)
    if not usable:
        return [], issues
    box_px = _NO_BOX
    if shape.tag == "box":
        geoms, geom_issues, box_px = _box(shape, ctx)
    elif shape.tag == "polyline":
        geoms, geom_issues = _polyline(shape, ctx, cfg.min_line_length_m)
    else:
        geoms, geom_issues = _polygons(shape, ctx)
    objs = [ImportedObject(label, tref.tile_id, g, attrs, box_px) for g in geoms]
    return objs, [*issues, *geom_issues]


@dataclass(frozen=True)
class _Prov:
    source: str
    run_id: str
    model_version: str

    def columns(self) -> dict[str, object]:
        return {"source": self.source, "run_id": self.run_id, "model_version": self.model_version,
                "confidence": IMPORT_CONFIDENCE, "qa_flags": ""}


# ---------------------------------------------------------------- structure


def _tile_images(doc: CvatDocument, tile_refs: Mapping[str, TileRef]) -> list[tuple[str, CvatImage]]:
    """(tile_id, image) in document order; unknown names, wrong sizes and duplicates raise."""
    out: list[tuple[str, CvatImage]] = []
    bad: list[str] = []
    for img in doc.images:
        tile_id = image_key(img.name).removesuffix(".tif")
        if tile_id not in tile_refs or (img.width, img.height) != (TILE_PX, TILE_PX):
            bad.append(f"{img.name} ({img.width}x{img.height})")
        out.append((tile_id, img))
    if bad:
        raise CvatFormatError("images with unknown names or a size other than 2048", images=bad[:20],
                              n_bad=len(bad))
    seen = [t for t, _ in out]
    dups = sorted({t for t in seen if seen.count(t) > 1})
    if dups:
        raise CvatFormatError("the same image appears twice in one document", tile_ids=dups[:20])
    return out


# ---------------------------------------------------------------- canopy -> row assignment


def _pick(canopy: Polygon, cands: Sequence[RowRef]) -> RowRef:
    def key(ref: RowRef) -> tuple[float, float, str]:
        return (round(canopy.distance(ref[2]), TIE_DECIMALS),
                round(ref[2].distance(canopy.centroid), TIE_DECIMALS), ref[1])

    return min(cands, key=key)


def assign_canopy_rows(canopies: Sequence[Polygon], rows: Sequence[RowRef], *,
                       max_m: float) -> list[tuple[str | None, str | None]]:
    """(row_id, piece_id) per canopy: nearest axis within max_m; (None, None) when there is none."""
    if not rows:
        return [(None, None)] * len(canopies)
    tree = STRtree([r[2] for r in rows])
    out: list[tuple[str | None, str | None]] = []
    for canopy in canopies:
        hits = tree.query(canopy, predicate="dwithin", distance=max_m)
        if len(hits) == 0:
            out.append((None, None))
            continue
        row_id, piece_id, _ = _pick(canopy, [rows[i] for i in sorted(hits)])
        out.append((row_id, piece_id))
    return out


def _along_m(poly: Polygon, axis: LineString | None) -> float:
    if axis is None:
        return float("nan")
    a, b = np.asarray(axis.coords[0]), np.asarray(axis.coords[-1])
    direction = (b - a) / np.linalg.norm(b - a)
    proj = np.asarray(poly.exterior.coords) @ direction
    return float(proj.max() - proj.min())


def _touches_edge(poly: Polygon, tref: TileRef) -> bool:
    uv = utm_to_px(tref, np.asarray(poly.exterior.coords))
    return bool(uv.min() <= EDGE_TOUCH_PX or uv.max() >= TILE_PX - EDGE_TOUCH_PX)


# ---------------------------------------------------------------- frames


def _frame(records: list[dict[str, object]], geoms: list[BaseGeometry], layer: str) -> gpd.GeoDataFrame:
    if not records:
        return empty_layer(layer)
    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=f"EPSG:{CRS_EPSG}")
    return coerce_layer(gdf, layer)


def _row_frame(objs: Sequence[ImportedObject], prov: _Prov) -> gpd.GeoDataFrame:
    records, geoms, seen = [], [], {}
    for obj in objs:
        key = (obj.attrs["row_id"], obj.tile_id)
        seen[key] = seen.get(key, 0) + 1
        records.append({
            "piece_id": format_row_piece_id(obj.attrs["row_id"], obj.tile_id, seen[key]),
            "row_id": obj.attrs["row_id"], "vineyard_id": obj.attrs["vineyard_id"], "tile_id": obj.tile_id,
            "row_structure": obj.attrs["row_structure"], "length_m": obj.geom.length, "max_gap_m": np.nan,
            "n_vertices": len(obj.geom.coords), **prov.columns()})
        geoms.append(obj.geom)
    return _frame(records, geoms, "row_pieces")


def _row_refs(rows: gpd.GeoDataFrame, tile_id: str) -> tuple[RowRef, ...]:
    sub = rows[rows["tile_id"] == tile_id]
    return tuple(zip(sub["row_id"], sub["piece_id"], sub.geometry, strict=True))


def _canopy_frame(objs: Sequence[ImportedObject], rows: gpd.GeoDataFrame, trefs: Mapping[str, TileRef],
                  cfg: ImportConfig, canopy_cfg: CanopyConfig, prov: _Prov) -> gpd.GeoDataFrame:
    records, geoms = [], []
    for tile_id in sorted({o.tile_id for o in objs}):
        tile_objs = [o for o in objs if o.tile_id == tile_id]
        refs = _row_refs(rows, tile_id)
        axes = {piece: line for _, piece, line in refs}
        assigned = assign_canopy_rows([o.geom for o in tile_objs], refs, max_m=cfg.canopy_row_assign_max_m)
        for k, (obj, (row_id, piece_id)) in enumerate(zip(tile_objs, assigned, strict=True), start=1):
            area = obj.geom.area
            records.append({
                "canopy_id": format_canopy_id(tile_id, k), "tile_id": tile_id,
                "vineyard_id": obj.attrs["vineyard_id"], "row_id": row_id, "area_m2": area,
                "n_vertices": len(obj.geom.exterior.coords) - 1,
                "along_m": _along_m(obj.geom, axes.get(piece_id) if piece_id else None),
                "is_clump": area > canopy_cfg.clump_area_m2,
                "touches_edge": _touches_edge(obj.geom, trefs[tile_id]), **prov.columns()})
            geoms.append(obj.geom)
    return _frame(records, geoms, "canopies")


def _interrow_frame(objs: Sequence[ImportedObject], prov: _Prov) -> gpd.GeoDataFrame:
    records, geoms, counters = [], [], {}
    for obj in objs:
        counters[obj.tile_id] = counters.get(obj.tile_id, 0) + 1
        records.append({
            "piece_id": format_marcaj_interrow_piece_id(obj.tile_id, counters[obj.tile_id]),
            "interrow_id": None, "vineyard_id": obj.attrs["vineyard_id"], "tile_id": obj.tile_id,
            "row_left_id": None, "row_right_id": None, "interrow_cover": obj.attrs["interrow_cover"],
            "veg_frac": np.nan, "shadow_frac": np.nan, "area_m2": obj.geom.area, "width_mean_m": np.nan,
            "n_notches": 0, **prov.columns()})
        geoms.append(obj.geom)
    return _frame(records, geoms, "interrow_pieces")


def _waste_frame(objs: Sequence[ImportedObject], prov: _Prov) -> gpd.GeoDataFrame:
    ordered = sorted(objs, key=lambda o: (o.tile_id, o.box_px[1], o.box_px[0], o.box_px[3], o.box_px[2]))
    records, geoms = [], []
    for k, obj in enumerate(ordered, start=1):
        xtl, ytl, xbr, ybr = obj.box_px
        records.append({
            "waste_id": format_waste_id(k), "tile_id": obj.tile_id, "vineyard_id": obj.attrs["vineyard_id"],
            "dist_block_m": np.nan, "px_xtl": xtl, "px_ytl": ytl, "px_xbr": xbr, "px_ybr": ybr,
            "area_m2": (xbr - xtl) * (ybr - ytl) * GSD_M**2, "category": WASTE_CATEGORY,
            "detector": WASTE_DETECTOR, "exported": True, **prov.columns()})
        geoms.append(obj.geom)
    return _frame(records, geoms, "waste")


# ---------------------------------------------------------------- entry point


def _convert_all(images: Sequence[tuple[str, CvatImage]], trefs: Mapping[str, TileRef], cfg: ImportConfig,
                 issues: list[QaIssue]) -> list[ImportedObject]:
    objs: list[ImportedObject] = []
    for tile_id, img in images:
        for k, shape in enumerate(img.shapes):
            converted, shape_issues = convert_shape(shape, trefs[tile_id], k, cfg)
            objs.extend(converted)
            issues.extend(shape_issues)
    return objs


def _of(objs: Sequence[ImportedObject], label: Label) -> list[ImportedObject]:
    return [o for o in objs if o.label == label]


def _collisions(objs: Sequence[ImportedObject]) -> tuple[QaIssue, ...]:
    return (*case_collision_issues("vineyard_id", sorted({o.attrs["vineyard_id"] for o in objs})),
            *case_collision_issues("row_id", sorted({o.attrs.get("row_id", "") for o in objs})))


def document_to_annset(
    doc: CvatDocument,
    *,
    tile_refs: Mapping[str, TileRef],
    source: Source,
    run_id: str,
    model_version: str,
    cfg: ImportConfig,
    canopy_cfg: CanopyConfig,
    inputs: Sequence[str] = (),
    extra_issues: Sequence[QaIssue] = (),
) -> tuple[AnnSet, gpd.GeoDataFrame]:
    """AnnSet(source) + qa_issues layer. `extra_issues` (e.g. reader issues) are merged into the layer.

    Structural problems raise CvatFormatError; per-object problems are qa issues (object skipped).
    AnnSetMeta.tile_ids lists every image, including empty ones.
    """
    src = Source(source)
    images = _tile_images(doc, tile_refs)
    issues: list[QaIssue] = list(extra_issues)
    objs = _convert_all(images, tile_refs, cfg, issues)
    prov = _Prov(src.value, run_id, model_version)
    rows = _row_frame(_of(objs, Label.ROW), prov)
    layers = {
        "canopies": _canopy_frame(_of(objs, Label.VINEYARD), rows, tile_refs, cfg, canopy_cfg, prov),
        "row_pieces": rows,
        "interrow_pieces": _interrow_frame(_of(objs, Label.INTERROW_AREA), prov),
        "waste": _waste_frame(_of(objs, Label.WASTE), prov),
    }
    issues.extend(_collisions(objs))
    annset = empty_annset(make_meta(src, run_id, model_version, [t for t, _ in images], inputs=inputs))
    for name, gdf in layers.items():
        validate_layer(gdf, name)
        annset = annset.with_layer(name, gdf)
    qa = issues_to_gdf(issues, source=src, run_id=run_id, model_version=model_version)
    return annset, qa
