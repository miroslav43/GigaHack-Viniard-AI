"""AnnSet (UTM) -> per-tile CvatImages (CVAT px) with the geometric cleaning of 01 §3.7 / arch §4.13.

Polygons: UTM->px, clip to the tile box, make_valid, explode, drop < min_polygon_px2, notch holes,
simplify (interrows only: canopies are already Douglas-Peucker'd), CCW in UTM (negative px
shoelace), no closing vertex, rounded to coord_decimals and clipped to [0, tile_px]; a polygon that
rounding breaks is repaired once, else dropped with an `invalid_after_rounding` warning.
Rows: clip_line + simplify + rounding. Waste: axis-aligned box from the geometry bounds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final, Literal

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from vineyard.annset.model import TILE_COLUMN, AnnSet
from vineyard.config.sections_io import CvatExportConfig
from vineyard.contracts.enums import LABEL_ATTRIBUTES, Label
from vineyard.contracts.ids import file_name_from_tile_id
from vineyard.cvat.model import CvatImage, CvatShape
from vineyard.cvat.report import Issue, info, warning
from vineyard.geo.ops import (
    clip_box,
    clip_line,
    clip_polygonal,
    drop_consecutive_duplicates,
    make_valid_polygonal,
    notch_holes,
)
from vineyard.geo.tiling import GSD_M, TILE_PX, TileRef, px_to_utm, utm_to_px
from vineyard.logging_setup import get_logger, log_failure

LABEL_LAYERS: Final[tuple[tuple[Label, str, str], ...]] = (
    (Label.ROW, "row_pieces", "piece_id"),
    (Label.INTERROW_AREA, "interrow_pieces", "piece_id"),
    (Label.VINEYARD, "canopies", "canopy_id"),
    (Label.WASTE, "waste", "waste_id"),
)
SIMPLIFIED_LABELS: Final = frozenset({Label.ROW, Label.INTERROW_AREA})
INTERROW_LAYER: Final = "interrow_pieces"
INTERROW_PK: Final = "piece_id"
INTERROW_UNSIMPLIFIED: Final = "interrow_unsimplified"
PX_AREA_M2: Final = GSD_M * GSD_M
PX_ORIENT_SIGN: Final = -1.0  # exterior CW in (u, v) px == CCW in UTM (v points south)
MIN_RING_VERTICES: Final = 3
MIN_LINE_VERTICES: Final = 2
PART_SUFFIX: Final = "#{k}"
INVALID_AFTER_ROUNDING: Final = "invalid_after_rounding"
TileErrorPolicy = Literal["raise", "empty"]

_log = get_logger("cvat.to_cvat")


@dataclass(frozen=True)
class TileWriteStats:
    tile_id: str
    n_in: Mapping[str, int]
    n_out: Mapping[str, int]
    n_dropped: Mapping[str, int]
    n_split: int
    issues: tuple[Issue, ...] = ()
    failed: str = ""  # error text when the tile could not be converted (image written empty)


@dataclass(frozen=True)
class _Params:
    decimals: int
    tile_px: float
    min_area: float
    min_len: float
    simplify: float
    notch: float
    source: str
    tile_box: Polygon = field(default_factory=lambda: box(0.0, 0.0, TILE_PX, TILE_PX))


@dataclass(frozen=True)
class _LabelOut:
    shapes: tuple[CvatShape, ...]
    n_in: int
    n_dropped: int
    n_split: int
    issues: tuple[Issue, ...]


def _params(cfg: CvatExportConfig) -> _Params:
    return _Params(cfg.coord_decimals, float(TILE_PX), cfg.min_polygon_px2, cfg.min_polyline_px, cfg.simplify_px,
                   cfg.notch_width_px, cfg.shape_source)


def px_geom_to_utm(geom: BaseGeometry, t: TileRef) -> BaseGeometry:
    return shapely.transform(geom, lambda uv: px_to_utm(t, uv))


def utm_geom_to_px(geom: BaseGeometry, t: TileRef) -> BaseGeometry:
    return shapely.transform(geom, lambda xy: utm_to_px(t, xy))


def _inside(geom: BaseGeometry, p: _Params) -> bool:
    minx, miny, maxx, maxy = geom.bounds
    return minx >= 0.0 and miny >= 0.0 and maxx <= p.tile_px and maxy <= p.tile_px


def _round(pts: np.ndarray, p: _Params) -> np.ndarray:
    # +0.0 turns -0.0 into 0.0 so the written text never carries a sign
    return np.clip(np.round(pts, p.decimals), 0.0, p.tile_px) + 0.0


def _shoelace(pts: np.ndarray) -> float:
    return 0.5 * float(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(np.roll(pts[:, 0], -1), pts[:, 1]))


# ---------------------------------------------------------------- polygons


def _rounded_ring(poly: Polygon, p: _Params) -> np.ndarray:
    shell = orient(Polygon(poly.exterior), PX_ORIENT_SIGN)
    return drop_consecutive_duplicates(_round(np.asarray(shell.exterior.coords)[:-1], p), closed=True)


def _ring_state(pts: np.ndarray, p: _Params) -> str:
    """'ok' | 'small' (valid but under the area floor) | 'invalid'."""
    if len(np.unique(pts, axis=0)) < MIN_RING_VERTICES:
        return "invalid"
    poly = Polygon(pts)
    if not poly.is_valid or _shoelace(pts) >= 0.0:
        return "invalid"
    return "ok" if poly.area >= p.min_area else "small"


def _finalize(poly: Polygon, p: _Params) -> tuple[list[np.ndarray], bool]:
    """Rounded rings for one clean polygon, and whether rounding broke it beyond one repair."""
    pts = _rounded_ring(poly, p)
    state = _ring_state(pts, p)
    if state != "invalid":
        return ([pts] if state == "ok" else []), False
    repaired = make_valid_polygonal(Polygon(pts)) if len(pts) >= MIN_RING_VERTICES else []
    rings = [r for r in (_rounded_ring(q, p) for q in repaired) if _ring_state(r, p) == "ok"]
    return rings, not rings


def _clean_parts(geom_utm: BaseGeometry, t: TileRef, p: _Params, simplify: bool) -> list[Polygon]:
    geom = utm_geom_to_px(geom_utm, t)
    parts = make_valid_polygonal(geom) if _inside(geom, p) else clip_polygonal(geom, p.tile_box)
    parts = [q for q in parts if q.area >= p.min_area]
    flat = [r for q in parts for r in (notch_holes(q, p.notch) if q.interiors else [q])]
    if simplify and p.simplify > 0.0:
        flat = [s for q in flat for s in make_valid_polygonal(q.simplify(p.simplify, preserve_topology=True))]
    return flat


def _polygon_rings(geom_utm: BaseGeometry, t: TileRef, p: _Params, simplify: bool) -> tuple[list[np.ndarray], int]:
    rings: list[np.ndarray] = []
    n_invalid = 0
    for part in _clean_parts(geom_utm, t, p, simplify):
        found, broken = _finalize(part, p)
        rings += found
        n_invalid += int(broken)
    return rings, n_invalid


# ---------------------------------------------------------------- rows and waste


def _row_points(geom_utm: BaseGeometry, t: TileRef, p: _Params) -> np.ndarray | None:
    geom = utm_geom_to_px(geom_utm, t)
    line = geom if _inside(geom, p) else clip_line(geom, p.tile_box)
    if line is None:
        return None
    if p.simplify > 0.0:
        line = line.simplify(p.simplify, preserve_topology=False)
    pts = drop_consecutive_duplicates(_round(np.asarray(line.coords), p), closed=False)
    if len(np.unique(pts, axis=0)) < MIN_LINE_VERTICES or LineString(pts).length < p.min_len:
        return None
    return pts


def _waste_points(geom_utm: BaseGeometry, t: TileRef, p: _Params) -> np.ndarray | None:
    minx, miny, maxx, maxy = geom_utm.bounds
    (u0, v1), (u1, v0) = utm_to_px(t, np.array([[minx, miny], [maxx, maxy]]))
    clipped = clip_box((u0, v0, u1, v1), (0.0, 0.0, p.tile_px, p.tile_px))
    if clipped is None:
        return None
    xtl, ytl, xbr, ybr = _round(np.asarray(clipped), p)
    return np.array([[xtl, ytl], [xbr, ybr]]) if xtl < xbr and ytl < ybr else None


# ---------------------------------------------------------------- shapes


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _shape(label: Label, pts: np.ndarray, record: Mapping[str, Any], ref: str, p: _Params) -> CvatShape:
    tag = {Label.ROW: "polyline", Label.WASTE: "box"}.get(label, "polygon")
    attrs = tuple((name, _text(record.get(name))) for name in LABEL_ATTRIBUTES[label])
    points = tuple((float(u), float(v)) for u, v in pts)
    return CvatShape(tag, label.value, points, attrs, source=p.source, ref_id=ref)  # type: ignore[arg-type]


def _object_points(label: Label, geom: BaseGeometry, t: TileRef, p: _Params) -> tuple[list[np.ndarray], int]:
    if label == Label.ROW:
        pts = _row_points(geom, t, p)
        return ([] if pts is None else [pts]), 0
    if label == Label.WASTE:
        pts = _waste_points(geom, t, p)
        return ([] if pts is None else [pts]), 0
    return _polygon_rings(geom, t, p, label in SIMPLIFIED_LABELS)


def _label_shapes(label: Label, frame: gpd.GeoDataFrame | None, pk: str, t: TileRef, p: _Params) -> _LabelOut:
    if frame is None or frame.empty:
        return _LabelOut((), 0, 0, 0, ())
    shapes: list[CvatShape] = []
    issues: list[Issue] = []
    dropped = split = 0
    records = frame.drop(columns=frame.geometry.name).to_dict("records")
    for record, geom in zip(records, frame.geometry, strict=True):
        ref = str(record[pk])
        parts, n_invalid = _object_points(label, geom, t, p)
        issues += [warning(INVALID_AFTER_ROUNDING, f"{label.value} dropped: invalid after rounding",
                           tile_id=t.tile_id, object_ref=ref)] * n_invalid
        dropped += int(not parts)
        split += max(len(parts) - 1, 0)
        shapes += [_shape(label, pts, record, ref if k == 0 else ref + PART_SUFFIX.format(k=k + 1), p)
                   for k, pts in enumerate(parts)]
    return _LabelOut(tuple(shapes), len(records), dropped, split, tuple(issues))


def _overlap_m2(canopies: Sequence[CvatShape], interrows: Sequence[CvatShape]) -> float:
    if not canopies or not interrows:
        return 0.0
    can = shapely.union_all([Polygon(s.points) for s in canopies])
    irs = shapely.union_all([Polygon(s.points) for s in interrows])
    return float(shapely.intersection(can, irs).area) * PX_AREA_M2


def _guard_interrow_simplify(outs: dict[Label, _LabelOut], frames: Mapping[str, gpd.GeoDataFrame], t: TileRef,
                             p: _Params, max_overlap_m2: float) -> dict[Label, _LabelOut]:
    """Simplifying an interrow that hugs a canopy can cut into it: keep that tile's interrows unsimplified."""
    irs = outs[Label.INTERROW_AREA]
    if p.simplify <= 0.0 or _overlap_m2(outs[Label.VINEYARD].shapes, irs.shapes) <= max_overlap_m2:
        return outs
    raw = _label_shapes(Label.INTERROW_AREA, frames.get(INTERROW_LAYER), INTERROW_PK, t, replace(p, simplify=0.0))
    note = info(INTERROW_UNSIMPLIFIED, "interrows kept unsimplified: simplify overlapped canopies",
                tile_id=t.tile_id)
    return outs | {Label.INTERROW_AREA: replace(raw, issues=(*raw.issues, note))}


def tile_to_image(
    frames: Mapping[str, gpd.GeoDataFrame], t: TileRef, image_id: int, cfg: CvatExportConfig
) -> tuple[CvatImage, TileWriteStats]:
    """One tile: `frames` maps AnnSet layer -> that tile's rows (missing layer = no objects)."""
    p = _params(cfg)
    outs = {label: _label_shapes(label, frames.get(layer), pk, t, p) for label, layer, pk in LABEL_LAYERS}
    outs = _guard_interrow_simplify(outs, frames, t, p, cfg.max_canopy_interrow_overlap_m2)
    shapes = tuple(s for label, _, _ in LABEL_LAYERS for s in outs[label].shapes)
    stats = TileWriteStats(
        tile_id=t.tile_id,
        n_in=MappingProxyType({lb.value: o.n_in for lb, o in outs.items()}),
        n_out=MappingProxyType({lb.value: len(o.shapes) for lb, o in outs.items()}),
        n_dropped=MappingProxyType({lb.value: o.n_dropped for lb, o in outs.items()}),
        n_split=sum(o.n_split for o in outs.values()),
        issues=tuple(i for o in outs.values() for i in o.issues),
    )
    image = CvatImage(image_id, file_name_from_tile_id(t.tile_id), TILE_PX, TILE_PX, shapes)
    return image, stats


def _group_by_tile(annset: AnnSet, tile_ids: frozenset[str]) -> dict[str, dict[str, gpd.GeoDataFrame]]:
    groups: dict[str, dict[str, gpd.GeoDataFrame]] = {}
    for _, layer, pk in LABEL_LAYERS:
        frame = annset.layer(layer)
        if frame.empty:
            continue
        kept = frame[frame[TILE_COLUMN].isin(tile_ids)]
        for tile_id, sub in kept.groupby(TILE_COLUMN, sort=True):
            ordered = sub.sort_values(pk, kind="mergesort").reset_index(drop=True)
            groups.setdefault(str(tile_id), {})[layer] = ordered
    return groups


def _failed_tile(t: TileRef, image_id: int, exc: Exception) -> tuple[CvatImage, TileWriteStats]:
    log_failure(_log, "cvat.tile_failed", exc, tile_id=t.tile_id)
    zero = MappingProxyType({label.value: 0 for label, _, _ in LABEL_LAYERS})
    message = f"{type(exc).__name__}: {exc}"
    issue = warning("tile_failed", f"tile exported empty: {message}", tile_id=t.tile_id)
    image = CvatImage(image_id, file_name_from_tile_id(t.tile_id), TILE_PX, TILE_PX, ())
    return image, TileWriteStats(t.tile_id, zero, zero, zero, 0, (issue,), message)


def annset_to_images(
    annset: AnnSet,
    tiles: Sequence[TileRef],
    cfg: CvatExportConfig,
    *,
    on_tile_error: TileErrorPolicy = "raise",
) -> tuple[tuple[CvatImage, ...], tuple[TileWriteStats, ...]]:
    """One CvatImage per tile, sorted by name, ids 0..n-1 (re-numbered per ZIP by the export)."""
    ordered = sorted(tiles, key=lambda t: t.tile_id)
    groups = _group_by_tile(annset, frozenset(t.tile_id for t in ordered))
    images: list[CvatImage] = []
    stats: list[TileWriteStats] = []
    for image_id, t in enumerate(ordered):
        try:
            image, stat = tile_to_image(groups.get(t.tile_id, {}), t, image_id, cfg)
        except Exception as exc:  # per-tile isolation: logged with traceback, exported empty, reported
            if on_tile_error == "raise":
                raise
            image, stat = _failed_tile(t, image_id, exc)
        images.append(image)
        stats.append(stat)
    return tuple(images), tuple(stats)
