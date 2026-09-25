"""Shared AnnSet factories for the post-Marcaj tests (derive, targets, passable, route, measure, web).

Two builders:
- `make_annset(BlockSpec(...), ...)`: synthetic straight-row blocks, split per 51.2 m tile like CVAT data
  (row pieces, rectangle canopies with optional gaps, interrow bands cut at the shorter row end).
- `reference_annset(examples_xml)`: AnnSet(reference) from the organizers' 2 example tiles, built with
  the independent test oracle `tests.helpers.examples` (no dependency on vineyard.cvat).

Everything returned is a fresh object; callers must not mutate frames they did not create.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Final

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from tests.helpers.examples import ExampleImage, load_examples, to_utm
from vineyard.annset.model import AnnSet, make_meta
from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import Source
from vineyard.contracts.ids import (
    format_canopy_id,
    format_marcaj_interrow_piece_id,
    format_row_id,
    format_row_piece_id,
    format_waste_id,
)
from vineyard.contracts.ordering import canonical_normal
from vineyard.contracts.schemas import coerce_layer, empty_layer, validate_layer
from vineyard.geo.ops import make_valid_polygonal, orient_ccw, split_multi
from vineyard.geo.tiling import (
    CRS_EPSG,
    GRID_ORIGIN_X,
    GRID_ORIGIN_Y,
    TILE_M,
    tile_box,
    tile_ref,
    tile_ref_from_grid,
    utm_to_px,
)


@lru_cache(maxsize=1)
def default_config() -> AppConfig:
    """configs/default.yaml (cached; AppConfig is frozen)."""
    return load_config()


EXAMPLE_TILES: Final = ("siret3_r006_c004", "siret3_r021_c012")
# Inside tiles r018_c011 / r018_c012 (both exist): a default block spans the two tiles.
DEFAULT_ORIGIN: Final = (629560.0, 5220290.0)
SYNTH_RUN_ID: Final = "20260926T0000-reference-000000"
SYNTH_MODEL_VERSION: Final = "synthetic@tests"
CANOPY_ROW_MAX_M: Final = 0.5  # contract §2.5.7
CLUMP_AREA_M2: Final = 2.0  # contract §2.5.7 is_clump
MIN_AREA_M2: Final = 1e-6
MIN_LINE_M: Final = 0.05
EDGE_EPS_M: Final = 1e-6


@dataclass(frozen=True)
class Prov:
    source: Source = Source.REFERENCE
    run_id: str = SYNTH_RUN_ID
    model_version: str = SYNTH_MODEL_VERSION
    confidence: float = 1.0

    def columns(self) -> dict[str, Any]:
        return {"source": Source(self.source).value, "run_id": self.run_id, "model_version": self.model_version,
                "confidence": self.confidence, "qa_flags": ""}


DEFAULT_PROV: Final = Prov()


def layer_frame(name: str, records: Sequence[Mapping[str, Any]], prov: Prov = DEFAULT_PROV) -> gpd.GeoDataFrame:
    """Contract layer `name` from dict records (each with a `geometry`); provenance filled from `prov`."""
    if not records:
        return empty_layer(name)
    rows = [prov.columns() | {k: v for k, v in r.items() if k != "geometry"} for r in records]
    geoms = gpd.GeoSeries([r["geometry"] for r in records], crs=CRS_EPSG)
    out = coerce_layer(gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS_EPSG), name)
    validate_layer(out, name)
    return out


def build_annset(
    canopies: Sequence[Mapping[str, Any]] = (),
    row_pieces: Sequence[Mapping[str, Any]] = (),
    interrow_pieces: Sequence[Mapping[str, Any]] = (),
    waste: Sequence[Mapping[str, Any]] = (),
    *,
    prov: Prov = DEFAULT_PROV,
    tile_ids: Iterable[str] = (),
) -> AnnSet:
    """AnnSet from records; meta.tile_ids = `tile_ids` plus every tile that carries an object."""
    layers = {
        "canopies": layer_frame("canopies", canopies, prov),
        "row_pieces": layer_frame("row_pieces", row_pieces, prov),
        "interrow_pieces": layer_frame("interrow_pieces", interrow_pieces, prov),
        "waste": layer_frame("waste", waste, prov),
    }
    present = {t for gdf in layers.values() for t in gdf["tile_id"].dropna()}
    meta = make_meta(Source(prov.source), prov.run_id, prov.model_version, sorted(present | set(tile_ids)),
                     created_at="2026-09-26T00:00:00+03:00")
    annset = AnnSet(meta=meta, **layers)
    return annset.with_layer("canopies", annset.canopies)  # refreshes meta.counts


# ------------------------------------------------------------------ per-object records


def row_piece_record(row_id: str, vineyard_id: str, tile_id: str, line: LineString, *, dup: int = 1,
                     row_structure: str = "regular", max_gap_m: float = math.nan) -> dict[str, Any]:
    return {"piece_id": format_row_piece_id(row_id, tile_id, dup), "row_id": row_id, "vineyard_id": vineyard_id,
            "tile_id": tile_id, "row_structure": row_structure, "length_m": line.length, "max_gap_m": max_gap_m,
            "n_vertices": len(line.coords), "geometry": line}


def canopy_record(canopy_id: str, tile_id: str, vineyard_id: str, poly: Polygon, *,
                  row_id: str | None = None) -> dict[str, Any]:
    touches = poly.exterior.distance(tile_box(tile_ref(tile_id)).exterior) < EDGE_EPS_M
    return {"canopy_id": canopy_id, "tile_id": tile_id, "vineyard_id": vineyard_id, "row_id": row_id,
            "area_m2": poly.area, "n_vertices": len(poly.exterior.coords) - 1, "along_m": math.nan,
            "is_clump": poly.area > CLUMP_AREA_M2, "touches_edge": bool(touches), "geometry": orient_ccw(poly)}


def interrow_piece_record(piece_id: str, tile_id: str, vineyard_id: str, poly: Polygon, *,
                          interrow_cover: str = "bare_soil", interrow_id: str | None = None,
                          row_left_id: str | None = None, row_right_id: str | None = None) -> dict[str, Any]:
    return {"piece_id": piece_id, "interrow_id": interrow_id, "vineyard_id": vineyard_id, "tile_id": tile_id,
            "row_left_id": row_left_id, "row_right_id": row_right_id, "interrow_cover": interrow_cover,
            "veg_frac": math.nan, "shadow_frac": math.nan, "area_m2": poly.area, "width_mean_m": math.nan,
            "n_notches": 0, "geometry": orient_ccw(poly)}


def waste_record(waste_id: str, tile_id: str, xy: tuple[float, float], size_m: float, *,
                 vineyard_id: str = "") -> dict[str, Any]:
    half = size_m / 2.0
    rect = box(xy[0] - half, xy[1] - half, xy[0] + half, xy[1] + half)
    (u0, v0), (u1, v1) = utm_to_px(tile_ref(tile_id), np.array([[xy[0] - half, xy[1] + half],
                                                                  [xy[0] + half, xy[1] - half]]))
    return {"waste_id": waste_id, "tile_id": tile_id, "vineyard_id": vineyard_id, "dist_block_m": 0.0,
            "px_xtl": u0, "px_ytl": v0, "px_xbr": u1, "px_ybr": v1, "area_m2": rect.area, "category": "unknown",
            "detector": "manual", "exported": True, "geometry": orient_ccw(rect)}


# ------------------------------------------------------------------ tile splitting


def tiles_overlapping(geom: BaseGeometry) -> list[str]:
    """Grid tiles (existing or not) whose square intersects `geom`'s bounds; sorted ids."""
    minx, miny, maxx, maxy = geom.bounds
    cols = range(math.floor((minx - GRID_ORIGIN_X) / TILE_M), math.floor((maxx - GRID_ORIGIN_X) / TILE_M) + 1)
    rows = range(math.floor((GRID_ORIGIN_Y - maxy) / TILE_M), math.floor((GRID_ORIGIN_Y - miny) / TILE_M) + 1)
    return sorted(tile_ref_from_grid(r, c).tile_id for r in rows for c in cols)


def split_by_tile(geom: BaseGeometry) -> list[tuple[str, BaseGeometry]]:
    """(tile_id, part) for every non-degenerate intersection of `geom` with a tile square."""
    out: list[tuple[str, BaseGeometry]] = []
    for tile_id in tiles_overlapping(geom):
        for part in split_multi(geom.intersection(tile_box(tile_ref(tile_id)))):
            if isinstance(part, LineString) and part.length >= MIN_LINE_M or isinstance(part, Polygon) and part.area > MIN_AREA_M2:
                out.append((tile_id, part))
    return out


# ------------------------------------------------------------------ synthetic blocks


@dataclass(frozen=True)
class BlockSpec:
    """Straight parallel rows; R001 is the max n·c row (contract §1.5), rows step by -spacing along n.

    `gaps` = (row_index, from_m, to_m) along the row from its start: no canopy overlaps that interval.
    `skip_rows` = row indices without pieces or canopies (a missing row); their ids are still reserved.
    """

    vineyard_id: str = "V01"
    n_rows: int = 4
    spacing_m: float = 2.5
    angle_deg: float = 0.0
    origin_xy: tuple[float, float] = DEFAULT_ORIGIN
    length_m: float | tuple[float, ...] = 60.0
    start_m: float | tuple[float, ...] = 0.0
    gaps: tuple[tuple[int, float, float], ...] = ()
    skip_rows: tuple[int, ...] = ()
    canopy_len_m: float = 0.8
    canopy_width_m: float = 0.5
    canopy_step_m: float = 1.0
    row_structure: str = "regular"
    interrow_cover: str = "bare_soil"
    interrow_margin_m: float = 0.3
    lateral_jitter_m: tuple[tuple[int, float], ...] = field(default=())  # (row_index, extra offset along n)

    def _per_row(self, value: float | tuple[float, ...], k: int) -> float:
        return float(value[k - 1]) if isinstance(value, tuple) else float(value)

    def direction(self) -> np.ndarray:
        t = math.radians(self.angle_deg)
        return np.array([math.cos(t), math.sin(t)])

    def normal(self) -> np.ndarray:
        return np.array(canonical_normal(self.angle_deg))

    def row_id(self, k: int) -> str:
        return format_row_id(self.vineyard_id, k)

    def span(self, k: int) -> tuple[float, float]:
        start = self._per_row(self.start_m, k)
        return start, start + self._per_row(self.length_m, k)

    def centre(self, k: int) -> np.ndarray:
        jitter = dict(self.lateral_jitter_m).get(k, 0.0)
        return np.asarray(self.origin_xy, float) - ((k - 1) * self.spacing_m - jitter) * self.normal()

    def point(self, k: int, along_m: float, across_m: float = 0.0) -> np.ndarray:
        return self.centre(k) + along_m * self.direction() + across_m * self.normal()

    def axis(self, k: int) -> LineString:
        a0, a1 = self.span(k)
        return LineString([self.point(k, a0), self.point(k, a1)])

    def present_rows(self) -> tuple[int, ...]:
        return tuple(k for k in range(1, self.n_rows + 1) if k not in self.skip_rows)


def _rect(spec: BlockSpec, k: int, a0: float, a1: float, c0: float, c1: float) -> Polygon:
    return Polygon([spec.point(k, a0, c0), spec.point(k, a1, c0), spec.point(k, a1, c1), spec.point(k, a0, c1)])


def _canopy_spans(spec: BlockSpec, k: int) -> list[tuple[float, float]]:
    a0, a1 = spec.span(k)
    half = spec.canopy_len_m / 2.0
    gaps = [(g0, g1) for (row, g0, g1) in spec.gaps if row == k]
    spans, centre = [], a0 + spec.canopy_step_m / 2.0
    while centre + half <= a1 + EDGE_EPS_M:
        lo, hi = centre - half, centre + half
        if not any(lo < g1 and hi > g0 for g0, g1 in gaps):
            spans.append((lo, hi))
        centre += spec.canopy_step_m
    return spans


def _interrow_band(spec: BlockSpec, k: int) -> Polygon | None:
    """Band between R k and R k+1: axis_k + margin to axis_{k+1} - margin, cut at the shorter row."""
    lo = max(spec.span(k)[0], spec.span(k + 1)[0])
    hi = min(spec.span(k)[1], spec.span(k + 1)[1])
    width = float(np.dot(spec.centre(k) - spec.centre(k + 1), spec.normal()))
    if hi - lo <= 0 or width <= 2 * spec.interrow_margin_m:
        return None
    return _rect(spec, k, lo, hi, -width + spec.interrow_margin_m, -spec.interrow_margin_m)


def _block_objects(spec: BlockSpec) -> tuple[list[tuple], list[tuple], list[tuple]]:
    pieces = [(spec.row_id(k), t, part) for k in spec.present_rows() for t, part in split_by_tile(spec.axis(k))]
    half_w = spec.canopy_width_m / 2.0
    canopies = [(spec.row_id(k), t, part) for k in spec.present_rows() for lo, hi in _canopy_spans(spec, k)
                for t, part in split_by_tile(_rect(spec, k, lo, hi, -half_w, half_w))]
    present = spec.present_rows()
    bands = [(k, _interrow_band(spec, k)) for k in range(1, spec.n_rows) if k in present and k + 1 in present]
    interrows = [(k, t, part) for k, band in bands if band is not None for t, part in split_by_tile(band)]
    return pieces, canopies, interrows


def make_annset(
    *specs: BlockSpec,
    prov: Prov = DEFAULT_PROV,
    waste: Sequence[tuple[str, tuple[float, float], float, str]] = (),
    tile_ids: Iterable[str] = (),
    linked: bool = False,
) -> AnnSet:
    """Synthetic AnnSet of one or more blocks (default: one 4-row block over 2 tiles).

    waste: (tile_id, (x, y), size_m, vineyard_id) boxes, numbered W0001.. in the given order.
    linked=True fills interrow_id / row_left_id / row_right_id like the model does.
    """
    blocks = specs or (BlockSpec(),)
    rows, cans, irs = [], [], []
    for spec in blocks:
        pieces, canopies, interrows = _block_objects(spec)
        rows += [row_piece_record(rid, spec.vineyard_id, t, line, row_structure=spec.row_structure)
                 for rid, t, line in pieces]
        cans += [(t, spec.vineyard_id, poly, rid) for rid, t, poly in canopies]
        irs += [(t, spec, k, poly) for k, t, poly in interrows]
    return build_annset(_number_canopies(cans), rows, _number_interrows(irs, linked),
                        [waste_record(format_waste_id(i), t, xy, s, vineyard_id=v)
                         for i, (t, xy, s, v) in enumerate(waste, start=1)],
                        prov=prov, tile_ids=tile_ids)


def _number_canopies(cans: list[tuple[str, str, Polygon, str]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    out = []
    for tile_id, vid, poly, rid in cans:
        counters[tile_id] = counters.get(tile_id, 0) + 1
        out.append(canopy_record(format_canopy_id(tile_id, counters[tile_id]), tile_id, vid, poly, row_id=rid))
    return out


def _number_interrows(irs: list[tuple[str, BlockSpec, int, Polygon]], linked: bool) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    out = []
    for tile_id, spec, k, poly in irs:
        counters[tile_id] = counters.get(tile_id, 0) + 1
        links = {"interrow_id": f"{spec.vineyard_id}-I{k:03d}", "row_left_id": spec.row_id(k),
                 "row_right_id": spec.row_id(k + 1)} if linked else {}
        out.append(interrow_piece_record(format_marcaj_interrow_piece_id(tile_id, counters[tile_id]), tile_id,
                                         spec.vineyard_id, poly, interrow_cover=spec.interrow_cover, **links))
    return out


def headland_passages(spec: BlockSpec, *, width_m: float = 4.0, gap_m: float = 0.0,
                      overhang_m: float = 3.0) -> Polygon:
    """U-shaped passage: a strip beyond each row end (`gap_m` from the ends) joined by a side strip."""
    k_last = spec.n_rows
    lo = min(spec.span(k)[0] for k in range(1, k_last + 1)) - gap_m
    hi = max(spec.span(k)[1] for k in range(1, k_last + 1)) + gap_m
    c_top, c_bot = overhang_m, -((k_last - 1) * spec.spacing_m) - overhang_m
    west = _rect(spec, 1, lo - width_m, lo, c_bot, c_top)
    east = _rect(spec, 1, hi, hi + width_m, c_bot, c_top)
    north = _rect(spec, 1, lo - width_m, hi + width_m, c_top, c_top + width_m)
    return orient_ccw(west.union(east).union(north))


def start_point(spec: BlockSpec, *, width_m: float = 4.0, gap_m: float = 0.0) -> Point:
    """A START in the middle of the west headland strip of `headland_passages(spec)`."""
    lo = min(spec.span(k)[0] for k in range(1, spec.n_rows + 1)) - gap_m
    return Point(spec.point(1, lo - width_m / 2.0))


# ------------------------------------------------------------------ AnnSet(reference) from the examples


def _canopy_row_id(poly: Polygon, lines: Sequence[tuple[str, LineString]]) -> str | None:
    """Nearest row piece within CANOPY_ROW_MAX_M; ties -> smaller centroid distance (design 04 §3.1)."""
    scored = [(poly.distance(line), poly.centroid.distance(line), rid) for rid, line in lines]
    near = sorted(s for s in scored if s[0] <= CANOPY_ROW_MAX_M)
    return near[0][2] if near else None


def _reference_rows(img: ExampleImage) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    out = []
    for shape in img.by_label("row"):
        attrs = shape.attributes
        rid = attrs["row_id"]
        seen[rid] = seen.get(rid, 0) + 1
        line = LineString(to_utm(img.tile_id, shape.points))
        out.append(row_piece_record(rid, attrs["vineyard_id"], img.tile_id, line, dup=seen[rid],
                                    row_structure=attrs["row_structure"]))
    return out


def _reference_polygons(img: ExampleImage, label: str) -> list[tuple[Mapping[str, str], Polygon]]:
    return [(s.attributes, orient_ccw(p)) for s in img.by_label(label)
            for p in make_valid_polygonal(Polygon(to_utm(img.tile_id, s.points)))]


def _reference_tile(img: ExampleImage) -> tuple[list, list, list]:
    rows = _reference_rows(img)
    lines = [(r["row_id"], r["geometry"]) for r in rows]
    canopies = [canopy_record(format_canopy_id(img.tile_id, k), img.tile_id, a["vineyard_id"], poly,
                              row_id=_canopy_row_id(poly, lines))
                for k, (a, poly) in enumerate(_reference_polygons(img, "vineyard"), start=1)]
    interrows = [interrow_piece_record(format_marcaj_interrow_piece_id(img.tile_id, k), img.tile_id,
                                       a["vineyard_id"], poly, interrow_cover=a["interrow_cover"])
                 for k, (a, poly) in enumerate(_reference_polygons(img, "interrow_area"), start=1)]
    return canopies, rows, interrows


@lru_cache(maxsize=2)
def _reference_cached(digest: str, examples_xml: bytes) -> AnnSet:
    images = load_examples(examples_xml)
    canopies, rows, interrows = [], [], []
    for tile_id in sorted(images):
        c, r, i = _reference_tile(images[tile_id])
        canopies, rows, interrows = canopies + c, rows + r, interrows + i
    prov = Prov(Source.REFERENCE, "20260926T0000-reference-examples", f"reference-examples@{digest[:8]}")
    return build_annset(canopies, rows, interrows, prov=prov, tile_ids=images)


def reference_annset(examples_xml: bytes) -> AnnSet:
    """AnnSet(reference) of the 2 example tiles (cached; do not mutate its frames)."""
    return _reference_cached(hashlib.sha256(examples_xml).hexdigest(), examples_xml)
