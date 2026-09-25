"""Synthetic and real inputs for the passable tests (domain, skeleton, centerlines, connectors, graph)."""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from vineyard.geo.ops import make_valid_polygonal
from vineyard.geo.tiling import CRS_EPSG
from vineyard.geo.vector_io import read_geojson

ROUTE_FILES = ("passages", "forbidden", "start", "study_area")
START_XY = (629504.70, 5220250.75)
CORRIDOR_HALF_M = 0.3


def data_root() -> Path:
    default = Path(__file__).resolve().parents[2] / ".." / ".." / "data & info"
    return Path(os.environ.get("VINEYARD_DATA_ROOT", str(default))).resolve()


@dataclass(frozen=True)
class RouteInputs:
    passages: MultiPolygon
    forbidden: MultiPolygon
    start_xy: tuple[float, float]


@functools.cache
def _load_route_inputs() -> RouteInputs | None:
    route = data_root() / "02_route"
    if not all((route / f"{name}.geojson").is_file() for name in ROUTE_FILES):
        return None
    passages = read_geojson(route / "passages.geojson").geometry.union_all()
    forbidden = read_geojson(route / "forbidden.geojson").geometry.union_all()
    start = read_geojson(route / "start.geojson").geometry.iloc[0]
    return RouteInputs(MultiPolygon(make_valid_polygonal(passages)), MultiPolygon(make_valid_polygonal(forbidden)),
                       (start.x, start.y))


def route_inputs() -> RouteInputs:
    loaded = _load_route_inputs()
    if loaded is None:
        pytest.skip(f"02_route inputs missing under {data_root()}")
    return loaded


# ------------------------------------------------------------------ synthetic mini-vineyard


@dataclass(frozen=True)
class MiniVineyard:
    rows: gpd.GeoDataFrame            # row_id, vineyard_id, row_index, geometry
    interrow_pieces: gpd.GeoDataFrame  # piece_id, interrow_id, vineyard_id, geometry
    passages: MultiPolygon
    start_xy: tuple[float, float]


def _rows_frame(lines: list[LineString], vid: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"row_id": [f"{vid}-R{k:03d}" for k in range(1, len(lines) + 1)], "vineyard_id": [vid] * len(lines),
         "row_index": np.arange(1, len(lines) + 1, dtype=np.int16)},
        geometry=lines, crs=CRS_EPSG,
    )


def interrow_strip(a: LineString, b: LineString, half: float = CORRIDOR_HALF_M) -> Polygon:
    """Quadrilateral between two parallel horizontal row axes, `half` metres off each axis."""
    (ax0, ay), (ax1, _) = a.coords[0], a.coords[-1]
    (bx0, by), (bx1, _) = b.coords[0], b.coords[-1]
    lo, hi = sorted((ay, by))
    return box(max(ax0, bx0), lo + half, min(ax1, bx1), hi - half)


def mini_vineyard(n_rows: int = 4, length_m: float = 60.0, spacing_m: float = 2.6, headland_m: float = 4.0,
                  end_gap_m: float = 0.0, x0: float = 1000.0, y0: float = 2000.0) -> MiniVineyard:
    """Horizontal rows (row 1 on top, like n.c ordering), passages of width `headland_m` at both ends.

    `end_gap_m` leaves a strip of that width between the interrow ends and the passages (headland).
    """
    ys = [y0 + (n_rows - k) * spacing_m for k in range(1, n_rows + 1)]
    lines = [LineString([(x0, y), (x0 + length_m, y)]) for y in ys]
    rows = _rows_frame(lines, "V01")
    strips = [interrow_strip(a, b) for a, b in zip(lines[:-1], lines[1:], strict=True)]
    pieces = gpd.GeoDataFrame(
        {"piece_id": [f"siret3_r010_c010:I{k:03d}" for k in range(1, len(strips) + 1)],
         "interrow_id": [f"V01-I{k:03d}" for k in range(1, len(strips) + 1)],
         "vineyard_id": ["V01"] * len(strips)},
        geometry=strips, crs=CRS_EPSG,
    )
    y_lo, y_hi = min(ys) - spacing_m, max(ys) + spacing_m
    west = box(x0 - end_gap_m - headland_m, y_lo, x0 - end_gap_m, y_hi)
    east = box(x0 + length_m + end_gap_m, y_lo, x0 + length_m + end_gap_m + headland_m, y_hi)
    start = (x0 - end_gap_m - headland_m / 2.0, y_lo + 1.0)
    return MiniVineyard(rows, pieces, MultiPolygon([west, east]), start)


# ------------------------------------------------------------------ reference examples (AnnSet stand-in)


def _row_normal(lines: list[LineString]) -> np.ndarray:
    angles = [math.atan2(ln.coords[-1][1] - ln.coords[0][1], ln.coords[-1][0] - ln.coords[0][0]) % math.pi
              for ln in lines]
    theta = float(np.median(angles))
    normal = np.array([-math.sin(theta), math.cos(theta)])
    return normal if normal[1] > 0 else -normal


def ordered_rows(lines: list[LineString], vid: str) -> gpd.GeoDataFrame:
    """Contract §1.5 ordering: row_index 1 = max n.c (geometric, not the id suffix)."""
    normal = _row_normal(lines)
    keys = [float(np.dot(normal, np.asarray(ln.centroid.coords[0]))) for ln in lines]
    order = sorted(range(len(lines)), key=lambda i: -keys[i])
    return _rows_frame([lines[i] for i in order], vid)


def example_reference(examples_xml: bytes) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """(rows, interrow_pieces, canopies) of the 2 organizer example tiles, one block per tile."""
    from tests.helpers.examples import (
        canopy_polygons_utm,
        interrow_polygons_utm,
        load_examples,
        row_lines_utm,
    )

    images = load_examples(examples_xml)
    rows, pieces, canopies = [], [], []
    for img in images.values():
        vid = img.by_label("row")[0].attributes["vineyard_id"]
        rows.append(ordered_rows(row_lines_utm(img), vid))
        polys = interrow_polygons_utm(img)
        pieces.append(gpd.GeoDataFrame(
            {"piece_id": [f"{img.tile_id}:I{k:03d}" for k in range(1, len(polys) + 1)],
             "vineyard_id": [vid] * len(polys), "tile_id": [img.tile_id] * len(polys)},
            geometry=polys, crs=CRS_EPSG))
        canopy_polys = canopy_polygons_utm(img)
        canopies.append(gpd.GeoDataFrame({"vineyard_id": [vid] * len(canopy_polys)}, geometry=canopy_polys,
                                          crs=CRS_EPSG))
    concat = functools.partial(gpd.pd.concat, ignore_index=True)
    return (gpd.GeoDataFrame(concat(rows), crs=CRS_EPSG), gpd.GeoDataFrame(concat(pieces), crs=CRS_EPSG),
            gpd.GeoDataFrame(concat(canopies), crs=CRS_EPSG))


def corridor(width_m: float, pts: list[tuple[float, float]]) -> Polygon:
    """Flat-capped corridor of `width_m` around a polyline (mitre joins keep the corners square)."""
    return LineString(pts).buffer(width_m / 2.0, cap_style="flat", join_style="mitre")


def point(xy: tuple[float, float]) -> Point:
    return Point(xy)


__all__ = [
    "CORRIDOR_HALF_M", "START_XY", "MiniVineyard", "RouteInputs", "corridor", "data_root", "example_reference",
    "interrow_strip", "mini_vineyard", "ordered_rows", "point", "route_inputs", "shapely",
]
