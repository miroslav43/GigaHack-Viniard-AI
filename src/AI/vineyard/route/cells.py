"""Quadtree cells of a large polygonal region: local clips, lengths and buffers.

The full walking domain is one polygon with millions of vertices and tens of thousands of holes
(every canopy). GEOS overlays of short lines against it cost O(all vertices) each, and GEOS buffer
assigns holes to shells quadratically (measured: 2.5 s for 9k canopies, 100 s for 27k). Cutting
the region into small disjoint cells keeps every operation local.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon
from shapely.geometry.base import BaseGeometry

from vineyard.geo.ops import make_valid_polygonal, split_multi

# Stop splitting a cell below this many vertices or this width.
CELL_MAX_COORDS: Final = 2048
CELL_MIN_M: Final = 16.0
# Clipped parts from neighbouring cells meet at computed points; a micrometre grid lets them merge.
CLIP_MERGE_GRID_M: Final = 1e-6
# Extra halo beyond an operation's reach, so the halo's cut edge never influences the cell.
HALO_MARGIN_M: Final = 0.5
# Neighbouring cell results overlap by this much (well below any buffer distance used).
CELL_OVERLAP_M: Final = 1e-3

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True, eq=False)
class Cell:
    bounds: Bounds
    geom: MultiPolygon


def _quadrants(bounds: Bounds) -> list[Bounds]:
    x0, y0, x1, y1 = bounds
    xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return [(x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)]


def quadtree_cells(region: BaseGeometry, max_coords: int = CELL_MAX_COORDS,
                   min_cell_m: float = CELL_MIN_M) -> tuple[Cell, ...]:
    """Disjoint polygonal pieces of `region`, each <= max_coords vertices or <= min_cell_m wide."""
    if max_coords < 1 or min_cell_m <= 0.0:
        raise ValueError(f"quadtree: max_coords ({max_coords}) and min_cell_m ({min_cell_m}) must be > 0")
    if region.is_empty:
        return ()
    x0, y0, x1, y1 = region.bounds
    side = max(x1 - x0, y1 - y0, min_cell_m)
    stack: list[tuple[BaseGeometry, Bounds]] = [(region, (x0, y0, x0 + side, y0 + side))]
    cells: list[Cell] = []
    while stack:
        geom, bounds = stack.pop()
        if shapely.get_num_coordinates(geom) <= max_coords or bounds[2] - bounds[0] <= min_cell_m:
            cells.append(Cell(bounds, MultiPolygon(make_valid_polygonal(geom))))
            continue
        for quad in reversed(_quadrants(bounds)):
            polys = make_valid_polygonal(shapely.intersection(geom, shapely.box(*quad)))
            if polys:
                stack.append((MultiPolygon(polys), quad))
    return tuple(cells)


def local_apply(region: BaseGeometry, fn: Callable[[BaseGeometry], BaseGeometry], reach_m: float,
                grid_size_m: float, max_coords: int = CELL_MAX_COORDS,
                min_cell_m: float = CELL_MIN_M) -> BaseGeometry:
    """`fn(region)` computed cell by cell. Valid for operations whose result at a point depends only
    on `region` within `reach_m` of it and never grows past the region's cells: erosions (negative
    buffers) and closings, not dilations. Small regions are processed whole."""
    if region.is_empty or shapely.get_num_coordinates(region) <= max_coords:
        return fn(region)
    cells = quadtree_cells(region, max_coords, min_cell_m)
    pieces = np.asarray([c.geom for c in cells], dtype=object)
    tree = shapely.STRtree(pieces)
    out: list[BaseGeometry] = []
    for cell in cells:
        halo = _grown_box(cell.bounds, reach_m + HALO_MARGIN_M)
        near = shapely.intersection(pieces[np.sort(tree.query(halo))], halo)
        local = shapely.union_all(near, grid_size=grid_size_m)
        # cells overlap by CELL_OVERLAP_M so the final union has no float slivers along cell borders
        out.extend(make_valid_polygonal(shapely.intersection(fn(local), _grown_box(cell.bounds, CELL_OVERLAP_M))))
    return shapely.union_all(np.asarray(out, dtype=object)) if out else MultiPolygon()


def _grown_box(bounds: Bounds, d: float) -> BaseGeometry:
    x0, y0, x1, y1 = bounds
    return shapely.box(x0 - d, y0 - d, x1 + d, y1 + d)


@dataclass(frozen=True, eq=False)
class CellIndex:
    """Region cut into small disjoint cells, so clips of short lines never overlay the whole domain."""

    cells: tuple[MultiPolygon, ...]
    tree: shapely.STRtree
    grid_size_m: float

    @classmethod
    def build(cls, region: BaseGeometry, max_coords: int = CELL_MAX_COORDS, min_cell_m: float = CELL_MIN_M,
              grid_size_m: float = CLIP_MERGE_GRID_M) -> CellIndex:
        cells = tuple(c.geom for c in quadtree_cells(region, max_coords, min_cell_m))
        prepared = np.asarray(cells, dtype=object)
        shapely.prepare(prepared)
        return cls(cells=cells, tree=shapely.STRtree(prepared), grid_size_m=grid_size_m)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def inside_length(self, lines: Sequence[BaseGeometry]) -> np.ndarray:
        """Length of each line inside the region (clamped to the line length)."""
        arr = np.asarray(list(lines), dtype=object)
        if len(arr) == 0 or not self.cells:
            return np.zeros(len(arr), dtype=np.float64)
        li, ci = self.tree.query(arr, predicate="intersects")
        cells = np.asarray(self.cells, dtype=object)
        lengths = shapely.length(shapely.intersection(arr[li], cells[ci]))
        inside = np.bincount(li, weights=lengths, minlength=len(arr))
        return np.minimum(inside, shapely.length(arr))

    def clip_line(self, line: BaseGeometry) -> list[LineString]:
        """Parts of `line` inside the region, merged across cell borders, in no particular order."""
        hits = sorted(self.tree.query(line, predicate="intersects")) if self.cells else []
        if not hits:
            return []
        parts = shapely.intersection(line, np.asarray([self.cells[i] for i in hits], dtype=object))
        pieces = [g for p in parts for g in split_multi(p) if g.geom_type == "LineString" and g.length > 0.0]
        if len(pieces) > 1:
            merged = shapely.line_merge(shapely.union_all(pieces, grid_size=self.grid_size_m))
            pieces = [g for g in split_multi(merged) if g.geom_type == "LineString" and g.length > 0.0]
        return pieces


__all__ = ["CELL_MAX_COORDS", "CELL_MIN_M", "Cell", "CellIndex", "local_apply", "quadtree_cells"]
