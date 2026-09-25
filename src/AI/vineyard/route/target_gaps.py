"""The single adapter between the targets rules and the row-gap engine (`vineyard.perception.attrs`).

The engine (owned by perception) measures gaps on canopy PIXEL sets projected on the row axis in 1-px
bins inside the +-half corridor, ends included (kinds interior / head / tail / full). It works in one
tile's pixel frame, so a global row is measured in a virtual frame anchored at the north-west grid cell
of its corridor: every tile's canopy pixel set is shifted by whole tiles (tiles are 2048 px apart
exactly). Unknown spans (outside the imaged coverage) are cut out of the engine's gaps afterwards; the
parts next to them are `censored`, like the engine does with its own nodata bins.

Targets never call the engine directly: they receive a `GapFn`, so tests inject a reference engine and
the pipeline uses `engine_gap_fn`, which imports the perception module lazily at its first call.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final, Literal, Protocol, get_args

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.errors import StageError
from vineyard.geo.tiling import (
    GRID_ORIGIN_X,
    GRID_ORIGIN_Y,
    GSD_M,
    TILE_M,
    TILE_PX,
    TileRef,
    tile_box,
    tile_ref_from_grid,
)

GapKind = Literal["interior", "head", "tail", "full"]
GAP_KINDS: Final[tuple[str, ...]] = get_args(GapKind)
ENGINE_MODULE: Final = "vineyard.perception.attrs"
ENGINE_FUNCTIONS: Final = ("axis_to_px", "canopy_pixels", "occupancy_profile", "find_gaps")
# Engines report along-axis metres; allow float noise when checking a gap lies on the axis.
ALONG_TOL_M: Final = 1e-6
_ENGINE_KIND_ATTRS: Final = ("kind", "position")


@dataclass(frozen=True)
class RowGap:
    """A gap along a row axis, in metres from the axis start."""

    start_m: float
    end_m: float
    kind: GapKind
    censored: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start_m) and math.isfinite(self.end_m)):
            raise ValueError(f"RowGap bounds must be finite, got ({self.start_m}, {self.end_m})")
        if self.end_m < self.start_m:
            raise ValueError(f"RowGap end {self.end_m} < start {self.start_m}")
        if self.kind not in GAP_KINDS:
            raise ValueError(f"RowGap kind must be one of {GAP_KINDS}, got {self.kind!r}")

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m

    @property
    def centre_m(self) -> float:
        return (self.start_m + self.end_m) / 2.0


class GapFn(Protocol):
    """Gaps (all kinds, length >= min_length_m) of `axis` given the canopies near it."""

    def __call__(
        self, axis: LineString, canopies: Sequence[Polygon], unknown: BaseGeometry | None, min_length_m: float
    ) -> Sequence[RowGap]: ...


# ------------------------------------------------------------------ interval helpers


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Sorted union of closed intervals (touching intervals merge)."""
    merged: list[list[float]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return tuple((lo, hi) for lo, hi in merged)


def along_interval(axis: LineString, part: BaseGeometry) -> tuple[float, float] | None:
    """[min, max] along-axis position of the vertices of `part` (None when empty)."""
    coords = shapely.get_coordinates(part)
    if len(coords) == 0:
        return None
    along = shapely.line_locate_point(axis, shapely.points(coords))
    return float(np.min(along)), float(np.max(along))


def unknown_intervals(axis: LineString, unknown: BaseGeometry | None) -> tuple[tuple[float, float], ...]:
    """Along-axis intervals where the axis runs through `unknown` (merged, sorted)."""
    if unknown is None or unknown.is_empty:
        return ()
    inside = axis.intersection(unknown)
    parts = [along_interval(axis, p) for p in getattr(inside, "geoms", [inside]) if not p.is_empty]
    return merge_intervals([p for p in parts if p is not None and p[1] > p[0]])


def _part_kind(lo: float, hi: float, length_m: float, parent: str) -> GapKind:
    head = lo <= ALONG_TOL_M and parent in ("head", "full")
    tail = hi >= length_m - ALONG_TOL_M and parent in ("tail", "full")
    if head and tail:
        return "full"
    return "head" if head else "tail" if tail else "interior"


def split_by_unknown(gaps: Sequence[RowGap], unknown: Sequence[tuple[float, float]],
                     length_m: float) -> tuple[RowGap, ...]:
    """Cut unknown spans out of `gaps`; a part bounded by an unknown span is censored and not an end."""
    out: list[RowGap] = []
    for gap in gaps:
        cursor, parts = gap.start_m, []
        for lo, hi in unknown:
            if hi <= gap.start_m or lo >= gap.end_m:
                continue
            if lo > cursor:
                parts.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < gap.end_m:
            parts.append((cursor, gap.end_m))
        for lo, hi in parts:
            # A part cut on one side starts/ends inside the axis, so _part_kind never makes that side an end.
            out.append(RowGap(lo, hi, _part_kind(lo, hi, length_m, gap.kind),
                              gap.censored or lo > gap.start_m or hi < gap.end_m))
    return tuple(out)


# ------------------------------------------------------------------ the perception engine


def _load_engine() -> ModuleType:
    try:
        module = importlib.import_module(ENGINE_MODULE)
    except ImportError as exc:
        raise StageError("row-gap engine is not available", module=ENGINE_MODULE, error=str(exc)) from exc
    missing = [name for name in ENGINE_FUNCTIONS if not callable(getattr(module, name, None))]
    if missing:
        raise StageError("row-gap engine lacks required functions", module=ENGINE_MODULE, missing=missing)
    return module


def _engine_kind(gap: Any) -> str:
    for attr in _ENGINE_KIND_ATTRS:
        value = getattr(gap, attr, None)
        if value is not None:
            return str(value)
    raise StageError("engine gap has no kind/position", module=ENGINE_MODULE, gap=repr(gap))


def _from_engine(gap: Any) -> RowGap:
    try:
        return RowGap(float(gap.start_m), float(gap.end_m), _engine_kind(gap),  # type: ignore[arg-type]
                      bool(getattr(gap, "censored", False)))
    except (AttributeError, ValueError, TypeError) as exc:
        raise StageError("engine returned an unusable gap", module=ENGINE_MODULE, gap=repr(gap)) from exc


@dataclass(frozen=True)
class PixelFrame:
    """Virtual pixel frame of a global row: `ref` = north-west grid cell of its corridor."""

    ref: TileRef
    tiles: tuple[TileRef, ...]

    def offset(self, tile: TileRef) -> np.ndarray:
        """(di, dj) px shift from `tile`'s own frame to the virtual one."""
        return np.array([(tile.grid_col - self.ref.grid_col) * TILE_PX,
                         (tile.grid_row - self.ref.grid_row) * TILE_PX], dtype=np.float64)


def pixel_frame(axis: LineString, half_m: float) -> PixelFrame:
    minx, miny, maxx, maxy = axis.bounds
    cols = range(math.floor((minx - half_m - GRID_ORIGIN_X) / TILE_M),
                 math.floor((maxx + half_m - GRID_ORIGIN_X) / TILE_M) + 1)
    rows = range(math.floor((GRID_ORIGIN_Y - maxy - half_m) / TILE_M),
                 math.floor((GRID_ORIGIN_Y - miny + half_m) / TILE_M) + 1)
    return PixelFrame(ref=tile_ref_from_grid(rows[0], cols[0]),
                      tiles=tuple(tile_ref_from_grid(r, c) for r in rows for c in cols))


def _no_pixels() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float64)


def canopy_ij(engine: ModuleType, canopies: Sequence[Polygon], frame: PixelFrame) -> np.ndarray:
    """Canopy pixel sets of every tile of the frame, shifted into the virtual frame."""
    parts = []
    for tile in frame.tiles:
        square = tile_box(tile)
        # A canopy only touching the tile edge would add a 1-px sliver of the neighbour's frame.
        mine = [c for c in canopies if c.intersects(square) and not c.touches(square)]
        if mine:
            parts.append(np.asarray(engine.canopy_pixels(mine, tile), dtype=np.float64) + frame.offset(tile))
    return np.vstack(parts) if parts else _no_pixels()


def engine_gap_fn(row_structure_cfg: Any, half_m: float) -> GapFn:
    """GapFn backed by `vineyard.perception.attrs` (pixel-set profile, ends included, 1-px bins)."""
    half_px = half_m / GSD_M

    def gap_fn(
        axis: LineString, canopies: Sequence[Polygon], unknown: BaseGeometry | None, min_length_m: float
    ) -> tuple[RowGap, ...]:
        engine = _load_engine()
        frame = pixel_frame(axis, half_m)
        axis_px = np.asarray(engine.axis_to_px(axis, frame.ref), dtype=np.float64)
        profile = engine.occupancy_profile(axis_px, canopy_ij(engine, canopies, frame), half_px=half_px)
        if getattr(row_structure_cfg, "occ_smoothing_enabled", False):
            profile = engine.smooth_profile(profile, row_structure_cfg)
        raw = tuple(_from_engine(g) for g in engine.find_gaps(profile, include_ends=True, min_length_m=0.0))
        gaps = split_by_unknown(raw, unknown_intervals(axis, unknown), axis.length)
        return tuple(g for g in gaps if g.length_m >= min_length_m)

    return gap_fn


# ------------------------------------------------------------------ validated access


def _checked(gap: object, length_m: float, row_id: str) -> RowGap:
    if not isinstance(gap, RowGap):
        raise StageError("gap function must return RowGap values", row_id=row_id, got=type(gap).__name__)
    if gap.start_m < -ALONG_TOL_M or gap.end_m > length_m + ALONG_TOL_M:
        raise StageError("gap lies outside its row axis", row_id=row_id, start_m=gap.start_m, end_m=gap.end_m,
                         axis_length_m=length_m)
    return gap


def row_gaps(
    gap_fn: GapFn,
    axis: LineString,
    canopies: Sequence[Polygon],
    unknown: BaseGeometry | None,
    *,
    min_length_m: float = 0.0,
    row_id: str = "",
) -> tuple[RowGap, ...]:
    """Validated gaps of one row, sorted by (start_m, end_m)."""
    raw = gap_fn(axis, canopies, unknown, min_length_m)
    gaps = [_checked(g, axis.length, row_id) for g in raw]
    return tuple(sorted(gaps, key=lambda g: (g.start_m, g.end_m, g.kind)))
