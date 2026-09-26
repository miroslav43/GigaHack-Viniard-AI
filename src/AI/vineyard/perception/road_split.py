"""Rows cut where a road or track crosses them (annotation rules §6: "a road or a track always separates blocks").

For every road line (OSM snapshot) that crosses at least `min_rows` rows of one component, each crossing row is
searched for a vine-free window within `search_m` of the crossing (veg evidence of the row band; the OSM line may
be a few metres off the real track). When at least `min_gap_frac` of the crossing rows show one, the road cuts
every row that has it: the row loses the vine-free stretch (grown while the evidence stays low), and the hull of
those stretches becomes a barrier no row-graph edge may cross, so the two sides become two blocks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import shapely
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

if TYPE_CHECKING:
    from vineyard.config.sections_perception import RoadSplitConfig

Evidence = Callable[[LineString, float], "float | None"]
GROW_STEP_M: Final = 0.5
CUT_HALF_WIDTH_M: Final = 0.3  # the row is cut by its vine-free stretch buffered by this (a thin polygon)


@dataclass(frozen=True)
class RoadSplitOptions:
    enabled: bool
    min_rows: int
    min_gap_frac: float
    search_m: float
    window_half_m: float
    gap_share_max: float
    gap_rel_max: float
    max_cut_m: float
    min_line_m: float
    min_side_m: float

    @classmethod
    def from_config(cls, cfg: RoadSplitConfig) -> RoadSplitOptions:
        return cls(enabled=cfg.enabled, min_rows=cfg.min_rows, min_gap_frac=cfg.min_gap_frac, search_m=cfg.search_m,
                   window_half_m=cfg.window_half_m, gap_share_max=cfg.gap_share_max, gap_rel_max=cfg.gap_rel_max,
                   max_cut_m=cfg.max_cut_m, min_line_m=cfg.min_line_m, min_side_m=cfg.min_side_m)


@dataclass(frozen=True)
class RoadCut:
    """One road across one component: the rows it cuts, their cut polygons and the edge barrier."""

    road_index: int
    rows: tuple[int, ...]
    polygon: BaseGeometry  # union of the per-row cut polygons
    barrier: BaseGeometry  # hull of the cuts: no row-graph edge may cross it
    n_crossing: int


def _window(line: LineString, centre: float, half: float) -> LineString | None:
    a, b = max(0.0, centre - half), min(line.length, centre + half)
    return substring(line, a, b) if b - a > 1e-6 else None


def _low(share: float | None, base: float | None, o: RoadSplitOptions) -> bool:
    if share is None:
        return False
    return share <= o.gap_share_max or (base is not None and share <= o.gap_rel_max * base)


def _share(evidence: Evidence, line: LineString, centre: float, half: float) -> float | None:
    win = _window(line, centre, half)
    return None if win is None else evidence(win, 0.0)


def gap_interval(line: LineString, along: float, evidence: Evidence, o: RoadSplitOptions) -> tuple[float, float] | None:
    """The vine-free stretch [a, b] (along the row) nearest the crossing at `along`, or None."""
    base = evidence(line, 0.0)
    steps = int(o.search_m / GROW_STEP_M)
    best: tuple[float, float] | None = None
    for k in sorted(range(-steps, steps + 1), key=abs):
        centre = along + k * GROW_STEP_M
        share = _share(evidence, line, centre, o.window_half_m)
        if _low(share, base, o) and (best is None or share < best[1]):  # type: ignore[operator]
            best = (centre, share)  # type: ignore[assignment]
    if best is None:
        return None
    a = b = best[0]
    while b - a < o.max_cut_m and a > 0 and _low(_share(evidence, line, a - GROW_STEP_M, GROW_STEP_M), base, o):
        a -= GROW_STEP_M
    while b - a < o.max_cut_m and b < line.length and _low(_share(evidence, line, b + GROW_STEP_M, GROW_STEP_M),
                                                           base, o):
        b += GROW_STEP_M
    return max(0.0, a - o.window_half_m), min(line.length, b + o.window_half_m)


def _crossing_along(line: LineString, road: LineString) -> float | None:
    if not line.crosses(road):
        return None
    hit = line.intersection(road)
    point = hit if isinstance(hit, Point) else hit.representative_point() if not hit.is_empty else None
    return None if point is None else line.project(point)


def road_cuts(lines: Sequence[LineString], members: Sequence[int], roads: Sequence[LineString], evidence: Evidence,
              o: RoadSplitOptions) -> tuple[RoadCut, ...]:
    """The accepted road cuts of one component (rows = indices into `lines`)."""
    cuts: list[RoadCut] = []
    for r, road in enumerate(roads):
        crossing = [(i, a) for i in members if (a := _crossing_along(lines[i], road)) is not None]
        if len(crossing) < o.min_rows:
            continue
        found = [(i, iv) for i, a in crossing if (iv := gap_interval(lines[i], a, evidence, o)) is not None]
        if len(found) < o.min_gap_frac * len(crossing):
            continue
        polys = [substring(lines[i], a, b).buffer(CUT_HALF_WIDTH_M) for i, (a, b) in found]
        union = shapely.union_all(polys)
        cuts.append(RoadCut(r, tuple(i for i, _ in found), union, union.convex_hull.buffer(0.2), len(crossing)))
    return tuple(cuts)


def road_lines(geoms: Sequence[BaseGeometry], min_line_m: float) -> tuple[LineString, ...]:
    """Every LineString part of the road geometries, at least `min_line_m` long, in a stable order."""
    parts = [p for g in geoms for p in getattr(g, "geoms", (g,)) if isinstance(p, LineString) and p.length >= min_line_m]
    return tuple(sorted(parts, key=lambda p: (round(p.bounds[0], 2), round(p.bounds[1], 2), round(p.length, 2))))
