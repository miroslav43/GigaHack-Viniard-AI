"""Farms: groups of vineyard blocks that touch or lie at most `gap_max_m` apart (a headland or a field track
between them), unless the gap between them meets a public road. F01 = northernmost farm (like V01).

Ownership is unknown from the imagery: a farm is a spatial grouping, not a cadastral unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import shapely
from shapely.geometry import LineString, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from vineyard.contracts.ordering import block_sort_key
from vineyard.errors import SchemaError
from vineyard.geo.ops import make_valid_polygonal, orient_ccw

if TYPE_CHECKING:
    from vineyard.config.sections_post import FarmsConfig

FARM_PREFIX: Final = "F"


@dataclass(frozen=True)
class FarmParams:
    gap_max_m: float
    touch_m: float
    link_width_m: float
    public_road_buffer_m: float
    outline_simplify_m: float

    @classmethod
    def from_config(cls, cfg: FarmsConfig) -> FarmParams:
        return cls(gap_max_m=cfg.gap_max_m, touch_m=cfg.touch_m, link_width_m=cfg.link_width_m,
                   public_road_buffer_m=cfg.public_road_buffer_m, outline_simplify_m=cfg.outline_simplify_m)


@dataclass(frozen=True)
class Farm:
    farm_id: str
    vineyard_ids: tuple[str, ...]
    outline: BaseGeometry  # Polygon | MultiPolygon, EPSG:32635

    @property
    def area_m2(self) -> float:
        return float(self.outline.area)


def farm_id(index: int, n_farms: int) -> str:
    """F01…F99, F001… once there are more than 99 farms (index is 0-based)."""
    return f"{FARM_PREFIX}{index + 1:0{max(2, len(str(n_farms)))}d}"


def _linked(a: BaseGeometry, b: BaseGeometry, barrier: BaseGeometry | None, params: FarmParams) -> bool:
    gap = a.distance(b)
    if gap > params.gap_max_m:
        return False
    if gap <= params.touch_m or barrier is None:
        return True
    p, q = nearest_points(a, b)
    return not LineString([p, q]).buffer(params.link_width_m).intersects(barrier)


def block_links(geoms: Sequence[BaseGeometry], barrier: BaseGeometry | None,
                params: FarmParams) -> tuple[tuple[int, int], ...]:
    """Pairs (i < j) of blocks of the same farm; candidates from an STRtree within gap_max_m."""
    tree = shapely.STRtree(list(geoms))
    left, right = tree.query(list(geoms), predicate="dwithin", distance=params.gap_max_m)
    pairs = sorted({(int(i), int(j)) for i, j in zip(left, right, strict=True) if i < j})
    return tuple((i, j) for i, j in pairs if _linked(geoms[i], geoms[j], barrier, params))


def components(n: int, links: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    """Connected components of 0..n-1 (union-find), each sorted, in order of their smallest member."""
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in links:
        parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return tuple(sorted(tuple(g) for g in groups.values()))


def farm_outline(geoms: Sequence[BaseGeometry], barrier: BaseGeometry | None, params: FarmParams) -> BaseGeometry:
    """Closing (gap_max_m / 2) of the blocks, minus the public roads, simplified; CCW, largest part first."""
    half = params.gap_max_m / 2.0
    closed = shapely.union_all(shapely.buffer(list(geoms), half)).buffer(-half)
    kept = closed.difference(barrier) if barrier is not None else closed
    simple = kept.simplify(params.outline_simplify_m) if params.outline_simplify_m > 0 else kept
    parts = sorted((orient_ccw(p) for p in make_valid_polygonal(simple)), key=lambda p: -p.area)
    if not parts:  # the roads ate the whole outline: fall back to the blocks themselves
        parts = sorted((orient_ccw(p) for p in make_valid_polygonal(shapely.union_all(list(geoms)))),
                       key=lambda p: -p.area)
    if not parts:
        raise SchemaError("farm has no polygonal outline", n_blocks=len(geoms))
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _natural(vid: str) -> tuple[str, int]:
    digits = "".join(ch for ch in vid if ch.isdigit())
    return (vid.rstrip("0123456789"), int(digits) if digits else -1)


def group_blocks(vineyard_ids: Sequence[str], geoms: Sequence[BaseGeometry], public_roads: BaseGeometry | None,
                 params: FarmParams) -> tuple[Farm, ...]:
    """Farms of the given blocks (EPSG:32635), numbered F01… northernmost first, then westernmost."""
    if len(vineyard_ids) != len(geoms):
        raise SchemaError("vineyard_ids and geometries differ in length", n_ids=len(vineyard_ids), n_geoms=len(geoms))
    if not geoms:
        return ()
    barrier = None
    if public_roads is not None and not public_roads.is_empty:
        barrier = public_roads.buffer(params.public_road_buffer_m) if params.public_road_buffer_m > 0 else public_roads
    groups = components(len(geoms), block_links(geoms, barrier, params))
    drafts = [(farm_outline([geoms[i] for i in g], barrier, params),
               tuple(sorted((str(vineyard_ids[i]) for i in g), key=_natural))) for g in groups]
    drafts.sort(key=lambda d: block_sort_key(*_anchor(d[0])))
    return tuple(Farm(farm_id(k, len(drafts)), ids, outline) for k, (outline, ids) in enumerate(drafts))


def _anchor(outline: BaseGeometry) -> tuple[float, float]:
    point = (outline.geoms[0] if isinstance(outline, MultiPolygon) else outline).representative_point()
    return float(point.x), float(point.y)


def farm_of_block(farms: Sequence[Farm]) -> dict[str, str]:
    """vineyard_id -> farm_id."""
    return {vid: farm.farm_id for farm in farms for vid in farm.vineyard_ids}

