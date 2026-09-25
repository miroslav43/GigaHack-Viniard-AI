"""Passage skeleton (arch §4.11.2, design 04 §3.7).

region.buffer(-erode) -> per-part raster at `res_m` (pixel centres) -> skimage skeletonize ->
sknw graph (multi=True: two branches between the same junctions around a hole must both survive)
-> UTM polylines -> iterative spur pruning -> line_merge at degree-2 nodes -> guarded simplify.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import shapely
import sknw
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry
from skimage.morphology import skeletonize

from vineyard.geo.ops import make_valid_polygonal

if TYPE_CHECKING:
    from vineyard.config.sections_post import RouteGraphConfig

DEFAULT_MAX_PRUNE_PASSES: Final = 3   # design 04 §3.7: "repeated until stable (at most 3 passes)"
PAD_PX: Final = 1
MIN_BRANCH_M: Final = 1e-9
MIN_RING_COORDS: Final = 4
_PIXEL_CENTRE: Final = 0.5


@dataclass(frozen=True)
class SkeletonParams:
    res_m: float
    erode_m: float
    spur_min_m: float
    simplify_tol_m: float
    max_prune_passes: int = DEFAULT_MAX_PRUNE_PASSES

    def __post_init__(self) -> None:
        if self.res_m <= 0.0:
            raise ValueError(f"res_m must be > 0, got {self.res_m}")
        if min(self.erode_m, self.spur_min_m, self.simplify_tol_m) < 0.0:
            raise ValueError("erode_m, spur_min_m and simplify_tol_m must be >= 0")
        if self.max_prune_passes < 1:
            raise ValueError(f"max_prune_passes must be >= 1, got {self.max_prune_passes}")

    @classmethod
    def from_config(cls, cfg: RouteGraphConfig) -> SkeletonParams:
        return cls(res_m=cfg.skeleton_res_m, erode_m=cfg.skeleton_erode_m, spur_min_m=cfg.spur_min_m,
                   simplify_tol_m=cfg.simplify_tol_m)


@dataclass(frozen=True, eq=False)
class Branch:
    """Skeleton branch between sknw nodes u and v; `xy` runs u -> v in UTM."""

    u: int
    v: int
    xy: np.ndarray

    @property
    def length(self) -> float:
        return float(np.hypot(*np.diff(self.xy, axis=0).T).sum()) if len(self.xy) > 1 else 0.0


def rasterize(poly: BaseGeometry, res: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Boolean mask (pixel centre inside `poly`) with a 1-px empty border; returns (mask, (minx, maxy))."""
    bx0, by0, bx1, by1 = poly.bounds
    minx = math.floor(bx0 / res) * res - PAD_PX * res
    maxy = math.ceil(by1 / res) * res + PAD_PX * res
    width = math.ceil((bx1 - minx) / res) + PAD_PX
    height = math.ceil((maxy - by0) / res) + PAD_PX
    transform = Affine(res, 0.0, minx, 0.0, -res, maxy)
    mask = features.rasterize([(poly, 1)], out_shape=(height, width), transform=transform, fill=0,
                              dtype=np.uint8, all_touched=False)
    return mask.astype(bool), (minx, maxy)


def _to_utm(rc: np.ndarray, origin: tuple[float, float], res: float) -> np.ndarray:
    minx, maxy = origin
    rc = np.asarray(rc, dtype=np.float64)
    return np.column_stack([minx + (rc[:, 1] + _PIXEL_CENTRE) * res, maxy - (rc[:, 0] + _PIXEL_CENTRE) * res])


def _branches(mask: np.ndarray, origin: tuple[float, float], res: float) -> tuple[Branch, ...]:
    ske = skeletonize(mask)
    if ske.sum() < 2:
        return ()
    graph = sknw.build_sknw(ske.astype(np.uint16), multi=True, iso=False, ring=True, full=True)
    out: list[Branch] = []
    for s, e, key in sorted(graph.edges(keys=True)):
        pts = np.asarray(graph[s][e][key]["pts"])
        first = np.asarray(graph.nodes[s]["o"])
        u, v = (s, e) if np.array_equal(pts[0], first) else (e, s)
        branch = Branch(int(u), int(v), _to_utm(pts, origin, res))
        if branch.length > MIN_BRANCH_M:
            out.append(branch)
    return tuple(out)


def _droppable(branches: Sequence[Branch], deg: Counter[int], spur_min: float) -> set[int]:
    drop: set[int] = set()
    for i, b in enumerate(branches):
        if b.u == b.v or b.length >= spur_min:
            continue
        if (deg[b.u] == 1 and deg[b.v] >= 3) or (deg[b.v] == 1 and deg[b.u] >= 3):
            drop.add(i)
    return drop


def _keep_one_per_junction(branches: Sequence[Branch], drop: set[int], deg: Counter[int]) -> set[int]:
    """Never prune every branch of a junction (a tiny star region would vanish): keep its longest."""
    incident: dict[int, list[int]] = {}
    for i, b in enumerate(branches):
        for node in {b.u, b.v}:
            incident.setdefault(node, []).append(i)
    rescued: set[int] = set()
    for node, idxs in incident.items():
        if deg[node] >= 3 and all(i in drop for i in idxs):
            rescued.add(max(idxs, key=lambda i: (branches[i].length, -i)))
    return drop - rescued


def prune_spurs(branches: Sequence[Branch], spur_min: float, max_passes: int) -> tuple[Branch, ...]:
    """Drop terminal branches shorter than `spur_min` hanging off a junction; repeat up to `max_passes`."""
    current = tuple(branches)
    for _ in range(max_passes):
        deg: Counter[int] = Counter()
        for b in current:
            deg[b.u] += 1
            deg[b.v] += 1
        drop = _keep_one_per_junction(current, _droppable(current, deg, spur_min), deg)
        if not drop:
            break
        current = tuple(b for i, b in enumerate(current) if i not in drop)
    return current


def _merged_lines(branches: Sequence[Branch]) -> list[LineString]:
    if not branches:
        return []
    merged = shapely.line_merge(MultiLineString([b.xy for b in branches]))
    return [g for g in shapely.get_parts(merged) if g.geom_type == "LineString" and g.length > MIN_BRANCH_M]


def _simplified(lines: list[LineString], tol: float, guard: BaseGeometry) -> list[LineString]:
    if tol <= 0.0 or not lines:
        return lines
    arr = np.asarray(lines, dtype=object)
    simple = shapely.simplify(arr, tol, preserve_topology=False)
    ok = shapely.covered_by(simple, guard) & (shapely.length(simple) > MIN_BRANCH_M)
    rings = shapely.is_closed(arr)
    ok &= ~rings | (shapely.get_num_coordinates(simple) >= MIN_RING_COORDS)
    return [s if keep else o for s, o, keep in zip(simple, arr, ok, strict=True)]


def _sort_key(line: LineString) -> tuple[float, float, float]:
    x, y = line.coords[0]
    return (round(x, 3), round(y, 3), round(line.length, 3))


def skeleton_lines(region: BaseGeometry, params: SkeletonParams,
                   guard: BaseGeometry | None = None) -> tuple[LineString, ...]:
    """Skeleton polylines of `region` (junction-to-junction/end), in UTM, deterministic order.

    `guard`: simplification is kept only where the simplified line stays covered by it
    (default: the eroded region itself).
    """
    if region.is_empty:
        return ()
    eroded = region.buffer(-params.erode_m) if params.erode_m > 0.0 else region
    parts: list[Polygon] = make_valid_polygonal(eroded) if not eroded.is_empty else []
    lines: list[LineString] = []
    for part in sorted(parts, key=lambda p: (round(p.bounds[0], 3), round(p.bounds[1], 3))):
        mask, origin = rasterize(part, params.res_m)
        branches = prune_spurs(_branches(mask, origin, params.res_m), params.spur_min_m,
                               params.max_prune_passes)
        lines.extend(_merged_lines(branches))
    fence = eroded if guard is None else guard
    if not fence.is_empty:
        shapely.prepare(fence)
    return tuple(sorted(_simplified(lines, params.simplify_tol_m, fence), key=_sort_key))


__all__ = ["Branch", "SkeletonParams", "prune_spurs", "rasterize", "skeleton_lines"]
