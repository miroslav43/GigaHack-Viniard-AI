"""Walking domain (arch §4.11.1, design 04 §3.6, contract §2.5.12).

raw    = make_valid(close_seams(union(interrow pieces, passages)) - forbidden - canopies)
inner  = raw.buffer(-inner_buffer_m)   validation domain ("domain_in")
eroded = raw.buffer(-eroded_buffer_m)  graph and string-pulling domain
Both buffers are `shapely.prepare`d. Everything is in EPSG:32635 metres.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.geo.ops import make_valid_polygonal, split_multi
from vineyard.route.cells import CellIndex, local_apply

if TYPE_CHECKING:
    from vineyard.config.sections_post import RouteDomainConfig

PART_ID_FMT: Final = "PP{:05d}"
PART_KIND_INTERROW: Final = "interrow"
PART_KIND_PASSAGE: Final = "passage"
# Seam closing restores square corners only with mitre joins; the limit must exceed 1/sin(45 deg).
MITRE_LIMIT: Final = 5.0


@dataclass(frozen=True)
class DomainParams:
    grid_size_m: float
    inner_buffer_m: float
    eroded_buffer_m: float
    seam_close_m: float
    subtract_canopies: bool

    def __post_init__(self) -> None:
        if self.grid_size_m <= 0.0:
            raise ValueError(f"grid_size_m must be > 0, got {self.grid_size_m}")
        if min(self.inner_buffer_m, self.eroded_buffer_m, self.seam_close_m) < 0.0:
            raise ValueError("domain buffers must be >= 0")
        if self.eroded_buffer_m < self.inner_buffer_m:
            raise ValueError(f"eroded_buffer_m ({self.eroded_buffer_m}) must be >= inner_buffer_m "
                             f"({self.inner_buffer_m})")

    @classmethod
    def from_config(cls, cfg: RouteDomainConfig) -> DomainParams:
        return cls(grid_size_m=cfg.grid_size_m, inner_buffer_m=cfg.inner_buffer_m,
                   eroded_buffer_m=cfg.eroded_buffer_m, seam_close_m=cfg.seam_close_m,
                   subtract_canopies=cfg.subtract_canopies)


@dataclass(frozen=True, eq=False)
class DomainSet:
    """raw / inner / eroded walking domain; inner and eroded are prepared for fast predicates."""

    raw: MultiPolygon
    inner: MultiPolygon
    eroded: MultiPolygon
    grid_size_m: float
    inner_buffer_m: float
    eroded_buffer_m: float

    @property
    def n_components(self) -> int:
        return len(self.raw.geoms)

    def components(self) -> tuple[Polygon, ...]:
        return _sorted_parts(list(self.raw.geoms))


@dataclass(frozen=True, eq=False)
class PassablePart:
    """One row of the `passable_parts` layer."""

    part_id: str
    kind: str
    ref_id: str
    area_m2: float
    erosion_m: float
    geom: Polygon | MultiPolygon


def _sorted_parts(parts: Sequence[Polygon]) -> tuple[Polygon, ...]:
    def key(p: Polygon) -> tuple[float, float, float]:
        minx, miny, _, _ = p.bounds
        return (-round(p.area, 6), round(minx, 6), round(miny, 6))

    return tuple(sorted(parts, key=key))


def _as_multipolygon(geom: BaseGeometry, what: str) -> MultiPolygon:
    parts = [p for p in split_multi(geom) if p.geom_type == "Polygon" and p.area > 0.0]
    stray = [p.geom_type for p in split_multi(geom) if p.geom_type != "Polygon"]
    if stray:
        raise ValueError(f"{what}: expected polygonal geometry, got parts {sorted(set(stray))}")
    return MultiPolygon(list(_sorted_parts(parts)))


def _polygons(geoms: Sequence[BaseGeometry | None]) -> list[Polygon]:
    out: list[Polygon] = []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        for part in split_multi(geom):
            if part.geom_type in ("Polygon", "MultiPolygon"):
                out.extend(make_valid_polygonal(part))
    return out


def _union(geoms: Sequence[BaseGeometry | None], grid: float) -> BaseGeometry:
    polys = _polygons(geoms)
    if not polys:
        return Polygon()
    return shapely.union_all(np.asarray(polys, dtype=object), grid_size=grid)


def _closing(seam_m: float) -> Callable[[BaseGeometry], BaseGeometry]:
    def run(geom: BaseGeometry) -> BaseGeometry:
        grown = shapely.buffer(geom, seam_m, join_style="mitre", mitre_limit=MITRE_LIMIT)
        return shapely.buffer(grown, -seam_m, join_style="mitre", mitre_limit=MITRE_LIMIT)

    return run


def close_seams(geom: BaseGeometry, seam_m: float, grid: float) -> BaseGeometry:
    """Morphological closing: fills gaps narrower than 2*seam_m between tile-edge pieces (cell-local)."""
    if seam_m <= 0.0 or geom.is_empty:
        return geom
    # a mitre join reaches up to MITRE_LIMIT * seam_m out; the closing then looks seam_m further
    closed = local_apply(geom, _closing(seam_m), seam_m * (MITRE_LIMIT + 1.0), grid)
    return shapely.set_precision(closed, grid)


def _subtract(geom: BaseGeometry, cutter: BaseGeometry, grid: float) -> BaseGeometry:
    if geom.is_empty or cutter.is_empty:
        return geom
    return shapely.difference(geom, cutter, grid_size=grid)


def build_domain(
    interrows: Sequence[BaseGeometry],
    passages: BaseGeometry,
    forbidden: BaseGeometry | None,
    canopies: Sequence[BaseGeometry],
    params: DomainParams,
) -> DomainSet:
    """Domain from AnnSet interrow pieces, `in_passages`, `in_forbidden` and AnnSet canopies."""
    grid = params.grid_size_m
    merged = _union([*interrows, passages], grid)
    raw = close_seams(merged, params.seam_close_m, grid)
    if forbidden is not None:
        raw = _subtract(raw, _union([forbidden], grid), grid)
    if params.subtract_canopies and len(canopies):
        raw = _subtract(raw, _union(canopies, grid), grid)
    return domain_from_raw(MultiPolygon(make_valid_polygonal(raw)) if not raw.is_empty else raw, params)


def _eroded(raw: MultiPolygon, dist: float, grid: float) -> MultiPolygon:
    if raw.is_empty:
        return MultiPolygon()
    shrunk = raw if dist == 0.0 else local_apply(raw, lambda g: shapely.buffer(g, -dist), dist, grid)
    out = MultiPolygon(make_valid_polygonal(shrunk)) if not shrunk.is_empty else MultiPolygon()
    shapely.prepare(out)
    return out


def domain_from_raw(raw: BaseGeometry, params: DomainParams) -> DomainSet:
    """DomainSet from an already built raw domain (e.g. a `passable_domain` layer read back)."""
    multi = _as_multipolygon(raw, "domain_from_raw") if not raw.is_empty else MultiPolygon()
    grid = params.grid_size_m
    return DomainSet(raw=multi, inner=_eroded(multi, params.inner_buffer_m, grid),
                     eroded=_eroded(multi, params.eroded_buffer_m, grid), grid_size_m=grid,
                     inner_buffer_m=params.inner_buffer_m, eroded_buffer_m=params.eroded_buffer_m)


def _clipped(geom: BaseGeometry, index: CellIndex, grid: float) -> BaseGeometry:
    hits = sorted(index.tree.query(geom, predicate="intersects"))
    if not hits:
        return Polygon()
    # clip per cell, then merge only the (small) clipped parts: never union the domain cells themselves
    parts = shapely.intersection(geom, np.asarray([index.cells[i] for i in hits], dtype=object), grid_size=grid)
    return shapely.union_all(parts, grid_size=grid)


def passable_parts(
    dom: DomainSet,
    interrow_geoms: Sequence[BaseGeometry],
    interrow_ids: Sequence[str],
    passages: BaseGeometry,
) -> tuple[PassablePart, ...]:
    """Per-source pieces of the domain: every interrow piece, then every passage polygon, clipped to raw."""
    if len(interrow_geoms) != len(interrow_ids):
        raise ValueError(f"passable_parts: {len(interrow_geoms)} interrow geometries vs {len(interrow_ids)} ids")
    index = CellIndex.build(dom.raw)
    sources = [(PART_KIND_INTERROW, str(ref), g) for ref, g in zip(interrow_ids, interrow_geoms, strict=True)]
    sources += [(PART_KIND_PASSAGE, str(k), g) for k, g in enumerate(split_multi(passages))]
    parts: list[PassablePart] = []
    for kind, ref, geom in sources:
        polys = make_valid_polygonal(_clipped(geom, index, dom.grid_size_m))
        if not polys:
            continue
        clipped = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        parts.append(PassablePart(PART_ID_FMT.format(len(parts) + 1), kind, ref, float(clipped.area), 0.0,
                                  clipped))
    return tuple(parts)


__all__ = [
    "PART_KIND_INTERROW", "PART_KIND_PASSAGE", "DomainParams", "DomainSet", "PassablePart",
    "build_domain", "close_seams", "domain_from_raw", "passable_parts",
]
