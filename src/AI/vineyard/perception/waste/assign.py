"""Waste bookkeeping (A§4.9 step 5, contract §2.4 / §1.6): vineyard_id of a box and final W-ids.

vineyard_id = block with the largest intersection, else the nearest block within max_dist_m, else "".
W-ids: sorted by (tile_id, ytl, xtl) -> W0001..; an object cut by a tile edge (two boxes touching the
shared edge within edge_tol_px whose extents along it overlap by >= half the smaller one) shares one
number with suffixes a / b (a = the lexically smaller tile_id).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from vineyard.contracts.ids import format_waste_candidate_id, format_waste_id, parse_tile_id
from vineyard.geo.tiling import CRS_EPSG, TILE_PX, TileRef, px_to_utm
from vineyard.perception.waste.types import BlockAssignment, BoxPx, Candidate, Category, Decision, Detector

EDGE_MIN_OVERLAP: Final = 0.5
MAX_CANDIDATES_PER_TILE: Final = 9999  # tile-local id width <tile>:W0001
NO_BLOCK: Final = ""
BOX_COLUMNS: Final = ("px_xtl", "px_ytl", "px_xbr", "px_ybr")
_PARTS: Final = ("a", "b")


def assign_vineyard_id(box_utm: Polygon, blocks: gpd.GeoDataFrame, max_dist_m: float) -> BlockAssignment:
    if blocks.empty:
        return BlockAssignment(NO_BLOCK, math.nan)
    ids = blocks["vineyard_id"].astype(str).to_numpy()
    inter = blocks.geometry.intersection(box_utm).area.to_numpy()
    if np.any(inter > 0):
        best = min(np.flatnonzero(inter == inter.max()), key=lambda i: ids[i])
        return BlockAssignment(str(ids[best]), 0.0)
    dist = blocks.geometry.distance(box_utm).to_numpy()
    best = min(np.flatnonzero(dist == dist.min()), key=lambda i: ids[i])
    d = float(dist[best])
    return BlockAssignment(str(ids[best]) if d <= max_dist_m else NO_BLOCK, d)


@dataclass(frozen=True)
class _Box:
    idx: int
    tile_id: str
    row: int
    col: int
    xyxy: tuple[float, float, float, float]


def _boxes(waste: gpd.GeoDataFrame) -> list[_Box]:
    out = []
    for i, (tile, *xyxy) in enumerate(zip(waste["tile_id"], *(waste[c] for c in BOX_COLUMNS), strict=True)):
        row, col = parse_tile_id(str(tile))
        out.append(_Box(i, str(tile), row, col, tuple(float(v) for v in xyxy)))
    return out


def _overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    ov = min(a[1], b[1]) - max(a[0], b[0])
    return ov / min(a[1] - a[0], b[1] - b[0]) if ov > 0 else 0.0


def _edge_ratio(a: _Box, b: _Box, tol: float) -> float:
    """Overlap ratio when b is the east or south neighbour of a and both touch the shared edge, else 0."""
    ax0, ay0, ax1, ay1 = a.xyxy
    bx0, by0, bx1, by1 = b.xyxy
    if (b.row, b.col) == (a.row, a.col + 1) and ax1 >= TILE_PX - tol and bx0 <= tol:
        return _overlap_ratio((ay0, ay1), (by0, by1))
    if (b.row, b.col) == (a.row + 1, a.col) and ay1 >= TILE_PX - tol and by0 <= tol:
        return _overlap_ratio((ax0, ax1), (bx0, bx1))
    return 0.0


def _edge_pairs(boxes: list[_Box], tol: float) -> dict[int, int]:
    found = [
        (r, a.idx, b.idx) for a in boxes for b in boxes if (r := _edge_ratio(a, b, tol)) >= EDGE_MIN_OVERLAP
    ]
    partner: dict[int, int] = {}
    for _, i, j in sorted(found, key=lambda t: (-t[0], t[1], t[2])):
        if i not in partner and j not in partner:
            partner[i], partner[j] = j, i
    return partner


def assign_waste_ids(waste: gpd.GeoDataFrame, edge_tol_px: float) -> gpd.GeoDataFrame:
    """New frame sorted by (tile_id, ytl, xtl) with a `waste_id` column (W0001.., a/b edge pairs)."""
    frame = waste.sort_values(["tile_id", "px_ytl", "px_xtl", "px_xbr", "px_ybr"], kind="mergesort")
    frame = frame.reset_index(drop=True)
    boxes = _boxes(frame)
    partner = _edge_pairs(boxes, edge_tol_px)
    ids: dict[int, str] = {}
    k = 0
    for b in boxes:
        if b.idx in ids:
            continue
        k += 1
        if b.idx not in partner:
            ids[b.idx] = format_waste_id(k)
            continue
        pair = sorted((b, boxes[partner[b.idx]]), key=lambda x: x.tile_id)
        ids.update({x.idx: format_waste_id(k, part) for x, part in zip(pair, _PARTS, strict=True)})
    return frame.assign(waste_id=[ids[i] for i in range(len(frame))])


# ------------------------------------------------------------------ waste_candidates layer


@dataclass(frozen=True)
class Provenance:
    source: str
    run_id: str
    model_version: str


def utm_box(tile: TileRef, box: BoxPx) -> Polygon:
    """The px box as an axis-aligned UTM polygon (the tile grid has no rotation)."""
    (x0, y1), (x1, y0) = px_to_utm(tile, np.array([[box.xtl, box.ytl], [box.xbr, box.ybr]]))
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _capped(cands: Sequence[Candidate], max_per_tile: int) -> tuple[list[Candidate], int]:
    """Per tile at most max_per_tile rows (id width): live ones first, then the largest rejected."""
    by_tile: dict[str, list[Candidate]] = {}
    for c in cands:
        by_tile.setdefault(c.tile_id, []).append(c)
    out: list[Candidate] = []
    dropped = 0
    for tile in sorted(by_tile):
        mine = by_tile[tile]
        live = [c for c in mine if not c.rejected]
        if len(live) > max_per_tile:
            raise ValueError(
                f"{tile}: {len(live)} live waste candidates exceed the id width ({max_per_tile})"
            )
        rejected = sorted((c for c in mine if c.rejected), key=lambda c: (-c.area_px, c.cand_key))
        room = max_per_tile - len(live)
        dropped += max(0, len(rejected) - room)
        keep = live + rejected[:room]
        out.extend(sorted(keep, key=lambda c: (c.box.ytl, c.box.xtl, c.cand_key)))
    return out, dropped


def _row(
    c: Candidate, k: int, d: Decision | None, review: bool, a: BlockAssignment, prov: Provenance, gsd_m: float
) -> dict[str, object]:
    rank = d.rank_score if d is not None else 0.0
    return {
        "waste_id": format_waste_candidate_id(c.tile_id, k),
        "tile_id": c.tile_id,
        "vineyard_id": a.vineyard_id,
        "dist_block_m": a.dist_block_m,
        "px_xtl": c.box.xtl,
        "px_ytl": c.box.ytl,
        "px_xbr": c.box.xbr,
        "px_ybr": c.box.ybr,
        "area_m2": c.box.area * gsd_m * gsd_m,
        "category": Category.UNKNOWN.value,
        "detector": (d.detector if d is not None else Detector.RULE).value,
        "exported": bool(d and d.auto),
        "reject_reason": None if c.reject_reason is None else c.reject_reason.value,
        "source": prov.source,
        "run_id": prov.run_id,
        "model_version": prov.model_version,
        "confidence": rank,
        "qa_flags": "",
        "cand_key": c.cand_key,
        "colour_class": c.colour_class.value,
        "probe_p": np.nan,
        "clip_pos_p": np.nan,
        "clip_margin": np.nan,
        "sam_score": np.nan,
        "rank_score": rank,
        "auto": bool(d and d.auto),
        "review": review,
        "blob_area_m2": c.area_m2,
        "aspect": c.aspect,
        "length_m": c.length_m,
        "width_m": c.width_m,
        "mean_h": c.mean_hsv[0],
        "mean_s": c.mean_hsv[1],
        "mean_v": c.mean_hsv[2],
    }


def candidates_layer(
    cands: Sequence[Candidate],
    decisions: Mapping[str, Decision],
    review_keys: Collection[str],
    blocks: gpd.GeoDataFrame,
    tiles: Mapping[str, TileRef],
    max_dist_m: float,
    prov: Provenance,
    *,
    gsd_m: float,
    max_per_tile: int = MAX_CANDIDATES_PER_TILE,
) -> tuple[gpd.GeoDataFrame, int]:
    """(`waste_candidates` layer rows with tile-local ids <tile>:W0001 by (ytl, xtl), n rejected dropped)."""
    kept, dropped = _capped(cands, max_per_tile)
    review = frozenset(review_keys)
    rows, geoms, k_of = [], [], dict.fromkeys((c.tile_id for c in kept), 0)
    for c in kept:
        k_of[c.tile_id] += 1
        geom = utm_box(tiles[c.tile_id], c.box)
        a = (
            BlockAssignment(NO_BLOCK, math.nan)
            if c.rejected
            else assign_vineyard_id(geom, blocks, max_dist_m)
        )
        rows.append(_row(c, k_of[c.tile_id], decisions.get(c.cand_key), c.cand_key in review, a, prov, gsd_m))
        geoms.append(geom)
    columns = list(rows[0]) if rows else []
    return gpd.GeoDataFrame(
        pd.DataFrame(rows, columns=columns), geometry=geoms, crs=f"EPSG:{CRS_EPSG}"
    ), dropped
