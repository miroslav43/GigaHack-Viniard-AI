"""Object layers of the web data bundle (src/Web/CLAUDE.md §6.3): canopies, interrows, waste (+ targets, route).

Every builder returns a new EPSG:32635 frame with exactly the contract properties (plus a few informative
ones), in a deterministic feature order. Inputs are never modified.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import InterrowCover
from vineyard.errors import SchemaError
from vineyard.web.rows_export import features_frame, finite_or_none, natural_key, normalize_enum, text_or_none
from vineyard.web.targets_export import RouteInfo, route_features, target_features

__all__ = [
    "CANOPY_COLUMNS", "INTERROW_COLUMNS", "WASTE_COLUMNS", "RouteInfo", "canopy_areas", "canopy_features",
    "canopy_web_ids", "interrow_features", "interrow_totals", "route_features", "target_features", "waste_features",
]

CANOPY_COLUMNS: Final = ("canopy_id", "vineyard_id", "tile", "row_id", "area_m2")
INTERROW_COLUMNS: Final = ("interrow_id", "vineyard_id", "interrow_cover", "tile", "row_ids", "area_m2",
                           "interrow_total_m2", "piece_id")
WASTE_COLUMNS: Final = ("waste_id", "vineyard_id", "tile", "confidence")
INTERROW_COVERS: Final[tuple[str, ...]] = tuple(c.value for c in InterrowCover)
WEB_CANOPY_SEPARATOR: Final = "#"
CANOPY_NUMBER_WIDTH: Final = 4
ROWS_PER_INTERROW: Final = 2  # an interrow lies between two rows
LINK_COLUMNS: Final = ("row_left_id", "row_right_id")
CONFIDENCE_RANGE: Final = (0.0, 1.0)
_CANOPY_ID_RE: Final = re.compile(r"^(?P<tile>.+):C(?P<num>\d+)$")


# ------------------------------------------------------------------ canopies


def _kept_numbers(ids: Sequence[str], tile: str) -> list[str] | None:
    """The `<n>` of `<tile>:C<n>` ids, or None when any id does not follow it (or numbers repeat)."""
    matches = [_CANOPY_ID_RE.fullmatch(str(cid)) for cid in ids]
    if any(m is None or m["tile"] != tile for m in matches):
        return None
    numbers = [m["num"] for m in matches if m is not None]
    return numbers if len({int(n) for n in numbers}) == len(numbers) else None


def _renumbered(ids: Sequence[str]) -> list[str]:
    order = sorted(range(len(ids)), key=lambda i: (natural_key(str(ids[i])), i))
    rank = {i: k for k, i in enumerate(order, start=1)}
    return [f"{rank[i]:0{CANOPY_NUMBER_WIDTH}d}" for i in range(len(ids))]


def canopy_web_ids(canopy_ids: Sequence[str], tiles: Sequence[str]) -> list[str]:
    """`<tile>#<n>`: n from `<tile>:C<n>` when every id of the tile has it, else 1..k by canopy id."""
    if len(canopy_ids) != len(tiles):
        raise SchemaError("canopy ids and tiles differ in length", n_ids=len(canopy_ids), n_tiles=len(tiles))
    positions: dict[str, list[int]] = {}
    for i, tile in enumerate(tiles):
        positions.setdefault(str(tile), []).append(i)
    web_ids: dict[int, str] = {}
    for tile, idx in positions.items():
        ids = [str(canopy_ids[i]) for i in idx]
        numbers = _kept_numbers(ids, tile) or _renumbered(ids)
        web_ids |= {i: f"{tile}{WEB_CANOPY_SEPARATOR}{n}" for i, n in zip(idx, numbers, strict=True)}
    return [web_ids[i] for i in range(len(canopy_ids))]


def canopy_areas(canopies: gpd.GeoDataFrame) -> list[float]:
    """The layer's `area_m2`; the polygon area where the column is missing or not finite."""
    given = canopies["area_m2"] if "area_m2" in canopies.columns else [None] * len(canopies)
    return [area if area is not None else float(geom.area)
            for area, geom in zip(map(finite_or_none, given), canopies.geometry, strict=True)]


def canopy_features(canopies: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """canopy_id `<tile>#<n>`, vineyard_id, tile, row_id, area_m2; ordered by tile then canopy id."""
    tiles = [str(t) for t in canopies["tile_id"]]
    web_ids = canopy_web_ids([str(c) for c in canopies["canopy_id"]], tiles)
    records = [{"canopy_id": cid, "vineyard_id": text_or_none(vid), "tile": tile, "row_id": text_or_none(rid),
                "area_m2": area}
               for cid, vid, tile, rid, area in zip(web_ids, canopies["vineyard_id"], tiles, canopies["row_id"],
                                                    canopy_areas(canopies), strict=True)]
    order = sorted(range(len(records)), key=lambda i: (natural_key(tiles[i]), natural_key(web_ids[i])))
    geoms = canopies.geometry.to_numpy()
    return features_frame([records[i] for i in order], [geoms[i] for i in order], CANOPY_COLUMNS)


# ------------------------------------------------------------------ interrows


class _RowIndex:
    """Row pieces for 'which rows bound this interrow piece' queries (same block, within a tolerance)."""

    def __init__(self, row_pieces: gpd.GeoDataFrame) -> None:
        valid = row_pieces[row_pieces["row_id"].notna()]
        self.row_ids = [str(r) for r in valid["row_id"]]
        self.blocks = [text_or_none(v) for v in valid["vineyard_id"]]
        self.geoms = valid.geometry.to_numpy()
        self.tree = shapely.STRtree(self.geoms)

    def near(self, poly: BaseGeometry, vineyard_id: str | None, tol_m: float) -> list[str]:
        idx = self.tree.query(poly, predicate="dwithin", distance=tol_m)
        idx = np.asarray([i for i in idx if vineyard_id is None or self.blocks[i] == vineyard_id], dtype=np.int64)
        best: dict[str, float] = {}
        for i, dist in zip(idx, shapely.distance(poly, self.geoms[idx]), strict=True):
            best[self.row_ids[i]] = min(float(dist), best.get(self.row_ids[i], math.inf))
        nearest = sorted(best, key=lambda r: (best[r], natural_key(r)))[:ROWS_PER_INTERROW]
        return sorted(nearest, key=natural_key)


def _linked_rows(rec: Mapping[str, Any]) -> list[str]:
    ids = {text_or_none(rec.get(col)) for col in LINK_COLUMNS}
    return sorted((i for i in ids if i is not None), key=natural_key)


def _interrow_record(rec: Mapping[str, Any], index: _RowIndex, tol_m: float) -> dict[str, Any]:
    piece_id = str(rec["piece_id"])
    vineyard_id = text_or_none(rec.get("vineyard_id"))
    geom = rec["geometry"]
    return {"interrow_id": text_or_none(rec.get("interrow_id")) or piece_id, "vineyard_id": vineyard_id,
            "interrow_cover": normalize_enum(rec.get("interrow_cover"), INTERROW_COVERS,
                                             InterrowCover.UNASSESSABLE.value),
            "tile": str(rec["tile_id"]), "row_ids": _linked_rows(rec) or index.near(geom, vineyard_id, tol_m),
            "area_m2": float(geom.area), "piece_id": piece_id}


def interrow_totals(interrow_ids: Sequence[str], geoms: Sequence[BaseGeometry]) -> list[float]:
    """Area of the union of all pieces sharing an interrow id, for every piece (one union per interrow)."""
    members: dict[str, list[BaseGeometry]] = {}
    for iid, geom in zip(interrow_ids, geoms, strict=True):
        members.setdefault(iid, []).append(geom)
    area = {iid: float(shapely.union_all(parts).area) for iid, parts in members.items()}
    return [area[iid] for iid in interrow_ids]


def interrow_features(interrow_pieces: gpd.GeoDataFrame, row_pieces: gpd.GeoDataFrame, *,
                      link_tol_m: float) -> gpd.GeoDataFrame:
    """One feature per interrow piece (tile part); `interrow_id` is the linked global id when derive set it,
    else the piece id. `row_ids` come from the links, else from the (<= 2) nearest rows of the same block.
    `area_m2` is the piece's area, `interrow_total_m2` the area of the whole interrow (union of its pieces)."""
    if link_tol_m <= 0:
        raise SchemaError("interrow row-link tolerance must be > 0", link_tol_m=link_tol_m)
    index = _RowIndex(row_pieces)
    pieces = [_interrow_record(rec, index, link_tol_m) for rec in interrow_pieces.to_dict("records")]
    geoms = list(interrow_pieces.geometry)
    totals = interrow_totals([r["interrow_id"] for r in pieces], geoms)
    records = [r | {"interrow_total_m2": total} for r, total in zip(pieces, totals, strict=True)]
    order = sorted(range(len(records)),
                   key=lambda i: (natural_key(records[i]["tile"]), natural_key(records[i]["piece_id"])))
    return features_frame([records[i] for i in order], [geoms[i] for i in order], INTERROW_COLUMNS)


# ------------------------------------------------------------------ waste


def _confidence(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    number = float(value)  # type: ignore[arg-type]
    return float(np.clip(number, *CONFIDENCE_RANGE)) if math.isfinite(number) else None


def _waste_block(vineyard_id: object, dist_block_m: object, max_dist_m: float) -> str | None:
    """The box's block, or None when it has none or lies farther than `max_dist_m` from it (§6.3)."""
    known = dist_block_m is not None and not pd.isna(dist_block_m)
    dist = float(dist_block_m) if known else math.nan  # type: ignore[arg-type]
    return None if math.isfinite(dist) and dist > max_dist_m else text_or_none(vineyard_id)


def waste_features(waste: gpd.GeoDataFrame, *, max_block_dist_m: float) -> gpd.GeoDataFrame:
    """waste_id, vineyard_id (null beyond `max_block_dist_m`), tile, confidence clipped to [0, 1]."""
    ids = [str(w) for w in waste["waste_id"]]
    records = [{"waste_id": wid,
                "vineyard_id": _waste_block(rec.get("vineyard_id"), rec.get("dist_block_m"), max_block_dist_m),
                "tile": str(rec["tile_id"]), "confidence": _confidence(rec.get("confidence"))}
               for wid, rec in zip(ids, waste.to_dict("records"), strict=True)]
    geoms = list(waste.geometry)
    order = sorted(range(len(records)), key=lambda i: natural_key(ids[i]))
    return features_frame([records[i] for i in order], [geoms[i] for i in order], WASTE_COLUMNS)
