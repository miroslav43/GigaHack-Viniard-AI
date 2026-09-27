"""Union-preserving canopy partition: cut canopy polygons where two plants touch (annotation rule 2.2).

Only existing polygons are cut, so the canopy union (class IoU, canopy area, interrows) is unchanged.
Cuts are perpendicular to the row axis at visible narrowing (`width`), at maxima of a neural contact
map (`nn`) or at narrowings the contact map confirms (`width+nn`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Polygon

from fte.canopy.partition_profile import (
    AxisFrame,
    Cut,
    axis_frame,
    combined_candidates,
    contact_candidates,
    contact_profile,
    s_extent,
    sample_positions,
    smooth,
    split_by_cuts,
    width_candidates,
    width_profile,
)

MODES: Final = ("width", "nn", "width+nn")
NN_MODES: Final = ("nn", "width+nn")
TOUCH_TOL_M: Final = 0.025 + 1e-9  # one pixel, as vineyard.perception.canopy
ContactFn = Callable[[str], np.ndarray | None]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartitionParams:
    mode: str = "width"
    alpha: float = 0.55
    t_c: float = 0.5
    min_len_m: float = 1.6
    step_m: float = 0.05
    half_len_m: float = 1.5
    smooth_k: int = 3
    min_piece_len_m: float = 0.6
    min_area_m2: float = 0.19
    clump_area_m2: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"partition mode {self.mode!r} not in {MODES}")
        if not (0 < self.alpha <= 1 and 0 <= self.t_c <= 1):
            raise ValueError(f"alpha must be in (0, 1] and t_c in [0, 1], got {self.alpha}, {self.t_c}")
        if self.step_m <= 0 or self.min_len_m <= 0 or self.min_piece_len_m <= 0:
            raise ValueError("step_m, min_len_m and min_piece_len_m must be > 0")

    @classmethod
    def from_mapping(cls, doc: Mapping[str, Any]) -> PartitionParams:
        unknown = sorted(set(doc) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown partition params: {unknown}")
        return cls(**dict(doc))

    @classmethod
    def from_json(cls, text: str) -> PartitionParams:
        return cls.from_mapping(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TileContext:
    tile_id: str
    x0: float
    y0: float
    contact: np.ndarray | None


# ------------------------------------------------------------------ one polygon


def _candidates(poly: Polygon, frame: AxisFrame, s0: float, s1: float, t_max: float, params: PartitionParams,
                tile: TileContext) -> list[Cut]:
    ss = sample_positions(s0, s1, params.step_m)
    if len(ss) < 3:
        return []
    widths = smooth(width_profile(poly, frame, ss, params.half_len_m), params.smooth_k)
    if params.mode == "width":
        return width_candidates(widths, ss, params.alpha)
    if tile.contact is None:
        return []
    contact = smooth(contact_profile(poly, frame, ss, t_max, tile.contact, tile.x0, tile.y0), params.smooth_k)
    if params.mode == "nn":
        return contact_candidates(contact, ss, params.t_c)
    return combined_candidates(widths, contact, ss, params.alpha, params.t_c)


def _select(poly: Polygon, frame: AxisFrame, cands: list[Cut], s0: float, s1: float, t_max: float,
            params: PartitionParams) -> list[Polygon]:
    """Greedy strongest-first cut selection under the min piece length / area constraints."""
    accepted: list[float] = []
    pieces: list[Polygon] = [poly]
    for cut in sorted(cands, key=lambda c: (-c.strength, c.s)):
        if cut.s - s0 < params.min_piece_len_m or s1 - cut.s < params.min_piece_len_m:
            continue
        if any(abs(cut.s - a) < params.min_piece_len_m for a in accepted):
            continue
        trial = sorted([*accepted, cut.s])
        split = split_by_cuts(poly, frame, trial, s0, s1, t_max)
        if split is None or any(p.area < params.min_area_m2 for p in split):
            continue
        accepted, pieces = trial, split
    return pieces


def partition_polygon(poly: Polygon, axis: LineString | None, params: PartitionParams,
                      tile: TileContext) -> list[Polygon]:
    """Pieces of `poly` ordered along the axis; [poly] when nothing is cut."""
    if not isinstance(poly, Polygon) or poly.is_empty or not poly.is_valid:
        return [poly]
    frame = axis_frame(poly, axis)
    s0, s1, t_max = s_extent(poly, frame)
    if s1 - s0 <= params.min_len_m:
        return [poly]
    cands = _candidates(poly, frame, s0, s1, t_max, params, tile)
    return _select(poly, frame, cands, s0, s1, t_max, params) if cands else [poly]


# ------------------------------------------------------------------ attributes of new pieces


def _along(piece: Polygon, axis: LineString | None, frame: AxisFrame) -> tuple[float, float]:
    xy = np.asarray(piece.exterior.coords)
    if axis is not None and not axis.is_empty and axis.length > 0:
        s = shapely.line_locate_point(axis, shapely.points(xy))
    else:
        s = frame.s_of(xy)
    return float(np.min(s)), float(np.max(s))


def _touch_flags(pieces: list[Polygon], touched: bool, tile_boundary: Any) -> list[bool]:
    if not touched:
        return [False] * len(pieces)
    near = [bool(p.distance(tile_boundary) <= TOUCH_TOL_M) for p in pieces]
    return near if any(near) else [True] * len(pieces)


def _piece_records(row: Mapping[str, Any], pieces: list[Polygon], axis: LineString | None,
                   params: PartitionParams, tile_boundary: Any) -> list[dict[str, Any]]:
    frame = axis_frame(row["geometry"], axis)
    spans = [_along(p, axis, frame) for p in pieces]
    order = np.argsort([lo for lo, _ in spans], kind="stable")
    touches = _touch_flags(pieces, bool(row["touches_edge"]), tile_boundary)
    out = []
    for k in order:
        piece = shapely.orient_polygons(pieces[k])
        lo, hi = spans[k]
        out.append({**row, "geometry": piece, "area_m2": float(piece.area),
                    "n_vertices": len(piece.exterior.coords) - 1, "along_m": hi - lo,
                    "is_clump": bool(piece.area > params.clump_area_m2), "touches_edge": touches[k]})
    return out


# ------------------------------------------------------------------ frame level


def _axis_lookup(row_pieces: gpd.GeoDataFrame) -> dict[tuple[str, str], list[LineString]]:
    out: dict[tuple[str, str], list[LineString]] = {}
    for tile_id, row_id, geom in zip(row_pieces["tile_id"], row_pieces["row_id"], row_pieces.geometry, strict=True):
        if isinstance(geom, LineString) and not geom.is_empty:
            out.setdefault((str(tile_id), str(row_id)), []).append(geom)
    return out


def _pick_axis(axes: dict[tuple[str, str], list[LineString]], tile_id: str, row_id: Any,
               poly: Polygon) -> LineString | None:
    cands = axes.get((tile_id, str(row_id)), []) if isinstance(row_id, str) else []
    if not cands:
        return None
    return min(cands, key=lambda a: a.distance(poly.centroid))


def _tile_context(tile_id: str, params: PartitionParams, contact: ContactFn | None) -> TileContext:
    from vineyard.geo.tiling import tile_ref

    ref = tile_ref(tile_id)
    prob = contact(tile_id) if (contact is not None and params.mode in NN_MODES) else None
    if params.mode in NN_MODES and prob is None:
        _log.warning("no contact map for %s: mode %s leaves the tile uncut", tile_id, params.mode)
    return TileContext(tile_id, float(ref.x0), float(ref.y0), prob)


def _renumber(records: list[dict[str, Any]], tile_id: str) -> list[dict[str, Any]]:
    from vineyard.contracts.ids import format_canopy_id

    return [{**r, "canopy_id": format_canopy_id(tile_id, k + 1)} for k, r in enumerate(records)]


def _partition_tile(group: gpd.GeoDataFrame, axes: dict[tuple[str, str], list[LineString]],
                    params: PartitionParams, tile: TileContext) -> tuple[list[dict[str, Any]], int]:
    from vineyard.geo.tiling import tile_box, tile_ref

    boundary = tile_box(tile_ref(tile.tile_id)).boundary
    records: list[dict[str, Any]] = []
    n_cut = 0
    for row in group.to_dict("records"):
        axis = _pick_axis(axes, tile.tile_id, row.get("row_id"), row["geometry"])
        pieces = partition_polygon(row["geometry"], axis, params, tile)
        if len(pieces) == 1:
            records.append(dict(row))
            continue
        n_cut += 1
        records.extend(_piece_records(row, pieces, axis, params, boundary))
    return (_renumber(records, tile.tile_id) if n_cut else records), n_cut


def _as_frame(records: list[dict[str, Any]], like: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom_col = like.geometry.name
    frame = pd.DataFrame.from_records(records, columns=list(like.columns)) if records else like.iloc[0:0].copy()
    out = gpd.GeoDataFrame(frame, geometry=geom_col, crs=like.crs)
    for col in like.columns:
        if col != geom_col:
            out[col] = out[col].astype(like[col].dtype)
    return out.reset_index(drop=True)


def partition_canopies(canopies: gpd.GeoDataFrame, row_pieces: gpd.GeoDataFrame, params: PartitionParams,
                       contact: ContactFn | None = None) -> gpd.GeoDataFrame:
    """New canopies frame (same columns and dtypes) with merged plants cut apart; inputs are not modified."""
    if "tile_id" not in canopies.columns or "tile_id" not in row_pieces.columns:
        raise ValueError("canopies and row_pieces need a tile_id column")
    axes = _axis_lookup(row_pieces)
    records: list[dict[str, Any]] = []
    n_cut_total = 0
    for tile_id, group in canopies.groupby("tile_id", sort=False):
        tile = _tile_context(str(tile_id), params, contact)
        tile_records, n_cut = _partition_tile(group, axes, params, tile)
        records.extend(tile_records)
        n_cut_total += n_cut
    _log.info("partition %s: %d of %d canopies cut -> %d canopies", params.mode, n_cut_total, len(canopies),
              len(records))
    return _as_frame(records, canopies)


def png_contact_loader(directory: Path) -> ContactFn:
    """ContactFn reading `<directory>/<tile_id>.png` (uint8 probability map); None when missing."""
    import cv2

    root = Path(directory)

    def load(tile_id: str) -> np.ndarray | None:
        path = root / f"{tile_id}.png"
        if not path.is_file():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"cannot read contact map {path}")
        return img if img.ndim == 2 else img[..., 0]

    return load
