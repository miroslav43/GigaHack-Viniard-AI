"""Physical rows of the web bundle (one feature per row_id, src/Web/CLAUDE.md §6.3) and shared frame helpers.

Row semantics match measurements.csv (§6.4, written by `vineyard.measure`): row length = sum of its pieces,
plant_count = canopy polygons with that row_id, the row belongs to the majority vineyard_id of its pieces.
The text / enum / ordering helpers are the measure package's, so both outputs agree on every id and order.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from vineyard.errors import SchemaError
from vineyard.geo.tiling import CRS_EPSG
from vineyard.measure.measurements import aggregate_structure, majority, natural_key, text_or_none

ROW_COLUMNS: Final = ("row_id", "vineyard_id", "row_structure", "length_m", "plant_count", "max_gap_m",
                      "tile_structures")
# line_merge must not change the geometry length the web recomputes (§6.2 checks 0.5 %).
MERGE_LENGTH_TOL_M: Final = 1e-6


def normalize_enum(value: object, allowed: Sequence[str], fallback: str) -> str:
    """Lower-cased enum value; anything outside `allowed` becomes `fallback` (the web only accepts the enum)."""
    text = text_or_none(value)
    lowered = text.lower() if text is not None else None
    return lowered if lowered is not None and lowered in allowed else fallback


def features_frame(records: Sequence[Mapping[str, Any]], geoms: Sequence[BaseGeometry],
                   columns: Sequence[str]) -> gpd.GeoDataFrame:
    """EPSG:32635 frame with object-dtype properties: ints stay ints and None stays null in the JSON."""
    if len(records) != len(geoms):
        raise SchemaError("records and geometries differ in length", n_records=len(records), n_geoms=len(geoms))
    data = {col: pd.Series([rec[col] for rec in records], dtype=object) for col in columns}
    return gpd.GeoDataFrame(data, geometry=gpd.GeoSeries(list(geoms), crs=CRS_EPSG), crs=CRS_EPSG)


def merged_line(lines: Sequence[LineString]) -> BaseGeometry:
    """Pieces as one LineString when they chain end to end, else a MultiLineString; length preserved."""
    if len(lines) == 1:
        return lines[0]
    multi = MultiLineString(list(lines))
    merged = shapely.line_merge(multi)
    kept = isinstance(merged, LineString | MultiLineString)
    if kept and math.isclose(merged.length, multi.length, rel_tol=0.0, abs_tol=MERGE_LENGTH_TOL_M):
        return merged
    return multi


@dataclass(frozen=True)
class _Piece:
    order: tuple[tuple[int, int | str], ...]
    tile_id: str
    vineyard_id: str
    row_structure: object
    max_gap_m: float
    geometry: LineString


def _finite_or_nan(value: object) -> float:
    number = math.nan if value is None or value is pd.NA else float(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else math.nan


def finite_or_none(value: object) -> float | None:
    """A finite number as float, else None (None, NA, NaN and inf are all written as JSON null)."""
    number = _finite_or_nan(value)
    return None if math.isnan(number) else number


def _pieces_by_row(row_pieces: gpd.GeoDataFrame) -> dict[str, list[_Piece]]:
    """row_id -> its pieces in (tile, piece id) order; pieces without a row_id are ignored."""
    columns = ("row_id", "tile_id", "piece_id", "vineyard_id", "row_structure", "max_gap_m")
    groups: dict[str, list[_Piece]] = {}
    for (rid, tile, pid, vid, structure, gap), geom in zip(zip(*(row_pieces[c] for c in columns), strict=True),
                                                           row_pieces.geometry, strict=True):
        key = text_or_none(rid)
        if key is not None:
            groups.setdefault(key, []).append(_Piece(natural_key(f"{tile}|{pid}"), str(tile), str(vid), structure,
                                                     _finite_or_nan(gap), geom))
    return {rid: sorted(pieces, key=lambda p: p.order) for rid, pieces in groups.items()}


def _max_gap(pieces: Sequence[_Piece], derived: float | None) -> float | None:
    if derived is not None and math.isfinite(derived):
        return float(derived)
    gaps = [p.max_gap_m for p in pieces if math.isfinite(p.max_gap_m)]
    return max(gaps) if gaps else None


def _tile_structures(pieces: Sequence[_Piece]) -> dict[str, str]:
    tiles = sorted({p.tile_id for p in pieces}, key=natural_key)
    return {t: aggregate_structure([p.row_structure for p in pieces if p.tile_id == t]) for t in tiles}


def _derived_gaps(rows: pd.DataFrame | None) -> dict[str, float]:
    if rows is None or rows.empty:
        return {}
    return {str(r): float(g) for r, g in zip(rows["row_id"], rows["max_gap_m"], strict=True) if pd.notna(r)}


def _row_record(row_id: str, pieces: Sequence[_Piece], plants: Mapping[str, int],
                derived_gap: float | None) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "vineyard_id": majority([p.vineyard_id for p in pieces]),
        "row_structure": aggregate_structure([p.row_structure for p in pieces]),
        "length_m": math.fsum(p.geometry.length for p in pieces),
        "plant_count": int(plants.get(row_id, 0)),
        "max_gap_m": _max_gap(pieces, derived_gap),
        "tile_structures": _tile_structures(pieces),
    }


def physical_rows(row_pieces: gpd.GeoDataFrame, canopies: pd.DataFrame, *,
                  rows: pd.DataFrame | None = None) -> gpd.GeoDataFrame:
    """One feature per row_id: merged tile pieces, aggregated structure, sum of piece lengths, canopy count.

    `rows` (derive's global rows, optional) supplies max_gap_m measured on the whole row; otherwise the
    largest per-piece max_gap_m is used.
    """
    plants = Counter(v for v in map(text_or_none, canopies["row_id"]) if v is not None)
    gaps = _derived_gaps(rows)
    groups = _pieces_by_row(row_pieces)
    ids = sorted(groups, key=natural_key)
    records = [_row_record(rid, groups[rid], plants, gaps.get(rid)) for rid in ids]
    geoms = [merged_line([p.geometry for p in groups[rid]]) for rid in ids]
    order = sorted(range(len(records)), key=lambda i: (natural_key(records[i]["vineyard_id"]),
                                                       natural_key(records[i]["row_id"])))
    return features_frame([records[i] for i in order], [geoms[i] for i in order], ROW_COLUMNS)


__all__ = [
    "ROW_COLUMNS", "aggregate_structure", "features_frame", "finite_or_none", "majority", "merged_line",
    "natural_key", "normalize_enum", "physical_rows", "text_or_none",
]
