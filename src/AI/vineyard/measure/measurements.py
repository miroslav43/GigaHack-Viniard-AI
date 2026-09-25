"""Measurements with the web contract semantics (src/Web/CLAUDE.md §6.4), from one AnnSet.

- canopy / interrow area = area of the UNION of the polygons (per tile, summed; a global union is used
  when a polygon leaves its own tile square, so overlaps across tile edges are never counted twice);
- row length = sum of its pieces; total = sum over rows; a row belongs to the majority vineyard_id of
  its pieces, so block lengths add up to the total;
- plant_count = number of canopy polygons (survey: all; block: by vineyard_id; row: by canopy row_id);
- row_structure = disrupted if any piece is disrupted, unassessable if all are, else regular.
Values are exact here; rounding happens only in the CSV / JSON writers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from vineyard.contracts.enums import RowStructure, Severity
from vineyard.contracts.qa import QaIssue
from vineyard.errors import SchemaError
from vineyard.geo.tiling import tile_ref
from vineyard.measure.csv_format import (
    HEADER,
    LEVEL_BLOCK,
    LEVEL_ROW,
    LEVEL_SURVEY,
    MeasurementRecord,
    cell,
)

ROW_STRUCTURES: Final[tuple[str, ...]] = tuple(s.value for s in RowStructure)
# Mirrors configs/default.yaml measure.area_union_sum_warn_frac (callers pass the configured value).
DEFAULT_UNION_SUM_WARN_FRAC: Final = 0.001
UNION_SUM_CODE: Final = "area_union_sum_mismatch"
# A per-tile union may poke out of its tile square by float noise only.
TILE_EDGE_TOL_M: Final = 1e-6
_DIGITS: Final = re.compile(r"(\d+)")
_INT_JSON: Final = frozenset({"block_count", "row_count", "plant_count"})


@dataclass(frozen=True)
class MeasureInputs:
    canopies: gpd.GeoDataFrame
    row_pieces: gpd.GeoDataFrame
    interrow_pieces: gpd.GeoDataFrame
    waste: gpd.GeoDataFrame | None = None
    rows: gpd.GeoDataFrame | None = None  # derive's global rows: max_gap_m measured on the whole row
    targets: gpd.GeoDataFrame | None = None


@dataclass(frozen=True)
class Measurements:
    survey: MeasurementRecord
    blocks: tuple[MeasurementRecord, ...]
    rows: tuple[MeasurementRecord, ...]
    extras: Mapping[str, Any]  # {"survey": {...}, "block": {vineyard_id: {...}}, "row": {row_id: {...}}}
    issues: tuple[QaIssue, ...]

    def records(self) -> tuple[MeasurementRecord, ...]:
        return (self.survey, *self.blocks, *self.rows)


# ------------------------------------------------------------------ small helpers


def natural_key(value: object) -> tuple[tuple[int, int | str], ...]:
    """Digit runs compare as numbers ('V01-R9' < 'V01-R10')."""
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in _DIGITS.split(str(value)) if p)


def text_or_none(value: object) -> str | None:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def aggregate_structure(values: Iterable[object]) -> str:
    """disrupted if any piece is disrupted; unassessable if all are (or unknown); otherwise regular."""
    unassessable = RowStructure.UNASSESSABLE.value
    found = [(text_or_none(v) or "").lower() for v in values]
    structures = [s if s in ROW_STRUCTURES else unassessable for s in found]
    if not structures:
        raise SchemaError("row_structure of a row without pieces")
    if RowStructure.DISRUPTED.value in structures:
        return RowStructure.DISRUPTED.value
    return unassessable if all(s == unassessable for s in structures) else RowStructure.REGULAR.value


def majority(values: Sequence[str]) -> str:
    """Most frequent value; ties go to the naturally smallest (deterministic)."""
    if not values:
        raise SchemaError("majority of an empty sequence")
    counts = Counter(values)
    best = max(counts.values())
    return min((v for v, n in counts.items() if n == best), key=natural_key)


def _within_tile(geom: shapely.Geometry, tile_id: str) -> bool:
    try:
        minx, miny, maxx, maxy = tile_ref(tile_id).bounds
    except (SchemaError, ValueError):
        return False
    gx0, gy0, gx1, gy1 = geom.bounds
    tol = TILE_EDGE_TOL_M
    return gx0 >= minx - tol and gy0 >= miny - tol and gx1 <= maxx + tol and gy1 <= maxy + tol


def union_area(frame: gpd.GeoDataFrame) -> float:
    """Area of the union of `frame`'s polygons (per-tile unions; global union if one leaves its tile)."""
    if frame.empty:
        return 0.0
    geoms = frame.geometry.to_numpy()
    groups = frame.groupby(frame["tile_id"].astype(str), sort=True).indices
    parts = {tile: shapely.union_all(geoms[idx]) for tile, idx in groups.items()}
    if all(_within_tile(g, tile) for tile, g in parts.items()):
        return float(sum(g.area for g in parts.values()))
    return float(shapely.union_all(list(parts.values())).area)


def _area_sum(frame: gpd.GeoDataFrame) -> float:
    return float(shapely.area(frame.geometry.to_numpy()).sum()) if len(frame) else 0.0


def _of_block(frame: pd.DataFrame | None, vineyard_id: str) -> pd.DataFrame:
    if frame is None or frame.empty or "vineyard_id" not in frame.columns:
        return frame.iloc[0:0] if frame is not None else pd.DataFrame()
    return frame[frame["vineyard_id"].map(text_or_none) == vineyard_id]


def _count(frame: pd.DataFrame | None) -> int:
    return 0 if frame is None else len(frame)


# ------------------------------------------------------------------ physical rows


@dataclass(frozen=True)
class _Row:
    row_id: str
    vineyard_id: str
    length_m: float
    structure: str
    n_pieces: int
    tile_ids: tuple[str, ...]
    max_gap_m: float | None


def _max_gap(pieces: pd.DataFrame, derived: float | None) -> float | None:
    if derived is not None and math.isfinite(derived):
        return derived
    gaps = pd.to_numeric(pieces.get("max_gap_m", pd.Series(dtype=float)), errors="coerce").to_numpy(np.float64)
    gaps = gaps[np.isfinite(gaps)]
    return float(gaps.max()) if gaps.size else None


def _derived_gaps(rows: pd.DataFrame | None) -> dict[str, float]:
    if rows is None or rows.empty or "max_gap_m" not in rows.columns:
        return {}
    return {str(r): float(g) for r, g in zip(rows["row_id"], rows["max_gap_m"], strict=True) if text_or_none(r)}


def physical_rows(row_pieces: gpd.GeoDataFrame, rows: pd.DataFrame | None = None) -> tuple[_Row, ...]:
    """One record per row_id, sorted by (vineyard_id, row_id) with natural order."""
    valid = row_pieces[row_pieces["row_id"].map(text_or_none).notna()]
    lengths = shapely.length(valid.geometry.to_numpy())
    gaps = _derived_gaps(rows)
    out = []
    for row_id, idx in valid.groupby(valid["row_id"].map(text_or_none), sort=False).indices.items():
        group = valid.iloc[idx]
        blocks = [v for v in map(text_or_none, group["vineyard_id"]) if v is not None]
        out.append(_Row(row_id=row_id, vineyard_id=majority(blocks) if blocks else "",
                        length_m=float(lengths[idx].sum()), structure=aggregate_structure(group["row_structure"]),
                        n_pieces=len(group), tile_ids=tuple(sorted(set(map(str, group["tile_id"])), key=natural_key)),
                        max_gap_m=_max_gap(group, gaps.get(row_id))))
    return tuple(sorted(out, key=lambda r: (natural_key(r.vineyard_id), natural_key(r.row_id))))


# ------------------------------------------------------------------ records


def block_ids(inputs: MeasureInputs) -> tuple[str, ...]:
    """Distinct non-empty vineyard_id over canopies, row pieces and interrow pieces (web convention)."""
    frames = (inputs.canopies, inputs.row_pieces, inputs.interrow_pieces)
    ids = {text_or_none(v) for f in frames for v in f["vineyard_id"]}
    return tuple(sorted((v for v in ids if v is not None), key=natural_key))


def _targets_by(targets: pd.DataFrame | None, column: str) -> Counter[str]:
    if targets is None or targets.empty or column not in targets.columns:
        return Counter()
    return Counter(v for v in map(text_or_none, targets[column]) if v is not None)


def _block_record(vid: str, inputs: MeasureInputs, rows: Sequence[_Row]) -> MeasurementRecord:
    mine = [r for r in rows if r.vineyard_id == vid]
    canopies = _of_block(inputs.canopies, vid)
    return MeasurementRecord(level=LEVEL_BLOCK, vineyard_id=vid, row_count=len(mine),
                             row_length_m=float(sum(r.length_m for r in mine)), canopy_area_m2=union_area(canopies),
                             interrow_area_m2=union_area(_of_block(inputs.interrow_pieces, vid)),
                             plant_count=len(canopies))


def _row_record(row: _Row, plants: Counter[str]) -> MeasurementRecord:
    return MeasurementRecord(level=LEVEL_ROW, vineyard_id=row.vineyard_id or None, row_id=row.row_id,
                             row_length_m=row.length_m, plant_count=plants.get(row.row_id, 0),
                             row_structure=row.structure)


def _extras(inputs: MeasureInputs, blocks: Sequence[str], rows: Sequence[_Row]) -> dict[str, Any]:
    t_block, t_row = _targets_by(inputs.targets, "vineyard_id"), _targets_by(inputs.targets, "row_id")
    tiles = {str(t) for f in (inputs.canopies, inputs.row_pieces, inputs.interrow_pieces) for t in f["tile_id"]}
    survey = {"n_row_pieces": len(inputs.row_pieces), "n_interrow_pieces": len(inputs.interrow_pieces),
              "n_waste": _count(inputs.waste), "n_targets": _count(inputs.targets), "n_tiles": len(tiles),
              "canopy_area_sum_m2": _area_sum(inputs.canopies),
              "interrow_area_sum_m2": _area_sum(inputs.interrow_pieces)}
    per_block = {vid: {"n_row_pieces": len(_of_block(inputs.row_pieces, vid)),
                       "n_interrow_pieces": len(_of_block(inputs.interrow_pieces, vid)),
                       "n_waste": len(_of_block(inputs.waste, vid)), "n_targets": t_block.get(vid, 0),
                       "canopy_area_sum_m2": _area_sum(_of_block(inputs.canopies, vid)),
                       "interrow_area_sum_m2": _area_sum(_of_block(inputs.interrow_pieces, vid))} for vid in blocks}
    per_row = {r.row_id: {"n_pieces": r.n_pieces, "tile_ids": list(r.tile_ids), "max_gap_m": r.max_gap_m,
                          "n_targets": t_row.get(r.row_id, 0)} for r in rows}
    return {"survey": survey, "block": per_block, "row": per_row}


def _union_sum_issues(survey: MeasurementRecord, extras: Mapping[str, Any], warn_frac: float) -> tuple[QaIssue, ...]:
    issues = []
    for layer, union in (("canopy", survey.canopy_area_m2), ("interrow", survey.interrow_area_m2)):
        total = extras[f"{layer}_area_sum_m2"]
        if union and abs(total - union) / union > warn_frac:
            issues.append(QaIssue(Severity.WARNING, UNION_SUM_CODE, "", layer,
                                  f"{layer} area: union {union:.2f} m2 vs sum {total:.2f} m2 (overlapping polygons)"))
    return tuple(issues)


def compute_measurements(inputs: MeasureInputs, *,
                         union_sum_warn_frac: float = DEFAULT_UNION_SUM_WARN_FRAC) -> Measurements:
    """survey + block + row records (exact values) and the JSON extras of one AnnSet."""
    rows = physical_rows(inputs.row_pieces, inputs.rows)
    blocks = block_ids(inputs)
    plants = Counter(v for v in map(text_or_none, inputs.canopies["row_id"]) if v is not None)
    survey = MeasurementRecord(level=LEVEL_SURVEY, block_count=len(blocks), row_count=len(rows),
                               row_length_m=float(sum(r.length_m for r in rows)),
                               canopy_area_m2=union_area(inputs.canopies),
                               interrow_area_m2=union_area(inputs.interrow_pieces), plant_count=len(inputs.canopies))
    extras = _extras(inputs, blocks, rows)
    return Measurements(survey=survey, blocks=tuple(_block_record(v, inputs, rows) for v in blocks),
                        rows=tuple(_row_record(r, plants) for r in rows), extras=extras,
                        issues=_union_sum_issues(survey, extras["survey"], union_sum_warn_frac))


# ------------------------------------------------------------------ JSON


def _json_value(record: MeasurementRecord, column: str, m_decimals: int, ha_decimals: int) -> Any:
    text = cell(record, column, m_decimals, ha_decimals)
    if record.value(column) is None:
        return None
    if column in _INT_JSON:
        return int(text)
    return float(text) if column.endswith(("_m", "_m2", "_ha")) else text


def _json_extra(value: Any, m_decimals: int) -> Any:
    if isinstance(value, float):
        return round(value, m_decimals) if math.isfinite(value) else None
    return value


def _record_json(record: MeasurementRecord, extra: Mapping[str, Any], m_decimals: int,
                 ha_decimals: int) -> dict[str, Any]:
    base = {col: _json_value(record, col, m_decimals, ha_decimals) for col in HEADER}
    return base | {k: _json_extra(v, m_decimals) for k, v in extra.items()}


def measurements_json(m: Measurements, meta: Mapping[str, Any], *, m_decimals: int,
                      ha_decimals: int) -> dict[str, Any]:
    """{header, total, blocks[], rows[], meta}: the CSV values (identically rounded) plus extras."""
    return {
        "header": list(HEADER),
        "total": _record_json(m.survey, m.extras["survey"], m_decimals, ha_decimals),
        "blocks": [_record_json(b, m.extras["block"][b.vineyard_id], m_decimals, ha_decimals) for b in m.blocks],
        "rows": [_record_json(r, m.extras["row"][r.row_id], m_decimals, ha_decimals) for r in m.rows],
        "meta": dict(meta),
    }


__all__ = [
    "DEFAULT_UNION_SUM_WARN_FRAC", "MeasureInputs", "Measurements", "aggregate_structure", "block_ids",
    "compute_measurements", "majority", "measurements_json", "natural_key", "physical_rows", "union_area",
]
