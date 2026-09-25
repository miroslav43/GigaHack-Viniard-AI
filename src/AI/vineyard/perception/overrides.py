"""Human QA overrides (configs/overrides.yaml, A§2.4), applied deterministically in UTM.

Order: exclude_areas -> force_empty_tiles (candidates, before linking) -> delete_rows -> extend_rows ->
add_rows (chains, before blocks). Every entry yields an `override_applied` or `override_unmatched` issue.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from vineyard.config import canonical_json
from vineyard.contracts.enums import Severity, Source
from vineyard.contracts.ids import IdKind, format_chain_id, is_valid_id
from vineyard.contracts.qa import QaIssue
from vineyard.contracts.schema_defs import MIN_LINE_LENGTH_M
from vineyard.errors import ConfigError
from vineyard.geo.tiling import CRS_EPSG, tile_box, tile_of_point, tile_ref, tiles_for_bounds

CODE_APPLIED: Final = "override_applied"
CODE_UNMATCHED: Final = "override_unmatched"
OVERRIDE_FLAG_PREFIX: Final = "override:"
FLAG_SEP: Final = ";"
_ENTRY_ID_PATTERN: Final = r"^[A-Z][A-Za-z0-9_-]{0,31}$"


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=_ENTRY_ID_PATTERN)
    note: str = ""


def _parse_wkt(text: str, kinds: tuple[str, ...]) -> BaseGeometry:
    try:
        geom = shapely.from_wkt(text)
    except shapely.errors.GEOSException as exc:
        raise ValueError(f"invalid WKT: {exc}") from exc
    if geom is None or geom.is_empty or geom.geom_type not in kinds:
        raise ValueError(f"WKT must be a non-empty {'/'.join(kinds)}, got {text[:60]!r}")
    return geom


class ExcludeArea(_Entry):
    wkt: str

    @field_validator("wkt")
    @classmethod
    def _poly(cls, v: str) -> str:
        _parse_wkt(v, ("Polygon", "MultiPolygon"))
        return v


class DeleteRow(_Entry):
    wkt: str
    tol_m: float | None = Field(default=None, gt=0.0)  # None: blocks.override_match_tol_m

    @field_validator("wkt")
    @classmethod
    def _geom(cls, v: str) -> str:
        _parse_wkt(v, ("Point", "LineString", "Polygon"))
        return v


class ExtendRow(_Entry):
    row_hint_wkt: str
    to_wkt: str
    tol_m: float | None = Field(default=None, gt=0.0)  # None: blocks.override_match_tol_m

    @field_validator("row_hint_wkt", "to_wkt")
    @classmethod
    def _point(cls, v: str) -> str:
        _parse_wkt(v, ("Point",))
        return v


class AddRow(_Entry):
    wkt: str

    @field_validator("wkt")
    @classmethod
    def _line(cls, v: str) -> str:
        geom = _parse_wkt(v, ("LineString",))
        if geom.length < MIN_LINE_LENGTH_M:
            raise ValueError(f"added row shorter than {MIN_LINE_LENGTH_M} m")
        return v


class Overrides(BaseModel):
    """configs/overrides.yaml; every entry id is unique across sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1]
    force_empty_tiles: tuple[str, ...] = ()
    exclude_areas: tuple[ExcludeArea, ...] = ()
    delete_rows: tuple[DeleteRow, ...] = ()
    extend_rows: tuple[ExtendRow, ...] = ()
    add_rows: tuple[AddRow, ...] = ()

    @field_validator("force_empty_tiles")
    @classmethod
    def _tiles(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        bad = [t for t in v if not is_valid_id(IdKind.TILE, t)]
        if bad:
            raise ValueError(f"invalid tile ids in force_empty_tiles: {bad[:5]}")
        return v

    @model_validator(mode="after")
    def _unique_ids(self) -> Overrides:
        ids = [e.id for sec in (self.exclude_areas, self.delete_rows, self.extend_rows, self.add_rows) for e in sec]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate override ids: {dup}")
        return self


EMPTY_OVERRIDES: Final = Overrides(version=1)


def load_overrides(path: Path) -> Overrides:
    source = Path(path)
    if not source.is_file():
        raise ConfigError("overrides file not found", path=str(source))
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        return Overrides.model_validate(data if data is not None else {"version": 1})
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"invalid overrides file: {exc}", path=str(source)) from exc


def overrides_digest(ov: Overrides) -> str:
    return hashlib.sha1(canonical_json(ov.model_dump(mode="json")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OverrideResult:
    """New frame after overrides, the rows it removed (with `reason`), and one QA issue per entry."""

    frame: gpd.GeoDataFrame
    removed: gpd.GeoDataFrame
    issues: tuple[QaIssue, ...]


def _tile_at(p: Point) -> str:
    try:
        return tile_of_point(p.x, p.y).tile_id
    except ValueError:  # outside the grid: the issue stays unlocated by tile
        return ""


def _issue(entry_id: str, matched: int, geom: BaseGeometry, what: str) -> QaIssue:
    p = geom.representative_point()
    code, sev = (CODE_APPLIED, Severity.INFO) if matched else (CODE_UNMATCHED, Severity.WARNING)
    return QaIssue(sev, code, _tile_at(p), entry_id, f"{what}: {matched} matched", p.x, p.y)


def _longest_part(geom: BaseGeometry) -> LineString | None:
    parts = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, LineString) and not g.is_empty]
    best = max(parts, key=lambda g: g.length, default=None)
    return best if best is not None and best.length >= MIN_LINE_LENGTH_M else None


def _exclude(frame: gpd.GeoDataFrame, area: ExcludeArea) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, QaIssue]:
    poly = shapely.from_wkt(area.wkt)
    hit = frame.geometry.intersects(poly).to_numpy(dtype=bool)
    new_geoms = [_longest_part(g.difference(poly)) if h else g for g, h in zip(frame.geometry, hit, strict=True)]
    gone = np.array([g is None for g in new_geoms], dtype=bool)
    kept = frame.assign(geometry=gpd.GeoSeries(new_geoms, index=frame.index, crs=frame.crs))[~gone]
    removed = frame[gone].assign(reason=f"{OVERRIDE_FLAG_PREFIX}{area.id}")
    return kept, removed, _issue(area.id, int(hit.sum()), poly, "exclude_area")


def _force_empty(frame: gpd.GeoDataFrame, tiles: Sequence[str]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list]:
    issues = []
    drop = frame.tile_id.isin(list(tiles)).to_numpy(dtype=bool)
    for t in tiles:
        n = int((frame.tile_id == t).sum())
        centre = tile_box(tile_ref(t)).centroid
        issues.append(QaIssue(Severity.INFO, CODE_APPLIED, t, t, f"force_empty_tile: {n} candidates dropped",
                              centre.x, centre.y))
    return frame[~drop], frame[drop].assign(reason="override:force_empty"), issues


def _concat(frames: Sequence[gpd.GeoDataFrame], like: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    parts = [f for f in frames if len(f)]
    if not parts:
        return like.iloc[0:0].assign(reason=pd.Series(dtype=object))
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=like.crs)


def apply_candidate_overrides(cands: gpd.GeoDataFrame, ov: Overrides) -> OverrideResult:
    """exclude_areas (candidate lines cut, keep the longest part) then force_empty_tiles (dropped)."""
    frame, removed, issues = cands.copy(), [], []
    for area in ov.exclude_areas:
        frame, gone, issue = _exclude(frame, area)
        removed.append(gone)
        issues.append(issue)
    frame, gone, tile_issues = _force_empty(frame, ov.force_empty_tiles)
    removed.append(gone)
    issues.extend(tile_issues)
    return OverrideResult(frame.reset_index(drop=True), _concat(removed, cands), tuple(issues))


def _nearest(frame: gpd.GeoDataFrame, geom: BaseGeometry, tol_m: float) -> int | None:
    if not len(frame):
        return None
    dist = frame.geometry.distance(geom).to_numpy(dtype=np.float64)
    order = np.lexsort((frame.chain_id.astype(str).to_numpy(), dist))
    best = int(order[0])
    return best if dist[best] <= tol_m else None


def _add_flag(flags: object, flag: str) -> str:
    text = "" if flags is None or (isinstance(flags, float) and math.isnan(flags)) else str(flags)
    return flag if not text else f"{text}{FLAG_SEP}{flag}"


def _extended(line: LineString, to: Point) -> LineString:
    coords = np.asarray(line.coords, dtype=np.float64)[:, :2]
    at_end = np.hypot(*(coords[-1] - to.coords[0])) < np.hypot(*(coords[0] - to.coords[0]))
    seg = coords[-2:] if at_end else coords[:2][::-1]
    d = (seg[1] - seg[0]) / np.hypot(*(seg[1] - seg[0]))
    new_end = seg[0] + float(np.dot(np.asarray(to.coords[0]) - seg[0], d)) * d
    out = coords.copy()
    out[-1 if at_end else 0] = new_end
    return LineString(out)


def _line_attrs(line: LineString) -> dict[str, object]:
    tiles = [t for t in tiles_for_bounds(*line.bounds) if line.intersection(tile_box(tile_ref(t))).length > 0]
    coords = np.asarray(line.coords)
    dx, dy = coords[-1, 0] - coords[0, 0], coords[-1, 1] - coords[0, 1]
    return {"extent_m": float(line.length), "angle_deg": math.degrees(math.atan2(dy, dx)) % 180.0,
            "tile_ids": ",".join(tiles), "n_tiles": len(tiles), "is_curved": len(coords) > 2}


def _next_chain_index(frame: gpd.GeoDataFrame) -> int:
    nums = [int(c[1:]) for c in frame.chain_id.astype(str) if c[1:].isdigit()]
    return max(nums, default=0) + 1


def _added_record(entry: AddRow, chain_id: str) -> dict[str, object]:
    line = shapely.from_wkt(entry.wkt)
    return {"chain_id": chain_id, "member_cand_ids": "", **_line_attrs(line), "support_frac": math.nan,
            "width_p80_m": math.nan, "along_duty": math.nan, "harmonic_frac": 0.0, "rescued_frac": 0.0,
            "gaps_json": "[]", "interp_tile_ids": "", "source": Source.MODEL.value, "confidence": 1.0,
            "qa_flags": f"{OVERRIDE_FLAG_PREFIX}{entry.id}", "geometry": line}


def _tol(entry_tol_m: float | None, default_tol_m: float) -> float:
    return default_tol_m if entry_tol_m is None else entry_tol_m


def _delete_rows(frame: gpd.GeoDataFrame, ov: Overrides, default_tol_m: float
                 ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list]:
    removed, issues = [], []
    for entry in ov.delete_rows:
        geom = shapely.from_wkt(entry.wkt)
        k = _nearest(frame, geom, _tol(entry.tol_m, default_tol_m))
        issues.append(_issue(entry.id, int(k is not None), geom, "delete_row"))
        if k is not None:
            removed.append(frame.iloc[[k]].assign(reason=f"{OVERRIDE_FLAG_PREFIX}{entry.id}"))
            frame = frame.drop(index=frame.index[k]).reset_index(drop=True)
    return frame, _concat(removed, frame), issues


def _extend_rows(frame: gpd.GeoDataFrame, ov: Overrides, default_tol_m: float) -> tuple[gpd.GeoDataFrame, list]:
    issues = []
    for entry in ov.extend_rows:
        hint, to = shapely.from_wkt(entry.row_hint_wkt), shapely.from_wkt(entry.to_wkt)
        k = _nearest(frame, hint, _tol(entry.tol_m, default_tol_m))
        issues.append(_issue(entry.id, int(k is not None), hint, "extend_row"))
        if k is None:
            continue
        line = _extended(frame.geometry.iloc[k], to)
        flags = _column(frame, "qa_flags", "")
        updates = {"qa_flags": _replaced(flags, k, _add_flag(flags.iloc[k], f"{OVERRIDE_FLAG_PREFIX}{entry.id}")),
                   **{c: _replaced(_column(frame, c, None), k, v) for c, v in _line_attrs(line).items()}}
        geoms = gpd.GeoSeries(_replaced(frame.geometry, k, line), index=frame.index, crs=frame.crs)
        frame = frame.assign(**updates).set_geometry(geoms)
    return frame, issues


def _column(frame: gpd.GeoDataFrame, name: str, fill: object) -> pd.Series:
    return frame[name] if name in frame.columns else pd.Series([fill] * len(frame), index=frame.index)


def _replaced(values: pd.Series, k: int, value: object) -> list[object]:
    out = list(values)
    out[k] = value
    return out


def _add_rows(frame: gpd.GeoDataFrame, ov: Overrides) -> tuple[gpd.GeoDataFrame, list]:
    start = _next_chain_index(frame)
    records = [_added_record(e, format_chain_id(start + i)) for i, e in enumerate(ov.add_rows)]
    issues = [_issue(e.id, 1, shapely.from_wkt(e.wkt), "add_row") for e in ov.add_rows]
    if not records:
        return frame, issues
    added = gpd.GeoDataFrame(records, geometry="geometry", crs=frame.crs or f"EPSG:{CRS_EPSG}")
    return gpd.GeoDataFrame(pd.concat([frame, added], ignore_index=True), geometry="geometry",
                            crs=frame.crs or added.crs), issues


def apply_row_overrides(rows_raw: gpd.GeoDataFrame, ov: Overrides, *, default_tol_m: float) -> OverrideResult:
    """delete_rows -> extend_rows -> add_rows on the `rows_raw` chains (nearest chain within the entry's
    tol_m, default_tol_m = blocks.override_match_tol_m when the entry has none)."""
    frame, removed, issues = _delete_rows(rows_raw.reset_index(drop=True), ov, default_tol_m)
    frame, ext_issues = _extend_rows(frame, ov, default_tol_m)
    frame, add_issues = _add_rows(frame, ov)
    return OverrideResult(frame, removed, (*issues, *ext_issues, *add_issues))
