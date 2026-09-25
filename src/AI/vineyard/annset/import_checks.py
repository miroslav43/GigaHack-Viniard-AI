"""Attribute-level QA of an imported AnnSet (contract §4.3-4.4, design 04 §3.1, critique X8).

Organizers score the RAW Marcaj values, so nothing is rewritten here: every value that is blank, not an
exact enum member, padded with whitespace or colliding with another id by letter case becomes a
QaIssue pointing at the object, for the annotators to fix in Marcaj. Row-level checks that need the
merged geometry (row_multi_block, dup_row_in_tile, ...) live in `merge_rows`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import geopandas as gpd
import numpy as np
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from vineyard.annset.model import AnnSet
from vineyard.config import AppConfig
from vineyard.contracts.enums import InterrowCover, RowStructure, Severity, Source
from vineyard.contracts.ids import tile_grid_ids
from vineyard.contracts.qa import QaIssue

PK_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {"canopies": "canopy_id", "row_pieces": "piece_id", "interrow_pieces": "piece_id", "waste": "waste_id"})
# Waste may legitimately carry a blank vineyard_id (outside every block, contract §2.5.10).
REQUIRED_ATTRS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "canopies": ("vineyard_id",),
    "row_pieces": ("vineyard_id", "row_id", "row_structure"),
    "interrow_pieces": ("vineyard_id", "interrow_cover"),
})
ENUM_ATTRS: Final[Mapping[tuple[str, str], frozenset[str]]] = MappingProxyType({
    ("row_pieces", "row_structure"): frozenset(m.value for m in RowStructure),
    ("interrow_pieces", "interrow_cover"): frozenset(m.value for m in InterrowCover),
})
ID_ATTRS: Final[tuple[tuple[str, str], ...]] = (
    ("canopies", "vineyard_id"), ("row_pieces", "vineyard_id"), ("interrow_pieces", "vineyard_id"),
    ("waste", "vineyard_id"), ("row_pieces", "row_id"),
)
MAX_LISTED: Final = 10  # tile names listed in one images_missing message


@dataclass(frozen=True)
class ImportCheckParams:
    require_all_tiles: bool
    accept_synonyms: bool
    enum_synonyms: Mapping[str, str]
    max_overlap_m2: float  # canopy ∩ interrow above this, per canopy, is canopy_interrow_overlap
    expected_tile_ids: tuple[str, ...] = field(default_factory=tile_grid_ids)

    @classmethod
    def from_config(cls, cfg: AppConfig) -> ImportCheckParams:
        imp = cfg.import_
        return cls(imp.require_all_tiles, imp.accept_enum_synonyms, MappingProxyType(dict(imp.enum_synonyms)),
                   cfg.export.cvat.max_canopy_interrow_overlap_m2)


@dataclass(frozen=True)
class _Obj:
    layer: str
    object_id: str
    tile_id: str
    geom: BaseGeometry


def _raw(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _objects(annset: AnnSet, layer: str, column: str) -> Iterator[tuple[_Obj, str | None]]:
    gdf = annset.layer(layer)
    for oid, tile, value, geom in zip(gdf[PK_COLUMNS[layer]], gdf["tile_id"], gdf[column], gdf.geometry,
                                      strict=True):
        yield _Obj(layer, str(oid), str(tile), geom), _raw(value)


def _issue(severity: Severity, code: str, obj: _Obj, message: str) -> QaIssue:
    p = obj.geom.representative_point() if obj.geom is not None and not obj.geom.is_empty else None
    return QaIssue(severity, code, obj.tile_id, obj.object_id, message,
                   None if p is None else p.x, None if p is None else p.y)


def missing_attr_issues(annset: AnnSet) -> list[QaIssue]:
    """missing_attr (error): a required attribute is absent or blank."""
    issues = []
    for layer, columns in REQUIRED_ATTRS.items():
        for column in columns:
            issues += [_issue(Severity.ERROR, "missing_attr", obj, f"{obj.object_id}: lipsește atributul {column}")
                       for obj, value in _objects(annset, layer, column) if value is None or not value.strip()]
    return issues


def normalize_enum(raw: str, allowed: frozenset[str], synonyms: Mapping[str, str], accept_synonyms: bool) -> str | None:
    """The allowed value `raw` stands for (strip, lower, spaces and '-' -> '_', synonyms), or None."""
    key = raw.strip().lower()
    if accept_synonyms and key in synonyms and synonyms[key] in allowed:
        return synonyms[key]
    canonical = key.replace(" ", "_").replace("-", "_")
    return canonical if canonical in allowed else None


def enum_issues(annset: AnnSet, params: ImportCheckParams) -> list[QaIssue]:
    """enum_normalized (warning) for fixable spellings, bad_enum (error) for anything else."""
    issues = []
    for (layer, column), allowed in ENUM_ATTRS.items():
        for obj, value in _objects(annset, layer, column):
            if value is None or not value.strip() or value in allowed:
                continue
            fixed = normalize_enum(value, allowed, params.enum_synonyms, params.accept_synonyms)
            if fixed is None:
                issues.append(_issue(Severity.ERROR, "bad_enum", obj,
                                     f"{obj.object_id}: {column}={value!r} nu e o valoare permisă "
                                     f"({', '.join(sorted(allowed))})"))
            else:
                issues.append(_issue(Severity.WARNING, "enum_normalized", obj,
                                     f"{obj.object_id}: {column}={value!r} trebuie scris exact {fixed!r}"))
    return issues


def whitespace_issues(annset: AnnSet) -> list[QaIssue]:
    """id_whitespace (warning): an id with leading/trailing spaces is counted as a different id."""
    return [_issue(Severity.WARNING, "id_whitespace", obj, f"{obj.object_id}: {column}={value!r} are spații")
            for layer, column in ID_ATTRS for obj, value in _objects(annset, layer, column)
            if value is not None and value.strip() and value != value.strip()]


def _collision_groups(values: Iterable[str]) -> list[list[tuple[str, int]]]:
    """Groups of distinct (stripped) ids equal up to letter case, each sorted by (count desc, value)."""
    counts = Counter(values)
    by_fold: dict[str, list[str]] = {}
    for value in counts:
        by_fold.setdefault(value.casefold(), []).append(value)
    groups = [sorted(((v, counts[v]) for v in vals), key=lambda vc: (-vc[1], vc[0]))
              for vals in by_fold.values() if len(vals) > 1]
    return sorted(groups, key=lambda g: g[0][0])


def case_collision_issues(annset: AnnSet) -> list[QaIssue]:
    """id_case_collision (error): `V03` and `v03` in one set are two ids for the organizers."""
    issues = []
    for column in ("vineyard_id", "row_id"):
        pairs = [(obj, value.strip()) for layer, col in ID_ATTRS if col == column
                 for obj, value in _objects(annset, layer, column) if value is not None and value.strip()]
        for group in _collision_groups(v for _, v in pairs):
            major = group[0][0]
            for minor, _ in group[1:]:
                obj = next(o for o, v in pairs if v == minor)
                issues.append(_issue(Severity.ERROR, "id_case_collision", obj,
                                     f"{column}={minor!r} diferă de {major!r} doar prin litere mari/mici"))
    return issues


def images_missing_issues(annset: AnnSet, params: ImportCheckParams) -> list[QaIssue]:
    """images_missing: a Marcaj export must list every tile (error, or warning for --partial)."""
    if annset.meta.source is not Source.MARCAJ or not params.expected_tile_ids:
        return []
    missing = sorted(set(params.expected_tile_ids) - set(annset.meta.tile_ids))
    if not missing:
        return []
    severity = Severity.ERROR if params.require_all_tiles else Severity.WARNING
    listed = ", ".join(missing[:MAX_LISTED]) + (" ..." if len(missing) > MAX_LISTED else "")
    return [QaIssue(severity, "images_missing", missing[0], missing[0],
                    f"Exportul are {len(annset.meta.tile_ids)} din {len(params.expected_tile_ids)} imagini; "
                    f"lipsesc {len(missing)}: {listed}")]


def _overlaps(canopies: gpd.GeoDataFrame, interrows: gpd.GeoDataFrame) -> np.ndarray:
    """Per canopy, the area it shares with the interrow pieces of its own tile."""
    area = np.zeros(len(canopies))
    if not len(canopies) or not len(interrows):
        return area
    geoms = interrows.geometry.to_numpy()
    tiles = interrows["tile_id"].to_numpy(dtype=object)
    left, right = STRtree(geoms).query(canopies.geometry.to_numpy(), predicate="intersects")
    can_tiles = canopies["tile_id"].to_numpy(dtype=object)
    can_geoms = canopies.geometry.to_numpy()
    for i, j in zip(left, right, strict=True):
        if can_tiles[i] == tiles[j]:
            area[i] += can_geoms[i].intersection(geoms[j]).area
    return area


def overlap_issues(annset: AnnSet, params: ImportCheckParams) -> list[QaIssue]:
    """canopy_interrow_overlap (warning): canopy ∩ interrow above max_overlap_m2 in one tile."""
    canopies = annset.canopies.reset_index(drop=True)
    area = _overlaps(canopies, annset.interrow_pieces.reset_index(drop=True))
    objs = [obj for obj, _ in _objects(annset, "canopies", "vineyard_id")]
    return [_issue(Severity.WARNING, "canopy_interrow_overlap", objs[i],
                   f"{objs[i].object_id} se suprapune cu inter-rândul pe {area[i]:.2f} m²")
            for i in np.flatnonzero(area > params.max_overlap_m2)]


def check_annset(annset: AnnSet, params: ImportCheckParams) -> tuple[QaIssue, ...]:
    """Every import-level issue of the AnnSet, sorted by QaIssue.sort_key."""
    issues: Sequence[QaIssue] = (
        missing_attr_issues(annset) + enum_issues(annset, params) + whitespace_issues(annset)
        + case_collision_issues(annset) + images_missing_issues(annset, params) + overlap_issues(annset, params)
    )
    return tuple(sorted(issues, key=lambda i: i.sort_key))
