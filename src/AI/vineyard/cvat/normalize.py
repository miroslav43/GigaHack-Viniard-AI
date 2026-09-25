"""Normalization of imported CVAT attribute values (contract §4.4 with the X8 policy).

Organizers score the RAW Marcaj values, so every change we make to a raw value emits a qa warning
pointing at the object (fix it in Marcaj). Unknown enum values keep the raw text and are errors.
IDs are only stripped: case is never changed and nothing is renumbered.
"""

import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final

from vineyard.config.sections_io import ImportConfig
from vineyard.contracts.enums import LABEL_ATTRIBUTES, Label, Severity
from vineyard.contracts.qa import QaIssue
from vineyard.cvat.template import LABEL_SPECS
from vineyard.errors import CvatFormatError

VINEYARD_ID: Final = "vineyard_id"
ROW_ID: Final = "row_id"
CODE_NORMALIZED: Final = "enum_normalized"
CODE_BAD_ENUM: Final = "bad_enum"
CODE_MISSING: Final = "missing_attr"
CODE_WHITESPACE: Final = "id_whitespace"
CODE_CASE: Final = "id_case_collision"
COLLISION_SEP: Final = "|"
_SEPARATORS_RE: Final = re.compile(r"[\s\-]+")


def _enum_values() -> Mapping[str, tuple[str, ...]]:
    table: dict[str, tuple[str, ...]] = {}
    for spec in LABEL_SPECS.values():
        for attr in spec.attributes:
            if attr.input_type == "select":
                table[attr.name] = attr.values
    return MappingProxyType(table)


ENUM_VALUES: Final[Mapping[str, tuple[str, ...]]] = _enum_values()


def canonical_enum_text(raw: str) -> str:
    """strip, lower, runs of whitespace/'-' -> '_'."""
    return _SEPARATORS_RE.sub("_", raw.strip().lower())


def _issue(severity: Severity, code: str, tile_id: str, object_id: str, message: str) -> QaIssue:
    return QaIssue(severity, code, tile_id, object_id, message)


def _allowed(attr: str) -> tuple[str, ...]:
    if attr not in ENUM_VALUES:
        raise CvatFormatError("not an enum attribute", attribute=attr, known=sorted(ENUM_VALUES))
    return ENUM_VALUES[attr]


def _synonym(canon: str, synonyms: Mapping[str, str]) -> str | None:
    table = {canonical_enum_text(k): v for k, v in synonyms.items()}
    return table.get(canon)


def normalize_enum(attr: str, raw: str | None, *, synonyms: Mapping[str, str], accept_synonyms: bool,
                   tile_id: str = "", object_id: str = "") -> tuple[str, QaIssue | None]:
    """(value, issue). Exact -> no issue; normalized/synonym -> warning; unknown -> raw kept + error."""
    allowed = _allowed(attr)
    if raw is None or not raw.strip():
        return "", _issue(Severity.ERROR, CODE_MISSING, tile_id, object_id, f"{attr} is missing")
    canon = canonical_enum_text(raw)
    if canon in allowed:
        if canon == raw:
            return canon, None
        return canon, _issue(Severity.WARNING, CODE_NORMALIZED, tile_id, object_id,
                             f"{attr} {raw!r} read as {canon!r}; fix the value in Marcaj")
    target = _synonym(canon, synonyms)
    if target in allowed and accept_synonyms:
        return target, _issue(Severity.WARNING, CODE_NORMALIZED, tile_id, object_id,
                              f"{attr} synonym {raw!r} read as {target!r}; fix the value in Marcaj")
    return raw, _issue(Severity.ERROR, CODE_BAD_ENUM, tile_id, object_id,
                       f"{attr} {raw!r} is not one of {', '.join(allowed)}")


def normalize_id(raw: str | None) -> str:
    """IDs are stripped only (case kept); None -> ''."""
    return "" if raw is None else raw.strip()


def normalize_id_value(attr: str, raw: str | None, *, tile_id: str = "", object_id: str = "",
                       required: bool = True) -> tuple[str, QaIssue | None]:
    """(stripped value, issue): missing & required -> error; whitespace removed -> warning."""
    value = normalize_id(raw)
    if not value:
        if required:
            return "", _issue(Severity.ERROR, CODE_MISSING, tile_id, object_id, f"{attr} is missing")
        return "", None
    if value != raw:
        return value, _issue(Severity.WARNING, CODE_WHITESPACE, tile_id, object_id,
                             f"{attr} {raw!r} has surrounding whitespace; fix it in Marcaj")
    return value, None


def _normalize_one(label: Label, attr: str, raw: str | None, cfg: ImportConfig, tile_id: str,
                   object_id: str) -> tuple[str, QaIssue | None]:
    if attr in ENUM_VALUES:
        return normalize_enum(attr, raw, synonyms=cfg.enum_synonyms, accept_synonyms=cfg.accept_enum_synonyms,
                              tile_id=tile_id, object_id=object_id)
    required = not (label == Label.WASTE and attr == VINEYARD_ID)  # blank waste vineyard_id is legal
    return normalize_id_value(attr, raw, tile_id=tile_id, object_id=object_id, required=required)


def normalize_shape_attributes(label: Label, attributes: Iterable[tuple[str, str]], *, cfg: ImportConfig,
                               tile_id: str = "", object_id: str = "",
                               ) -> tuple[Mapping[str, str], tuple[QaIssue, ...], bool]:
    """Normalized contract attributes of one shape (first occurrence of each name wins).

    Returns (values, issues, usable); a row without row_id is unusable (no piece id can be built).
    """
    raw = {}
    for name, value in attributes:
        raw.setdefault(name, value)
    values: dict[str, str] = {}
    issues: list[QaIssue] = []
    for attr in LABEL_ATTRIBUTES[label]:
        values[attr], issue = _normalize_one(label, attr, raw.get(attr), cfg, tile_id, object_id)
        if issue is not None:
            issues.append(issue)
    usable = not (label == Label.ROW and not values[ROW_ID])
    return MappingProxyType(values), tuple(issues), usable


def find_case_collisions(values: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    """Groups of distinct non-empty values equal up to case (e.g. V03/v03), sorted."""
    groups: dict[str, set[str]] = {}
    for value in values:
        if value:
            groups.setdefault(value.casefold(), set()).add(value)
    return tuple(sorted(tuple(sorted(g)) for g in groups.values() if len(g) > 1))


def case_collision_issues(attr: str, values: Iterable[str]) -> tuple[QaIssue, ...]:
    return tuple(
        _issue(Severity.ERROR, CODE_CASE, "", COLLISION_SEP.join(group),
               f"{attr} values differ only by case: {', '.join(group)}")
        for group in find_case_collisions(values)
    )
