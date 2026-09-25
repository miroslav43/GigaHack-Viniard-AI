"""measurements.csv exactly as the web data contract (src/Web/CLAUDE.md §6.4): writer, reader, checker.

UTF-8, comma, decimal point, header; m and m² at `m_decimals`, ha at `ha_decimals` (from the exact m²);
cells that do not apply to a level are empty. Levels: one `survey` row, then `block`, then `row` rows.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

HEADER: Final[tuple[str, ...]] = (
    "level", "vineyard_id", "row_id", "block_count", "row_count", "row_length_m", "canopy_area_m2",
    "canopy_area_ha", "interrow_area_m2", "interrow_area_ha", "plant_count", "row_structure",
)
HEADER_LINE: Final = ",".join(HEADER)
LEVEL_SURVEY: Final = "survey"
LEVEL_BLOCK: Final = "block"
LEVEL_ROW: Final = "row"
LEVELS: Final = (LEVEL_SURVEY, LEVEL_BLOCK, LEVEL_ROW)
M2_PER_HA: Final = 10_000.0
HA_OF: Final[Mapping[str, str]] = {"canopy_area_ha": "canopy_area_m2", "interrow_area_ha": "interrow_area_m2"}
METRE_COLUMNS: Final = frozenset({"row_length_m", "canopy_area_m2", "interrow_area_m2"})
INT_COLUMNS: Final = frozenset({"block_count", "row_count", "plant_count"})
NUMERIC_COLUMNS: Final = METRE_COLUMNS | INT_COLUMNS | frozenset(HA_OF)
# A step is a power of ten when log10 is within this of an integer (0.01 -> 2 decimals).
_POW10_TOL: Final = 1e-9


@dataclass(frozen=True)
class MeasurementRecord:
    """One CSV line with exact (unrounded) values; None = empty cell."""

    level: str
    vineyard_id: str | None = None
    row_id: str | None = None
    block_count: int | None = None
    row_count: int | None = None
    row_length_m: float | None = None
    canopy_area_m2: float | None = None
    interrow_area_m2: float | None = None
    plant_count: int | None = None
    row_structure: str | None = None

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"measurement level must be one of {LEVELS}, got {self.level!r}")

    def value(self, column: str) -> float | int | str | None:
        if column in HA_OF:
            area = getattr(self, HA_OF[column])
            return None if area is None else area / M2_PER_HA
        return getattr(self, column)


@dataclass(frozen=True)
class CsvCheck:
    name: str
    ok: bool
    detail: str = ""


def decimals_for(step: float) -> int:
    """Number of decimals of a rounding step that is a power of ten (0.01 -> 2, 0.0001 -> 4, 1 -> 0)."""
    if not (math.isfinite(step) and 0.0 < step <= 1.0):
        raise ValueError(f"rounding step must be in (0, 1], got {step}")
    exponent = -math.log10(step)
    if abs(exponent - round(exponent)) > _POW10_TOL:
        raise ValueError(f"rounding step must be a power of ten, got {step}")
    return int(round(exponent))


def cell(record: MeasurementRecord, column: str, m_decimals: int, ha_decimals: int) -> str:
    value = record.value(column)
    if value is None:
        return ""
    if column in HA_OF:
        return f"{float(value):.{ha_decimals}f}"
    if column in METRE_COLUMNS:
        return f"{float(value):.{m_decimals}f}"
    return str(value)


def format_measurements_csv(records: Iterable[MeasurementRecord], *, m_decimals: int, ha_decimals: int) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    for record in records:
        writer.writerow([cell(record, col, m_decimals, ha_decimals) for col in HEADER])
    return buffer.getvalue()


def parse_measurements_csv(text: str) -> tuple[dict[str, str], ...]:
    """Data lines as dicts keyed by the header (no conversion)."""
    return tuple(csv.DictReader(io.StringIO(text)))


# ------------------------------------------------------------------ checks (publish)


def _cells_ok(rows: Sequence[Mapping[str | None, str | None]]) -> CsvCheck:
    """Every data line has exactly len(HEADER) cells (DictReader: None key = extra cells, None value = missing)."""
    bad = [f"line {k}" for k, row in enumerate(rows, start=2)
           if None in row or any(v is None for v in row.values())]
    return CsvCheck("cells", not bad, "; ".join(bad[:5]))


def _numbers_ok(rows: Sequence[Mapping[str, str]]) -> CsvCheck:
    bad = []
    for k, row in enumerate(rows, start=2):
        for col in NUMERIC_COLUMNS:
            text = row.get(col) or ""
            try:
                if text and not math.isfinite(float(text)):
                    bad.append(f"line {k} {col}")
            except ValueError:
                bad.append(f"line {k} {col}")
    return CsvCheck("numbers", not bad, "; ".join(bad[:5]))


def _lengths(rows: Sequence[Mapping[str, str]], level: str) -> list[str]:
    return [r.get("row_length_m") or "" for r in rows if r.get("level") == level]


def _half_unit(text: str) -> float:
    """Worst rounding error of a written value: half a unit of its last decimal."""
    decimals = len(text.split(".", 1)[1]) if "." in text else 0
    return 0.5 * 10.0 ** (-decimals)


def _sum_checks(rows: Sequence[Mapping[str, str]], sum_tol_m: float) -> list[CsvCheck]:
    (survey_text,) = _lengths(rows, LEVEL_SURVEY)
    survey = float(survey_text or 0.0)
    checks = []
    for level, name in ((LEVEL_BLOCK, "block_sum"), (LEVEL_ROW, "row_sum")):
        values = _lengths(rows, level)
        total = sum(float(v or 0.0) for v in values)
        # Every written value (and the survey one) may be off by half a unit: allow that on top.
        tol = sum_tol_m + sum(_half_unit(v) for v in (*values, survey_text))
        checks.append(CsvCheck(name, abs(total - survey) <= tol,
                               f"sum({level}.row_length_m)={total:.2f} survey={survey:.2f} tol={tol:.3f}"))
    return checks


def check_measurements_csv(text: str, *, sum_tol_m: float) -> tuple[CsvCheck, ...]:
    """Failed checks only (empty = the file may be published)."""
    first = text.splitlines()[0] if text else ""
    if first != HEADER_LINE:
        return (CsvCheck("header", False, f"got {first[:120]!r}"),)
    rows = parse_measurements_csv(text)
    levels = [r.get("level") for r in rows]
    survey = CsvCheck("survey_rows", levels.count(LEVEL_SURVEY) == 1, f"n={levels.count(LEVEL_SURVEY)}")
    cells = _cells_ok(rows)
    checks = [CsvCheck("levels", set(levels) <= set(LEVELS), f"levels={sorted(set(map(str, levels)))}"),
              survey, cells]
    if cells.ok:
        numbers = _numbers_ok(rows)
        checks.append(numbers)
        if numbers.ok and survey.ok:
            checks += _sum_checks(rows, sum_tol_m)
    return tuple(c for c in checks if not c.ok)


def check_measurements_bytes(data: bytes, *, sum_tol_m: float) -> tuple[CsvCheck, ...]:
    """check_measurements_csv on raw file bytes; a non-UTF-8 file fails the `utf8` check."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return (CsvCheck("utf8", False, str(exc)),)
    return check_measurements_csv(text, sum_tol_m=sum_tol_m)


def describe_failed(failed: Sequence[CsvCheck]) -> str:
    return "; ".join(f"{c.name}: {c.detail}" for c in failed)


__all__ = [
    "HEADER", "HEADER_LINE", "LEVELS", "LEVEL_BLOCK", "LEVEL_ROW", "LEVEL_SURVEY", "M2_PER_HA", "CsvCheck",
    "MeasurementRecord", "cell", "check_measurements_bytes", "check_measurements_csv", "decimals_for",
    "describe_failed", "format_measurements_csv", "parse_measurements_csv",
]
