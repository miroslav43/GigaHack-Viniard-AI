"""Validation issues of the CVAT export path (01 §2.5) and the export's report files (01 §2.7).

Issue/ValidationReport are immutable with a deterministic order; the file helpers build the
upload manifest, empty_tiles.csv and id_registry.json rows.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from vineyard.annset.model import ANNSET_LAYERS
from vineyard.contracts.enums import Label, Severity
from vineyard.contracts.ids import interrow_index_of, row_index_of
from vineyard.cvat.writer import serialize_image
from vineyard.errors import CvatFormatError, ExportBlocked
from vineyard.pipeline.atomic import atomic_write_text

if TYPE_CHECKING:
    import geopandas as gpd

    from vineyard.annset.model import AnnSet
    from vineyard.cvat.model import CvatDocument

_CODE_RE: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_SEVERITY_RANK: Final = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
DEFAULT_SUMMARY_LIMIT: Final = 20


@dataclass(frozen=True)
class Issue:
    """One finding. `object_ref` names the object (AnnSet id, shape index, ZIP entry)."""

    severity: Severity
    code: str
    message: str
    zip_name: str = ""
    tile_id: str = ""
    object_ref: str = ""

    def __post_init__(self) -> None:
        try:
            severity = Severity(self.severity)
        except ValueError:
            raise CvatFormatError("invalid issue severity", severity=self.severity, code=self.code) from None
        if not isinstance(self.code, str) or not _CODE_RE.fullmatch(self.code):
            raise CvatFormatError("issue code must be snake_case", code=self.code)
        # Frozen dataclass: normalise a plain-string severity once, at construction.
        object.__setattr__(self, "severity", severity)

    @property
    def sort_key(self) -> tuple[int, str, str, str, str, str]:
        return (_SEVERITY_RANK[self.severity], self.zip_name, self.tile_id, self.code, self.object_ref,
                self.message)

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "zip_name": self.zip_name,
            "tile_id": self.tile_id,
            "object_ref": self.object_ref,
        }

    def __str__(self) -> str:
        where = " ".join(part for part in (self.zip_name, self.tile_id, self.object_ref) if part)
        return f"[{self.severity.value}] {self.code}: {self.message}" + (f" ({where})" if where else "")


def _make(severity: Severity, code: str, message: str, **where: str) -> Issue:
    return Issue(severity, code, message, **where)


def error(code: str, message: str, **where: str) -> Issue:
    return _make(Severity.ERROR, code, message, **where)


def warning(code: str, message: str, **where: str) -> Issue:
    return _make(Severity.WARNING, code, message, **where)


def info(code: str, message: str, **where: str) -> Issue:
    return _make(Severity.INFO, code, message, **where)


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[Issue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def ok(self) -> bool:
        return self.n_errors == 0

    def _of(self, severity: Severity) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity is severity)

    @property
    def errors(self) -> tuple[Issue, ...]:
        return self._of(Severity.ERROR)

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return self._of(Severity.WARNING)

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    @property
    def n_warnings(self) -> int:
        return len(self.warnings)

    def merge(self, other: ValidationReport) -> ValidationReport:
        return ValidationReport(self.issues + other.issues)

    def with_issues(self, *issues: Issue) -> ValidationReport:
        return ValidationReport(self.issues + tuple(issues))

    def has_code(self, code: str) -> bool:
        return any(i.code == code for i in self.issues)

    def count_by_code(self) -> dict[str, int]:
        return dict(sorted(Counter(i.code for i in self.issues).items()))

    def sorted_issues(self) -> tuple[Issue, ...]:
        return tuple(sorted(self.issues, key=lambda i: i.sort_key))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "n_info": len(self._of(Severity.INFO)),
            "by_code": self.count_by_code(),
            "issues": [i.to_dict() for i in self.sorted_issues()],
        }

    def summary_lines(self, *, limit: int = DEFAULT_SUMMARY_LIMIT) -> list[str]:
        ordered = self.sorted_issues()
        lines = [str(i) for i in ordered[:limit]]
        if len(ordered) > limit:
            lines.append(f"... {len(ordered) - limit} more")
        return lines

    def raise_if_errors(self, what: str) -> None:
        """ExportBlocked with the first errors when the report is not ok."""
        if self.ok:
            return
        raise ExportBlocked(f"{what}: {self.n_errors} error(s)", n_errors=self.n_errors,
                            first="; ".join(str(i) for i in self.errors[:3]))


def report_of(issues: Iterable[Issue]) -> ValidationReport:
    return ValidationReport(tuple(issues))


# ---------------------------------------------------------------- export report files (01 §2.7)

MANIFEST_CSV: Final = "upload_manifest.csv"
EMPTY_TILES_CSV: Final = "empty_tiles.csv"
ID_REGISTRY_JSON: Final = "id_registry.json"
VALIDATION_JSON: Final = "validation_report.json"
SUMMARY_JSON: Final = "upload_summary.json"
MANIFEST_COLUMNS: Final = ("zip_name", "image_id", "tile_id", "sha256", "n_vineyard", "n_row", "n_interrow_area",
                           "n_waste", "xml_bytes")
EMPTY_TILES_COLUMNS: Final = ("tile_id", "zip_name", "image_id", "reason")
REASON_NO_OBJECTS: Final = "no_objects"
STATUS_OK: Final = "ok"
_IMAGE_EXT: Final = ".tif"
_HASH_CHUNK: Final = 1 << 20


@dataclass(frozen=True)
class PartInfo:
    zip_name: str
    n_tiles: int
    size: int
    first_tile: str
    last_tile: str
    sha256: str

    @classmethod
    def of(cls, path: Path, tile_ids: Sequence[str], size: int) -> PartInfo:
        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
                digest.update(chunk)
        return cls(Path(path).name, len(tile_ids), size, tile_ids[0], tile_ids[-1], digest.hexdigest())


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    """UTF-8 CSV with a header, '\\n' line ends, columns in the given order (atomic)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return atomic_write_text(path, buf.getvalue())


def _tile_of(name: str) -> str:
    return name[: -len(_IMAGE_EXT)]


def manifest_rows(docs: Sequence[tuple[str, CvatDocument]], shas: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = []
    for zip_name, doc in docs:
        for img in doc.images:
            counts = {f"n_{label.value}": img.count(label.value) for label in Label}
            rows.append({"zip_name": zip_name, "image_id": img.id, "tile_id": _tile_of(img.name),
                         "sha256": shas[_tile_of(img.name)], **counts,
                         "xml_bytes": len(serialize_image(img).encode("utf-8"))})
    return rows


def _status_reasons(tile_status: gpd.GeoDataFrame | None) -> dict[str, str]:
    if tile_status is None or tile_status.empty:
        return {}
    return {str(t): str(s) for t, s in zip(tile_status["tile_id"], tile_status["status"], strict=True)}


def empty_tile_rows(docs: Sequence[tuple[str, CvatDocument]], reasons: Mapping[str, str],
                    tile_status: gpd.GeoDataFrame | None) -> list[dict[str, Any]]:
    """Every image without shapes; reason = export failure, else tile_status, else no_objects."""
    status = _status_reasons(tile_status)
    rows = []
    for zip_name, doc in docs:
        for img in doc.images:
            if img.shapes:
                continue
            tile_id = _tile_of(img.name)
            from_status = status.get(tile_id, REASON_NO_OBJECTS)
            reason = reasons.get(tile_id, REASON_NO_OBJECTS if from_status == STATUS_OK else from_status)
            rows.append({"tile_id": tile_id, "zip_name": zip_name, "image_id": img.id, "reason": reason})
    return sorted(rows, key=lambda r: r["tile_id"])


def _block_number(vid: str) -> int:
    return int(vid[1:]) if vid[1:].isdigit() else -1


def _max_index(values: Iterable[Any], parse: Any) -> int | None:
    found = [n for n in (parse(v) for v in values if isinstance(v, str)) if n is not None]
    return max(found) if found else None


def id_registry(annset: AnnSet, contract_version: str, run_id: str, manual_row_start: int) -> dict[str, Any]:
    """Last V used and last R / I per block, so rows and blocks added after Publish get fresh numbers."""
    vids = sorted({v for name in ANNSET_LAYERS for v in annset.layer(name)["vineyard_id"].dropna()
                   if isinstance(v, str) and v.strip()}, key=lambda v: (_block_number(v), v))
    rows, irs = annset.row_pieces, annset.interrow_pieces
    blocks = {
        vid: {"last_row": _max_index(rows.loc[rows["vineyard_id"] == vid, "row_id"], row_index_of),
              "last_interrow": _max_index(irs.loc[irs["vineyard_id"] == vid, "interrow_id"], interrow_index_of)}
        for vid in vids
    }
    return {"contract_version": contract_version, "run_id": run_id, "annset_run_id": annset.meta.run_id,
            "last_vineyard": vids[-1] if vids else None, "blocks": blocks, "manual_row_start": manual_row_start}


def label_totals(docs: Sequence[tuple[str, CvatDocument]]) -> dict[str, int]:
    return {label.value: sum(img.count(label.value) for _, doc in docs for img in doc.images) for label in Label}
