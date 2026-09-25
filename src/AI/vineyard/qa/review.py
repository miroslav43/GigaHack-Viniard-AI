"""Human review triage: per-tile priority (02 §3.10 + plan S3/S4/S8 codes) and qa/review_queue.csv.

Priority 1 = an error, or an urgent code (failed tile, empty tile to confirm, possible false vineyard,
low SNR with rows, suspect/interpolated rows, overrides, band cuts, tree/orchard removals);
2 = warnings only; 3 = clean. empty_tiles.csv belongs to the CVAT export, not here (plan S9).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from vineyard.contracts.enums import Severity
from vineyard.contracts.qa import QaIssue
from vineyard.pipeline.atomic import atomic_write_text

PRIORITY_URGENT: Final = 1
PRIORITY_WARN: Final = 2
PRIORITY_CLEAN: Final = 3
CODE_SEP: Final = ";"
PREVIEW_DIR: Final = "previews"
PREVIEW_EXT: Final = ".jpg"
URGENT_CODES: Final[frozenset[str]] = frozenset({
    "tile_failed", "empty_tile_confirm", "canopy_on_non_vineyard_tile", "low_snr", "missing_row_suspect",
    "row_interpolated", "override_applied", "override_unmatched", "orchard_rejected", "tree_removed",
    "transverse_band_cut",
})
STATUS_COLUMNS: Final = ("tile_id", "status", "has_vineyard", "n_canopies", "n_row_pieces", "n_interrow_pieces",
                         "n_waste", "review_priority", "issues")
ISSUE_COLUMNS: Final = ("severity", "code", "tile_id", "object_id", "message")
QUEUE_COLUMNS: Final = ("review_priority", "tile_id", "status", "has_vineyard", "n_canopies", "n_row_pieces",
                        "n_interrow_pieces", "n_waste", "n_errors", "n_warnings", "issues", "preview")


@dataclass(frozen=True)
class TileIssueSummary:
    n_errors: int
    n_warnings: int
    codes: tuple[str, ...]  # sorted, unique
    priority: int


def review_priority(issues: Iterable[QaIssue]) -> int:
    """1 for errors or urgent codes, 2 for warnings only, 3 otherwise (info issues are not triage)."""
    items = list(issues)
    if any(i.severity == Severity.ERROR or i.code in URGENT_CODES for i in items):
        return PRIORITY_URGENT
    return PRIORITY_WARN if any(i.severity == Severity.WARNING for i in items) else PRIORITY_CLEAN


def summarize_issues(issues: Iterable[QaIssue]) -> Mapping[str, TileIssueSummary]:
    """Per tile_id (issues without a tile are ignored), keys sorted."""
    by_tile: dict[str, list[QaIssue]] = {}
    for issue in issues:
        if issue.tile_id:
            by_tile.setdefault(issue.tile_id, []).append(issue)
    return MappingProxyType({
        tile_id: TileIssueSummary(
            n_errors=sum(i.severity == Severity.ERROR for i in items),
            n_warnings=sum(i.severity == Severity.WARNING for i in items),
            codes=tuple(sorted({i.code for i in items})),
            priority=review_priority(items),
        )
        for tile_id, items in sorted(by_tile.items())
    })


def _location(geom: object) -> tuple[float | None, float | None]:
    if isinstance(geom, Point) and not geom.is_empty:
        return float(geom.x), float(geom.y)
    return None, None


def issues_from_frame(qa_issues: gpd.GeoDataFrame) -> tuple[QaIssue, ...]:
    """QaIssue objects (with their point location) of a `qa_issues` layer read back from disk."""
    missing = [c for c in ISSUE_COLUMNS if c not in qa_issues.columns]
    if missing:
        raise ValueError(f"qa_issues lacks columns {missing}")
    columns = [qa_issues[c].fillna("").astype(str) for c in ISSUE_COLUMNS]
    geoms = list(qa_issues.geometry) if isinstance(qa_issues, gpd.GeoDataFrame) else [None] * len(qa_issues)
    return tuple(QaIssue(Severity(sev), code, tile_id, object_id, message, *_location(geom))
                 for sev, code, tile_id, object_id, message, geom in zip(*columns, geoms, strict=True))


def summaries_from_frame(qa_issues: pd.DataFrame) -> Mapping[str, TileIssueSummary]:
    """summarize_issues over a `qa_issues` layer read back from disk."""
    return summarize_issues(issues_from_frame(qa_issues))


def preview_relpath(tile_id: str) -> str:
    """Preview path relative to runs/<id>/qa/."""
    return f"{PREVIEW_DIR}/{tile_id}{PREVIEW_EXT}"


def review_queue(tile_status: pd.DataFrame, summaries: Mapping[str, TileIssueSummary]) -> pd.DataFrame:
    """One row per tile, sorted by (review_priority, tile_id); QUEUE_COLUMNS in order."""
    missing = [c for c in STATUS_COLUMNS if c not in tile_status.columns]
    if missing:
        raise ValueError(f"tile_status lacks columns {missing}")
    base = pd.DataFrame({c: tile_status[c].to_numpy() for c in STATUS_COLUMNS})
    ids = [str(t) for t in base["tile_id"]]
    empty = TileIssueSummary(0, 0, (), PRIORITY_CLEAN)
    queue = base.assign(
        n_errors=[summaries.get(t, empty).n_errors for t in ids],
        n_warnings=[summaries.get(t, empty).n_warnings for t in ids],
        preview=[preview_relpath(t) for t in ids],
    )
    ordered = queue.sort_values(["review_priority", "tile_id"], kind="mergesort").reset_index(drop=True)
    return ordered[list(QUEUE_COLUMNS)]


def write_review_queue(queue: pd.DataFrame, path: Path) -> Path:
    """Deterministic UTF-8 CSV (LF line ends), written atomically."""
    return atomic_write_text(Path(path), queue.to_csv(index=False, lineterminator="\n"))
