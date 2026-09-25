"""Review priority rules and review_queue.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vineyard.contracts.enums import Severity
from vineyard.contracts.qa import QaIssue
from vineyard.qa import review

A, B, C, D = "siret3_r001_c001", "siret3_r001_c002", "siret3_r001_c003", "siret3_r001_c004"


def _issue(sev: Severity, code: str, tile: str) -> QaIssue:
    return QaIssue(sev, code, tile, "", "m")


def test_priority_rules() -> None:
    assert review.review_priority([]) == review.PRIORITY_CLEAN
    assert review.review_priority([_issue(Severity.INFO, "cover_borderline", A)]) == review.PRIORITY_CLEAN
    assert review.review_priority([_issue(Severity.WARNING, "structure_borderline", A)]) == review.PRIORITY_WARN
    assert review.review_priority([_issue(Severity.ERROR, "row_multi_block", A)]) == review.PRIORITY_URGENT
    for code in ("empty_tile_confirm", "canopy_on_non_vineyard_tile", "low_snr", "missing_row_suspect",
                 "row_interpolated", "override_applied", "tile_failed"):
        assert review.review_priority([_issue(Severity.INFO, code, A)]) == review.PRIORITY_URGENT, code


def test_summaries_group_by_tile() -> None:
    issues = [_issue(Severity.WARNING, "structure_borderline", A), _issue(Severity.ERROR, "tile_failed", A),
              _issue(Severity.WARNING, "structure_borderline", A), _issue(Severity.INFO, "cover_borderline", B)]
    summary = review.summarize_issues(issues)
    assert summary[A].n_errors == 1 and summary[A].n_warnings == 2
    assert summary[A].codes == ("structure_borderline", "tile_failed")
    assert summary[A].priority == review.PRIORITY_URGENT
    assert summary[B].codes == ("cover_borderline",) and summary[B].priority == review.PRIORITY_CLEAN


def _status() -> pd.DataFrame:
    return pd.DataFrame({
        "tile_id": [D, C, B, A], "status": ["ok", "no_vineyard", "ok", "ok"], "has_vineyard": [True, False, True, True],
        "n_canopies": [5, 0, 3, 1], "n_row_pieces": [2, 0, 1, 1], "n_interrow_pieces": [1, 0, 0, 0],
        "n_waste": [0, 0, 0, 0], "review_priority": [3, 1, 2, 1], "issues": ["", "empty_tile_confirm", "x", "y"],
    })


def test_review_queue_order_and_columns(tmp_path: Path) -> None:
    queue = review.review_queue(_status(), review.summarize_issues([_issue(Severity.ERROR, "y", A)]))
    assert list(queue.columns) == list(review.QUEUE_COLUMNS)
    assert list(queue["tile_id"]) == [A, C, B, D]
    assert list(queue["n_errors"]) == [1, 0, 0, 0]
    assert queue["preview"].iloc[0] == f"previews/{A}.jpg"
    path = review.write_review_queue(queue, tmp_path / "qa" / "review_queue.csv")
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(review.QUEUE_COLUMNS)
    assert review.write_review_queue(queue, tmp_path / "again.csv").read_bytes() == path.read_bytes()


def test_review_queue_empty() -> None:
    queue = review.review_queue(_status().iloc[:0], {})
    assert list(queue.columns) == list(review.QUEUE_COLUMNS) and len(queue) == 0
