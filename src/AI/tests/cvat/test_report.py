"""cvat.report: Issue and ValidationReport (immutable, deterministic)."""

from __future__ import annotations

import dataclasses

import pytest

from vineyard.contracts.enums import Severity
from vineyard.cvat.report import (
    Issue,
    ValidationReport,
    error,
    info,
    report_of,
    warning,
)
from vineyard.errors import CvatFormatError, ExportBlocked


def test_issue_normalises_string_severity() -> None:
    issue = Issue("error", "bad_enum", "row_structure='Regular'", tile_id="siret3_r021_c012")
    assert issue.severity is Severity.ERROR
    assert issue.zip_name == ""


def test_issue_rejects_unknown_severity_and_bad_code() -> None:
    with pytest.raises(CvatFormatError):
        Issue("fatal", "bad_enum", "x")
    with pytest.raises(CvatFormatError):
        Issue(Severity.ERROR, "Bad Code", "x")


def test_issue_is_frozen() -> None:
    issue = error("x_code", "msg")
    with pytest.raises(dataclasses.FrozenInstanceError):
        issue.code = "other"  # type: ignore[misc]


def test_helpers_build_the_right_severity() -> None:
    assert error("a", "m").severity is Severity.ERROR
    assert warning("a", "m").severity is Severity.WARNING
    assert info("a", "m", tile_id="t", zip_name="z.zip", object_ref="o").object_ref == "o"


def test_report_ok_only_without_errors() -> None:
    assert ValidationReport().ok
    assert report_of([warning("w", "m"), info("i", "m")]).ok
    assert not report_of([warning("w", "m"), error("e", "m")]).ok


def test_merge_returns_new_report_and_keeps_inputs() -> None:
    a = report_of([error("e1", "m")])
    b = report_of([warning("w1", "m")])
    merged = a.merge(b)
    assert merged is not a and len(merged.issues) == 2
    assert len(a.issues) == 1 and len(b.issues) == 1


def test_with_issues_appends() -> None:
    base = ValidationReport()
    grown = base.with_issues(error("e", "m"))
    assert base.issues == () and len(grown.issues) == 1


def test_counts_and_filters() -> None:
    rep = report_of([error("e", "m"), error("e", "n"), warning("w", "m")])
    assert rep.n_errors == 2 and rep.n_warnings == 1
    assert rep.count_by_code() == {"e": 2, "w": 1}
    assert [i.code for i in rep.errors] == ["e", "e"]
    assert [i.code for i in rep.warnings] == ["w"]
    assert rep.has_code("w") and not rep.has_code("zz")


def test_json_dict_is_sorted_and_complete() -> None:
    rep = report_of([warning("w", "m2", tile_id="b"), error("e", "m1", tile_id="a", zip_name="z.zip")])
    doc = rep.to_json_dict()
    assert doc["ok"] is False
    assert doc["n_errors"] == 1 and doc["n_warnings"] == 1 and doc["n_info"] == 0
    assert doc["issues"][0]["severity"] == "error"
    assert doc["issues"][0]["zip_name"] == "z.zip"
    assert set(doc["issues"][0]) == {"severity", "code", "message", "zip_name", "tile_id", "object_ref"}


def test_raise_if_errors() -> None:
    report_of([warning("w", "m")]).raise_if_errors("validate")
    with pytest.raises(ExportBlocked) as excinfo:
        report_of([error("e", "boom", tile_id="t1")]).raise_if_errors("validate")
    assert "validate" in str(excinfo.value)
    assert excinfo.value.context["n_errors"] == 1


def test_summary_lines_limit() -> None:
    rep = report_of([error(f"e{i}", "m") for i in range(10)])
    lines = rep.summary_lines(limit=3)
    assert len(lines) == 4  # 3 issues + "... 7 more"
    assert "7 more" in lines[-1]
