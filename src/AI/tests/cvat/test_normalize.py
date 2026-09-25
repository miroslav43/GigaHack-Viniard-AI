"""cvat.normalize: enum/ID normalization for imported CVAT values (contract §4.4, X8 policy)."""

import pytest

from vineyard.contracts.enums import Severity
from vineyard.cvat.normalize import (
    ENUM_VALUES,
    canonical_enum_text,
    case_collision_issues,
    find_case_collisions,
    normalize_enum,
    normalize_id,
    normalize_id_value,
)
from vineyard.errors import CvatFormatError

SYN = {"bare soil": "bare_soil", "bare": "bare_soil", "grass": "vegetation", "veg": "vegetation",
       "unknown": "unassessable"}


def _norm(attr: str, raw: str | None, accept: bool = True):
    return normalize_enum(attr, raw, synonyms=SYN, accept_synonyms=accept, tile_id="siret3_r021_c012",
                          object_id="obj")


def test_enum_values_come_from_meta() -> None:
    assert ENUM_VALUES["row_structure"] == ("regular", "disrupted", "unassessable")
    assert ENUM_VALUES["interrow_cover"] == ("bare_soil", "vegetation", "mixed", "unassessable")


@pytest.mark.parametrize(("raw", "canon"), [(" Bare-Soil ", "bare_soil"), ("bare  soil", "bare_soil"), ("X", "x")])
def test_canonical_enum_text(raw: str, canon: str) -> None:
    assert canonical_enum_text(raw) == canon


def test_exact_value_has_no_issue() -> None:
    assert _norm("row_structure", "regular") == ("regular", None)


@pytest.mark.parametrize(
    ("attr", "raw", "value"),
    [
        ("row_structure", " Regular ", "regular"),
        ("row_structure", "Regular", "regular"),
        ("interrow_cover", "bare soil", "bare_soil"),
        ("interrow_cover", "bare-soil", "bare_soil"),
        ("interrow_cover", "grass", "vegetation"),
        ("interrow_cover", "Veg", "vegetation"),
        ("row_structure", "unknown", "unassessable"),
    ],
)
def test_changed_values_warn(attr: str, raw: str, value: str) -> None:
    out, issue = _norm(attr, raw)
    assert out == value
    assert issue is not None and issue.severity == Severity.WARNING and issue.code == "enum_normalized"
    assert (issue.tile_id, issue.object_id) == ("siret3_r021_c012", "obj")
    assert repr(raw) in issue.message


def test_synonym_refused_when_not_accepted() -> None:
    out, issue = _norm("interrow_cover", "grass", accept=False)
    assert out == "grass"
    assert issue is not None and (issue.severity, issue.code) == (Severity.ERROR, "bad_enum")


def test_synonym_for_other_attribute_is_bad() -> None:
    out, issue = _norm("row_structure", "grass")
    assert out == "grass" and issue is not None and issue.code == "bad_enum"


def test_unknown_value_keeps_raw() -> None:
    out, issue = _norm("row_structure", "Foo")
    assert out == "Foo"
    assert issue is not None and (issue.severity, issue.code) == (Severity.ERROR, "bad_enum")


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_missing_enum(raw: str | None) -> None:
    out, issue = _norm("interrow_cover", raw)
    assert out == ""
    assert issue is not None and (issue.severity, issue.code) == (Severity.ERROR, "missing_attr")


def test_not_an_enum_attribute() -> None:
    with pytest.raises(CvatFormatError):
        _norm("vineyard_id", "V01")


def test_normalize_id() -> None:
    assert normalize_id(" V01 ") == "V01"
    assert normalize_id(None) == ""
    assert normalize_id("v01") == "v01"  # case is never changed


def test_normalize_id_value_issues() -> None:
    assert normalize_id_value("vineyard_id", "V01", tile_id="t", object_id="o") == ("V01", None)
    value, issue = normalize_id_value("vineyard_id", " V01 ", tile_id="t", object_id="o")
    assert value == "V01" and issue is not None and (issue.severity, issue.code) == (Severity.WARNING, "id_whitespace")
    value, issue = normalize_id_value("row_id", None, tile_id="t", object_id="o")
    assert value == "" and issue is not None and (issue.severity, issue.code) == (Severity.ERROR, "missing_attr")
    assert normalize_id_value("vineyard_id", "", tile_id="t", object_id="o", required=False) == ("", None)


def test_find_case_collisions() -> None:
    assert find_case_collisions(["V03", "v03", "V01", "V03", "V02", "v02"]) == (("V02", "v02"), ("V03", "v03"))
    assert find_case_collisions(["V01", "V02", ""]) == ()


def test_case_collision_issues() -> None:
    issues = case_collision_issues("vineyard_id", ["V03", "v03", "V01"])
    assert len(issues) == 1
    assert (issues[0].severity, issues[0].code, issues[0].object_id) == (Severity.ERROR, "id_case_collision", "V03|v03")
