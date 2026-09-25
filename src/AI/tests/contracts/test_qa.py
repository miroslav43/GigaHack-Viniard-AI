import pytest

from vineyard.contracts.enums import Severity, Source
from vineyard.contracts.qa import QA_CODES, SEVERITY_RANK, QaIssue, issues_to_gdf
from vineyard.contracts.schemas import validate_layer
from vineyard.errors import SchemaError

TILE = "siret3_r021_c012"


def _issues() -> list[QaIssue]:
    return [
        QaIssue(Severity.INFO, "otsu_fallback", TILE, "", "info"),
        QaIssue(Severity.WARNING, "structure_borderline", TILE, "V01-R015@" + TILE, "w1", 10.0, 20.0),
        QaIssue(Severity.ERROR, "tile_failed", "siret3_r006_c004", "", "boom"),
        QaIssue(Severity.WARNING, "bad_enum", TILE, "b", "w2", 1.0, 2.0),
        QaIssue(Severity.WARNING, "bad_enum", TILE, "a", "w3"),
        QaIssue(Severity.ERROR, "canopy_interrow_overlap", TILE, "x", "e2", 5.0, 6.0),
    ]


def test_qa_codes_reexported() -> None:
    assert "tile_failed" in QA_CODES
    assert SEVERITY_RANK[Severity.ERROR] < SEVERITY_RANK[Severity.WARNING] < SEVERITY_RANK[Severity.INFO]


def test_issue_is_frozen_and_normalizes_severity() -> None:
    issue = QaIssue("warning", "bad_enum", TILE, "x", "msg")  # type: ignore[arg-type]
    assert issue.severity is Severity.WARNING
    with pytest.raises(AttributeError):
        issue.code = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"severity": "fatal"},
        {"code": "Bad Code"},
        {"code": ""},
        {"x": 1.0, "y": None},
        {"x": float("nan"), "y": 1.0},
    ],
)
def test_issue_validation(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "severity": Severity.ERROR, "code": "tile_failed", "tile_id": TILE, "object_id": "", "message": "m",
    }
    base.update(kwargs)
    with pytest.raises(SchemaError):
        QaIssue(**base)  # type: ignore[arg-type]


def test_issues_to_gdf_numbering_is_deterministic() -> None:
    issues = _issues()
    a = issues_to_gdf(issues)
    b = issues_to_gdf(list(reversed(issues)))
    assert a.drop(columns="geometry").equals(b.drop(columns="geometry"))
    assert a.geometry.equals(b.geometry)
    assert a["issue_id"].tolist() == [f"Q{i:05d}" for i in range(1, 7)]
    assert a["severity"].tolist() == ["error", "error", "warning", "warning", "warning", "info"]
    assert a["code"].tolist()[:2] == ["canopy_interrow_overlap", "tile_failed"]
    assert a["object_id"].tolist()[2:4] == ["a", "b"]
    assert a.crs.to_epsg() == 32635
    validate_layer(a, "qa_issues")


def test_issues_to_gdf_geometry_and_provenance() -> None:
    gdf = issues_to_gdf(_issues(), source=Source.MARCAJ, run_id="r9", model_version="marcaj-export@x")
    first = gdf.iloc[0]
    assert first["code"] == "canopy_interrow_overlap"
    assert (first.geometry.x, first.geometry.y) == (5.0, 6.0)
    assert gdf.iloc[1].geometry.is_empty
    assert set(gdf["source"]) == {"marcaj"}
    assert set(gdf["run_id"]) == {"r9"}
    assert set(gdf["confidence"]) == {1.0}
    validate_layer(gdf, "qa_issues")


def test_issues_to_gdf_empty() -> None:
    gdf = issues_to_gdf([])
    assert len(gdf) == 0
    validate_layer(gdf, "qa_issues")
