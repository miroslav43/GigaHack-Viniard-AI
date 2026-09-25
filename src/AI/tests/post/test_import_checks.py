"""annset.import_checks: raw-value QA of an imported AnnSet (contract §4.3-4.4, critique X8)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from shapely.geometry import box

from tests.post.factories import (
    BlockSpec,
    Prov,
    build_annset,
    canopy_record,
    default_config,
    interrow_piece_record,
    make_annset,
)
from vineyard.annset.derive import DeriveParams
from vineyard.annset.import_checks import ImportCheckParams, check_annset, normalize_enum
from vineyard.annset.model import AnnSet
from vineyard.contracts.enums import InterrowCover, RowStructure, Severity, Source
from vineyard.contracts.ids import tile_grid_ids

T11 = "siret3_r018_c011"
T12 = "siret3_r018_c012"
X0, Y0 = 629560.0, 5220290.0
SYNONYMS = {"bare soil": "bare_soil", "bare": "bare_soil", "grass": "vegetation", "veg": "vegetation",
            "unknown": "unassessable"}


@pytest.fixture(scope="module")
def params() -> ImportCheckParams:
    return DeriveParams.from_config(default_config()).checks


def _set(annset: AnnSet, layer: str, column: str, index: int, value: object) -> AnnSet:
    gdf = annset.layer(layer)
    values = list(gdf[column])
    values[index] = value
    return annset.with_layer(layer, gdf.assign(**{column: values}))


def _codes(issues) -> list[str]:
    return [i.code for i in issues]


def test_clean_annset_has_no_issues(params) -> None:
    assert check_annset(make_annset(BlockSpec(n_rows=3)), params) == ()


def test_config_params() -> None:
    p = DeriveParams.from_config(default_config()).checks
    assert p.require_all_tiles and p.accept_synonyms and p.max_overlap_m2 == pytest.approx(0.05)
    assert p.enum_synonyms["bare soil"] == "bare_soil"
    assert p.expected_tile_ids == tile_grid_ids() and len(p.expected_tile_ids) == 311


@pytest.mark.parametrize(("layer", "column"), [
    ("row_pieces", "row_id"), ("row_pieces", "row_structure"), ("row_pieces", "vineyard_id"),
    ("canopies", "vineyard_id"), ("interrow_pieces", "interrow_cover"),
])
@pytest.mark.parametrize("blank", ["", "  ", None])
def test_missing_attr(params, layer: str, column: str, blank: object) -> None:
    annset = _set(make_annset(BlockSpec(n_rows=3)), layer, column, 0, blank)
    issues = [i for i in check_annset(annset, params) if i.code == "missing_attr"]
    assert len(issues) == 1 and issues[0].severity is Severity.ERROR
    assert column in issues[0].message
    assert issues[0].x is not None


def test_blank_waste_block_is_allowed(params) -> None:
    annset = make_annset(BlockSpec(n_rows=3), waste=[(T11, (X0 + 5, Y0 + 5), 1.0, "")])
    assert check_annset(annset, params) == ()


@pytest.mark.parametrize(("value", "code"), [
    ("Regular", "enum_normalized"), (" disrupted ", "enum_normalized"), ("regular?", "bad_enum"),
])
def test_row_structure_values(params, value: str, code: str) -> None:
    annset = _set(make_annset(BlockSpec(n_rows=3)), "row_pieces", "row_structure", 1, value)
    issues = check_annset(annset, params)
    assert _codes(issues) == [code]
    assert issues[0].severity is (Severity.ERROR if code == "bad_enum" else Severity.WARNING)
    assert issues[0].object_id == annset.row_pieces["piece_id"].iloc[1]


@pytest.mark.parametrize(("value", "accept", "code"), [
    ("bare soil", True, "enum_normalized"), ("Bare-Soil", False, "enum_normalized"),
    ("grass", True, "enum_normalized"), ("grass", False, "bad_enum"), ("mud", True, "bad_enum"),
])
def test_interrow_cover_values(params, value: str, accept: bool, code: str) -> None:
    annset = _set(make_annset(BlockSpec(n_rows=3)), "interrow_pieces", "interrow_cover", 0, value)
    assert _codes(check_annset(annset, replace(params, accept_synonyms=accept))) == [code]


def test_normalize_enum() -> None:
    covers = frozenset(m.value for m in InterrowCover)
    assert normalize_enum("Bare Soil", covers, SYNONYMS, True) == "bare_soil"
    assert normalize_enum("veg", covers, SYNONYMS, True) == "vegetation"
    assert normalize_enum("veg", covers, SYNONYMS, False) is None
    structures = frozenset(m.value for m in RowStructure)
    assert normalize_enum("UNASSESSABLE", structures, SYNONYMS, True) == "unassessable"
    assert normalize_enum("unknown", structures, SYNONYMS, True) == "unassessable"
    assert normalize_enum("broken", structures, SYNONYMS, True) is None


def test_id_whitespace(params) -> None:
    annset = _set(make_annset(BlockSpec(n_rows=3)), "canopies", "vineyard_id", 2, " V01")
    issues = check_annset(annset, params)
    assert _codes(issues) == ["id_whitespace"]
    assert issues[0].severity is Severity.WARNING and issues[0].object_id == annset.canopies["canopy_id"].iloc[2]


def test_case_collision_points_at_the_minority_value(params) -> None:
    annset = _set(make_annset(BlockSpec(n_rows=3)), "canopies", "vineyard_id", 4, "v01")
    issues = check_annset(annset, params)
    assert _codes(issues) == ["id_case_collision"]
    assert issues[0].severity is Severity.ERROR
    assert issues[0].object_id == annset.canopies["canopy_id"].iloc[4]
    assert "'v01'" in issues[0].message and "'V01'" in issues[0].message


def test_row_id_case_collision(params) -> None:
    annset = _set(make_annset(BlockSpec(n_rows=3)), "row_pieces", "row_id", 0, "v01-r001")
    assert _codes(check_annset(annset, params)) == ["id_case_collision"]


def _marcaj(tile_ids: tuple[str, ...]) -> AnnSet:
    return build_annset(prov=Prov(Source.MARCAJ, "20260926T0000-marcaj-000000", "marcaj-export@00000000"),
                        tile_ids=tile_ids)


def test_images_missing_error_or_warning(params) -> None:
    all_tiles = tile_grid_ids()
    assert check_annset(_marcaj(all_tiles), params) == ()
    partial = _marcaj(all_tiles[1:])
    (issue,) = check_annset(partial, params)
    assert issue.code == "images_missing" and issue.severity is Severity.ERROR
    assert issue.tile_id == all_tiles[0] and "lipsesc 1" in issue.message
    (warn,) = check_annset(partial, replace(params, require_all_tiles=False))
    assert warn.severity is Severity.WARNING


def test_images_missing_lists_at_most_ten(params) -> None:
    (issue,) = check_annset(_marcaj(tile_grid_ids()[20:]), params)
    assert "lipsesc 20" in issue.message and issue.message.endswith("...")


def test_images_missing_only_for_marcaj(params) -> None:
    assert check_annset(build_annset(tile_ids=tile_grid_ids()[:2]), params) == ()


def test_canopy_interrow_overlap(params) -> None:
    canopy = box(X0, Y0, X0 + 1.0, Y0 + 1.0)
    interrow = box(X0 + 0.8, Y0 - 3.0, X0 + 3.0, Y0 + 3.0)  # 0.2 m² inside the canopy
    far = box(X0 + 10.0, Y0, X0 + 11.0, Y0 + 1.0)
    annset = build_annset(
        canopies=[canopy_record(f"{T11}:C0001", T11, "V01", canopy), canopy_record(f"{T11}:C0002", T11, "V01", far)],
        interrow_pieces=[interrow_piece_record(f"{T11}:I001", T11, "V01", interrow)],
    )
    (issue,) = check_annset(annset, params)
    assert issue.code == "canopy_interrow_overlap" and issue.object_id == f"{T11}:C0001"
    assert "0.20 m²" in issue.message
    assert check_annset(annset, replace(params, max_overlap_m2=0.25)) == ()
    other_tile = annset.with_layer("interrow_pieces", annset.interrow_pieces.assign(tile_id=[T12]))
    assert check_annset(other_tile, params) == ()


def test_issues_are_sorted(params) -> None:
    annset = _set(make_annset(BlockSpec(n_rows=3)), "row_pieces", "row_structure", 0, "Regular")
    annset = _set(annset, "canopies", "vineyard_id", 0, "")
    issues = check_annset(annset, params)
    assert _codes(issues) == ["missing_attr", "enum_normalized"]
    assert list(issues) == sorted(issues, key=lambda i: i.sort_key)
