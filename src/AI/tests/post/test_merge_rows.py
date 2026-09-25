"""annset.merge_rows: contract §4.5 steps 1-6 (merge per row_id, qa issues)."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString

from tests.post.factories import (
    BlockSpec,
    build_annset,
    default_config,
    make_annset,
    reference_annset,
    row_piece_record,
)
from vineyard.annset.derive import DeriveParams, build_coverage
from vineyard.annset.merge_rows import MergedRow, merge_rows, structure_any
from vineyard.contracts.enums import Severity

T10, T11, T12 = "siret3_r018_c010", "siret3_r018_c011", "siret3_r018_c012"
X10, X11, X12, X13 = 629504.0, 629555.2, 629606.4, 629657.6
Y = 5220280.0


@pytest.fixture(scope="module")
def params():
    return DeriveParams.from_config(default_config()).merge


def _pieces(*specs: tuple[str, str, str, list[tuple[float, float]]], **kw):
    seen: dict[tuple[str, str], int] = {}
    records = []
    for row_id, vid, tile, coords in specs:
        seen[(row_id, tile)] = seen.get((row_id, tile), 0) + 1
        records.append(row_piece_record(row_id, vid, tile, LineString(coords), dup=seen[(row_id, tile)], **kw))
    return build_annset(row_pieces=records).row_pieces


def _codes(issues) -> list[str]:
    return sorted(i.code for i in issues)


def test_collinear_pieces_merge_into_one_row(params) -> None:
    spec = BlockSpec(n_rows=1, origin_xy=(X11, Y), length_m=102.4)
    rows, issues = merge_rows(make_annset(spec).row_pieces, params, build_coverage([T11, T12]))
    assert issues == ()
    (row,) = rows
    assert isinstance(row, MergedRow)
    assert row.row_id == "V01-R001" and row.vineyard_id == "V01" and row.n_pieces == 2
    assert row.length_m == pytest.approx(102.4) and row.extent_m == pytest.approx(102.4)
    assert row.tile_ids == (T11, T12)
    assert row.angle_deg == pytest.approx(0.0)
    assert row.structure_any == "regular"
    assert [tuple(c) for c in row.line.coords] == [pytest.approx((x, Y)) for x in (X11, X12, X13)]


def test_pieces_drawn_backwards_are_ordered_along_the_axis(params) -> None:
    pieces = _pieces(("V01-R001", "V01", T12, [(X13 - 1, Y), (X12, Y)]),
                     ("V01-R001", "V01", T11, [(X12, Y), (X11 + 1, Y)]))
    (row,), issues = merge_rows(pieces, params)
    assert issues == ()
    assert row.tile_ids == (T11, T12)
    assert row.line.coords[0] == (X11 + 1, Y) and row.line.coords[-1] == (X13 - 1, Y)
    assert row.extent_m == pytest.approx(row.length_m)


@pytest.mark.parametrize(("offset", "expected"), [(0.6, ["row_piece_misaligned"]), (0.3, [])])
def test_lateral_offset_at_junction(params, offset: float, expected: list[str]) -> None:
    pieces = _pieces(("V01-R001", "V01", T11, [(X11 + 1, Y), (X12, Y)]),
                     ("V01-R001", "V01", T12, [(X12, Y + offset), (X13 - 1, Y + offset)]))
    _, issues = merge_rows(pieces, params)
    assert _codes(issues) == expected
    if expected:
        assert issues[0].severity is Severity.WARNING and issues[0].object_id == "V01-R001"


def test_short_piece_uses_row_direction_for_the_junction(params) -> None:
    # A 0.3 m stub at a tile corner is not a reliable support line: extended 10 m it would miss by 1.7 m.
    pieces = _pieces(("V01-R001", "V01", T11, [(X12 - 0.5, Y + 0.05), (X12 - 0.2, Y)]),
                     ("V01-R001", "V01", T12, [(X12 + 10, Y), (X13 - 1, Y)]))
    _, issues = merge_rows(pieces, params)
    assert issues == ()


def test_duplicate_row_in_tile(params) -> None:
    pieces = _pieces(("V01-R001", "V01", T11, [(X11 + 1, Y), (X11 + 20, Y)]),
                     ("V01-R001", "V01", T11, [(X11 + 25, Y), (X12 - 1, Y)]))
    assert sorted(pieces["piece_id"]) == [f"V01-R001@{T11}", f"V01-R001@{T11}#2"]
    (row,), issues = merge_rows(pieces, params)
    assert _codes(issues) == ["dup_row_in_tile"]
    assert issues[0].tile_id == T11
    assert row.n_pieces == 2 and row.length_m == pytest.approx(19.0 + 25.2)
    assert row.extent_m == pytest.approx(49.2)


def test_missing_middle_piece_only_inside_coverage(params) -> None:
    pieces = _pieces(("V01-R001", "V01", T10, [(X10 + 5, Y), (X11, Y)]),
                     ("V01-R001", "V01", T12, [(X12, Y), (X13 - 5, Y)]))
    _, issues = merge_rows(pieces, params, build_coverage([T10, T11, T12]))
    assert _codes(issues) == ["row_missing_in_tile"]
    assert issues[0].tile_id == T11 and issues[0].object_id == "V01-R001"
    assert issues[0].x == pytest.approx((X11 + X12) / 2) and issues[0].y == pytest.approx(Y)
    _, none = merge_rows(pieces, params, build_coverage([T10, T12]))
    assert none == ()
    _, no_cov = merge_rows(pieces, params, None)
    assert no_cov == ()


def test_row_multi_block_takes_the_majority(params) -> None:
    pieces = _pieces(("V01-R001", "V01", T10, [(X10 + 5, Y), (X11, Y)]),
                     ("V01-R001", "V01", T11, [(X11, Y), (X12, Y)]),
                     ("V01-R001", "V02", T12, [(X12, Y), (X13 - 5, Y)]))
    (row,), issues = merge_rows(pieces, params)
    assert _codes(issues) == ["row_multi_block"]
    assert issues[0].severity is Severity.ERROR
    assert row.vineyard_id == "V01" and row.vineyard_ids == ("V01", "V02")


def test_blank_row_ids_are_skipped(params) -> None:
    pieces = _pieces(("V01-R001", "V01", T11, [(X11 + 1, Y), (X12 - 1, Y)]))
    blank = pieces.assign(row_id=[" "])
    rows, issues = merge_rows(blank, params)
    assert rows == () and issues == ()


@pytest.mark.parametrize(("values", "expected"), [
    (["regular", "disrupted"], "disrupted"),
    (["unassessable", "unassessable"], "unassessable"),
    (["regular", "unassessable"], "regular"),
    (["Regular?"], "regular"),
])
def test_structure_any(values: list[str], expected: str) -> None:
    assert structure_any(values) == expected


def test_tile_structures_follow_pieces(params) -> None:
    pieces = _pieces(("V01-R001", "V01", T11, [(X11 + 1, Y), (X12, Y)]),
                     ("V01-R001", "V01", T12, [(X12, Y), (X13 - 1, Y)]))
    pieces = pieces.assign(row_structure=["regular", "disrupted"])
    (row,), _ = merge_rows(pieces, params)
    assert row.structure_any == "disrupted"
    assert row.tile_structures == ((T11, "regular"), (T12, "disrupted"))


@pytest.mark.examples
def test_reference_gives_51_rows(params, examples_xml: bytes) -> None:
    annset = reference_annset(examples_xml)
    rows, issues = merge_rows(annset.row_pieces, params, build_coverage(annset.meta.tile_ids))
    assert len(rows) == 51 and issues == ()
    by_block = {vid: sum(r.length_m for r in rows if r.vineyard_id == vid) for vid in ("V01", "V02")}
    assert by_block["V01"] == pytest.approx(910.10, abs=0.005)
    assert by_block["V02"] == pytest.approx(1031.45, abs=0.005)
    assert sum(r.length_m for r in rows) == pytest.approx(1941.55, abs=0.01)
    assert all(r.n_pieces == 1 and r.extent_m == pytest.approx(r.length_m) for r in rows)
    disrupted = sorted(r.row_id for r in rows if r.structure_any == "disrupted")
    assert len(disrupted) == 5 and all(r.startswith("V02-") for r in disrupted)
