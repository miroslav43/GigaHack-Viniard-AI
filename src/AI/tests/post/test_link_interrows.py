"""annset.link_interrows: contract §4.5.7 with geometric interrow numbering (critique C4)."""

from __future__ import annotations

import math

import pytest

from tests.post.factories import BlockSpec, default_config, make_annset, reference_annset
from vineyard.annset.derive import DeriveParams, build_coverage
from vineyard.annset.link_interrows import LinkParams, LinkResult, interrow_numbers, link_interrows
from vineyard.annset.merge_rows import merge_rows
from vineyard.annset.model import AnnSet
from vineyard.annset.row_order import block_axes, order_rows
from vineyard.contracts.enums import Severity

T11, T12 = "siret3_r018_c011", "siret3_r018_c012"


@pytest.fixture(scope="module")
def params() -> DeriveParams:
    return DeriveParams.from_config(default_config())


def _link(annset: AnnSet, params: DeriveParams, link: LinkParams | None = None) -> LinkResult:
    rows, _ = merge_rows(annset.row_pieces, params.merge)
    return link_interrows(annset.interrow_pieces, annset.row_pieces, order_rows(rows), block_axes(rows),
                          link or params.link)


def _rename_rows(annset: AnnSet, mapping: dict[str, str]) -> AnnSet:
    pieces = annset.row_pieces
    return annset.with_layer("row_pieces", pieces.assign(row_id=pieces["row_id"].map(lambda r: mapping.get(r, r))))


def test_synthetic_block_links_every_piece(params) -> None:
    result = _link(make_annset(BlockSpec(n_rows=4)), params)
    linked = result.pieces
    assert len(linked) == 6 and result.issues == ()
    assert sorted(set(linked["interrow_id"])) == ["V01-I001", "V01-I002", "V01-I003"]
    for k in (1, 2, 3):
        rows = linked[linked["interrow_id"] == f"V01-I{k:03d}"]
        assert set(rows["row_left_id"]) == {f"V01-R{k:03d}"}
        assert set(rows["row_right_id"]) == {f"V01-R{k + 1:03d}"}
        assert set(rows["tile_id"]) == {T11, T12}
    assert set(linked["link_method"]) == {"tile"}
    # band edges are 0.3 m inside each axis: centroid offsets are half the spacing
    assert linked["left_offset_m"].to_numpy() == pytest.approx([1.25] * 6)
    assert linked["width_mean_m"].to_numpy() == pytest.approx([1.9] * 6)


def test_global_interrows_union_the_pieces(params) -> None:
    result = _link(make_annset(BlockSpec(n_rows=3, length_m=60.0)), params)
    assert [r["interrow_id"] for r in result.interrows] == ["V01-I001", "V01-I002"]
    first = result.interrows[0]
    assert first["n_pieces"] == 2 and first["tile_ids"] == f"{T11},{T12}"
    assert first["geometry"].geom_type == "Polygon"
    assert first["area_m2"] == pytest.approx(60.0 * 1.9)
    assert first["length_m"] == pytest.approx(60.0)
    assert first["width_mean_m"] == pytest.approx(1.9)
    assert (first["row_left_id"], first["row_right_id"]) == ("V01-R001", "V01-R002")


def test_free_text_row_ids_use_row_index(params) -> None:
    annset = _rename_rows(make_annset(BlockSpec(n_rows=4)), {"V01-R001": "V01-RA", "V01-R002": "V01-R900",
                                                             "V01-R003": "V01-Rc"})
    linked = _link(annset, params).pieces
    third = linked[linked["row_left_id"] == "V01-Rc"]
    assert set(third["interrow_id"]) == {"V01-I003"}
    assert third["k_min_ids"].isna().all()
    first = linked[linked["row_left_id"] == "V01-RA"]
    assert set(first["interrow_id"]) == {"V01-I001"}


def test_inserted_r900_row_never_collides(params) -> None:
    # Geometric order R001..R004, R900, R005, R006: min(k) gives I005 twice, row_index order does not.
    annset = _rename_rows(make_annset(BlockSpec(n_rows=7, length_m=30.0)),
                          {"V01-R005": "V01-R900", "V01-R006": "V01-R005", "V01-R007": "V01-R006"})
    result = _link(annset, params)
    per_pair = result.pieces.groupby(["row_left_id", "row_right_id"])["interrow_id"].unique()
    ids = [v[0] for v in per_pair]
    assert len(ids) == 6 and len(set(ids)) == 6
    assert per_pair[("V01-R900", "V01-R005")][0] == "V01-I005"
    assert per_pair[("V01-R005", "V01-R006")][0] == "V01-I006"
    kmin = result.pieces.drop_duplicates("interrow_id")["k_min_ids"].tolist()
    assert kmin.count(5) == 2  # the contract rule would have collided


def test_one_sided_piece_is_unlinked(params) -> None:
    annset = make_annset(BlockSpec(n_rows=2))
    pieces = annset.row_pieces
    one_row = annset.with_layer("row_pieces", pieces[pieces["row_id"] == "V01-R001"].reset_index(drop=True))
    result = _link(one_row, params)
    assert result.pieces["interrow_id"].isna().all()
    assert set(result.pieces["row_left_id"]) == {"V01-R001"}
    assert result.pieces["row_right_id"].isna().all()
    assert set(result.pieces["link_method"]) == {"none"}
    assert [i.code for i in result.issues] == ["interrow_unlinked", "interrow_unlinked"]
    assert all(i.severity is Severity.WARNING for i in result.issues)
    assert {i.object_id for i in result.issues} == set(result.pieces["piece_id"])
    assert result.interrows == ()


def test_rows_too_far_are_not_linked(params) -> None:
    result = _link(make_annset(BlockSpec(n_rows=2)), params, LinkParams(max_offset_m=1.0, along_tol_m=1.0, seam_close_m=0.05))
    assert result.pieces["interrow_id"].isna().all() and len(result.issues) == 2


def test_missing_same_tile_rows_fall_back_to_block_rows(params) -> None:
    annset = make_annset(BlockSpec(n_rows=2))
    pieces = annset.interrow_pieces
    moved = annset.with_layer("interrow_pieces", pieces.assign(tile_id=["siret3_r017_c011"] * len(pieces)))
    result = _link(moved, params)
    assert set(result.pieces["link_method"]) == {"block"}
    assert set(result.pieces["interrow_id"]) == {"V01-I001"}


def test_block_without_rows_is_unlinked(params) -> None:
    annset = make_annset(BlockSpec(n_rows=2))
    pieces = annset.interrow_pieces
    other = annset.with_layer("interrow_pieces", pieces.assign(vineyard_id=["V09"] * len(pieces)))
    result = _link(other, params)
    assert set(result.pieces["link_method"]) == {"none"}
    assert result.pieces["width_mean_m"].isna().all()
    assert len(result.issues) == 2 and "niciun rând" in result.issues[0].message


def test_input_frame_is_not_mutated(params) -> None:
    annset = make_annset(BlockSpec(n_rows=2))
    before = annset.interrow_pieces.copy()
    _link(annset, params)
    assert annset.interrow_pieces.equals(before)
    assert "link_method" not in annset.interrow_pieces.columns


@pytest.mark.parametrize(("pairs", "expected"), [
    ([(1, 2), (2, 3), (3, 4)], {(1, 2): 1, (2, 3): 2, (3, 4): 3}),
    ([(2, 3), (3, 4)], {(2, 3): 2, (3, 4): 3}),
    ([(4, 5), (4, 6), (5, 6), (6, 7)], {(4, 5): 4, (4, 6): 5, (5, 6): 6, (6, 7): 7}),
    ([], {}),
])
def test_interrow_numbers(pairs, expected) -> None:
    assert interrow_numbers(pairs) == expected


@pytest.mark.examples
def test_reference_links_all_49_pieces(params, examples_xml: bytes) -> None:
    annset = reference_annset(examples_xml)
    rows, _ = merge_rows(annset.row_pieces, params.merge, build_coverage(annset.meta.tile_ids))
    result = link_interrows(annset.interrow_pieces, annset.row_pieces, order_rows(rows), block_axes(rows),
                            params.link)
    linked = result.pieces
    assert len(linked) == 49 and linked["interrow_id"].notna().all() and result.issues == ()
    first = linked[linked["piece_id"] == "siret3_r021_c012:I001"].iloc[0]
    assert (first["interrow_id"], first["row_left_id"], first["row_right_id"]) == ("V01-I001", "V01-R01", "V01-R02")
    assert linked.groupby("vineyard_id")["interrow_id"].nunique().to_dict() == {"V01": 24, "V02": 25}
    # On clean data the geometric numbering equals the contract's min(k) rule.
    assert all(int(i[-3:]) == k for i, k in zip(linked["interrow_id"], linked["k_min_ids"], strict=True))
    assert len(result.interrows) == 49
    assert all(not math.isnan(r["width_mean_m"]) and r["width_mean_m"] > 0 for r in result.interrows)
