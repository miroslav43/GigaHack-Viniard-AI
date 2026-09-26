"""Contract §2.7 invariants (plan S5 overlap limit) and conditional canopy subtraction from interrows."""

from __future__ import annotations

import pytest
import shapely

from tests.qa import annset_factory as af
from vineyard.contracts.enums import Severity
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.perception import topology

A = af.TILE_A
LIMIT = 0.05
CLIPS = {A: tile_box(tile_ref(A))}


def _rows():
    return af.row_pieces([(A, "V01-R001", af.hline(A, 10.0)), (A, "V01-R002", af.hline(A, 12.5)),
                          (A, "V01-R003", af.hline(A, 15.0))])


def _irs():
    return af.interrow_pieces([(A, "V01-I001", af.rect(A, 0, 10.3, 51.2, 12.2)),
                               (A, "V01-I002", af.rect(A, 0, 12.8, 51.2, 14.7))])


def _canopies(*extra):
    base = [(A, "V01", af.rect(A, 5, 9.8, 6, 10.2)), (A, "V01", af.rect(A, 8, 12.3, 9, 12.7))]
    return af.canopies(base + list(extra))


def _codes(issues) -> list[str]:
    return [i.code for i in issues]


def test_clean_annset_has_no_issues() -> None:
    ann = af.annset([A], can=_canopies(), rows=_rows(), irs=_irs())
    assert topology.check_invariants(ann, CLIPS, max_overlap_m2=LIMIT) == ()


def test_overlap_by_tile_measures_intersection() -> None:
    can = _canopies((A, "V01", af.rect(A, 0, 9.8, 0.8, 10.4)))
    assert topology.overlap_by_tile(can, _irs())[A] == pytest.approx(0.08)
    assert topology.overlap_by_tile(_canopies(), _irs()) == {A: 0.0}


def test_small_overlap_is_left_alone() -> None:
    can = _canopies((A, "V01", af.rect(A, 0, 9.8, 0.2, 10.4)))  # 0.02 m2, like the reference
    irs = _irs()
    out, fixed = topology.remove_canopy_overlap(irs, can, max_overlap_m2=LIMIT, min_piece_m2=0.25, clearance_m=0.0)
    assert fixed == ()
    assert out.geometry.geom_equals(irs.geometry).all()
    ann = af.annset([A], can=can, rows=_rows(), irs=irs)
    assert topology.check_invariants(ann, CLIPS, max_overlap_m2=LIMIT) == ()


def test_large_overlap_is_error_until_removed() -> None:
    can = _canopies((A, "V01", af.rect(A, 0, 9.8, 0.8, 10.4)))
    ann = af.annset([A], can=can, rows=_rows(), irs=_irs())
    issues = topology.check_invariants(ann, CLIPS, max_overlap_m2=LIMIT)
    assert _codes(issues) == ["canopy_interrow_overlap"]
    assert issues[0].severity == Severity.ERROR and issues[0].x is not None
    out, fixed = topology.remove_canopy_overlap(_irs(), can, max_overlap_m2=LIMIT, min_piece_m2=0.25, clearance_m=0.0)
    assert fixed == (A,)
    assert topology.overlap_by_tile(can, out)[A] < 1e-9
    assert list(out["piece_id"]) == list(_irs()["piece_id"])
    assert out["area_m2"].iloc[0] == pytest.approx(out.geometry.iloc[0].area)
    assert topology.check_invariants(af.annset([A], can=can, rows=_rows(), irs=out), CLIPS,
                                     max_overlap_m2=LIMIT) == ()


def test_split_piece_gets_dup_suffix() -> None:
    can = _canopies((A, "V01", af.rect(A, 20, 10.2, 21, 12.3)))
    out, fixed = topology.remove_canopy_overlap(_irs(), can, max_overlap_m2=LIMIT, min_piece_m2=0.25, clearance_m=0.0)
    assert fixed == (A,)
    ids = sorted(out["piece_id"])
    assert ids == [f"V01-I001@{A}", f"V01-I001@{A}#2", f"V01-I002@{A}"]
    assert out.geometry.is_valid.all()


def test_row_multi_block() -> None:
    rows = _rows()
    rows = rows.assign(vineyard_id=["V01", "V02", "V01"], row_id=["V01-R001", "V01-R001", "V01-R003"],
                       piece_id=["a@x", "b@x", "c@x"])
    ann = af.annset([A], rows=rows)
    issues = topology.check_invariants(ann, CLIPS, max_overlap_m2=LIMIT)
    assert _codes(issues) == ["row_multi_block"]
    assert issues[0].object_id == "V01-R001"


def test_waste_block_must_exist() -> None:
    wst = af.waste([(A, "V09", af.rect(A, 1, 1, 2, 2)), (A, "", af.rect(A, 3, 3, 4, 4)),
                    (A, "V01", af.rect(A, 5, 5, 6, 6))])
    ann = af.annset([A], can=_canopies(), rows=_rows(), irs=_irs(), wst=wst)
    issues = topology.check_invariants(ann, CLIPS, max_overlap_m2=LIMIT)
    assert _codes(issues) == ["waste_block_unknown"]
    assert issues[0].object_id == "W0001"


def test_object_outside_clip_tolerance() -> None:
    ok = _canopies((A, "V01", af.rect(A, -0.0005, 9.8, 0.5, 10.2)))
    assert topology.check_invariants(af.annset([A], can=ok), CLIPS, max_overlap_m2=LIMIT) == ()
    bad = _canopies((A, "V01", af.rect(A, -0.002, 9.8, 0.5, 10.2)))
    issues = topology.check_invariants(af.annset([A], can=bad), CLIPS, max_overlap_m2=LIMIT)
    assert _codes(issues) == [topology.OUTSIDE_CLIP_CODE]


def test_row_pieces_checked_against_tile_box_not_valid_clip() -> None:
    half_clip = {A: af.rect(A, 10, 0, 51.2, 51.2)}  # nodata on the west side of the tile
    ann = af.annset([A], rows=_rows())
    assert topology.check_invariants(ann, half_clip, max_overlap_m2=LIMIT) == ()
    ann2 = af.annset([A], can=_canopies())
    issues = topology.check_invariants(ann2, half_clip, max_overlap_m2=LIMIT)
    assert _codes(issues) == [topology.OUTSIDE_CLIP_CODE] * 2


def test_interrow_outside_extreme_rows() -> None:
    irs = af.interrow_pieces([(A, "V01-I001", af.rect(A, 0, 10.3, 51.2, 12.2)),
                              (A, "V01-I003", af.rect(A, 0, 15.3, 51.2, 17.2))])
    ann = af.annset([A], rows=_rows(), irs=irs)
    issues = topology.check_invariants(ann, CLIPS, max_overlap_m2=LIMIT)
    assert _codes(issues) == [topology.INTERROW_OUTSIDE_CODE]
    assert issues[0].object_id == f"V01-I003@{A}"


def test_interrow_uses_global_rows_when_given() -> None:
    irs = af.interrow_pieces([(A, "V01-I001", af.rect(A, 0, 10.3, 51.2, 12.2))])
    ann = af.annset([A], irs=irs)
    assert _codes(topology.check_invariants(ann, CLIPS, max_overlap_m2=LIMIT)) == [topology.INTERROW_OUTSIDE_CODE]
    assert topology.check_invariants(ann, CLIPS, max_overlap_m2=LIMIT, rows=_rows()) == ()


def test_issue_order_is_stable() -> None:
    wst = af.waste([(A, "V09", af.rect(A, 1, 1, 2, 2))])
    can = _canopies((A, "V01", af.rect(A, 0, 9.8, 0.8, 10.4)))
    ann = af.annset([A], can=can, rows=_rows(), irs=_irs(), wst=wst)
    first = topology.check_invariants(ann, CLIPS, max_overlap_m2=LIMIT)
    assert first == topology.check_invariants(ann, CLIPS, max_overlap_m2=LIMIT)
    assert _codes(first) == sorted(_codes(first))


def test_removed_overlap_keeps_a_clearance_from_canopies() -> None:
    can = _canopies((A, "V01", af.rect(A, 0, 9.8, 0.8, 10.4)))
    out, fixed = topology.remove_canopy_overlap(_irs(), can, max_overlap_m2=LIMIT, min_piece_m2=0.25,
                                                clearance_m=0.01)
    assert fixed == (A,)
    gap = shapely.distance(shapely.union_all(can.geometry.values), shapely.union_all(out.geometry.values))
    assert gap == pytest.approx(0.01, abs=1e-3)
    assert topology.overlap_by_tile(can, out)[A] == 0.0
