"""Cross-block inter-row overlap removal: block interrow areas must add up to the survey union."""

from __future__ import annotations

import pandas as pd
import pytest
import shapely

from tests.qa import annset_factory as af
from vineyard.contracts.enums import Severity
from vineyard.perception import block_overlap
from vineyard.perception.block_overlap import BlockOverlapParams, remove_block_overlap

A = af.TILE_A
PARAMS = BlockOverlapParams(min_overlap_m2=1e-4, support_buffer_m=1.0, min_piece_m2=0.25)
V01_A, V01_B = f"V01-I001@{A}", f"V01-I002@{A}"


def _irs(*extra):
    base = [(A, "V01-I001", af.rect(A, 0, 10.3, 51.2, 12.2)), (A, "V01-I002", af.rect(A, 0, 12.8, 51.2, 14.7))]
    return af.interrow_pieces(base + list(extra))


def _v02_over_v01():
    """V02 band crossing the east half of both V01 bands (25.44 + 4.24 m2 of overlap)."""
    return _irs((A, "V02-I001", af.rect(A, 30, 11.0, 51.2, 13.0)))


def _vines(vid: str, n: int = 6, x0: float = 32.0) -> list[tuple]:
    """n vines of block `vid` along y = 12.5 from x0 on, 3 m apart."""
    return [(A, vid, af.rect(A, x0 + 3 * k, 12.3, x0 + 1 + 3 * k, 12.7)) for k in range(n)]


def _canopies(vid: str, n: int = 6):
    """Vines of one block next to the contested area of `_v02_over_v01` (x 30..51.2)."""
    return af.canopies(_vines(vid, n))


def _by_id(frame) -> dict[str, shapely.Geometry]:
    return dict(zip(frame["piece_id"], frame.geometry, strict=True))


def _cross_overlap(frame) -> float:
    total = 0.0
    rows = list(zip(frame["vineyard_id"], frame.geometry, strict=True))
    for i, (va, ga) in enumerate(rows):
        total += sum(ga.intersection(gb).area for vb, gb in rows[i + 1:] if vb != va)
    return total


def _block_union_sum(frame) -> float:
    return sum(shapely.union_all(list(g.geometry)).area for _, g in frame.groupby("vineyard_id"))


def _union(frame) -> float:
    return shapely.union_all(list(frame.geometry)).area


def test_disjoint_blocks_are_unchanged() -> None:
    irs = _irs((A, "V02-I001", af.rect(A, 0, 20.0, 51.2, 22.0)))
    out = remove_block_overlap(irs, _canopies("V01"), PARAMS)
    assert out.pieces is irs and out.issues == ()
    assert (out.n_cut, out.n_dropped, out.n_split) == (0, 0, 0) and out.moved_m2 == 0.0


def test_same_block_overlap_is_left_alone() -> None:
    irs = _irs((A, "V01-I003", af.rect(A, 10, 11.0, 20, 13.0)))
    out = remove_block_overlap(irs, _canopies("V01"), PARAMS)
    assert out.pieces is irs and out.issues == ()


def test_overlap_below_the_minimum_is_ignored() -> None:
    irs = _irs((A, "V02-I001", af.rect(A, 51.19, 12.1995, 51.2, 12.8)))  # 0.01 m x 0.5 mm over V01-I001
    out = remove_block_overlap(irs, _canopies("V02"), PARAMS)
    assert out.pieces is irs and out.issues == ()


def test_overlap_goes_to_the_block_whose_vines_flank_it() -> None:
    irs = _v02_over_v01()
    out = remove_block_overlap(irs, _canopies("V02"), PARAMS)
    got = _by_id(out.pieces)
    assert got[f"V02-I001@{A}"].equals(_by_id(irs)[f"V02-I001@{A}"])
    assert got[V01_A].area == pytest.approx(51.2 * 1.9 - 21.2 * 1.2)
    assert got[V01_B].area == pytest.approx(51.2 * 1.9 - 21.2 * 0.2)
    assert _cross_overlap(out.pieces) == pytest.approx(0.0, abs=1e-9)
    assert _union(out.pieces) == pytest.approx(_union(irs), abs=1e-9)
    assert _block_union_sum(out.pieces) == pytest.approx(_union(out.pieces), abs=1e-9)
    assert (out.n_cut, out.n_dropped, out.n_split) == (2, 0, 0)
    assert out.moved_m2 == pytest.approx(21.2 * 1.4)
    area = dict(zip(out.pieces["piece_id"], out.pieces["area_m2"], strict=True))
    assert area[V01_A] == pytest.approx(got[V01_A].area)
    code = block_overlap.BLOCK_OVERLAP_CODE
    assert [(i.code, i.severity, i.object_id) for i in out.issues] == [(code, Severity.WARNING, V01_A),
                                                                        (code, Severity.WARNING, V01_B)]
    assert all(i.tile_id == A and i.x is not None and "V02" in i.message for i in out.issues)


def test_support_decides_not_the_block_id() -> None:
    out = remove_block_overlap(_v02_over_v01(), _canopies("V01"), PARAMS)
    got = _by_id(out.pieces)
    assert got[V01_A].area == pytest.approx(51.2 * 1.9)
    assert got[f"V02-I001@{A}"].area == pytest.approx(21.2 * 2.0 - 21.2 * 1.2 - 21.2 * 0.2)
    assert [i.object_id for i in out.issues] == [f"V02-I001@{A}"]


def test_without_vines_the_lower_block_id_keeps_the_overlap() -> None:
    out = remove_block_overlap(_v02_over_v01(), af.canopies([]), PARAMS)
    assert [i.object_id for i in out.issues] == [f"V02-I001@{A}"]
    assert _cross_overlap(out.pieces) == pytest.approx(0.0, abs=1e-9)


def test_attributes_and_row_order_are_kept() -> None:
    irs = _v02_over_v01()
    out = remove_block_overlap(irs, _canopies("V02"), PARAMS).pieces
    assert list(out["piece_id"]) == list(irs["piece_id"]) and list(out.columns) == list(irs.columns)
    keep = [c for c in irs.columns if c not in ("geometry", "area_m2", "width_mean_m")]
    pd.testing.assert_frame_equal(out[keep].reset_index(drop=True), irs[keep].reset_index(drop=True))
    assert out.geometry.is_valid.all() and set(out.geom_type) == {"Polygon"}


def test_remainder_below_the_minimum_area_is_dropped() -> None:
    irs = _irs((A, "V02-I001", af.rect(A, 30, 10.3, 34, 12.25)))  # 4 x 0.05 m = 0.2 m2 left over V01-I001
    out = remove_block_overlap(irs, _canopies("V01"), PARAMS)
    assert f"V02-I001@{A}" not in set(out.pieces["piece_id"])
    assert (out.n_cut, out.n_dropped) == (0, 1)
    assert out.dropped_m2 == pytest.approx(0.2)
    assert "eliminat" in out.issues[0].message


def test_split_piece_keeps_its_id_on_the_largest_part() -> None:
    irs = _irs((A, "V02-I001", af.rect(A, 20, 9.0, 22, 12.5)))  # cuts V01-I001 in two, misses V01-I002
    out = remove_block_overlap(irs, af.canopies(_vines("V02", 1, x0=20.5)), PARAMS)
    got = _by_id(out.pieces)
    assert got[V01_A].area == pytest.approx(29.2 * 1.9) and got[f"{V01_A}#2"].area == pytest.approx(20.0 * 1.9)
    assert out.n_split == 1 and out.n_cut == 1


def test_three_blocks_keep_the_union() -> None:
    irs = _irs((A, "V02-I001", af.rect(A, 30, 11.0, 51.2, 13.0)), (A, "V03-I001", af.rect(A, 40, 9.0, 45, 16.0)))
    out = remove_block_overlap(irs, af.canopies(_vines("V02", 3) + _vines("V03", 2)), PARAMS)
    assert _cross_overlap(out.pieces) == pytest.approx(0.0, abs=1e-9)
    assert _union(out.pieces) == pytest.approx(_union(irs), abs=1e-9)
    assert _block_union_sum(out.pieces) == pytest.approx(_union(out.pieces), abs=1e-9)


def test_result_does_not_depend_on_the_input_order() -> None:
    irs = _irs((A, "V02-I001", af.rect(A, 30, 11.0, 51.2, 13.0)), (A, "V03-I001", af.rect(A, 40, 9.0, 45, 16.0)))
    first = remove_block_overlap(irs, _canopies("V02"), PARAMS)
    second = remove_block_overlap(irs.iloc[::-1], _canopies("V02"), PARAMS)
    a, b = _by_id(first.pieces), _by_id(second.pieces)
    assert a.keys() == b.keys()
    assert all(shapely.normalize(a[k]).equals_exact(shapely.normalize(b[k]), 0.0) for k in a)
    assert sorted(first.issues, key=lambda i: i.object_id) == sorted(second.issues, key=lambda i: i.object_id)


def test_input_frames_are_not_mutated() -> None:
    irs, can = _v02_over_v01(), _canopies("V02")
    before, before_can = irs.copy(), can.copy()
    remove_block_overlap(irs, can, PARAMS)
    assert irs.equals(before) and can.equals(before_can)


def test_params_come_from_config() -> None:
    from tests.post.factories import default_config

    cfg = default_config()
    params = BlockOverlapParams.from_config(cfg)
    assert params.min_overlap_m2 == cfg.derive.interrow_overlap_min_m2
    assert params.support_buffer_m == cfg.derive.interrow_overlap_support_m
    assert params.min_piece_m2 == cfg.export.min_interrow_piece_m2
