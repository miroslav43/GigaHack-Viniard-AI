"""annset.row_order: R001 = max n·c per block (contract §1.5), spacing, block axes."""

from __future__ import annotations

import math
import statistics

import pytest

from tests.post.factories import BlockSpec, default_config, make_annset, reference_annset
from vineyard.annset.derive import DeriveParams, build_coverage
from vineyard.annset.merge_rows import merge_rows
from vineyard.annset.row_order import OrderedRow, block_axes, order_rows, row_record
from vineyard.contracts.ids import row_index_of


@pytest.fixture(scope="module")
def params():
    return DeriveParams.from_config(default_config()).merge


def _ordered(annset, params) -> tuple[OrderedRow, ...]:
    rows, _ = merge_rows(annset.row_pieces, params)
    return order_rows(rows)


def test_synthetic_block_indices_and_spacing(params) -> None:
    ordered = _ordered(make_annset(BlockSpec(n_rows=5, angle_deg=127.5, spacing_m=2.5, length_m=30.0)), params)
    assert [o.row_index for o in ordered] == [1, 2, 3, 4, 5]
    assert [o.row.row_id for o in ordered] == [f"V01-R00{k}" for k in range(1, 6)]
    assert math.isnan(ordered[0].spacing_prev_m) and math.isnan(ordered[-1].spacing_next_m)
    assert all(o.spacing_next_m == pytest.approx(2.5) for o in ordered[:-1])
    assert all(o.spacing_prev_m == pytest.approx(2.5) for o in ordered[1:])


def test_indices_are_geometric_not_from_ids(params) -> None:
    annset = make_annset(BlockSpec(n_rows=3))
    renamed = annset.row_pieces.assign(row_id=annset.row_pieces["row_id"].map(
        {"V01-R001": "V01-R900", "V01-R002": "V01-R005", "V01-R003": "V01-R001"}))
    ordered = order_rows(merge_rows(renamed, params)[0])
    assert [(o.row.row_id, o.row_index) for o in ordered] == [("V01-R900", 1), ("V01-R005", 2), ("V01-R001", 3)]


def test_indices_restart_per_block(params) -> None:
    annset = make_annset(BlockSpec(n_rows=2), BlockSpec(vineyard_id="V02", n_rows=3, origin_xy=(629560.0, 5220265.0)))
    ordered = _ordered(annset, params)
    assert [(o.row.vineyard_id, o.row_index) for o in ordered] == [
        ("V01", 1), ("V01", 2), ("V02", 1), ("V02", 2), ("V02", 3)]


def test_unequal_rows_use_their_overlap_for_spacing(params) -> None:
    spec = BlockSpec(n_rows=2, spacing_m=2.6, length_m=(50.0, 40.0), start_m=(0.0, 30.0))
    ordered = _ordered(make_annset(spec), params)
    assert ordered[0].spacing_next_m == pytest.approx(2.6)


def test_collinear_rows_tie_break_on_row_id(params) -> None:
    annset = make_annset(BlockSpec(n_rows=1))  # 2 collinear pieces over 2 tiles
    split = annset.row_pieces.assign(row_id=["V01-R002", "V01-R001"])
    ordered = order_rows(merge_rows(split, params)[0])
    assert [o.row.row_id for o in ordered] == ["V01-R001", "V01-R002"]
    assert [o.row_index for o in ordered] == [1, 2]


def test_block_axes_use_the_axial_median(params) -> None:
    rows, _ = merge_rows(make_annset(BlockSpec(n_rows=3, angle_deg=179.5)).row_pieces, params)
    axes = block_axes(rows)
    assert set(axes) == {"V01"}
    assert axes["V01"].angle_deg == pytest.approx(179.5)
    assert axes["V01"].normal[1] > 0


def test_row_record_matches_rows_schema_columns(params) -> None:
    ordered = _ordered(make_annset(BlockSpec(n_rows=2)), params)
    rec = row_record(ordered[0], plant_count=7)
    assert rec["row_id"] == "V01-R001" and rec["row_index"] == 1 and rec["n_pieces"] == 2
    assert rec["tile_ids"] == "siret3_r018_c011,siret3_r018_c012"
    assert rec["plant_count"] == 7 and rec["structure_any"] == "regular"
    assert rec["tile_structures"] == '{"siret3_r018_c011": "regular", "siret3_r018_c012": "regular"}'
    assert math.isnan(rec["max_gap_m"]) and rec["n_gaps_ge5"] is None


@pytest.mark.examples
def test_reference_order_angles_spacing(params, examples_xml: bytes) -> None:
    annset = reference_annset(examples_xml)
    rows, _ = merge_rows(annset.row_pieces, params, build_coverage(annset.meta.tile_ids))
    ordered = order_rows(rows)
    assert all(o.row_index == row_index_of(o.row.row_id) for o in ordered)
    v01 = [o for o in ordered if o.row.vineyard_id == "V01"]
    v02 = [o for o in ordered if o.row.vineyard_id == "V02"]
    assert all(126.5 <= o.row.angle_deg <= 133.5 for o in v01)
    assert all(111.5 <= o.row.angle_deg <= 113.5 for o in v02)
    assert statistics.median(o.spacing_next_m for o in v02[:-1]) == pytest.approx(2.53, abs=0.02)
    assert statistics.median(o.spacing_next_m for o in v01[:-1]) == pytest.approx(2.78, abs=0.05)
