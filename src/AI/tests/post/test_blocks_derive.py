"""annset.blocks: outlines from any AnnSet, block stats and block-level qa (contract §2.5.6, §4.5.8)."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, MultiPolygon, box

from tests.post.factories import (
    BlockSpec,
    build_annset,
    default_config,
    make_annset,
    reference_annset,
    row_piece_record,
)
from vineyard.annset.blocks import Block, block_ids, block_record, build_blocks, outline_of
from vineyard.annset.derive import DeriveParams, build_coverage
from vineyard.annset.merge_rows import merge_rows
from vineyard.annset.model import AnnSet
from vineyard.annset.row_order import block_axes, order_rows
from vineyard.contracts.enums import Severity

X0, Y0 = 629560.0, 5220290.0
TILE_AREA_M2 = 51.2 * 51.2


@pytest.fixture(scope="module")
def params() -> DeriveParams:
    return DeriveParams.from_config(default_config())


def _blocks(annset: AnnSet, params: DeriveParams, **kw) -> tuple[tuple[Block, ...], tuple]:
    rows, _ = merge_rows(annset.row_pieces, params.merge)
    return build_blocks(annset, order_rows(rows), block_axes(rows), params.blocks, **kw)


def _relabel(annset: AnnSet, old: str, new: str, row_offset: int) -> AnnSet:
    """Objects of block `old` moved to block `new`, row numbers shifted by `row_offset`."""
    def fix_row(rid: object) -> object:
        if not isinstance(rid, str) or not rid.startswith(f"{old}-R"):
            return rid
        return f"{new}-R{int(rid[-3:]) + row_offset:03d}"

    out = annset
    for name in ("canopies", "row_pieces", "interrow_pieces"):
        gdf = out.layer(name)
        gdf = gdf.assign(vineyard_id=gdf["vineyard_id"].replace(old, new))
        if "row_id" in gdf.columns:
            gdf = gdf.assign(row_id=gdf["row_id"].map(fix_row))
        out = out.with_layer(name, gdf)
    return out


def _codes(issues) -> list[str]:
    return sorted(i.code for i in issues)


def test_single_block_stats(params) -> None:
    annset = make_annset(BlockSpec(n_rows=3, length_m=40.0))
    (block,), issues = _blocks(annset, params)
    assert issues == ()
    assert block.vineyard_id == "V01" and block.n_rows == 3 and block.n_row_pieces == 3
    assert block.n_canopies == len(annset.canopies) and block.n_tiles == 1
    assert block.row_length_m == pytest.approx(120.0)
    assert block.canopy_area_m2 == pytest.approx(float(annset.canopies.geometry.area.sum()))
    assert block.interrow_area_m2 == pytest.approx(2 * 40.0 * 1.9)
    assert block.angle_deg == pytest.approx(0.0) and block.spacing_med_m == pytest.approx(2.5)
    assert not block.is_garden
    # closing fills the space between rows: about 40 m x (2 x 2.5 + canopy width) m
    assert block.outline.geom_type == "Polygon"
    assert block.outline_area_m2 == pytest.approx(40.0 * 5.5, rel=0.03)


def test_same_id_8m_apart_is_split(params) -> None:
    a = make_annset(BlockSpec(n_rows=3, length_m=30.0),
                    BlockSpec(vineyard_id="V09", n_rows=3, length_m=30.0, origin_xy=(X0, Y0 - 5.0 - 8.5)))
    (block,), issues = _blocks(_relabel(a, "V09", "V01", 10), params)
    assert isinstance(block.outline, MultiPolygon) and len(block.outline.geoms) == 2
    assert _codes(issues) == ["block_split"]
    assert issues[0].severity is Severity.WARNING and issues[0].object_id == "V01"
    assert "8.0 m" in issues[0].message


def test_blocks_3m_apart_should_merge_unless_a_passage_separates_them(params) -> None:
    annset = make_annset(BlockSpec(n_rows=3, length_m=30.0),
                         BlockSpec(vineyard_id="V02", n_rows=3, length_m=30.0, origin_xy=(X0, Y0 - 5.0 - 3.5)))
    blocks, issues = _blocks(annset, params)
    assert [b.vineyard_id for b in blocks] == ["V01", "V02"]
    assert _codes(issues) == ["blocks_should_merge"]
    assert issues[0].object_id == "V01+V02" and "3.0 m" in issues[0].message
    strip = box(X0 - 5, Y0 - 8.0, X0 + 40, Y0 - 5.5)
    _, with_passage = _blocks(annset, params, passages=strip)
    assert with_passage == ()


def test_waste_only_id_is_unknown_block(params) -> None:
    annset = make_annset(BlockSpec(n_rows=3, length_m=30.0),
                         waste=[("siret3_r018_c011", (X0 + 10, Y0 + 10), 1.0, "V07"),
                                ("siret3_r018_c011", (X0 + 12, Y0 + 10), 1.0, "V01"),
                                ("siret3_r018_c011", (X0 + 14, Y0 + 10), 1.0, "")])
    blocks, issues = _blocks(annset, params)
    assert block_ids(annset) == ("V01",) and len(blocks) == 1
    assert _codes(issues) == ["waste_block_unknown"] and issues[0].object_id == "W0001"


def test_block_with_few_rows_warns(params) -> None:
    _, issues = _blocks(make_annset(BlockSpec(n_rows=2, length_m=30.0)), params)
    assert _codes(issues) == ["block_too_few_rows"]


def test_garden_near_forbidden(params) -> None:
    annset = make_annset(BlockSpec(n_rows=3, length_m=30.0))
    near = box(X0 - 30, Y0 - 20, X0 - 15, Y0)
    far = box(X0 - 200, Y0 - 20, X0 - 150, Y0)
    (garden,), _ = _blocks(annset, params, forbidden=near)
    (field_block,), _ = _blocks(annset, params, forbidden=far)
    assert garden.is_garden and not field_block.is_garden


def test_lone_row_block_gets_a_corridor_outline(params) -> None:
    line = LineString([(X0, Y0), (X0 + 20, Y0)])
    annset = build_annset(row_pieces=[row_piece_record("V05-R001", "V05", "siret3_r018_c011", line)])
    (block,), issues = _blocks(annset, params)
    assert block.outline.geom_type == "Polygon"
    assert block.outline_area_m2 == pytest.approx(20.0 * 0.6 + math.pi * 0.09, rel=0.01)
    assert block.n_canopies == 0 and math.isnan(block.spacing_med_m)
    assert _codes(issues) == ["block_too_few_rows"]


def test_outline_of_is_a_closing() -> None:
    a, b = box(0, 0, 10, 1), box(0, 3, 10, 4)
    closed = outline_of([a, b], 2.5, 0.3)
    assert closed.geom_type == "Polygon"
    # round buffers (contract formula) shave the convex corners slightly
    assert closed.area == pytest.approx(40.0, rel=0.02) and closed.area <= 40.0 + 1e-6
    apart = outline_of([a, box(0, 10, 10, 11)], 2.5, 0.3)
    assert isinstance(apart, MultiPolygon)


def test_block_record_columns(params) -> None:
    (block,), _ = _blocks(make_annset(BlockSpec(n_rows=3, length_m=30.0)), params)
    rec = block_record(block)
    assert set(rec) >= {"vineyard_id", "n_rows", "n_row_pieces", "n_canopies", "n_tiles", "row_length_m",
                        "canopy_area_m2", "interrow_area_m2", "outline_area_m2", "angle_deg", "spacing_med_m",
                        "is_garden", "geometry"}
    assert rec["outline_area_m2"] == pytest.approx(rec["geometry"].area)


@pytest.mark.examples
def test_reference_gives_two_blocks(params, examples_xml: bytes) -> None:
    annset = reference_annset(examples_xml)
    rows, _ = merge_rows(annset.row_pieces, params.merge, build_coverage(annset.meta.tile_ids))
    blocks, issues = build_blocks(annset, order_rows(rows), block_axes(rows), params.blocks)
    assert issues == ()
    by_id = {b.vineyard_id: b for b in blocks}
    assert (by_id["V01"].n_rows, by_id["V01"].n_canopies) == (25, 399)
    assert (by_id["V02"].n_rows, by_id["V02"].n_canopies) == (26, 251)
    assert by_id["V01"].row_length_m == pytest.approx(910.10, abs=0.005)
    assert by_id["V02"].row_length_m == pytest.approx(1031.45, abs=0.005)
    assert by_id["V01"].canopy_area_m2 == pytest.approx(237.1194, abs=0.01)
    assert by_id["V02"].interrow_area_m2 == pytest.approx(1996.3612, abs=0.01)
    assert all(b.outline_area_m2 <= TILE_AREA_M2 and b.outline.geom_type == "Polygon" for b in blocks)


def test_overlapping_blocks_should_merge(params) -> None:
    annset = make_annset(BlockSpec(n_rows=3, length_m=30.0),
                         BlockSpec(vineyard_id="V02", n_rows=3, length_m=30.0, origin_xy=(X0 + 10.0, Y0 - 1.25)))
    _, issues = _blocks(annset, params)
    merge = [i for i in issues if i.code == "blocks_should_merge"]
    assert len(merge) == 1 and "0.0 m" in merge[0].message and merge[0].x is not None
