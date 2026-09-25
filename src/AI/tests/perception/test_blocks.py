"""Blocks: components, IDs, row order, skip-one, passages, bands, spacing cut, orchard, frames, examples."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from vineyard.config import load_config
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.geo.tiling import CRS_EPSG, TILE_M, tile_box, tile_ref
from vineyard.geo.vector_io import write_layer
from vineyard.perception.blocks import (
    CODE_BAND_CUT,
    CODE_MISSING_ROW,
    REASON_ORCHARD,
    REASON_TOO_FEW,
    BlockSettings,
    build_blocks,
)
from vineyard.perception.blocks_graph import EDGE_SKIP_ONE
from vineyard.perception.rows_link import FLAG_INTERPOLATED, LinkSettings, link_candidates

TILE = "siret3_r011_c005"
T = tile_ref(TILE)
X0, Y0 = T.x0 + 2.0, T.y0 - 5.0
R021_SPACINGS = (2.58, 2.89, 2.71, 3.04, 2.82, 2.85, 3.04, 2.78, 3.74, 3.61, 2.49, 2.68, 2.60, 2.70, 2.52)


@pytest.fixture(scope="module")
def settings() -> BlockSettings:
    return BlockSettings.from_config(load_config())


def hrow(y: float, x0: float = 0.0, x1: float = 40.0) -> LineString:
    return LineString([(X0 + x0, Y0 - y), (X0 + x1, Y0 - y)])


def chains(lines: Sequence[LineString], gaps: Sequence[str] | None = None, **cols: Sequence[object]
           ) -> gpd.GeoDataFrame:
    n = len(lines)
    data = {"chain_id": [f"L{k + 1:05d}" for k in range(n)], "gaps_json": list(gaps or ["[]"] * n),
            "qa_flags": [""] * n, "interp_tile_ids": [""] * n, "confidence": [0.9] * n, **cols}
    return gpd.GeoDataFrame(data, geometry=list(lines), crs=CRS_EPSG)


def run(frame: gpd.GeoDataFrame, s: BlockSettings, passages: object = None, clips: dict | None = None):
    return build_blocks(frame, passages, clips or {}, s, run_id="test-run", model_version="pipe@test")


def test_ten_rows_one_block_ordered_north_first(settings: BlockSettings) -> None:
    res = run(chains([hrow(2.5 * k) for k in range(10)]), settings)
    assert list(res.blocks.vineyard_id) == ["V01"]
    assert list(res.rows.row_id) == [f"V01-R{k:03d}" for k in range(1, 11)]
    ys = [g.coords[0][1] for g in res.rows.geometry]
    assert ys == sorted(ys, reverse=True)
    assert list(res.rows.chain_id) == [f"L{k:05d}" for k in range(1, 11)]
    blk = res.blocks.iloc[0]
    assert blk.n_rows == 10 and blk.row_length_m == pytest.approx(400.0)
    assert blk.spacing_med_m == pytest.approx(2.5) and blk.angle_deg == pytest.approx(0.0)
    assert blk.outline_area_m2 == pytest.approx(40.0 * 23.1, rel=0.05)
    assert len(res.row_pairs) == 9 and list(res.row_pairs.k_a) == list(range(1, 10))
    assert res.rows_rejected.empty and res.issues == ()


def test_ids_match_contract(settings: BlockSettings) -> None:
    res = run(chains([hrow(2.5 * k) for k in range(4)]), settings)
    assert all(is_valid_id(IdKind.ROW, r) for r in res.rows.row_id)
    assert all(is_valid_id(IdKind.BLOCK, v) for v in res.blocks.vineyard_id)


def test_passage_between_plantings_gives_two_blocks(settings: BlockSettings) -> None:
    ys = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 17.5, 20.0, 22.5, 25.0]
    passage = box(X0 - 5, Y0 - 14.0, X0 + 50, Y0 - 11.0)
    res = run(chains([hrow(y) for y in ys]), settings, passages=passage)
    assert list(res.blocks.vineyard_id) == ["V01", "V02"]
    north = res.rows[res.rows.vineyard_id == "V01"]
    assert min(g.coords[0][1] for g in north.geometry) > Y0 - 11.0
    joined = run(chains([hrow(y) for y in ys]), settings)
    assert len(joined.blocks) == 1 and CODE_MISSING_ROW in joined.blocks.qa_flags.iloc[0]


def test_row_inside_passage_polygon_stays(settings: BlockSettings) -> None:
    passage = box(X0 - 5, Y0 - 9.0, X0 + 50, Y0 - 6.0)
    res = run(chains([hrow(2.5 * k) for k in range(4)]), settings, passages=passage)
    assert len(res.blocks) == 1 and len(res.rows) == 4


def test_passages_ignored_when_cut_disabled(settings: BlockSettings) -> None:
    ys = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 17.5, 20.0, 22.5, 25.0]
    passage = box(X0 - 5, Y0 - 14.0, X0 + 50, Y0 - 11.0)
    res = run(chains([hrow(y) for y in ys]), replace(settings, cut_by_passages=False), passages=passage)
    assert len(res.blocks) == 1


def test_two_row_block_dropped(settings: BlockSettings) -> None:
    lines = [hrow(2.5 * k) for k in range(4)] + [hrow(40.0), hrow(42.5)]
    res = run(chains(lines), settings)
    assert len(res.blocks) == 1 and len(res.rows) == 4
    assert list(res.rows_rejected.chain_id) == ["L00005", "L00006"]
    assert set(res.rows_rejected.reason) == {REASON_TOO_FEW}
    assert [i.code for i in res.issues] == [REASON_TOO_FEW]


def test_single_stray_row_has_no_issue(settings: BlockSettings) -> None:
    res = run(chains([hrow(2.5 * k) for k in range(3)] + [hrow(60.0)]), settings)
    assert list(res.rows_rejected.chain_id) == ["L00004"] and res.issues == ()


def test_collinear_pieces_count_once(settings: BlockSettings) -> None:
    lines = [hrow(0.0), hrow(2.5, 0, 18), hrow(2.5, 21, 40)]
    res = run(chains(lines), settings)
    assert res.blocks.empty and len(res.rows_rejected) == 3


def test_missing_middle_row_one_block_flagged(settings: BlockSettings) -> None:
    ys = [0.0, 2.5, 5.0, 10.0, 12.5, 15.0]
    res = run(chains([hrow(y) for y in ys]), settings)
    assert len(res.blocks) == 1 and res.blocks.n_skip_one.iloc[0] == 1
    skip = res.row_pairs[res.row_pairs.kind == EDGE_SKIP_ONE]
    assert list(zip(skip.row_a, skip.row_b, strict=True)) == [("V01-R003", "V01-R004")]
    assert skip.qa_flags.iloc[0] == CODE_MISSING_ROW and skip.spacing_m.iloc[0] == pytest.approx(5.0)
    assert [i.code for i in res.issues] == [CODE_MISSING_ROW]
    r3 = res.rows[res.rows.row_id == "V01-R003"].iloc[0]
    assert r3.spacing_prev_m == pytest.approx(2.5) and r3.spacing_next_m == pytest.approx(5.0)


def test_full_width_band_splits_block(settings: BlockSettings) -> None:
    lines = [hrow(2.5 * k, 0, 60) for k in range(5)]
    res = run(chains(lines, ["[[20.0, 26.0]]"] * 5), settings)
    assert list(res.blocks.vineyard_id) == ["V01", "V02"]
    west = res.rows[res.rows.vineyard_id == "V01"]
    assert all(g.bounds[2] < X0 + 21.0 for g in west.geometry)
    assert all(CODE_BAND_CUT in f for f in res.rows.qa_flags)
    assert res.rows.chain_id.value_counts().max() == 2
    assert [i.code for i in res.issues] == [CODE_BAND_CUT]


def test_r006_like_partial_band_not_split(settings: BlockSettings) -> None:
    gaps = ["[[12.6, 23.9]]", "[[14.3, 22.1]]", "[[17.4, 30.3]]", "[]", "[]", "[]"]
    res = run(chains([hrow(2.5 * k, 0, 60) for k in range(6)], gaps), settings)
    assert len(res.blocks) == 1 and len(res.rows) == 6
    assert json.loads(res.rows.gaps_json.iloc[0]) == [[12.6, 23.9]]


def _r021_lines() -> list[LineString]:
    ys = np.concatenate(([0.0], np.cumsum(R021_SPACINGS)))
    return [hrow(float(y)) for y in ys]


def test_r021_spacing_sequence_one_block_by_default(settings: BlockSettings) -> None:
    res = run(chains(_r021_lines()), settings)
    assert len(res.blocks) == 1 and len(res.rows) == len(R021_SPACINGS) + 1


def test_r021_spacing_cut_documents_split(settings: BlockSettings) -> None:
    on = replace(settings, spacing_cut=replace(settings.spacing_cut, enabled=True))
    assert len(run(chains(_r021_lines()), on).blocks) > 1


def test_spacing_cut_keeps_regular_block(settings: BlockSettings) -> None:
    on = replace(settings, spacing_cut=replace(settings.spacing_cut, enabled=True))
    assert len(run(chains([hrow(2.5 * k) for k in range(12)]), on).blocks) == 1


def test_orchard_rejection_behind_flag(settings: BlockSettings) -> None:
    lines = [hrow(3.5 * k) for k in range(4)]
    frame = chains(lines, width_p80_m=[3.0] * 4, along_period_m=[5.0] * 4, along_duty=[0.4] * 4,
                   harmonic_frac=[1.0] * 4)
    assert len(run(frame, settings).blocks) == 1
    on = replace(settings, orchard=replace(settings.orchard, enabled=True))
    res = run(frame, on)
    assert res.blocks.empty and set(res.rows_rejected.reason) == {REASON_ORCHARD}
    assert [i.code for i in res.issues] == [REASON_ORCHARD]
    clean = chains([hrow(2.5 * k) for k in range(4)], width_p80_m=[0.5] * 4)
    assert len(run(clean, on).blocks) == 1


def test_shuffled_input_is_identical(settings: BlockSettings) -> None:
    ys = [0.0, 2.5, 5.0, 10.0, 12.5, 15.0, 40.0, 42.5, 45.0, 47.5]
    frame = chains([hrow(y) for y in ys])
    a = run(frame, settings)
    b = run(frame.sample(frac=1.0, random_state=3), settings)
    for name in ("rows", "blocks", "row_pairs"):
        fa, fb = getattr(a, name), getattr(b, name)
        assert fa.drop(columns="geometry").equals(fb.drop(columns="geometry"))
        assert [g.wkb for g in fa.geometry] == [g.wkb for g in fb.geometry]


def test_lengths_count_pieces_inside_clips(settings: BlockSettings) -> None:
    right = tile_ref("siret3_r011_c006")
    lines = [LineString([(T.x0 + 10, Y0 - y), (T.x0 + TILE_M + 20, Y0 - y)]) for y in (0.0, 2.5, 5.0)]
    hole = box(right.x0, T.y0 - TILE_M, right.x0 + 5, T.y0)
    clips = {TILE: tile_box(T), right.tile_id: tile_box(right).difference(hole)}
    res = run(chains(lines), settings, clips=clips)
    row = res.rows.iloc[0]
    assert row.n_pieces == 2 and row.tile_ids == f"{TILE},{right.tile_id}"
    assert row.extent_m == pytest.approx(TILE_M + 10.0)
    assert row.length_m == pytest.approx(TILE_M + 5.0)
    assert res.blocks.n_tiles.iloc[0] == 2 and res.blocks.n_row_pieces.iloc[0] == 6


def test_interpolated_flag_kept_only_on_its_tiles(settings: BlockSettings) -> None:
    lines = [hrow(2.5 * k, 0, 60) for k in range(4)]
    other = "siret3_r011_c006"
    frame = chains(lines, qa_flags=["row_interpolated;curved_row", "", "", ""],
                   interp_tile_ids=[other, "", "", ""])
    res = run(frame, settings, clips={})
    first = res.rows.iloc[0]
    assert first.interp_tile_ids == other and FLAG_INTERPOLATED in first.qa_flags
    frame2 = chains(lines, qa_flags=["row_interpolated", "", "", ""], interp_tile_ids=["siret3_r099_c099", "", "", ""])
    assert FLAG_INTERPOLATED not in run(frame2, settings).rows.qa_flags.iloc[0]


def test_empty_input_gives_empty_layers(settings: BlockSettings, tmp_path: Path) -> None:
    res = run(chains([]), settings)
    for name in ("rows", "blocks", "row_pairs", "rows_rejected"):
        frame = getattr(res, name)
        assert frame.empty
        write_layer(frame, name, tmp_path / f"{name}.parquet")


def test_layers_validate(settings: BlockSettings, tmp_path: Path) -> None:
    ys = [0.0, 2.5, 5.0, 10.0, 12.5, 15.0, 40.0, 42.5]
    res = run(chains([hrow(y) for y in ys]), settings)
    for name in ("rows", "blocks", "row_pairs", "rows_rejected"):
        write_layer(getattr(res, name), name, tmp_path / f"{name}.parquet")


# ---------------------------------------------------------------- reference axes (examples)


def _reference(examples_xml: bytes, tile_id: str) -> tuple[list[str], list[LineString], list[Polygon]]:
    from tests.helpers.examples import canopy_polygons_utm, load_examples, row_lines_utm

    img = load_examples(examples_xml)[tile_id]
    ids = [s.attributes["row_id"] for s in img.by_label("row")]
    return ids, row_lines_utm(img), canopy_polygons_utm(img)


def _interior_gaps(axis: LineString, canopies: Sequence[Polygon], half_m: float, min_m: float) -> str:
    corridor = axis.buffer(half_m, cap_style="flat")
    spans = sorted((min(t), max(t)) for c in canopies if c.intersects(corridor)
                   for t in [[axis.project(shapely_point) for shapely_point in _points(c.intersection(corridor))]])
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gaps = [[round(x[1], 3), round(y[0], 3)] for x, y in zip(merged, merged[1:], strict=False) if y[0] - x[1] >= min_m]
    return json.dumps(gaps)


def _points(geom: object) -> list:
    from shapely.geometry import Point

    parts = getattr(geom, "geoms", [geom])
    return [Point(c) for p in parts if not p.is_empty for c in np.asarray(p.exterior.coords)]


def _cands(tile_id: str, lines: Sequence[LineString], gaps: Sequence[str]) -> gpd.GeoDataFrame:
    order = np.random.default_rng(5).permutation(len(lines))
    return gpd.GeoDataFrame({"cand_id": [f"{tile_id}:K{k + 1:02d}" for k in range(len(lines))],
                             "tile_id": tile_id, "rejected_reason": None,
                             "gaps_json": [gaps[i] for i in order]},
                            geometry=[lines[i] for i in order], crs=CRS_EPSG)


@pytest.mark.examples
@pytest.mark.parametrize(("tile_id", "n_rows", "total_m"),
                         [("siret3_r021_c012", 25, 910.1), ("siret3_r006_c004", 26, 1031.5)])
def test_reference_axes_as_rows_raw(examples_xml: bytes, settings: BlockSettings, tile_id: str, n_rows: int,
                                    total_m: float) -> None:
    ids, lines, _ = _reference(examples_xml, tile_id)
    perm = np.random.default_rng(11).permutation(len(lines))
    frame = chains([lines[i] for i in perm])
    res = run(frame, settings, clips={tile_id: tile_box(tile_ref(tile_id))})
    assert len(res.blocks) == 1 and len(res.rows) == n_rows
    by_chain = {f"L{k + 1:05d}": ids[i] for k, i in enumerate(perm)}
    ref_order = [by_chain[c] for c in res.rows.chain_id]
    assert [int(r.split("-R")[1]) for r in ref_order] == list(range(1, n_rows + 1))
    assert res.rows.length_m.sum() == pytest.approx(total_m, abs=1.0)
    again = run(frame, settings, clips={tile_id: tile_box(tile_ref(tile_id))})
    assert list(again.rows.row_id) == list(res.rows.row_id) and list(again.rows.chain_id) == list(res.rows.chain_id)


@pytest.mark.examples
@pytest.mark.parametrize("tile_id", ["siret3_r021_c012", "siret3_r006_c004"])
def test_reference_axes_through_linking(examples_xml: bytes, settings: BlockSettings, tile_id: str) -> None:
    cfg = load_config()
    ids, lines, canopies = _reference(examples_xml, tile_id)
    gaps = [_interior_gaps(ln, canopies, cfg.canopy.corridor_half_m, cfg.rows.detect.gap_record_min_m) for ln in lines]
    clips = {tile_id: tile_box(tile_ref(tile_id))}
    linked = link_candidates(_cands(tile_id, lines, gaps), None, clips, LinkSettings.from_config(cfg))
    assert len(linked.rows_raw) == len(lines)
    if tile_id == "siret3_r006_c004":  # the partial R06-R09 gaps must reach the band test
        assert (linked.rows_raw.gaps_json != "[]").sum() >= 3
    res = run(linked.rows_raw, settings, clips=clips)
    assert len(res.blocks) == 1 and len(res.rows) == len(lines) and res.rows_rejected.empty
    cand_of_line = {f"{tile_id}:K{k + 1:02d}": int(i) for k, i in
                    enumerate(np.random.default_rng(5).permutation(len(lines)))}
    member = dict(zip(linked.rows_raw.chain_id, linked.rows_raw.member_cand_ids, strict=True))
    got = [ids[cand_of_line[member[c]]] for c in res.rows.chain_id]
    assert [int(r.split("-R")[1]) for r in got] == list(range(1, len(lines) + 1))
    assert not any(i.code == CODE_BAND_CUT for i in res.issues)
    assert math.isfinite(res.blocks.spacing_med_m.iloc[0])
