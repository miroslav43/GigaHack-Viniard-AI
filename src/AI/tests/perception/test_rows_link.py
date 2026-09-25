"""Global row linking: support-line criteria, greedy constrained union, refit, end snap, S4 hole split."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from vineyard.config import load_config
from vineyard.errors import StageError
from vineyard.geo.tiling import CRS_EPSG, TILE_M, tile_box, tile_ref
from vineyard.perception.rows_link import (
    DECISION_DUPLICATE,
    DECISION_HARD,
    DECISION_RESCUED,
    DECISION_SOFT_CHAIN,
    DECISION_SOFT_DROPPED,
    FLAG_INTERPOLATED,
    FLAG_RESCUED,
    LinkSettings,
    extend_chain_ends,
    link_candidates,
    parse_gaps,
    refit_chain,
)

ROW = 11
TILES = [f"siret3_r{ROW:03d}_c{c:03d}" for c in range(4, 9)]
X0 = tile_ref(TILES[0]).x0
YC = tile_ref(TILES[0]).y0 - 25.6


@pytest.fixture(scope="module")
def settings() -> LinkSettings:
    return LinkSettings.from_config(load_config())


def clips(tiles: Sequence[str] = TILES) -> dict:
    return {t: tile_box(tile_ref(t)) for t in tiles}


def seg(col: int, u0: float, u1: float, dy0: float = 0.0, dy1: float | None = None) -> LineString:
    """Line in tile column `col` (index into TILES) from local x u0 to u1 metres, y offsets from the mid-line."""
    x = X0 + col * TILE_M
    return LineString([(x + u0, YC + dy0), (x + u1, YC + (dy0 if dy1 is None else dy1))])


def frame(items: Sequence[tuple], extra: dict | None = None) -> gpd.GeoDataFrame:
    """items: (col, k, line[, rejected_reason])."""
    recs = []
    for it in items:
        col, k, line = it[:3]
        rec = {"cand_id": f"{TILES[col]}:K{k:02d}", "tile_id": TILES[col], "angle_deg": 0.0,
               "rejected_reason": it[3] if len(it) > 3 else None, "geometry": line}
        rec.update((extra or {}).get(rec["cand_id"], {}))
        recs.append(rec)
    if not recs:
        return gpd.GeoDataFrame({"cand_id": [], "tile_id": [], "angle_deg": [], "rejected_reason": []},
                                geometry=gpd.GeoSeries([], crs=CRS_EPSG), crs=CRS_EPSG)
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=CRS_EPSG)


def chains_of(res) -> list[set[str]]:
    return [set(m.split(",")) for m in res.rows_raw.member_cand_ids]


def test_collinear_pieces_in_adjacent_tiles_link(settings: LinkSettings) -> None:
    rot = 10.0 * math.tan(math.radians(1.0))
    cands = frame([(0, 1, seg(0, 5.0, TILE_M)), (1, 1, seg(1, 0.0, 10.0, 0.1, 0.1 + rot))])
    res = link_candidates(cands, None, clips(), settings)
    assert len(res.rows_raw) == 1
    row = res.rows_raw.iloc[0]
    assert row.chain_id == "L00001" and row.n_tiles == 2
    assert row.tile_ids == ",".join(TILES[:2])


def test_half_metre_offset_does_not_link(settings: LinkSettings) -> None:
    cands = frame([(0, 1, seg(0, 5.0, TILE_M)), (1, 1, seg(1, 0.0, 30.0, 0.5))])
    assert len(link_candidates(cands, None, clips(), settings).rows_raw) == 2


def _three_rows(cols: Sequence[int], skip_row0_cols: Sequence[int] = ()) -> list[tuple]:
    items = []
    for r, dy in enumerate((0.0, 2.5, -2.5)):
        for col in cols:
            if r == 0 and col in skip_row0_cols:
                continue
            items.append((col, r + 1, seg(col, 0.0, TILE_M, dy)))
    return items


def test_gap_over_one_supported_tile_links_and_flags_interpolated(settings: LinkSettings) -> None:
    res = link_candidates(frame(_three_rows([0, 1, 2], skip_row0_cols=[1])), None, clips(), settings)
    assert len(res.rows_raw) == 3
    mid = res.rows_raw[res.rows_raw.member_cand_ids.str.contains(":K01")].iloc[0]
    assert mid.n_tiles == 3 and mid.interp_tile_ids == TILES[1]
    assert FLAG_INTERPOLATED in mid.qa_flags
    assert json.loads(mid.gaps_json) == [[pytest.approx(TILE_M), pytest.approx(2 * TILE_M)]]


def test_unsupported_hole_splits_chain(settings: LinkSettings) -> None:
    res = link_candidates(frame([(0, 1, seg(0, 0.0, TILE_M)), (2, 1, seg(2, 0.0, TILE_M))]), None, clips(),
                          settings)
    assert len(res.rows_raw) == 2
    assert all(f == "" for f in res.rows_raw.interp_tile_ids)


def test_gap_over_two_tiles_not_linked(settings: LinkSettings) -> None:
    items = [it for it in _three_rows([0, 1, 2, 3]) if not (it[1] == 1 and it[0] in (1, 2))]
    res = link_candidates(frame(items), None, clips(), settings)
    k01 = [c for c in chains_of(res) if any(":K01" in m for m in c)]
    assert len(k01) == 2


def test_connector_crossing_passage_blocks_link(settings: LinkSettings) -> None:
    x = X0 + TILE_M
    passage = box(x + 0.5, YC - 10, x + 3.5, YC + 10)
    cands = frame([(0, 1, seg(0, 5.0, TILE_M)), (1, 1, seg(1, 4.0, 40.0))])
    assert len(link_candidates(cands, passage, clips(), settings).rows_raw) == 2
    assert len(link_candidates(cands, None, clips(), settings).rows_raw) == 1


def test_transitive_drift_refused(settings: LinkSettings) -> None:
    cands = frame([(0, 1, seg(0, 0.0, TILE_M)), (1, 1, seg(1, 0.0, 20.0, 0.25)), (1, 2, seg(1, 0.0, 20.0, -0.25))])
    res = link_candidates(cands, None, clips(), settings)
    b, c = f"{TILES[1]}:K01", f"{TILES[1]}:K02"
    assert not any(b in ch and c in ch for ch in chains_of(res))
    assert len(res.rows_raw) == 2


def test_short_stub_links_on_offset_alone(settings: LinkSettings) -> None:
    t = math.radians(10.0)
    stub = seg(1, 0.05, 0.05 + 0.8 * math.cos(t), 0.05, 0.05 + 0.8 * math.sin(t))
    res = link_candidates(frame([(0, 1, seg(0, 0.0, TILE_M)), (1, 1, stub)]), None, clips(), settings)
    assert len(res.rows_raw) == 1


def test_three_collinear_pieces_refit_to_two_vertices(settings: LinkSettings) -> None:
    items = [(c, 1, seg(c, 0.0, TILE_M, 0.05 * (c % 2))) for c in range(3)]
    res = link_candidates(frame(items), None, clips(), settings)
    assert len(res.rows_raw) == 1
    line = res.rows_raw.geometry.iloc[0]
    assert len(line.coords) == 2
    assert line.length == pytest.approx(3 * TILE_M, abs=0.01)


def test_arc_pieces_refit_to_polyline(settings: LinkSettings) -> None:
    radius, cx = 300.0, X0 + 1.5 * TILE_M
    cy = YC - radius

    def arc(x: float) -> float:
        return cy + math.sqrt(radius**2 - (x - cx) ** 2)

    items = []
    for c in range(3):
        xa, xb = X0 + c * TILE_M, X0 + (c + 1) * TILE_M
        xs = np.linspace(xa, xb, 5)
        pts = [(x, arc(x)) for x in xs]
        items.append((c, 1, LineString(pts)))
    lines = [it[2] for it in items]
    line = refit_chain(lines, settings)
    assert len(line.coords) > 2
    dev = max(line.distance(Point(p)) for ln in lines for p in ln.coords)
    assert dev <= 0.15


def test_soft_only_chain_dropped_and_rescue(settings: LinkSettings) -> None:
    items = [(0, 1, seg(0, 0.0, TILE_M)), (1, 1, seg(1, 0.0, 30.0), "low_snr"),
             (3, 1, seg(3, 0.0, 30.0, 10.0), "low_snr"), (4, 1, seg(4, 0.0, 30.0), "too_wide")]
    res = link_candidates(frame(items), None, clips(), settings)
    assert len(res.rows_raw) == 1
    row = res.rows_raw.iloc[0]
    assert FLAG_RESCUED in row.qa_flags and 0 < row.rescued_frac < 1
    dec = dict(zip(res.decisions.cand_id, res.decisions.link_decision, strict=True))
    assert dec[f"{TILES[1]}:K01"] == DECISION_RESCUED
    assert dec[f"{TILES[3]}:K01"] == DECISION_SOFT_CHAIN
    assert dec[f"{TILES[4]}:K01"] == DECISION_HARD
    assert any(i.code == FLAG_RESCUED for i in res.issues)


def test_rescue_disabled_drops_soft(settings: LinkSettings) -> None:
    from dataclasses import replace

    items = [(0, 1, seg(0, 0.0, TILE_M)), (1, 1, seg(1, 0.0, 30.0), "low_snr")]
    res = link_candidates(frame(items), None, clips(), replace(settings, rescue_soft=False))
    dec = dict(zip(res.decisions.cand_id, res.decisions.link_decision, strict=True))
    assert dec[f"{TILES[1]}:K01"] == DECISION_SOFT_DROPPED
    assert res.rows_raw.member_cand_ids.iloc[0] == f"{TILES[0]}:K01"


def test_same_tile_duplicate_removed(settings: LinkSettings) -> None:
    items = [(0, 1, seg(0, 0.0, TILE_M)), (0, 2, seg(0, 10.0, 30.0, 0.1))]
    res = link_candidates(frame(items), None, clips(), settings)
    assert len(res.rows_raw) == 1
    dec = dict(zip(res.decisions.cand_id, res.decisions.link_decision, strict=True))
    assert dec[f"{TILES[0]}:K02"] == DECISION_DUPLICATE


def test_shuffled_input_is_identical(settings: LinkSettings) -> None:
    items = _three_rows([0, 1, 2]) + [(3, 9, seg(3, 0.0, 40.0, 7.5))]
    a = link_candidates(frame(items), None, clips(), settings).rows_raw
    shuffled = frame(items).sample(frac=1.0, random_state=7)
    b = link_candidates(shuffled, None, clips(), settings).rows_raw
    assert list(a.chain_id) == list(b.chain_id)
    assert list(a.member_cand_ids) == list(b.member_cand_ids)
    assert [g.wkb for g in a.geometry] == [g.wkb for g in b.geometry]


def test_member_gaps_mapped_to_chain(settings: LinkSettings) -> None:
    cid = f"{TILES[0]}:K01"
    cands = frame([(0, 1, seg(0, 0.0, 40.0))], extra={cid: {"gaps_json": "[[10.0, 16.0]]"}})
    row = link_candidates(cands, None, clips(), settings).rows_raw.iloc[0]
    assert json.loads(row.gaps_json) == [[10.0, 16.0]] or json.loads(row.gaps_json) == [[24.0, 30.0]]


def test_parse_gaps_formats() -> None:
    assert parse_gaps(None, "x") == ()
    assert parse_gaps("", "x") == ()
    assert parse_gaps(float("nan"), "x") == ()
    assert parse_gaps("[[5, 3]]", "x") == ((3.0, 5.0),)
    assert parse_gaps('[{"start_m": 1, "end_m": 2}]', "x") == ((1.0, 2.0),)
    with pytest.raises(StageError):
        parse_gaps("[[1]]", "x")
    with pytest.raises(StageError):
        parse_gaps("{nope", "x")


def test_end_snap_to_clip(settings: LinkSettings) -> None:
    near = seg(0, 2.0, 30.0)
    out = extend_chain_ends(near, clips(), settings.snap_m)
    assert out.coords[0][0] == pytest.approx(X0)
    far = seg(0, 4.0, 30.0)
    assert extend_chain_ends(far, clips(), settings.snap_m).equals(far)
    corner = tile_box(tile_ref(TILES[0])).difference(box(X0 - 1, YC - 30, X0 + 1.0, YC + 30))
    out2 = extend_chain_ends(near, {TILES[0]: corner}, settings.snap_m)
    assert out2.coords[0][0] == pytest.approx(X0 + 1.0)


def test_empty_input(settings: LinkSettings) -> None:
    res = link_candidates(frame([]), None, clips(), settings)
    assert len(res.rows_raw) == 0 and "chain_id" in res.rows_raw.columns
