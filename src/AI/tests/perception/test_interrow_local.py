"""perception.interrow_local: per-tile interrow pieces from the tile's own row pieces (annotation rule 5.1)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString

from vineyard.config import load_config
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.geo.tiling import CRS_EPSG, TILE_PX, px_to_utm, tile_box, tile_ref
from vineyard.perception.corridor import clip_rows_to_tile
from vineyard.perception.interrow import build_interrow_bands, consecutive_pairs, row_positions
from vineyard.perception.interrow_local import LocalPairOptions, adjacent_pairs, tile_interrow_pieces

TILE = tile_ref("siret3_r021_c012")
OPTS = LocalPairOptions(offset_m=0.30, max_spacing_m=4.0, min_overlap_frac=0.2, edge_extend_m=10.0, edge_tol_m=0.05,
                        hole_min_area_m2=1.0, notch_width_m=0.025, min_piece_m2=0.25)


def _rows(specs: list[tuple[str, float, float, float]]) -> gpd.GeoDataFrame:
    """(row_id, v, u0, u1): horizontal rows in px of TILE (u may leave the tile)."""
    lines = [LineString(px_to_utm(TILE, np.array([[u0, v], [u1, v]]))) for _, v, u0, u1 in specs]
    ids = [r for r, *_ in specs]
    return gpd.GeoDataFrame({"row_id": ids, "vineyard_id": "V01", "row_index": [int(r[-3:]) for r in ids]},
                            geometry=lines, crs=f"EPSG:{CRS_EPSG}")


def _bands(rows: gpd.GeoDataFrame, pairs: list[tuple[str, str]] | None = None) -> gpd.GeoDataFrame:
    frame = consecutive_pairs(rows) if pairs is None else pd.DataFrame(
        [{"vineyard_id": "V01", "row_a": a, "row_b": b} for a, b in pairs], columns=["vineyard_id", "row_a", "row_b"])
    return build_interrow_bands(rows, frame, None, load_config().interrow, 0.025)


def _pieces(rows: gpd.GeoDataFrame, bands: gpd.GeoDataFrame, clip: shapely.Geometry | None = None) -> gpd.GeoDataFrame:
    local = clip_rows_to_tile(rows, TILE, tile_box(TILE))
    return tile_interrow_pieces(local, bands, TILE, tile_box(TILE) if clip is None else clip, None,
                                row_positions(rows), OPTS)


def test_three_full_rows_give_two_pieces_with_the_global_ids() -> None:
    rows = _rows([("V01-R001", 500.0, -400.0, 2400.0), ("V01-R002", 620.0, -400.0, 2400.0),
                  ("V01-R003", 740.0, -400.0, 2400.0)])
    out = _pieces(rows, _bands(rows))
    assert out["piece_id"].tolist() == [f"V01-I001@{TILE.tile_id}", f"V01-I002@{TILE.tile_id}"]
    assert out["interrow_id"].tolist() == ["V01-I001", "V01-I002"]
    assert out["row_left_id"].tolist() == ["V01-R001", "V01-R002"]
    assert out["row_right_id"].tolist() == ["V01-R002", "V01-R003"]
    assert all(is_valid_id(IdKind.INTERROW_PIECE, p) for p in out["piece_id"])
    # the rows cross the tile: each band reaches both tile edges, 51.2 m long and 3 m - 2 x 0.30 m wide
    assert out["area_m2"].tolist() == pytest.approx([51.2 * 2.4] * 2, rel=1e-6)
    assert out["width_mean_m"].tolist() == pytest.approx([2.4, 2.4], rel=1e-6)
    assert (out["tile_id"] == TILE.tile_id).all() and (out["n_notches"] == 0).all()
    assert out.geometry.iloc[0].exterior.is_ccw


def test_a_row_of_the_neighbour_tile_adds_no_corner_piece() -> None:
    # R003 lies in the tile to the south: the global band R002-R003 straddles the edge, but R003 has no
    # piece here, so the tile only gets R001-R002 (the global cut would add a strip of I002)
    rows = _rows([("V01-R001", 1900.0, -400.0, 2400.0), ("V01-R002", 2020.0, -400.0, 2400.0),
                  ("V01-R003", 2140.0, 0.0, 2400.0)])
    bands = _bands(rows)
    assert bands.intersects(tile_box(TILE)).sum() == 2
    out = _pieces(rows, bands)
    assert out["interrow_id"].tolist() == ["V01-I001"]


def test_collinear_fragments_are_never_paired_and_each_pairs_with_its_neighbour() -> None:
    rows = _rows([("V01-R001", 500.0, -400.0, 900.0), ("V01-R002", 500.0, 1000.0, 2400.0),
                  ("V01-R003", 620.0, -400.0, 2400.0)])
    local = clip_rows_to_tile(rows, TILE, tile_box(TILE))
    assert adjacent_pairs(list(local.geometry), list(local["row_id"]), set(), OPTS) == [(0, 2), (1, 2)]
    out = _pieces(rows, _bands(rows, [("V01-R002", "V01-R003")]))
    assert list(zip(out["interrow_id"], out["row_left_id"], out["row_right_id"], strict=True)) == [
        ("V01-I002", "V01-R002", "V01-R003"), ("V01-I003", "V01-R001", "V01-R003")]  # I003: a fresh id


def test_wide_pairs_need_a_global_pair() -> None:
    rows = _rows([("V01-R001", 500.0, -400.0, 2400.0), ("V01-R003", 740.0, -400.0, 2400.0)])  # 6 m apart
    assert _pieces(rows, _bands(rows, [])).empty
    skip = _bands(rows, [("V01-R001", "V01-R003")])
    out = _pieces(rows, skip)
    assert out["row_left_id"].tolist() == ["V01-R001"] and out["interrow_id"].tolist() == skip["interrow_id"].tolist()


def test_ends_on_a_nodata_edge_are_extended_to_the_clip() -> None:
    rows = _rows([("V01-R001", 500.0, 300.0, 2400.0), ("V01-R002", 620.0, 300.0, 2400.0)])
    clip = shapely.transform(shapely.box(300.0, 0.0, float(TILE_PX), float(TILE_PX)), lambda uv: px_to_utm(TILE, uv))
    local = clip_rows_to_tile(rows, TILE, tile_box(TILE))
    out = tile_interrow_pieces(local, _bands(rows), TILE, clip, None, row_positions(rows), OPTS)
    west = LineString(px_to_utm(TILE, np.array([[300.0, 0.0], [300.0, float(TILE_PX)]])))
    assert len(out) == 1 and out.geometry.iloc[0].intersection(west).length == pytest.approx(2.4, abs=1e-6)


def test_pairs_without_their_own_band_take_the_overlapping_band_id() -> None:
    # R002 continues R001's line in this tile without being linked; the global pairs are R001-R003 and
    # R002-R004, and the local pair R002-R003 lies inside the band R002-R004 of the other side
    rows = _rows([("V01-R001", 500.0, -400.0, 1000.0), ("V01-R002", 500.0, 1100.0, 2400.0),
                  ("V01-R003", 620.0, -400.0, 2400.0)])
    bands = _bands(rows, [("V01-R001", "V01-R003")])
    wide = bands.assign(geometry=[bands.geometry.iloc[0].union(shapely.transform(
        shapely.box(1100.0, 500.0, 2048.0, 620.0), lambda uv: px_to_utm(TILE, uv)))])
    out = _pieces(rows, wide)
    assert out["interrow_id"].tolist() == ["V01-I001", "V01-I001"]
    assert out["piece_id"].tolist() == [f"V01-I001@{TILE.tile_id}", f"V01-I001@{TILE.tile_id}#2"]
    assert out["row_left_id"].tolist() == ["V01-R001", "V01-R002"]


def test_no_rows_or_one_row_give_no_pieces_and_inputs_are_not_modified() -> None:
    rows = _rows([("V01-R001", 500.0, -400.0, 2400.0)])
    local = clip_rows_to_tile(rows, TILE, tile_box(TILE))
    before = local.copy()
    assert tile_interrow_pieces(local, _bands(rows), TILE, tile_box(TILE), None, row_positions(rows), OPTS).empty
    assert tile_interrow_pieces(local.iloc[0:0], _bands(rows), TILE, tile_box(TILE), None, {}, OPTS).empty
    assert local.equals(before)
