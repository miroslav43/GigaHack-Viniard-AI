"""perception.axis_refine: lateral shift of local row pieces onto the canopy-mask centre line."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_box, tile_ref, utm_to_px
from vineyard.perception.axis_refine import AxisRefineOptions, refine_offsets_m, refine_pieces
from vineyard.perception.corridor import clip_rows_to_tile

TILE = tile_ref("siret3_r021_c012")
OPTS = AxisRefineOptions(band_m=0.35, iterations=3, max_shift_m=0.10, min_px=200)


def _pieces(vs: list[float], u0: float = -100.0, u1: float = 2200.0) -> gpd.GeoDataFrame:
    lines = [LineString(px_to_utm(TILE, np.array([[u0, v], [u1, v]]))) for v in vs]
    rows = gpd.GeoDataFrame({"row_id": [f"V01-R{k + 1:03d}" for k in range(len(vs))], "vineyard_id": "V01"},
                            geometry=lines, crs=f"EPSG:{CRS_EPSG}")
    return clip_rows_to_tile(rows, TILE, tile_box(TILE), margin_m=1.0)


def _band(v_lo: int, v_hi: int, u0: int = 0, u1: int = TILE_PX) -> np.ndarray:
    m = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    m[v_lo:v_hi, u0:u1] = True
    return m


def _v_of(geom: LineString) -> np.ndarray:
    return utm_to_px(TILE, np.asarray(geom.coords))[:, 1]


def test_piece_moves_onto_the_mask_centre_and_keeps_its_direction() -> None:
    pieces = _pieces([500.0])
    mask = _band(497, 509)  # pixel centres 497.5 .. 508.5 -> mean 503.0, i.e. 3 px (7.5 cm) south of the axis
    out = refine_pieces(pieces, mask, TILE, OPTS)
    v = _v_of(out.geometry.iloc[0])
    assert v == pytest.approx([503.0, 503.0], abs=1e-6)
    assert out.geometry.iloc[0].length == pytest.approx(pieces.geometry.iloc[0].length)
    assert list(out["row_id"]) == list(pieces["row_id"]) and list(out.columns) == list(pieces.columns)
    shift = refine_offsets_m(pieces, mask, TILE, OPTS)
    assert abs(shift[0]) == pytest.approx(3 * GSD_M, abs=1e-9)


def test_shift_is_capped_and_needs_support() -> None:
    pieces = _pieces([500.0, 900.0])
    mask = _band(503, 515) | _band(898, 902, u0=0, u1=40)  # row 1: 9 px off; row 2: 160 px < min_px
    out = refine_pieces(pieces, mask, TILE, OPTS)
    assert _v_of(out.geometry.iloc[0]) == pytest.approx([504.0, 504.0], abs=1e-6)  # capped at 0.10 m = 4 px
    assert _v_of(out.geometry.iloc[1]) == pytest.approx([900.0, 900.0], abs=1e-9)


def test_only_pixels_along_the_piece_and_inside_the_band_count() -> None:
    pieces = _pieces([500.0], u0=500.0, u1=1500.0)
    mask = _band(496, 504, 500, 1500) | _band(520, 540, 0, 400) | _band(560, 600, 500, 1500)
    out = refine_pieces(pieces, mask, TILE, OPTS)
    assert _v_of(out.geometry.iloc[0]) == pytest.approx([500.0, 500.0], abs=1e-6)


def test_no_options_or_no_pieces_return_the_input_and_never_mutate() -> None:
    pieces = _pieces([500.0])
    before = pieces.copy()
    mask = _band(497, 509)
    assert refine_pieces(pieces, mask, TILE, None) is pieces
    assert refine_pieces(pieces.iloc[0:0], mask, TILE, OPTS).empty
    refine_pieces(pieces, mask, TILE, OPTS)
    assert pieces.equals(before)
    with pytest.raises(ValueError):
        AxisRefineOptions(band_m=0.0, iterations=3, max_shift_m=0.1, min_px=200)
