"""row_trim: row ends that are not continued stop at the last canopy (label audit 2026-09-26)."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from vineyard.config import load_config
from vineyard.geo.tiling import CRS_EPSG
from vineyard.perception.row_trim import TrimOptions, canopy_span, trim_row_pieces, trimmed_span

OPTS = TrimOptions(enabled=True, band_m=0.35, margin_m=0.2, min_overshoot_m=1.0)


def _vines(x0: float, x1: float, step: float = 1.0) -> list:
    out, x = [], x0
    while x <= x1 + 1e-9:
        out.append(box(x - 0.3, -0.3, x + 0.3, 0.3))
        x += step
    return out


def test_options_from_default_config() -> None:
    opts = TrimOptions.from_config(load_config().row_structure)
    assert opts.enabled and opts.band_m > 0 and opts.min_overshoot_m >= opts.margin_m


def test_canopy_span_ignores_other_rows() -> None:
    line = LineString([(0, 0), (50, 0)])
    far = [box(0, 2.2, 50, 2.8)]
    assert canopy_span(line, far, 0.35) is None
    assert canopy_span(line, _vines(10, 30), 0.35) == pytest.approx((9.7, 30.3), abs=0.05)


def test_free_ends_pulled_back_to_last_vine() -> None:
    line = LineString([(0, 0), (50, 0)])
    lo, hi = trimmed_span(line, True, True, _vines(10, 30), OPTS)
    assert lo == pytest.approx(9.5, abs=0.05) and hi == pytest.approx(30.5, abs=0.05)


def test_continued_end_kept_and_small_overshoot_kept() -> None:
    line = LineString([(0, 0), (50, 0)])
    assert trimmed_span(line, False, False, _vines(10, 30), OPTS) == (0.0, 50.0)
    lo, hi = trimmed_span(line, True, True, _vines(0.8, 49.2), OPTS)
    assert (lo, hi) == (0.0, 50.0)


def test_trim_row_pieces_updates_along_and_keeps_continued_end() -> None:
    pieces = gpd.GeoDataFrame(
        {"row_id": ["V01-R001"], "along_from_m": [0.0], "along_to_m": [50.0]},
        geometry=gpd.GeoSeries([LineString([(0, 0), (50, 0)])], crs=CRS_EPSG), crs=CRS_EPSG)
    canopies = gpd.GeoDataFrame(geometry=gpd.GeoSeries(_vines(10, 30), crs=CRS_EPSG), crs=CRS_EPSG)
    out = trim_row_pieces(pieces, {"V01-R001": 80.0}, canopies, OPTS)  # the row continues past x=50
    assert out.along_from_m.iloc[0] == pytest.approx(9.5, abs=0.05)
    assert out.along_to_m.iloc[0] == pytest.approx(50.0)
    assert out.geometry.iloc[0].length == pytest.approx(40.5, abs=0.05)
    assert pieces.geometry.iloc[0].length == 50.0  # input untouched


def test_disabled_or_no_canopy_is_identity() -> None:
    pieces = gpd.GeoDataFrame(
        {"row_id": ["V01-R001"], "along_from_m": [0.0], "along_to_m": [50.0]},
        geometry=gpd.GeoSeries([LineString([(0, 0), (50, 0)])], crs=CRS_EPSG), crs=CRS_EPSG)
    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=CRS_EPSG), crs=CRS_EPSG)
    assert trim_row_pieces(pieces, {"V01-R001": 50.0}, empty, OPTS) is pieces
    off = TrimOptions(enabled=False, band_m=0.35, margin_m=0.2, min_overshoot_m=1.0)
    canopies = gpd.GeoDataFrame(geometry=gpd.GeoSeries(_vines(10, 30), crs=CRS_EPSG), crs=CRS_EPSG)
    assert trim_row_pieces(pieces, {"V01-R001": 50.0}, canopies, off) is pieces


def test_sparse_canopy_young_planting_is_not_trimmed() -> None:
    """Canopies cover < min_occupancy of their span: the canopy stage missed most vines, keep the row."""
    line = LineString([(0, 0), (50, 0)])
    sparse = _vines(10, 30, step=5.0)  # 0.6 m of every 5 m
    opts = TrimOptions(enabled=True, band_m=0.35, margin_m=0.2, min_overshoot_m=1.0, min_occupancy=0.3)
    assert trimmed_span(line, True, True, sparse, opts) == (0.0, 50.0)
    assert trimmed_span(line, True, True, _vines(10, 30), opts)[0] == pytest.approx(9.5, abs=0.05)


def test_end_on_tile_edge_is_kept() -> None:
    pieces = gpd.GeoDataFrame(
        {"row_id": ["V01-R001"], "along_from_m": [0.0], "along_to_m": [50.0]},
        geometry=gpd.GeoSeries([LineString([(0, 0), (50, 0)])], crs=CRS_EPSG), crs=CRS_EPSG)
    canopies = gpd.GeoDataFrame(geometry=gpd.GeoSeries(_vines(10, 30), crs=CRS_EPSG), crs=CRS_EPSG)
    edge = box(-10, -10, 50, 10).boundary  # x = 50 lies on the tile edge
    out = trim_row_pieces(pieces, {"V01-R001": 50.0}, canopies, OPTS, edge)
    assert out.along_from_m.iloc[0] == pytest.approx(9.5, abs=0.05)
    assert out.along_to_m.iloc[0] == pytest.approx(50.0)
