"""Unit tests for the c2 interrow cover override."""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from fte.cover.override import (
    CoverThresholds,
    classify_frac,
    is_borderline,
    override_cover,
    veg_frac2,
)

TILE = "siret3_r006_c004"
X0, Y0 = 629196.8, 5220915.2
TH = CoverThresholds(0.25, 0.75)


def _piece(cx: float, cy: float) -> object:
    return box(X0 + cx - 0.5, Y0 - cy - 0.5, X0 + cx + 0.5, Y0 - cy + 0.5)


def test_classify_frac_matches_rule_thresholds() -> None:
    assert classify_frac(0.1, TH) == "bare_soil"
    assert classify_frac(0.25, TH) == "mixed"
    assert classify_frac(0.75, TH) == "mixed"
    assert classify_frac(0.76, TH) == "vegetation"


def test_is_borderline() -> None:
    assert is_borderline("cover_borderline")
    assert is_borderline("missing_row_suspect;cover_borderline")
    assert not is_borderline("")
    assert not is_borderline(None)


def test_veg_frac2_counts_valid_pixels_only() -> None:
    c2 = np.zeros((2048, 2048), dtype=np.uint8)
    c2[:, : int(10 / 0.025)] = 255  # left of x = 10 m is vegetation
    poly = _piece(10.0, 10.0)  # half left, half right
    assert abs(veg_frac2(poly, c2, None, X0, Y0) - 0.5) < 0.03
    valid = np.zeros_like(c2, dtype=bool)
    assert math.isnan(veg_frac2(poly, c2, valid, X0, Y0))


def test_override_only_borderline_assessable_pieces() -> None:
    polys = [_piece(5, 5), _piece(10, 5), _piece(15, 5)]
    frame = gpd.GeoDataFrame({
        "piece_id": ["V01-I001@" + TILE, "V01-I002@" + TILE, "V01-I003@" + TILE], "tile_id": [TILE] * 3,
        "interrow_cover": ["bare_soil", "mixed", "unassessable"], "veg_frac": [0.22, 0.30, 0.3],
        "qa_flags": ["cover_borderline", "", "cover_borderline"],
    }, geometry=polys, crs="EPSG:32635")
    before = frame.copy(deep=True)
    c2 = np.full((2048, 2048), 255, dtype=np.uint8)
    table = override_cover(frame, lambda t: c2 if t == TILE else None, TH)
    assert list(table["piece_id"]) == ["V01-I001@" + TILE]
    row = table.iloc[0]
    assert row["interrow_cover"] == "mixed" and bool(row["changed"])
    assert abs(row["veg_frac2"] - 1.0) < 1e-9
    assert frame.equals(before)
    empty = override_cover(frame, lambda t: None, TH)
    assert len(empty) == 0
