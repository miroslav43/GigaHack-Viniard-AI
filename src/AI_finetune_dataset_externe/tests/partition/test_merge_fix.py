"""Unit tests for the optional merge-fix of split plant fragments."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box

from fte.canopy.merge_fix import MergeParams, bridge, merge_fragments

TILE = "siret3_r006_c004"
X0, Y0 = 629196.8, 5220915.2
CX, CY = X0 + 20.0, Y0 - 20.0


def _frames(polys: list[Polygon]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    n = len(polys)
    can = gpd.GeoDataFrame({
        "canopy_id": [f"{TILE}:C{k + 1:04d}" for k in range(n)], "tile_id": [TILE] * n,
        "vineyard_id": ["V01"] * n, "row_id": ["V01-R001"] * n, "area_m2": [p.area for p in polys],
        "n_vertices": np.array([4] * n, dtype="int16"), "along_m": np.ones(n, dtype="float32"),
        "is_clump": [False] * n, "touches_edge": [False] * n, "confidence": np.ones(n, dtype="float32"),
        "qa_flags": [""] * n,
    }, geometry=polys, crs="EPSG:32635")
    rows = gpd.GeoDataFrame({"piece_id": [f"V01-R001@{TILE}"], "row_id": ["V01-R001"], "tile_id": [TILE]},
                            geometry=[LineString([(CX - 10, CY), (CX + 10, CY)])], crs="EPSG:32635")
    return can, rows


def test_bridge_is_one_polygon_with_tiny_extra_area() -> None:
    a, b = box(CX, CY, CX + 0.5, CY + 0.4), box(CX + 0.6, CY, CX + 1.1, CY + 0.4)
    merged = bridge(a, b, 0.05)
    assert isinstance(merged, Polygon)
    assert 0 < merged.area - (a.area + b.area) < 0.01  # gap 0.1 m x 0.05 m bridge (+ square caps)


def test_merge_close_fragments_only() -> None:
    a = box(CX, CY - 0.2, CX + 0.5, CY + 0.2)
    b = box(CX + 0.6, CY - 0.2, CX + 1.1, CY + 0.2)  # gap 0.1 m
    far = box(CX + 3.0, CY - 0.2, CX + 3.5, CY + 0.2)
    can, rows = _frames([a, b, far])
    before = can.copy(deep=True)
    out = merge_fragments(can, rows, MergeParams(max_gap_m=0.15, max_len_m=1.4))
    assert len(out) == 2
    assert out["canopy_id"].is_unique and out["canopy_id"].str.match(r"^siret3_r\d{3}_c\d{3}:C\d{4}$").all()
    assert out.dtypes.to_dict() == can.dtypes.to_dict()
    assert abs(shapely.union_all(out.geometry.values).area - shapely.union_all(can.geometry.values).area) < 0.01
    assert can.equals(before)


def test_touching_or_too_long_pairs_are_kept() -> None:
    a = box(CX, CY - 0.2, CX + 0.5, CY + 0.2)
    touching = box(CX + 0.5, CY - 0.2, CX + 1.0, CY + 0.2)
    can, rows = _frames([a, touching])
    assert len(merge_fragments(can, rows, MergeParams(max_gap_m=0.15, max_len_m=3.0))) == 2
    long_b = box(CX + 0.6, CY - 0.2, CX + 2.0, CY + 0.2)
    can2, rows2 = _frames([a, long_b])
    assert len(merge_fragments(can2, rows2, MergeParams(max_gap_m=0.15, max_len_m=1.4))) == 2


def test_contact_map_blocks_merge() -> None:
    a = box(CX, CY - 0.2, CX + 0.5, CY + 0.2)
    b = box(CX + 0.6, CY - 0.2, CX + 1.1, CY + 0.2)
    can, rows = _frames([a, b])
    contact = np.full((2048, 2048), 255, dtype=np.uint8)
    out = merge_fragments(can, rows, MergeParams(max_gap_m=0.15, max_len_m=1.4), lambda t: contact)
    assert len(out) == 2
