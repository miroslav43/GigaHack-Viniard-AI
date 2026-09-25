from __future__ import annotations

import math
import re

import geopandas as gpd
import pytest
from shapely.geometry import box

from vineyard.perception.waste.assign import assign_vineyard_id, assign_waste_ids

CRS = "EPSG:32635"


def blocks(*items: tuple[str, object]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"vineyard_id": [v for v, _ in items]}, geometry=[g for _, g in items], crs=CRS)


V1 = box(0, 0, 100, 100)
V2 = box(200, 0, 300, 100)


def test_box_inside_a_block() -> None:
    a = assign_vineyard_id(box(210, 10, 211, 11), blocks(("V01", V1), ("V02", V2)), 10.0)
    assert (a.vineyard_id, a.dist_block_m) == ("V02", 0.0)


def test_nearest_within_and_beyond_max_distance() -> None:
    b = blocks(("V01", V1))
    near = assign_vineyard_id(box(109.99, 10, 110.99, 11), b, 10.0)
    assert near.vineyard_id == "V01" and near.dist_block_m == pytest.approx(9.99)
    far = assign_vineyard_id(box(110.01, 10, 111, 11), b, 10.0)
    assert far.vineyard_id == "" and far.dist_block_m == pytest.approx(10.01)


def test_overlap_two_blocks_takes_larger_intersection() -> None:
    a = assign_vineyard_id(box(90, 10, 230, 20), blocks(("V01", V1), ("V02", V2)), 10.0)
    assert a.vineyard_id == "V02"


def test_no_blocks() -> None:
    a = assign_vineyard_id(box(0, 0, 1, 1), blocks(), 10.0)
    assert a.vineyard_id == "" and math.isnan(a.dist_block_m)


def frame(rows: list[tuple[str, float, float, float, float]]) -> gpd.GeoDataFrame:
    tiles, xtl, ytl, xbr, ybr = zip(*rows, strict=True)
    return gpd.GeoDataFrame(
        {"tile_id": tiles, "px_xtl": xtl, "px_ytl": ytl, "px_xbr": xbr, "px_ybr": ybr},
        geometry=[box(0, 0, 1, 1)] * len(rows),
        crs=CRS,
    )


def test_ids_sorted_and_edge_pairs() -> None:
    out = assign_waste_ids(
        frame(
            [
                ("siret3_r010_c006", 0.0, 480.0, 30.0, 530.0),
                ("siret3_r010_c005", 900.0, 50.0, 950.0, 90.0),
                ("siret3_r010_c005", 100.0, 50.0, 150.0, 90.0),
                ("siret3_r009_c007", 10.0, 1000.0, 60.0, 1040.0),
                ("siret3_r010_c005", 2000.0, 500.0, 2048.0, 540.0),
            ]
        ),
        1.0,
    )
    got = dict(zip(zip(out["tile_id"], out["px_xtl"], strict=True), out["waste_id"], strict=True))
    assert got[("siret3_r009_c007", 10.0)] == "W0001"
    assert got[("siret3_r010_c005", 100.0)] == "W0002"
    assert got[("siret3_r010_c005", 900.0)] == "W0003"
    assert got[("siret3_r010_c005", 2000.0)] == "W0004a"
    assert got[("siret3_r010_c006", 0.0)] == "W0004b"
    assert all(re.fullmatch(r"^W\d{4}[ab]?$", w) for w in out["waste_id"])


def test_vertical_pair_and_insufficient_overlap() -> None:
    out = assign_waste_ids(
        frame(
            [
                ("siret3_r001_c001", 100.0, 2000.0, 140.0, 2048.0),
                ("siret3_r002_c001", 110.0, 0.0, 150.0, 30.0),
                ("siret3_r001_c002", 1000.0, 2000.0, 1040.0, 2048.0),
                ("siret3_r002_c002", 1030.0, 0.0, 1070.0, 30.0),
            ]
        ),
        1.0,
    )
    assert list(out["waste_id"]) == ["W0001a", "W0002", "W0001b", "W0003"]


def test_edge_tolerance() -> None:
    out = assign_waste_ids(
        frame(
            [
                ("siret3_r001_c001", 2000.0, 100.0, 2046.0, 140.0),
                ("siret3_r001_c002", 0.0, 100.0, 30.0, 140.0),
            ]
        ),
        1.0,
    )
    assert list(out["waste_id"]) == ["W0001", "W0002"]


def test_empty_frame() -> None:
    empty = gpd.GeoDataFrame(
        {"tile_id": [], "px_xtl": [], "px_ytl": [], "px_xbr": [], "px_ybr": []}, geometry=[], crs=CRS
    )
    assert assign_waste_ids(empty, 1.0).empty
