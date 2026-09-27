"""Waste detections inside the blocks for the web map (vineyard.web.detected_waste)."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from vineyard.web.detected_waste import DetectedWasteParams, detected_waste

CRS = "EPSG:32635"
GSD = 0.025
P = DetectedWasteParams(10, 15)


def cand(x: float, y: float, w_px: float, h_px: float, reason: str | None = "near_axis", probe=0.4,
         tile: str = "siret3_r010_c010") -> dict:
    return {"tile_id": tile, "px_xtl": 0.0, "px_ytl": y, "px_xbr": w_px, "px_ybr": y + h_px, "probe_p": probe,
            "reject_reason": reason, "geometry": box(x, 0, x + w_px * GSD, h_px * GSD)}


def frame(rows: list[dict]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


BLOCKS = gpd.GeoDataFrame({"vineyard_id": ["V01"]}, geometry=[box(0, -5, 50, 5)], crs=CRS)
NO_WASTE = gpd.GeoDataFrame({"waste_id": []}, geometry=[], crs=CRS)


def test_keeps_every_filter_reason_inside_the_blocks_at_least_10_by_15_px():
    cands = frame([cand(1, 0, 10, 15, "bright_soil"), cand(5, 1, 16, 10, "too_small_white", None),
                   cand(9, 2, 12, 20, None), cand(20, 3, 9, 30), cand(25, 4, 14, 14), cand(80, 5, 20, 20)])
    out = detected_waste(cands, BLOCKS, NO_WASTE, P)
    assert list(out.waste_id) == ["D00001", "D00002", "D00003"]
    assert set(out.vineyard_id) == {"V01"}
    assert list(out.confidence) == pytest.approx([0.4, 0.0, 0.4])


def test_drops_nms_duplicates_and_boxes_overlapping_annotated_waste():
    cands = frame([cand(1, 0, 20, 20, "nms"), cand(10, 1, 20, 20), cand(30, 2, 20, 20)])
    annotated = gpd.GeoDataFrame({"waste_id": ["W0001"]}, geometry=[box(10, 0, 10.2, 0.2)], crs=CRS)
    out = detected_waste(cands, BLOCKS, annotated, P)
    assert len(out) == 1
    assert out.geometry.iloc[0].bounds[0] == pytest.approx(30)


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        DetectedWasteParams(15, 10)
