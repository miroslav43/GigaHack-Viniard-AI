"""Confirmed waste as extra route targets (vineyard.route.extra_waste)."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from vineyard.contracts.enums import TargetKind
from vineyard.geo.tiling import TileRef
from vineyard.perception.waste.types import BoxPx, Category, Confirmation
from vineyard.route.extra_waste import confirmation_box, extra_waste, with_extra_waste
from vineyard.route.target_waste import waste_drafts

GSD = 0.025
TILE = TileRef("siret3_r010_c010", 10, 10, 1000.0, 5000.0)
CRS = "EPSG:32635"


def conf(xtl: float, ytl: float, xbr: float, ybr: float, tile: str = TILE.tile_id) -> Confirmation:
    return Confirmation(1, tile, BoxPx(xtl, ytl, xbr, ybr), "add", Category.BAG, "test", "")


def annset_waste(*geoms, ids=None) -> gpd.GeoDataFrame:
    ids = ids or [f"W{k + 1:04d}" for k in range(len(geoms))]
    return gpd.GeoDataFrame({"waste_id": ids, "vineyard_id": ["V01"] * len(geoms)}, geometry=list(geoms), crs=CRS)


def test_confirmation_box_maps_tile_px_to_utm():
    g = confirmation_box(conf(0, 0, 40, 80), TILE, GSD)
    assert g.bounds == pytest.approx((1000.0, 4998.0, 1001.0, 5000.0))


def test_extra_waste_skips_boxes_already_in_the_annset():
    near = confirmation_box(conf(100, 100, 120, 120), TILE, GSD)
    extra = extra_waste(annset_waste(near.buffer(0.2)), [conf(100, 100, 120, 120), conf(1000, 1000, 1020, 1020)],
                        {TILE.tile_id: TILE}, GSD, 1.0)
    assert list(extra.waste_id) == ["W9001"]
    assert extra.geometry.iloc[0].bounds == pytest.approx(confirmation_box(conf(1000, 1000, 1020, 1020), TILE, GSD).bounds)


def test_extra_waste_dedupes_confirmations_and_skips_used_ids_and_unknown_tiles():
    confs = [conf(500, 500, 520, 520), conf(505, 505, 525, 525), conf(900, 900, 920, 920),
             conf(10, 10, 20, 20, tile="siret3_r099_c099")]
    far = box(0, 0, 1, 1)
    extra = extra_waste(annset_waste(far, ids=["W9001"]), confs, {TILE.tile_id: TILE}, GSD, 1.0)
    assert list(extra.waste_id) == ["W9002", "W9003"]
    assert set(extra.vineyard_id) == {""}


def test_merged_waste_gives_one_target_per_object():
    base = annset_waste(box(1100, 4900, 1101, 4901))
    extra = extra_waste(base, [conf(200, 200, 240, 240)], {TILE.tile_id: TILE}, GSD, 1.0)
    drafts = waste_drafts(with_extra_waste(base, extra))
    assert [d.waste_id for d in drafts] == ["W0001", "W9001"]
    assert all(d.kind is TargetKind.WASTE for d in drafts)
    assert (drafts[1].x, drafts[1].y) == pytest.approx((1000 + 220 * GSD, 5000 - 220 * GSD))


def test_no_extra_keeps_the_annset_frame():
    base = annset_waste(box(0, 0, 1, 1))
    assert with_extra_waste(base, extra_waste(base, [], {}, GSD, 1.0)) is base
