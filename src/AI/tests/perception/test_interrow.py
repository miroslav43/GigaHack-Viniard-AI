"""perception.interrow: bands between neighbour rows, per-tile pieces, cover classification."""

from __future__ import annotations

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString, Polygon, box

from tests.helpers.examples import interrow_polygons_utm, load_examples, to_utm
from vineyard.config import load_config
from vineyard.contracts.enums import InterrowCover
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.geo.raster import read_tile, valid_mask
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_box, tile_ref
from vineyard.perception.interrow import (
    BORDERLINE_FLAG,
    INTERROW_COLUMNS,
    PIECE_COLUMNS,
    CoverStats,
    band_polygon,
    build_interrow_bands,
    classify_cover,
    classify_pieces,
    consecutive_pairs,
    cover_stats,
    cut_interrows_to_tile,
    geometric_positions,
    punch_holes,
    row_positions,
)
from vineyard.perception.types import VIS_NODATA, VIS_OK, VIS_SHADOW
from vineyard.perception.vegmask import compute_tile_masks

CFG = load_config()
TILE = tile_ref("siret3_r021_c012")
NOTCH_M = CFG.export.cvat.notch_width_px * GSD_M


def _rows(ys: list[float], x0: float = 0.0, x1: float = 30.0, ids: list[str] | None = None) -> gpd.GeoDataFrame:
    lines = [LineString([(x0, y), (x1, y)]) for y in ys]
    return gpd.GeoDataFrame(
        {"row_id": ids or [f"V01-R{k + 1:03d}" for k in range(len(ys))], "vineyard_id": "V01"},
        geometry=lines, crs=f"EPSG:{CRS_EPSG}",
    )


# ------------------------------------------------------------------ band geometry


def test_band_polygon_parallel_rows() -> None:
    band = band_polygon(LineString([(0, 0), (10, 0)]), LineString([(0, 3), (10, 3)]), 0.3)
    assert band is not None and band.exterior.is_ccw
    assert band.area == pytest.approx(10 * 2.4)
    assert band.bounds == pytest.approx((0, 0.3, 10, 2.7))


def test_band_polygon_shorter_row_rule_and_reversed_direction() -> None:
    band = band_polygon(LineString([(0, 0), (20, 0)]), LineString([(15, 3), (5, 3)]), 0.3)
    assert band is not None
    assert band.bounds == pytest.approx((5, 0.3, 15, 2.7))
    assert band_polygon(LineString([(0, 0), (4, 0)]), LineString([(5, 3), (9, 3)]), 0.3) is None


def test_band_polygon_oblique_rows() -> None:
    left = LineString([(0, 0), (10, 10)])
    right = LineString([(3, 0), (13, 10)])
    band = band_polygon(left, right, 0.3)
    assert band is not None and band.is_valid
    width = 3 / np.sqrt(2) - 0.6
    assert band.area == pytest.approx(width * (np.sqrt(200) - 3 / np.sqrt(2)), rel=1e-6)


def test_punch_holes_notches_interior_forbidden_areas() -> None:
    band = box(0, 0, 20, 2)
    small = box(1, 0.5, 1.5, 1.0)  # 0.25 m2 < 1 m2: ignored
    parts, n = punch_holes(band, small, 1.0, NOTCH_M)
    assert n == 0 and parts[0].equals(band)
    hole = box(8, 0.5, 10, 1.5)
    parts, n = punch_holes(band, hole, 1.0, NOTCH_M)
    assert n == 1 and all(not p.interiors for p in parts)
    assert sum(p.area for p in parts) == pytest.approx(40 - 2 - NOTCH_M * 0.5, abs=0.05)
    edge = box(15, -1, 17, 1)  # touches the boundary: plain subtraction
    parts, n = punch_holes(band, edge, 1.0, NOTCH_M)
    assert n == 0 and sum(p.area for p in parts) == pytest.approx(38)
    assert punch_holes(band, None, 1.0, NOTCH_M) == ([band], 0)


def test_band_design_case_shorter_row_quad() -> None:
    band = band_polygon(LineString([(0, 0), (40, 0)]), LineString([(5, 2.5), (35, 2.5)]), 0.3)
    assert band is not None
    assert len(band.exterior.coords) - 1 == 4
    assert band.area == pytest.approx(57.0) and band.bounds == pytest.approx((5, 0.3, 35, 2.2))


# ------------------------------------------------------------------ numbering


def test_geometric_positions_and_consecutive_pairs() -> None:
    rows = _rows([0.0, 6.0, 3.0], ids=["V01-R001", "V01-R003", "V01-R002"])
    pos = geometric_positions(rows)
    assert pos == {"V01-R003": 1, "V01-R002": 2, "V01-R001": 3}
    pairs = consecutive_pairs(rows)
    assert pairs[["row_a", "row_b"]].values.tolist() == [["V01-R003", "V01-R002"], ["V01-R002", "V01-R001"]]


def test_row_positions_prefer_the_blocks_row_index() -> None:
    rows = _rows([0.0, 3.0, 6.0]).assign(row_index=[1, 2, 3])  # blocks' own order (differs from n·c here)
    assert row_positions(rows) == {"V01-R001": 1, "V01-R002": 2, "V01-R003": 3}
    bands = build_interrow_bands(rows, consecutive_pairs(rows), None, CFG.interrow, NOTCH_M)
    assert bands[["interrow_id", "row_left_id", "row_right_id"]].values.tolist() == [
        ["V01-I001", "V01-R001", "V01-R002"], ["V01-I002", "V01-R002", "V01-R003"]]
    dup = rows.assign(row_index=[1, 1, 2])
    assert row_positions(dup) == geometric_positions(dup)
    missing = rows.assign(row_index=[1.0, float("nan"), 3.0])
    assert row_positions(missing) == geometric_positions(missing)
    stray = pd.DataFrame({"vineyard_id": ["V01"], "row_a": ["V01-R001"], "row_b": ["V01-R777"]})
    with pytest.raises(ValueError, match="V01-R777"):
        build_interrow_bands(rows, stray, None, CFG.interrow, NOTCH_M)


def test_build_bands_ids_follow_geometric_order_not_row_numbers() -> None:
    rows = _rows([9.0, 6.0, 3.0], ids=["V01-R900", "V01-R001", "V01-R002"])
    pairs = consecutive_pairs(rows)
    bands = build_interrow_bands(rows, pairs, None, CFG.interrow, NOTCH_M)
    assert list(bands.columns) == [*INTERROW_COLUMNS, "geometry"]
    assert bands["interrow_id"].tolist() == ["V01-I001", "V01-I002"]
    assert bands["row_left_id"].tolist() == ["V01-R900", "V01-R001"]
    assert all(is_valid_id(IdKind.INTERROW, i) for i in bands["interrow_id"])
    assert bands["width_mean_m"].iloc[0] == pytest.approx(2.4)
    assert bands["length_m"].iloc[0] == pytest.approx(30.0)


def test_skip_one_pair_and_collisions_get_unique_ids() -> None:
    rows = _rows([10.0, 5.0, 2.5])
    pairs = pd.DataFrame({"vineyard_id": ["V01", "V01"], "row_a": ["V01-R001", "V01-R001"],
                          "row_b": ["V01-R002", "V01-R003"]})
    bands = build_interrow_bands(rows, pairs, None, CFG.interrow, NOTCH_M)
    assert bands["interrow_id"].tolist() == ["V01-I001", "V01-I002"]
    assert bands.loc[1, "row_right_id"] == "V01-R003"


def test_build_bands_empty_and_forbidden() -> None:
    rows = _rows([0.0, 3.0])
    empty = build_interrow_bands(rows, pd.DataFrame(columns=["vineyard_id", "row_a", "row_b"]), None,
                                 CFG.interrow, NOTCH_M)
    assert len(empty) == 0 and list(empty.columns) == [*INTERROW_COLUMNS, "geometry"]
    forbidden = box(10, 1, 12, 2)
    bands = build_interrow_bands(rows, consecutive_pairs(rows), forbidden, CFG.interrow, NOTCH_M)
    assert bands["n_notches"].iloc[0] == 1 and bands.geometry.iloc[0].is_valid


def test_n_rows_give_n_minus_one_bands_and_none_across_blocks() -> None:
    a = _rows([0.0, 3.0, 6.0, 9.0])
    b = _rows([12.0, 15.0], ids=["V02-R001", "V02-R002"]).assign(vineyard_id="V02")
    rows = gpd.GeoDataFrame(pd.concat([a, b], ignore_index=True), crs=a.crs)
    bands = build_interrow_bands(rows, consecutive_pairs(rows), None, CFG.interrow, NOTCH_M)
    assert bands["interrow_id"].tolist() == ["V01-I001", "V01-I002", "V01-I003", "V02-I001"]
    for vid, left, right in zip(bands["vineyard_id"], bands["row_left_id"], bands["row_right_id"], strict=True):
        assert left.startswith(vid) and right.startswith(vid)


# ------------------------------------------------------------------ tiles and cover


def _tile_rows(vs: list[float]) -> gpd.GeoDataFrame:
    lines = [LineString(px_to_utm(TILE, np.array([[-400.0, v], [2500.0, v]]))) for v in vs]
    return gpd.GeoDataFrame({"row_id": [f"V01-R{k + 1:03d}" for k in range(len(vs))], "vineyard_id": "V01"},
                            geometry=lines, crs=f"EPSG:{CRS_EPSG}")


def test_cut_to_tile_explodes_with_hash_ids() -> None:
    rows = _tile_rows([500.0, 620.0])
    bands = build_interrow_bands(rows, consecutive_pairs(rows), None, CFG.interrow, NOTCH_M)
    hole = shapely.transform(box(1000, 0, 1100, 2048), lambda uv: px_to_utm(TILE, uv))
    clip = tile_box(TILE).difference(hole)
    pieces = cut_interrows_to_tile(bands, TILE, clip, CFG.export.min_interrow_piece_m2)
    assert len(pieces) == 2
    tid = TILE.tile_id
    assert pieces["piece_id"].tolist() == [f"V01-I001@{tid}", f"V01-I001@{tid}#2"]
    assert all(is_valid_id(IdKind.INTERROW_PIECE, p) for p in pieces["piece_id"])
    assert pieces.geometry.iloc[0].centroid.x < pieces.geometry.iloc[1].centroid.x
    assert pieces["width_mean_m"].iloc[0] == pytest.approx(3.0 - 0.6, rel=1e-3)
    other = cut_interrows_to_tile(bands, tile_ref("siret3_r005_c000"), tile_box(tile_ref("siret3_r005_c000")), 0.25)
    assert len(other) == 0


def test_band_across_a_tile_edge_splits_into_two_pieces_with_the_same_area() -> None:
    lines = [LineString(px_to_utm(TILE, np.array([[1500.0, v], [2600.0, v]]))) for v in (500.0, 620.0)]
    rows = gpd.GeoDataFrame({"row_id": ["V01-R001", "V01-R002"], "vineyard_id": "V01"}, geometry=lines,
                            crs=f"EPSG:{CRS_EPSG}")
    bands = build_interrow_bands(rows, consecutive_pairs(rows), None, CFG.interrow, NOTCH_M)
    right = tile_ref("siret3_r021_c013")
    here = cut_interrows_to_tile(bands, TILE, tile_box(TILE), 0.25)
    there = cut_interrows_to_tile(bands, right, tile_box(right), 0.25)
    assert len(here) == len(there) == 1
    assert here.area.sum() + there.area.sum() == pytest.approx(bands.area.sum(), rel=1e-9)


def test_cover_stats_and_classify_pieces() -> None:
    rows = _tile_rows([500.0, 620.0])
    bands = build_interrow_bands(rows, consecutive_pairs(rows), None, CFG.interrow, NOTCH_M)
    pieces = cut_interrows_to_tile(bands, TILE, tile_box(TILE), 0.25)
    veg = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    veg[:, :1024] = True
    vis = np.full((TILE_PX, TILE_PX), VIS_OK, dtype=np.uint8)
    vis[:, 1800:] = VIS_NODATA
    s = cover_stats(veg, vis, pieces.geometry.iloc[0], TILE)
    assert s.veg_frac == pytest.approx(1024 / 1800, abs=0.01) and s.shadow_frac == 0.0 and s.n_px > 0
    out = classify_pieces(pieces, veg, vis, TILE, CFG.interrow)
    assert list(out.columns) == [*PIECE_COLUMNS, "geometry"]
    assert out["interrow_cover"].tolist() == [InterrowCover.MIXED.value]
    assert out["qa_flags"].tolist() == [""]


@pytest.mark.parametrize(
    ("veg", "shadow", "width", "cover", "flags"),
    [
        (0.10, 0.0, 2.0, InterrowCover.BARE_SOIL, ()),
        (0.50, 0.0, 2.0, InterrowCover.MIXED, ()),
        (0.80, 0.0, 2.0, InterrowCover.VEGETATION, (BORDERLINE_FLAG,)),
        (0.90, 0.0, 2.0, InterrowCover.VEGETATION, ()),
        (0.27, 0.0, 2.0, InterrowCover.MIXED, (BORDERLINE_FLAG,)),
        (0.10, 0.6, 2.0, InterrowCover.UNASSESSABLE, ()),
        (0.10, 0.0, 0.35, InterrowCover.UNASSESSABLE, ()),
        (float("nan"), 0.0, 2.0, InterrowCover.UNASSESSABLE, ()),
    ],
)
def test_classify_cover(veg: float, shadow: float, width: float, cover: InterrowCover, flags: tuple[str, ...]) -> None:
    assert classify_cover(CoverStats(veg, shadow, 100), width, CFG.interrow) == (cover, flags)


def test_cover_stats_shadow_and_outside_tile() -> None:
    veg = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    vis = np.full((TILE_PX, TILE_PX), VIS_SHADOW, dtype=np.uint8)
    piece = shapely.transform(box(10, 10, 50, 50), lambda uv: px_to_utm(TILE, uv))
    s = cover_stats(veg, vis, piece, TILE)
    assert s.shadow_frac == 1.0 and s.veg_frac == 0.0 and s.n_px == 1600
    far = Polygon([(0, 0), (1, 0), (1, 1)])
    empty = cover_stats(veg, vis, far, TILE)
    assert empty.n_px == 0 and np.isnan(empty.veg_frac)


# ------------------------------------------------------------------ acceptance on the 2 reference examples

EXAMPLE_PIECES = {"siret3_r021_c012": 24, "siret3_r006_c004": 25}
MIN_IOU = 0.98
MAX_BARE_VEG_FRAC = 0.16  # 02 §7: measured 0.13 / 0.15
MIXED_VEG_FRAC = (0.51, 0.68)  # 02 §7: measured 0.517..0.678 on r006
EDGE_EXTEND_M = 10.0  # reference axes stop at the tile edge; the real rows continue into the next tile


def _extend_edge_ends(uv: np.ndarray) -> np.ndarray:
    p = np.asarray(uv, dtype=float)
    out = p.copy()
    for idx, nb in ((0, 1), (-1, -2)):
        if np.isclose(p[idx], 0.0).any() or np.isclose(p[idx], float(TILE_PX)).any():
            d = (p[idx] - p[nb]) / np.linalg.norm(p[idx] - p[nb])
            out[idx] = p[idx] + d * EDGE_EXTEND_M / GSD_M
    return out


def _example_pieces(img: object, tif: object) -> gpd.GeoDataFrame:
    nd = CFG.nodata
    rgb = read_tile(tif)
    valid = valid_mask(rgb, max_rgb=nd.max_rgb, min_area_px=nd.min_area_m2 / GSD_M**2, close_px=nd.close_px,
                       dilate_px=nd.dilate_px)
    eroded = cv2.erode(valid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * nd.veg_erode_px + 1,) * 2)) > 0
    masks = compute_tile_masks(rgb, eroded, CFG.veg)
    shapes = img.by_label("row")  # type: ignore[attr-defined]
    tid = img.tile_id  # type: ignore[attr-defined]
    rows = gpd.GeoDataFrame({"row_id": [s.attributes["row_id"] for s in shapes], "vineyard_id": "V01"},
                            geometry=[LineString(to_utm(tid, _extend_edge_ends(s.points))) for s in shapes],
                            crs=f"EPSG:{CRS_EPSG}")
    bands = build_interrow_bands(rows, consecutive_pairs(rows), None, CFG.interrow, NOTCH_M)
    tile = tile_ref(tid)
    pieces = cut_interrows_to_tile(bands, tile, tile_box(tile), CFG.export.min_interrow_piece_m2)
    return classify_pieces(pieces, masks.veg, masks.vis, tile, CFG.interrow)


@pytest.mark.examples
@pytest.mark.parametrize("tile_id", sorted(EXAMPLE_PIECES))
def test_interrow_acceptance_with_reference_axes(tile_id: str, examples_xml: bytes, example_tif: object) -> None:
    img = load_examples(examples_xml)[tile_id]
    pieces = _example_pieces(img, example_tif(tile_id))  # type: ignore[operator]
    refs = img.by_label("interrow_area")
    ref_polys = interrow_polygons_utm(img)
    assert len(pieces) == EXAMPLE_PIECES[tile_id] == len(refs)
    pred_u, ref_u = shapely.union_all(list(pieces.geometry)), shapely.union_all(ref_polys)
    assert pred_u.intersection(ref_u).area / pred_u.union(ref_u).area >= MIN_IOU
    for shape, poly in zip(refs, ref_polys, strict=True):
        j = int(np.argmax([poly.intersection(g).area for g in pieces.geometry]))
        assert pieces["interrow_cover"].iloc[j] == shape.attributes["interrow_cover"]
    by_cover = pieces.groupby("interrow_cover")["veg_frac"]
    assert by_cover.max()["bare_soil"] <= MAX_BARE_VEG_FRAC
    if "mixed" in by_cover.groups:
        assert MIXED_VEG_FRAC[0] <= by_cover.min()["mixed"] and by_cover.max()["mixed"] <= MIXED_VEG_FRAC[1]
