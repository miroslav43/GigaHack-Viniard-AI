"""perception.corridor: per-tile row pieces, flat-cap corridors, corridor label raster."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from vineyard.geo.raster import rasterize_px
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_M, TILE_PX, px_to_utm, tile_box, tile_ref, utm_to_px
from vineyard.perception.corridor import (
    PIECE_COLUMNS,
    clip_rows_to_tile,
    corridor_label_raster,
    corridor_labels,
    corridor_polygon,
    extend_censored_ends,
)

TILE = tile_ref("siret3_r021_c012")
HALF = 0.30


def _utm_line(uv: list[tuple[float, float]]) -> LineString:
    return LineString(px_to_utm(TILE, np.asarray(uv, dtype=float)))


def _rows(lines: list[LineString], ids: list[str] | None = None) -> gpd.GeoDataFrame:
    ids = ids or [f"V01-R{k + 1:03d}" for k in range(len(lines))]
    return gpd.GeoDataFrame(
        {"row_id": ids, "vineyard_id": ["V01"] * len(lines), "row_index": list(range(1, len(lines) + 1))},
        geometry=lines, crs=f"EPSG:{CRS_EPSG}",
    )


# ------------------------------------------------------------------ clip_rows_to_tile


def test_clip_rows_keeps_one_piece_per_row_and_adds_columns() -> None:
    long_row = _utm_line([(-500.0, 1000.0), (2600.0, 1000.0)])
    rows = _rows([long_row])
    out = clip_rows_to_tile(rows, TILE, tile_box(TILE))
    assert len(out) == 1
    assert set(PIECE_COLUMNS) <= set(out.columns)
    rec = out.iloc[0]
    assert rec["tile_id"] == TILE.tile_id
    assert rec["piece_id"] == f"V01-R001@{TILE.tile_id}"
    assert out.geometry.iloc[0].length == pytest.approx(TILE_M, abs=1e-6)
    assert rec["along_from_m"] == pytest.approx(500 * GSD_M, abs=1e-6)
    assert rec["along_to_m"] == pytest.approx(2548 * GSD_M, abs=1e-6)
    assert out.crs == rows.crs


def test_clip_rows_margin_extends_pieces_beyond_the_tile() -> None:
    rows = _rows([_utm_line([(-500.0, 1000.0), (2600.0, 1000.0)])])
    out = clip_rows_to_tile(rows, TILE, tile_box(TILE), margin_m=1.0)
    assert out.geometry.iloc[0].length == pytest.approx(TILE_M + 2.0, abs=1e-6)


def test_clip_rows_first_to_last_valid_vertex_across_a_nodata_cut() -> None:
    rows = _rows([_utm_line([(-10.0, 1000.0), (2100.0, 1000.0)])])
    tb = tile_box(TILE)
    hole = shapely.transform(box(900, 900, 1100, 1100), lambda uv: px_to_utm(TILE, uv))
    clip = tb.difference(hole)
    out = clip_rows_to_tile(rows, TILE, clip)
    geom = out.geometry.iloc[0]
    assert isinstance(geom, LineString)
    assert geom.length == pytest.approx(TILE_M, abs=1e-6)


def test_clip_rows_drops_rows_outside_and_degenerate_pieces() -> None:
    outside = _utm_line([(3000.0, 10.0), (4000.0, 10.0)])
    corner = _utm_line([(-5.0, 1.0), (1.0, -5.0)])  # touches the tile in < 5 cm
    inside = _utm_line([(100.0, 100.0), (400.0, 400.0)])
    out = clip_rows_to_tile(_rows([outside, corner, inside]), TILE, tile_box(TILE))
    assert out["row_id"].tolist() == ["V01-R003"]


def test_clip_rows_empty_input_and_multipolygon_clip() -> None:
    empty = _rows([])
    out = clip_rows_to_tile(empty, TILE, tile_box(TILE))
    assert len(out) == 0 and set(PIECE_COLUMNS) <= set(out.columns)
    clip = MultiPolygon([tile_box(TILE)])
    out2 = clip_rows_to_tile(_rows([_utm_line([(0.0, 5.0), (2048.0, 5.0)])]), TILE, clip)
    assert len(out2) == 1


def test_clip_rows_does_not_mutate_input() -> None:
    rows = _rows([_utm_line([(-500.0, 1000.0), (2600.0, 1000.0)])])
    before = rows.copy()
    clip_rows_to_tile(rows, TILE, tile_box(TILE), margin_m=1.0)
    assert rows.equals(before)


def test_clip_rows_rejects_negative_margin_and_bad_geometry() -> None:
    rows = _rows([_utm_line([(0.0, 5.0), (2048.0, 5.0)])])
    with pytest.raises(ValueError, match="margin_m"):
        clip_rows_to_tile(rows, TILE, tile_box(TILE), margin_m=-1.0)
    bad = rows.set_geometry([box(0, 0, 1, 1)], crs=rows.crs)
    with pytest.raises(TypeError, match="LineString"):
        clip_rows_to_tile(bad, TILE, tile_box(TILE))


def test_clip_rows_duplicate_row_id_gets_hash_suffix() -> None:
    lines = [_utm_line([(0.0, 5.0), (2048.0, 5.0)]), _utm_line([(0.0, 50.0), (2048.0, 50.0)])]
    out = clip_rows_to_tile(_rows(lines, ["V01-R001", "V01-R001"]), TILE, tile_box(TILE))
    assert out["piece_id"].tolist() == [f"V01-R001@{TILE.tile_id}", f"V01-R001@{TILE.tile_id}#2"]


# ------------------------------------------------------------------ corridor_polygon


def test_corridor_polygon_flat_caps() -> None:
    axis = LineString([(0.0, 0.0), (10.0, 0.0)])
    poly = corridor_polygon(axis, HALF)
    assert isinstance(poly, Polygon)
    assert poly.area == pytest.approx(10.0 * 2 * HALF)
    assert poly.bounds == pytest.approx((0.0, -HALF, 10.0, HALF))


def test_corridor_polygon_polyline_and_errors() -> None:
    axis = LineString([(0.0, 0.0), (10.0, 0.0), (20.0, 1.0)])
    poly = corridor_polygon(axis, HALF)
    assert poly.is_valid and poly.contains(axis.interpolate(0.5, normalized=True))
    with pytest.raises(ValueError, match="half_m"):
        corridor_polygon(axis, 0.0)
    with pytest.raises(ValueError, match="length"):
        corridor_polygon(LineString([(1.0, 1.0), (1.0, 1.0)]), HALF)


# ------------------------------------------------------------------ corridor_labels


def test_corridor_labels_values_and_continuous_convention() -> None:
    lines = [_utm_line([(0.0, 500.0), (2048.0, 500.0)]), _utm_line([(0.0, 800.0), (2048.0, 800.0)])]
    pieces = clip_rows_to_tile(_rows(lines), TILE, tile_box(TILE))
    lab = corridor_labels(pieces, TILE, HALF)
    assert lab.dtype == np.int32 and lab.shape == (TILE_PX, TILE_PX)
    assert set(np.unique(lab).tolist()) == {0, 1, 2}
    half_px = HALF / GSD_M  # 12 px: centres v+0.5 in (488, 512) -> rows 488..511
    assert (lab[:, 1000] == 1).sum() == round(2 * half_px)
    assert lab[488, 1000] == 1 and lab[487, 1000] == 0 and lab[511, 1000] == 1 and lab[512, 1000] == 0
    assert lab[800, 10] == 2


def test_corridor_labels_match_rasterize_px_per_piece() -> None:
    lines = [_utm_line([(10.0, 30.0), (2000.0, 1900.0)]), _utm_line([(100.0, 30.0), (2040.0, 1800.0)])]
    pieces = clip_rows_to_tile(_rows(lines), TILE, tile_box(TILE), margin_m=1.0)
    lab = corridor_labels(pieces, TILE, HALF)
    for k, axis in enumerate(pieces.geometry):
        ring = utm_to_px(TILE, np.asarray(corridor_polygon(axis, HALF).exterior.coords))
        ref = rasterize_px([ring], value=1, convention="continuous") > 0
        assert np.array_equal(lab == k + 1, ref)


def test_corridor_labels_empty() -> None:
    lab = corridor_labels(_rows([]), TILE, HALF)
    assert lab.dtype == np.int32 and not lab.any()


def test_corridor_label_raster_index_convention_uses_pixel_index_points() -> None:
    pieces = clip_rows_to_tile(_rows([_utm_line([(0.0, 100.3), (2048.0, 100.3)])]), TILE, tile_box(TILE))
    cont = corridor_label_raster(pieces, TILE, HALF, convention="continuous")
    idx = corridor_label_raster(pieces, TILE, HALF, convention="index")
    assert np.array_equal(cont, corridor_labels(pieces, TILE, HALF))
    # corridor v in [88.3, 112.3]: centres j+0.5 -> 88..111; index points j -> 89..112
    assert np.flatnonzero(cont[:, 1000]).tolist() == list(range(88, 112))
    assert np.flatnonzero(idx[:, 1000]).tolist() == list(range(89, 113))
    assert idx.dtype == np.int32


def test_corridor_label_raster_index_points_lie_inside_the_corridor() -> None:
    axis = _utm_line([(10.0, 30.0), (2000.0, 1900.0)])
    pieces = clip_rows_to_tile(_rows([axis]), TILE, tile_box(TILE))
    idx = corridor_label_raster(pieces, TILE, HALF, convention="index")
    vv, uu = np.nonzero(idx)
    pts = px_to_utm(TILE, np.column_stack((uu, vv)).astype(float))
    dist = shapely.distance(pieces.geometry.iloc[0], shapely.points(pts))
    assert float(dist.max()) <= HALF + 1e-9
    with pytest.raises(ValueError, match="convention"):
        corridor_label_raster(pieces, TILE, HALF, convention="bogus")


# ------------------------------------------------------------------ extend_censored_ends


def test_extend_censored_ends_only_extends_ends_on_the_boundary() -> None:
    row = _utm_line([(0.0, 1000.0), (1500.0, 1000.0)])  # starts on the tile edge, ends inside
    rows = _rows([row])
    out = extend_censored_ends(rows, tile_box(TILE).boundary, extend_m=10.0, tol_m=0.05)
    line = out.geometry.iloc[0]
    assert line.length == pytest.approx(row.length + 10.0, abs=1e-6)
    end = np.asarray(line.coords)[-1]
    assert np.allclose(end, np.asarray(row.coords)[-1])
    assert rows.geometry.iloc[0].equals(row)  # input untouched


def test_extend_censored_ends_follows_the_end_segment_direction() -> None:
    row = _utm_line([(0.0, 0.0), (1000.0, 1000.0), (2048.0, 1500.0)])
    out = extend_censored_ends(_rows([row]), tile_box(TILE).boundary, extend_m=5.0, tol_m=0.05)
    xy = np.asarray(out.geometry.iloc[0].coords)
    assert len(xy) == 3  # ends move along their segment, no vertex added
    head = (xy[0] - xy[1]) / np.linalg.norm(xy[0] - xy[1])
    tail = (xy[-1] - xy[-2]) / np.linalg.norm(xy[-1] - xy[-2])
    src = np.asarray(row.coords)
    assert np.allclose(head, (src[0] - src[1]) / np.linalg.norm(src[0] - src[1]))
    assert np.allclose(tail, (src[-1] - src[-2]) / np.linalg.norm(src[-1] - src[-2]))
    assert np.linalg.norm(xy[0] - src[0]) == pytest.approx(5.0)


def test_extend_censored_ends_noop_cases_and_errors() -> None:
    rows = _rows([_utm_line([(0.0, 1000.0), (2048.0, 1000.0)])])
    same = extend_censored_ends(rows, tile_box(TILE).boundary, extend_m=0.0, tol_m=0.05)
    assert same.geometry.iloc[0].equals(rows.geometry.iloc[0])
    empty = extend_censored_ends(rows, None, extend_m=10.0, tol_m=0.05)
    assert empty.geometry.iloc[0].equals(rows.geometry.iloc[0])
    none = extend_censored_ends(rows.iloc[:0], tile_box(TILE).boundary, extend_m=10.0, tol_m=0.05)
    assert none.empty
    with pytest.raises(ValueError, match="extend_m"):
        extend_censored_ends(rows, tile_box(TILE).boundary, extend_m=-1.0, tol_m=0.05)
    with pytest.raises(ValueError, match="tol_m"):
        extend_censored_ends(rows, tile_box(TILE).boundary, extend_m=1.0, tol_m=-0.05)
