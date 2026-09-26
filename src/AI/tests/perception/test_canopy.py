"""perception.canopy: synthetic masks + row pieces -> canopy polygons (S1 raw convention, S4 evidence)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Polygon, box

from tests.helpers import proto_metrics
from tests.helpers.examples import (
    ExampleImage,
    canopy_polygons_utm,
    interrow_polygons_utm,
    load_examples,
    to_utm,
)
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.eval.metrics import canopy_metrics
from vineyard.geo.raster import read_tile, valid_mask
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_box, tile_ref, utm_to_px
from vineyard.perception.axis_refine import AxisRefineOptions
from vineyard.perception.canopy import (
    CANOPY_COLUMNS,
    EVIDENCE_COLUMNS,
    CanopyOptions,
    eligible_labels,
    extract_canopies,
    interpolated_mask,
    label_veg_fractions,
    vine_mask,
)
from vineyard.perception.corridor import clip_rows_to_tile, corridor_labels
from vineyard.perception.vegmask import compute_tile_masks, exg_mask

TILE = tile_ref("siret3_r021_c012")
OPTS = CanopyOptions(
    corridor_half_m=0.30, min_area_m2=0.19, connectivity=8, approx_eps_px=2.5, offset_px=0.0, outset_px=0.0,
    clump_area_m2=2.0, clip_to_corridor=False, interpolated_min_veg_frac=0.15, label_convention="index",
)


def _rows(vs: list[float], flags: list[str] | None = None) -> gpd.GeoDataFrame:
    lines = [LineString(px_to_utm(TILE, np.array([[-100.0, v], [2200.0, v]]))) for v in vs]
    return gpd.GeoDataFrame(
        {"row_id": [f"V01-R{k + 1:03d}" for k in range(len(vs))], "vineyard_id": "V01",
         "row_index": list(range(1, len(vs) + 1)), "qa_flags": flags or [""] * len(vs)},
        geometry=lines, crs=f"EPSG:{CRS_EPSG}",
    )


def _pieces(vs: list[float], flags: list[str] | None = None) -> gpd.GeoDataFrame:
    return clip_rows_to_tile(_rows(vs, flags), TILE, tile_box(TILE), margin_m=1.0)


def _blobs(v: int, us: list[tuple[int, int]], half_px: int = 10) -> np.ndarray:
    m = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    for u0, u1 in us:
        m[v - half_px : v + half_px, u0:u1] = True
    return m


def test_raw_convention_integer_vertices_and_attributes() -> None:
    mask = _blobs(500, [(100, 180), (400, 460)])
    res = extract_canopies(mask, _pieces([500.0]), TILE, tile_box(TILE), OPTS)
    can = res.canopies
    assert list(can.columns) == [*CANOPY_COLUMNS, "geometry"]
    assert len(can) == 2
    assert can["canopy_id"].tolist() == [f"{TILE.tile_id}:C0001", f"{TILE.tile_id}:C0002"]
    assert all(is_valid_id(IdKind.CANOPY, c) for c in can["canopy_id"])
    first = can.geometry.iloc[0]
    uv = (np.asarray(first.exterior.coords) - (TILE.x0, TILE.y0)) * (1 / GSD_M, -1 / GSD_M)
    assert np.allclose(uv, np.round(uv), atol=1e-6)  # raw findContours: integer vertices
    assert first.exterior.is_ccw
    assert can["area_m2"].iloc[0] == pytest.approx(79 * 19 * GSD_M**2, rel=1e-6)
    assert can["along_m"].iloc[0] == pytest.approx(79 * GSD_M, abs=1e-6)
    assert can["row_id"].tolist() == ["V01-R001", "V01-R001"]
    assert not can["is_clump"].any() and not can["touches_edge"].any()
    assert res.stats.n_components == 2 and res.stats.n_pieces == 1


def test_mask_outside_corridor_and_small_components_dropped() -> None:
    mask = _blobs(500, [(100, 180)]) | _blobs(700, [(100, 400)]) | _blobs(500, [(600, 610)])
    res = extract_canopies(mask, _pieces([500.0]), TILE, tile_box(TILE), OPTS)
    assert len(res.canopies) == 1
    assert res.stats.n_small_dropped == 1


def test_vector_min_area_applies_after_vectorization() -> None:
    # 20 x 16 = 320 px (>= 304 px) but the raw contour area is 19 x 15 px = 0.178 m2 < 0.19
    mask = _blobs(500, [(100, 116)])
    res = extract_canopies(mask, _pieces([500.0]), TILE, tile_box(TILE), OPTS)
    assert len(res.canopies) == 0 and res.stats.n_components == 1


def test_clump_edge_and_sorting_by_row_then_along() -> None:
    mask = _blobs(500, [(0, 300), (1000, 1100)]) | _blobs(900, [(50, 150)])
    res = extract_canopies(mask, _pieces([900.0, 500.0]), TILE, tile_box(TILE), OPTS)
    can = res.canopies
    assert can["row_id"].tolist() == ["V01-R001", "V01-R002", "V01-R002"]
    assert can["is_clump"].tolist() == [False, True, False]
    assert can["touches_edge"].tolist() == [False, True, False]


def test_offset_and_outset_variants_grow_area() -> None:
    mask = _blobs(500, [(100, 180)])
    base = extract_canopies(mask, _pieces([500.0]), TILE, tile_box(TILE), OPTS).canopies
    grown_opts = dataclasses.replace(OPTS, offset_px=0.5, outset_px=0.5)
    grown = extract_canopies(mask, _pieces([500.0]), TILE, tile_box(TILE), grown_opts).canopies
    assert grown["area_m2"].iloc[0] == pytest.approx(80 * 20 * GSD_M**2, rel=1e-6)
    assert grown["area_m2"].iloc[0] > base["area_m2"].iloc[0]


def test_clip_to_corridor_flag_trims_to_the_corridor() -> None:
    mask = _blobs(500, [(100, 180)], half_px=10)
    opts = dataclasses.replace(OPTS, offset_px=0.5, outset_px=2.0, clip_to_corridor=True)
    can = extract_canopies(mask, _pieces([500.0]), TILE, tile_box(TILE), opts).canopies
    miny, maxy = can.geometry.iloc[0].bounds[1], can.geometry.iloc[0].bounds[3]
    assert maxy - miny <= 0.60 + 1e-9


def _oblique_pieces() -> gpd.GeoDataFrame:
    axis = LineString(px_to_utm(TILE, np.array([[100.0, 1500.3], [1900.0, 700.7]])))
    rows = gpd.GeoDataFrame({"row_id": ["V01-R001"], "vineyard_id": "V01", "row_index": [1], "qa_flags": [""]},
                            geometry=[axis], crs=f"EPSG:{CRS_EPSG}")
    return clip_rows_to_tile(rows, TILE, tile_box(TILE), margin_m=1.0)


def test_index_label_convention_keeps_every_vertex_inside_the_corridor() -> None:
    pieces = _oblique_pieces()
    full = np.ones((TILE_PX, TILE_PX), dtype=bool)
    axis = pieces.geometry.iloc[0]

    def max_dist(opts: CanopyOptions) -> float:
        can = extract_canopies(full, pieces, TILE, tile_box(TILE), opts).canopies
        return max(float(shapely.distance(axis, shapely.points(np.asarray(g.exterior.coords))).max())
                   for g in can.geometry)

    assert OPTS.label_convention == "index"
    assert max_dist(OPTS) <= OPTS.corridor_half_m + 1e-6
    assert max_dist(dataclasses.replace(OPTS, label_convention="continuous")) > OPTS.corridor_half_m + 1e-3


def test_clip_polygon_cuts_canopies() -> None:
    mask = _blobs(500, [(100, 300)])
    clip = shapely.transform(box(0, 0, 200, 2048), lambda uv: px_to_utm(TILE, uv))
    can = extract_canopies(mask, _pieces([500.0]), TILE, clip, OPTS).canopies
    assert len(can) == 1 and bool(can["touches_edge"].iloc[0])
    assert can.geometry.iloc[0].bounds[2] <= clip.bounds[2] + 1e-9


def test_interpolated_piece_needs_vegetation_evidence() -> None:
    sparse = _blobs(500, [(100, 180)])
    flags = ["row_interpolated"]
    res = extract_canopies(sparse, _pieces([500.0], flags), TILE, tile_box(TILE), OPTS)
    assert len(res.canopies) == 0 and res.stats.n_interpolated_skipped == 1
    dense = _blobs(500, [(0, 2048)])
    res2 = extract_canopies(dense, _pieces([500.0], flags), TILE, tile_box(TILE), OPTS)
    assert len(res2.canopies) == 1 and res2.canopies["qa_flags"].iloc[0] == "row_interpolated"


def test_interpolated_mask_sources() -> None:
    pieces = _pieces([500.0, 900.0], ["x;row_interpolated", ""])
    assert interpolated_mask(pieces).tolist() == [True, False]
    assert interpolated_mask(pieces.drop(columns="qa_flags")).tolist() == [False, False]
    with_bool = pieces.assign(row_interpolated=[False, True])
    assert interpolated_mask(with_bool).tolist() == [False, True]


def test_interpolated_mask_is_per_tile_with_interp_tile_ids() -> None:
    # blocks flags the whole row but lists the unsupported tiles: only those pieces need evidence
    here, other = TILE.tile_id, "siret3_r021_c013"
    pieces = _pieces([500.0, 900.0, 1300.0], ["row_interpolated"] * 3).assign(
        interp_tile_ids=[f"{other},{here}", other, None])
    assert interpolated_mask(pieces).tolist() == [True, False, False]
    sparse = _blobs(500, [(100, 180)]) | _blobs(900, [(100, 180)])
    res = extract_canopies(sparse, pieces, TILE, tile_box(TILE), OPTS)
    assert res.canopies["row_id"].tolist() == ["V01-R002"] and res.stats.n_interpolated_skipped == 1


def test_label_fractions_eligibility_and_vine_mask() -> None:
    pieces = _pieces([500.0, 900.0])
    labels = corridor_labels(pieces, TILE, 0.30)
    mask = _blobs(500, [(0, 1024)])
    frac = label_veg_fractions(mask, labels, 3)
    assert frac[0] == pytest.approx(0.5 * 20 / 24, abs=0.01) and frac[1] == 0.0 and np.isnan(frac[2])
    ok = eligible_labels(np.array([True, True, True]), frac, 0.15)
    assert ok.tolist() == [False, True, False, False]
    tree = np.zeros_like(mask)
    tree[:, :100] = True
    vm = vine_mask(mask, labels, tree)
    assert vm.dtype == np.int32 and vm[500, 50] == 0 and vm[500, 500] == 1 and vm[900, 500] == 0


def test_axis_refinement_moves_the_corridor_onto_off_axis_canopies() -> None:
    mask = _blobs(510, [(100, 400)], half_px=10)  # rows 500..519: centred 10 px (0.25 m) south of the axis
    pieces = _pieces([500.0])
    plain = extract_canopies(mask, pieces, TILE, tile_box(TILE), OPTS).canopies
    refine = AxisRefineOptions(band_m=0.35, iterations=3, max_shift_m=0.10, min_px=200)
    moved = extract_canopies(mask, pieces, TILE, tile_box(TILE), dataclasses.replace(OPTS, axis_refine=refine))
    can = moved.canopies
    assert len(plain) == len(can) == 1 and can["row_id"].tolist() == ["V01-R001"]
    # the corridor moves 4 px (capped at 0.10 m) south: 4 more pixel rows of the blob are inside it
    assert can["area_m2"].iloc[0] == pytest.approx(plain["area_m2"].iloc[0] + 4 * 299 * GSD_M**2, rel=1e-6)
    assert can["along_m"].iloc[0] == pytest.approx(plain["along_m"].iloc[0])
    assert pieces.geometry.iloc[0].equals(_pieces([500.0]).geometry.iloc[0])


def test_gap_evidence_keeps_small_plants_out_of_the_canopies() -> None:
    # 18 x 16 px = 288 px (< 304) with a raw contour of 17 x 15 px = 0.159 m2: evidence, never a canopy
    mask = _blobs(500, [(100, 180)]) | _blobs(500, [(600, 616)], half_px=9) | _blobs(500, [(900, 908)])
    off = extract_canopies(mask, _pieces([500.0]), TILE, tile_box(TILE), OPTS)
    assert len(off.canopies) == 1 and off.evidence.empty
    on = extract_canopies(mask, _pieces([500.0]), TILE, tile_box(TILE),
                          dataclasses.replace(OPTS, evidence_min_area_m2=0.14))
    assert on.canopies.drop(columns="geometry").equals(off.canopies.drop(columns="geometry"))
    ev = on.evidence
    assert list(ev.columns) == [*EVIDENCE_COLUMNS, "geometry"] and len(ev) == 1
    assert ev["area_m2"].iloc[0] == pytest.approx(17 * 15 * GSD_M**2, rel=1e-6)
    assert ev["row_id"].tolist() == ["V01-R001"] and ev["tile_id"].tolist() == [TILE.tile_id]
    assert on.stats.n_components == off.stats.n_components and on.stats.n_small_dropped == off.stats.n_small_dropped


def test_empty_inputs_and_no_mutation() -> None:
    pieces = _pieces([500.0])
    before = pieces.copy()
    mask = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    res = extract_canopies(mask, pieces, TILE, tile_box(TILE), OPTS, valid=~mask)
    assert len(res.canopies) == 0 and list(res.canopies.columns) == [*CANOPY_COLUMNS, "geometry"]
    assert res.stats.corridor_veg_frac == 0.0
    assert pieces.equals(before)
    none = extract_canopies(mask, pieces.iloc[0:0], TILE, tile_box(TILE), OPTS)
    assert len(none.canopies) == 0


def test_options_from_config() -> None:
    from vineyard.config import load_config

    cfg = load_config()
    opts = CanopyOptions.from_config(cfg.canopy, clip_to_corridor=True)
    assert opts.corridor_half_m == cfg.canopy.corridor_half_m and opts.clip_to_corridor
    assert opts.min_component_px == 304 and opts.label_convention == "index"
    assert CanopyOptions.from_config(cfg.canopy, label_convention="continuous").label_convention == "continuous"
    with pytest.raises(ValueError, match="convention"):
        dataclasses.replace(OPTS, label_convention="bogus")
    assert (opts.interpolated_min_veg_frac, CanopyOptions.from_config(cfg.canopy).clip_to_corridor) == (0.15, False)
    future = cfg.canopy.model_copy(update={"clip_to_corridor": True, "corridor_label_convention": "continuous",
                                           "interpolated_min_veg_frac": 0.3})
    picked = CanopyOptions.from_config(future)
    assert picked.clip_to_corridor and picked.label_convention == "continuous"
    assert picked.interpolated_min_veg_frac == 0.3


# ------------------------------------------------------------------ acceptance on the 2 reference examples

EXAMPLE_TILES = ("siret3_r021_c012", "siret3_r006_c004")
REF_CANOPY_AREA_M2 = {"siret3_r021_c012": 237.12, "siret3_r006_c004": 299.06}
AREA_TOL = 0.03
MIN_MEAN_SCORE = 0.84
MIN_TILE_RASTER_SCORE = 0.80
MAX_INTERROW_OVERLAP_M2 = 0.01
MIN_INTEGER_VERTEX_FRAC = 0.99
MAX_RAW_VERTEX_PX = TILE_PX - 1
# canopy.simplify_px went 2.5 -> 2.0 (P1-CAN sweep): +0.005 vector score, vertex density closer to the reference.
PREVIOUS_SIMPLIFY_PX = 2.5
EDGE_EXTEND_M = 10.0  # reference axes stop at the tile edge; production rows continue into the next tile

Predicted = dict[str, tuple[list[Polygon], ExampleImage]]


def _example_masks(cfg: object, tif: Path) -> tuple[np.ndarray, np.ndarray]:
    """(canopy mask as the canopy stage builds it for cfg.canopy.mask_method, valid mask)."""
    nd, can = cfg.nodata, cfg.canopy  # type: ignore[attr-defined]
    rgb = read_tile(tif)
    valid = valid_mask(rgb, max_rgb=nd.max_rgb, min_area_px=nd.min_area_m2 / GSD_M**2, close_px=nd.close_px,
                       dilate_px=nd.dilate_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * nd.veg_erode_px + 1,) * 2)
    eroded = cv2.erode(valid.astype(np.uint8), k) > 0
    if can.mask_method == "exg":
        return exg_mask(rgb, eroded, blur_sigma_px=can.exg_blur_sigma_px, threshold=can.exg_threshold), valid
    return compute_tile_masks(rgb, eroded, cfg.veg).veg, valid  # type: ignore[attr-defined]


def _extend_edge_ends(uv: np.ndarray) -> np.ndarray:
    p = np.asarray(uv, dtype=float)
    out = p.copy()
    for idx, nb in ((0, 1), (-1, -2)):
        if np.isclose(p[idx], 0.0).any() or np.isclose(p[idx], float(TILE_PX)).any():
            d = (p[idx] - p[nb]) / np.linalg.norm(p[idx] - p[nb])
            out[idx] = p[idx] + d * EDGE_EXTEND_M / GSD_M
    return out


def _reference_rows(img: ExampleImage) -> gpd.GeoDataFrame:
    shapes = img.by_label("row")
    return gpd.GeoDataFrame(
        {"row_id": [s.attributes["row_id"] for s in shapes], "vineyard_id": [s.attributes["vineyard_id"] for s in shapes]},
        geometry=[LineString(to_utm(img.tile_id, _extend_edge_ends(s.points))) for s in shapes],
        crs=f"EPSG:{CRS_EPSG}",
    )


@pytest.fixture(scope="module")
def example_canopies(examples_xml: bytes, example_tif: Callable[[str], Path]) -> dict[float, Predicted]:
    """{simplify_px: {tile_id: (predicted canopies, reference image)}} for the config default and 2.5."""
    from vineyard.config import load_config

    cfg = load_config()
    imgs = load_examples(examples_xml)
    # the reference axes are exact: the axis refinement exists to correct model axes fitted on a*
    base = dataclasses.replace(CanopyOptions.from_config(cfg.canopy), axis_refine=None)
    inputs = {tid: _example_masks(cfg, example_tif(tid)) for tid in EXAMPLE_TILES}
    out: dict[float, Predicted] = {}
    for eps in (base.approx_eps_px, PREVIOUS_SIMPLIFY_PX):
        opts = dataclasses.replace(base, approx_eps_px=eps)
        out[eps] = {}
        for tid, (veg, valid) in inputs.items():
            tile = tile_ref(tid)
            pieces = clip_rows_to_tile(_reference_rows(imgs[tid]), tile, tile_box(tile),
                                       margin_m=cfg.canopy.rows_margin_m)
            res = extract_canopies(veg, pieces, tile, tile_box(tile), opts, valid=valid)
            out[eps][tid] = (list(res.canopies.geometry), imgs[tid])
    return out


def _vector_scores(pred: Predicted) -> list[float]:
    from vineyard.config import load_config

    ev = load_config().eval
    return [canopy_metrics(p, canopy_polygons_utm(img), match_iou=ev.canopy_match_iou, w_iou=ev.canopy_w_iou,
                           w_f1=ev.canopy_w_f1).score for p, img in pred.values()]


def _raster_score(pred: list[Polygon], img: ExampleImage) -> float:
    tile = tile_ref(img.tile_id)
    ref_lab = proto_metrics.reference_labels(img.points("vineyard"))
    pred_lab = proto_metrics.reference_labels([utm_to_px(tile, np.asarray(p.exterior.coords)) for p in pred])
    iou, f1, _ = proto_metrics.score(pred_lab, ref_lab, len(img.points("vineyard")))
    return proto_metrics.canopy_score(iou, f1)


def _check_geometry(pred: Predicted) -> None:
    for tid, (polys, img) in pred.items():
        area = sum(p.area for p in polys)
        assert abs(area / REF_CANOPY_AREA_M2[tid] - 1.0) <= AREA_TOL, (tid, area)
        uv = utm_to_px(tile_ref(tid), np.vstack([np.asarray(p.exterior.coords) for p in polys]))
        integer = np.all(np.abs(uv - np.round(uv)) < 1e-6, axis=1)
        assert integer.mean() >= MIN_INTEGER_VERTEX_FRAC  # make_valid splits add a few fractional vertices
        assert uv.max() <= MAX_RAW_VERTEX_PX + 1e-6
        overlap = shapely.union_all(polys).intersection(shapely.union_all(interrow_polygons_utm(img))).area
        assert overlap <= MAX_INTERROW_OVERLAP_M2, (tid, overlap)


@pytest.mark.examples
def test_canopy_acceptance_with_reference_axes(example_canopies: dict[float, Predicted]) -> None:
    default = next(eps for eps in example_canopies if eps != PREVIOUS_SIMPLIFY_PX)
    pred = example_canopies[default]
    _check_geometry(pred)
    raster = [_raster_score(polys, img) for polys, img in pred.values()]
    assert float(np.mean(raster)) >= MIN_MEAN_SCORE and min(raster) >= MIN_TILE_RASTER_SCORE, raster
    scores = _vector_scores(pred)
    assert float(np.mean(scores)) >= MIN_MEAN_SCORE, scores


@pytest.mark.examples
def test_canopy_default_simplify_scores_higher_than_previous(example_canopies: dict[float, Predicted]) -> None:
    default = next(eps for eps in example_canopies if eps != PREVIOUS_SIMPLIFY_PX)
    _check_geometry(example_canopies[PREVIOUS_SIMPLIFY_PX])
    previous = float(np.mean(_vector_scores(example_canopies[PREVIOUS_SIMPLIFY_PX])))
    assert float(np.mean(_vector_scores(example_canopies[default]))) > previous
