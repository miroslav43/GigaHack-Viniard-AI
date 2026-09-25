"""geo.polygonize: raster re-exports and per-label vectorization in the raw contour convention (S1)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from tests.helpers.examples import ExampleImage, canopy_polygons_utm, load_examples
from vineyard.geo import polygonize, raster
from vineyard.geo.polygonize import polygonize_labels, polygonize_mask
from vineyard.geo.tiling import GSD_M, TILE_PX, px_to_utm, tile_box, tile_ref

T = tile_ref("siret3_r021_c012")
PX2 = GSD_M * GSD_M
SHAPE = (256, 256)


def _utm_box(u0: float, v0: float, u1: float, v1: float) -> Polygon:
    (x0, y0), (x1, y1) = px_to_utm(T, np.array([[u0, v1], [u1, v0]], dtype=np.float64))
    return box(x0, y0, x1, y1)


def _labels() -> np.ndarray:
    lab = np.zeros(SHAPE, dtype=np.int32)
    lab[10:30, 10:30] = 1  # 20x20 px -> raw contour area 19x19 px
    lab[50:70, 0:20] = 1  # touches the left image border
    lab[100:140, 100:140] = 7  # sparse label value, with a hole
    lab[115:125, 115:125] = 0
    lab[236:256, 236:256] = 3  # bottom-right corner
    return lab


def _same(a: Polygon, b: Polygon) -> bool:
    return a.symmetric_difference(b).area < 1e-9 and a.exterior.is_ccw == b.exterior.is_ccw


def test_reexports_are_the_raster_functions() -> None:
    assert polygonize.mask_to_polygons is raster.mask_to_polygons
    assert polygonize.valid_polygon is raster.valid_polygon
    assert {"mask_to_polygons", "valid_polygon", "polygonize_labels", "polygonize_mask"} == set(
        polygonize.__all__
    )


def test_polygonize_labels_matches_full_raster_vectorization_per_label() -> None:
    lab = _labels()
    out = polygonize_labels(lab, T, min_area_m2=0.0, eps_px=0.0)
    assert [k for k, _ in out] == [1, 1, 3, 7]
    for value in (1, 3, 7):
        mine = [p for k, p in out if k == value]
        full = raster.mask_to_polygons(lab == value, T, approx_eps_px=0.0, min_area_px=0.0)
        assert len(mine) == len(full)
        assert all(_same(a, b) for a, b in zip(sorted(mine, key=_key), sorted(full, key=_key), strict=True))
    holed = [p for k, p in out if k == 7][0]
    assert len(holed.interiors) == 1 and holed.exterior.is_ccw and holed.is_valid


def _key(p: Polygon) -> tuple[float, float]:
    return (round(p.bounds[0], 6), round(p.bounds[1], 6))


def test_polygonize_labels_raw_convention_areas_and_offset_outset() -> None:
    lab = _labels()
    raw = dict(polygonize_labels(lab, T, min_area_m2=0.0, eps_px=0.0)[2:3])
    assert raw[3].area == pytest.approx(19 * 19 * PX2)
    grown = polygonize_labels(lab, T, min_area_m2=0.0, eps_px=0.0, offset_px=0.5, outset_px=0.5)
    area3 = [p.area for k, p in grown if k == 3][0]
    assert area3 == pytest.approx(20 * 20 * PX2)


def test_polygonize_labels_min_area_is_on_the_vector_area() -> None:
    lab = np.zeros(SHAPE, dtype=np.uint16)
    lab[10:30, 10:30] = 1  # 19^2 px = 0.2256 m^2 kept
    lab[50:67, 50:67] = 2  # 16^2 px = 0.16 m^2 dropped
    out = polygonize_labels(lab, T, min_area_m2=0.19, eps_px=2.5)
    assert [k for k, _ in out] == [1]


def test_polygonize_labels_optional_per_label_clip_then_min_area() -> None:
    lab = _labels()
    clips = {1: _utm_box(0, 0, 20, 256), 3: tile_box(T), 7: _utm_box(100, 100, 120, 140)}
    out = polygonize_labels(lab, T, min_area_m2=0.0, eps_px=0.0, clips=clips)
    by = {k: [p for kk, p in out if kk == k] for k in (1, 3, 7)}
    assert sum(p.area for p in by[1]) == pytest.approx((10 * 19 + 19 * 19) * PX2)
    assert by[7][0].area < 20 * 39 * PX2 and all(p.exterior.is_ccw for p in by[7])
    dropped = polygonize_labels(lab, T, min_area_m2=0.2, eps_px=0.0, clips=clips)
    assert [k for k, _ in dropped].count(1) == 1  # the 10x19 px clipped part falls below 0.2 m^2
    with pytest.raises(ValueError, match="clip"):
        polygonize_labels(lab, T, min_area_m2=0.0, eps_px=0.0, clips={1: tile_box(T)})


def test_polygonize_labels_validates_input_and_handles_empty() -> None:
    assert polygonize_labels(np.zeros(SHAPE, np.int32), T, min_area_m2=0.0, eps_px=0.0) == []
    with pytest.raises(ValueError, match="2-D"):
        polygonize_labels(np.zeros((4, 4, 1), np.int32), T, min_area_m2=0.0, eps_px=0.0)
    with pytest.raises(ValueError, match="integer"):
        polygonize_labels(np.zeros(SHAPE, np.float32), T, min_area_m2=0.0, eps_px=0.0)
    with pytest.raises(ValueError, match="negative"):
        polygonize_labels(np.full(SHAPE, -1, np.int32), T, min_area_m2=0.0, eps_px=0.0)
    with pytest.raises(ValueError, match="min_area_m2"):
        polygonize_labels(np.zeros(SHAPE, np.int32), T, min_area_m2=-1.0, eps_px=0.0)


def test_polygonize_labels_does_not_mutate_and_is_deterministic() -> None:
    lab = _labels()
    before = lab.copy()
    a = polygonize_labels(lab, T, min_area_m2=0.0, eps_px=2.5)
    b = polygonize_labels(lab, T, min_area_m2=0.0, eps_px=2.5)
    np.testing.assert_array_equal(lab, before)
    assert [(k, p.wkb) for k, p in a] == [(k, p.wkb) for k, p in b]


def test_polygonize_mask_single_label_and_bool_input() -> None:
    mask = _labels() > 0
    out = polygonize_mask(mask, T, min_area_m2=0.19, eps_px=2.5)
    ref = [p for p in raster.mask_to_polygons(mask, T, approx_eps_px=2.5, min_area_px=0.0) if p.area >= 0.19]
    assert len(out) == len(ref) and sum(p.area for p in out) == pytest.approx(sum(p.area for p in ref))
    clipped = polygonize_mask(mask, T, min_area_m2=0.0, eps_px=0.0, clip=_utm_box(0, 0, 20, 256))
    assert all(p.bounds[2] <= _utm_box(0, 0, 20, 256).bounds[2] + 1e-9 for p in clipped)


@pytest.mark.examples
def test_reference_canopy_labels_round_trip(examples_xml: bytes) -> None:
    examples: dict[str, ExampleImage] = load_examples(examples_xml)
    img = examples["siret3_r021_c012"]
    ref_utm = canopy_polygons_utm(img)
    lab = np.zeros((TILE_PX, TILE_PX), dtype=np.int32)
    for k, pts in enumerate(img.points("vineyard"), start=1):
        cv2.fillPoly(lab, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], k)  # index convention
    out = polygonize_labels(lab, tile_ref(img.tile_id), min_area_m2=0.0, eps_px=0.0)
    assert {k for k, _ in out} == set(range(1, len(ref_utm) + 1))
    assert sum(p.area for _, p in out) == pytest.approx(sum(p.area for p in ref_utm), rel=0.005)
