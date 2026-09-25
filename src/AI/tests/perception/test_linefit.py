"""perception.linefit: Huber band fits, rejects, clip extension and curved-row tracking."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from tests.helpers.synth import arc_mask, striped_mask
from vineyard.config import RowsDetectConfig, load_config
from vineyard.geo.tiling import GSD_M, TILE_PX
from vineyard.perception.linefit import (
    REJECT_ANGLE_GATE,
    REJECT_MIN_BAND_AREA,
    BandFit,
    SlabIndex,
    axial_diff_deg,
    centreline_residual_px,
    extend_to_clip,
    fit_band_line,
    max_deviation_px,
    near_edge,
    track_row,
)
from vineyard.perception.profile import mask_points

SHAPE = (TILE_PX, TILE_PX)
FRAME = box(0.0, 0.0, float(TILE_PX), float(TILE_PX))
ARC_R_PX = 300.0 / GSD_M
RAY = load_config().rows.detect.snap_ray_max_factor


@pytest.fixture(scope="module")
def det() -> RowsDetectConfig:
    return load_config().rows.detect


def _single_stripe(angle: float, offset_px: float, width_px: float = 20.0) -> np.ndarray:
    return striped_mask(SHAPE, angle_deg=angle, spacing_px=8000.0, width_px=width_px, offset_px=offset_px)


def _fit(mask: np.ndarray, angle: float, peak_px: float, det: RowsDetectConfig, **kw: float) -> BandFit | None:
    pts = mask_points(mask)
    slabs = SlabIndex.build(pts, angle)
    idx = slabs.slab(peak_px, det.band_max_drift_m / GSD_M)
    return fit_band_line(pts[idx], slabs.offsets[idx], peak_px, angle, det, **kw)


@pytest.fixture(scope="module")
def arc() -> tuple[np.ndarray, LineString]:
    centre = (1024.0, 1024.0 + ARC_R_PX)
    mask = arc_mask(SHAPE, centre=centre, r0_px=ARC_R_PX, spacing_px=100.0, width_px=16.0, n_rows=1)
    th = np.linspace(-0.2, 0.2, 20001)
    truth = LineString(np.column_stack((centre[0] + ARC_R_PX * np.sin(th), centre[1] - ARC_R_PX * np.cos(th))))
    return mask, truth


def test_axial_diff() -> None:
    assert axial_diff_deg(179.0, 1.0) == pytest.approx(2.0)
    assert axial_diff_deg(0.0, 90.0) == pytest.approx(90.0)
    assert axial_diff_deg(30.0, 210.0) == pytest.approx(0.0)


def test_slab_index_query() -> None:
    pts = np.array([[0.0, 5.0], [0.0, 1.0], [0.0, 3.0], [0.0, 2.0]], dtype=np.float32)
    slabs = SlabIndex.build(pts, 0.0)
    np.testing.assert_array_equal(slabs.slab(2.0, 1.5), [1, 2, 3])
    np.testing.assert_array_equal(slabs.slab(10.0, 1.0), [])


def test_straight_band_fit(det: RowsDetectConfig) -> None:
    fit = _fit(_single_stripe(30.0, 500.0), 30.0, 500.0, det)
    assert fit is not None and fit.reject is None
    assert axial_diff_deg(fit.angle_px_deg, 30.0) < 0.1
    assert fit.residual_p95_px * GSD_M < 0.05
    a, b = fit.endpoints()
    assert fit.line().length == pytest.approx(float(np.hypot(b[0] - a[0], b[1] - a[1])))
    assert fit.line().length * GSD_M > 40.0


def test_arc_has_large_residual_and_tracking_follows_it(arc: tuple[np.ndarray, LineString],
                                                        det: RowsDetectConfig) -> None:
    mask, truth = arc
    pts = mask_points(mask)
    slabs = SlabIndex.build(pts, 0.0)
    fit = fit_band_line(pts, slabs.offsets, float(np.median(slabs.offsets)), 0.0, det)
    assert fit is not None
    assert fit.residual_p95_px * GSD_M > det.residual_split_m
    tracked = track_row(pts, fit, det)
    assert len(tracked.coords) > 2
    assert max_deviation_px(tracked, truth) * GSD_M <= 0.1
    assert tracked.length > fit.line().length  # follows the arc beyond the chord band


def test_track_row_falls_back_to_the_fit_without_points(det: RowsDetectConfig) -> None:
    fit = BandFit(point=(100.0, 100.0), direction=(1.0, 0.0), angle_px_deg=0.0, t_lo=-50.0, t_hi=50.0,
                  n_points=0, residual_p95_px=0.0)
    far = np.array([[900.0, 900.0], [901.0, 900.0]], dtype=np.float32)
    assert track_row(far, fit, det).equals(fit.line())


def test_tiny_band_is_rejected_min_band_area(det: RowsDetectConfig) -> None:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[1000:1004, 1000:1040] = True  # 160 px = 0.1 m2 < 0.2 m2
    fit = _fit(mask, 0.0, 1002.0, det)
    assert fit is not None and fit.reject == REJECT_MIN_BAND_AREA
    assert fit.angle_px_deg == 0.0


def test_too_few_points_gives_none(det: RowsDetectConfig) -> None:
    pts = np.array([[10.0, 10.0]], dtype=np.float32)
    assert fit_band_line(pts, np.array([10.0]), 10.0, 0.0, det) is None


def test_angle_gate_rejects_a_tilted_long_band(det: RowsDetectConfig) -> None:
    mask = _single_stripe(36.0, 600.0)
    peak = float(np.median(SlabIndex.build(mask_points(mask), 30.0).offsets))
    fit = _fit(mask, 30.0, peak, det, stub_len_m=5.0)
    assert fit is not None and fit.reject == REJECT_ANGLE_GATE
    assert axial_diff_deg(fit.angle_px_deg, 36.0) < 1.0


def test_short_stub_keeps_the_profile_angle(det: RowsDetectConfig) -> None:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[1000:1016, 1000:1120] = True  # 3 m stub
    fit = _fit(mask, 2.0, float(np.array([-np.sin(np.radians(2.0)), np.cos(np.radians(2.0))]) @ [1060, 1008]),
               det, stub_len_m=5.0)
    assert fit is not None and fit.reject is None
    assert fit.angle_px_deg == pytest.approx(2.0)


def test_centreline_residual() -> None:
    t = np.arange(1000.0)
    assert centreline_residual_px(t, np.zeros_like(t), 100.0) == pytest.approx(0.0)
    assert np.isnan(centreline_residual_px(np.zeros(0), np.zeros(0), 100.0))


def test_end_near_edge_snaps_exactly_far_end_unchanged(det: RowsDetectConfig) -> None:
    snap_px = det.snap_to_edge_m / GSD_M
    near, far = 2.0 / GSD_M, 4.0 / GSD_M
    line = LineString([(near, 700.0), (TILE_PX - far, 700.0)])
    out = extend_to_clip(line, FRAME, snap_px, ray_max_factor=RAY)
    assert out.coords[0] == (0.0, 700.0)
    assert out.coords[-1] == pytest.approx((TILE_PX - far, 700.0))
    vertical = extend_to_clip(LineString([(900.0, 400.0), (900.0, TILE_PX - near)]), FRAME, snap_px, ray_max_factor=RAY)
    assert vertical.coords[-1] == (900.0, float(TILE_PX))


def test_nodata_corner_stops_the_extension_at_the_valid_boundary(det: RowsDetectConfig) -> None:
    clip = FRAME.difference(box(0.0, 0.0, 300.0, 300.0))
    snap_px = det.snap_to_edge_m / GSD_M
    in_bay_row = extend_to_clip(LineString([(340.0, 250.0), (1500.0, 250.0)]), clip, snap_px, ray_max_factor=RAY)
    assert in_bay_row.coords[0] == pytest.approx((300.0, 250.0))
    below_bay_row = extend_to_clip(LineString([(40.0, 500.0), (1500.0, 500.0)]), clip, snap_px, ray_max_factor=RAY)
    assert below_bay_row.coords[0] == (0.0, 500.0)


def test_end_outside_the_clip_is_pulled_back(det: RowsDetectConfig) -> None:
    line = LineString([(500.0, 700.0), (TILE_PX + 2.0, 701.0)])
    out = extend_to_clip(line, FRAME, det.snap_to_edge_m / GSD_M, ray_max_factor=RAY)
    assert out.coords[-1][0] == float(TILE_PX)
    assert out.coords[0] == (500.0, 700.0)
    assert extend_to_clip(line, box(0, 0, 0, 0).buffer(0), 120.0, ray_max_factor=RAY).equals(line)


def test_extension_parallel_to_an_edge_is_refused(det: RowsDetectConfig) -> None:
    line = LineString([(500.0, 20.0), (1500.0, 20.1)])
    out = extend_to_clip(line, FRAME, det.snap_to_edge_m / GSD_M, ray_max_factor=RAY)
    assert out.coords[0] == pytest.approx((500.0, 20.0))


def test_near_edge_and_max_deviation() -> None:
    assert near_edge((10.0, 500.0), FRAME, 20.0)
    assert not near_edge((500.0, 500.0), FRAME, 20.0)
    ref = LineString([(0.0, 0.0), (100.0, 0.0)])
    assert max_deviation_px(LineString([(0.0, 1.0), (50.0, 3.0)]), ref) == pytest.approx(3.0)
    assert Point(0, 0).distance(ref) == 0.0
