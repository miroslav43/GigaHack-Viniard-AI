"""perception.profile: dominant angle, offset profile, autocorr spacing + SNR, peaks and lattice flags."""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers.synth import blobs_mask, row_axes, striped_mask
from vineyard.config import RowsDetectConfig, load_config
from vineyard.geo.tiling import GSD_M
from vineyard.perception.profile import (
    Profile,
    SpacingEstimate,
    dominant_angle_px,
    estimate_spacing,
    find_row_offsets,
    lattice_deviation,
    mask_points,
    offset_profile,
    px_angle_to_utm,
    sample_points,
    spectral_snr,
    subsample,
    unit_vectors,
)

SHAPE = (1024, 1024)
SPACING_PX = 100.0  # 2.5 m
WIDTH_PX = 16.0  # 0.4 m


@pytest.fixture(scope="module")
def det() -> RowsDetectConfig:
    return load_config().rows.detect


def _analyse(mask: np.ndarray, d: RowsDetectConfig) -> tuple[float, Profile, SpacingEstimate]:
    pts = sample_points(mask, d.max_sample_points, 0)
    angle = dominant_angle_px(pts, coarse_step_deg=d.angle_coarse_step_deg, fine_step_deg=d.angle_step_deg,
                              bin_px=d.angle_bin_m / GSD_M)
    prof = offset_profile(pts, angle, d.profile_bin_m / GSD_M, d.smooth_sigma_bins)
    return angle, prof, estimate_spacing(prof, d)


def _axial_err(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def test_unit_vectors_and_utm_angle() -> None:
    d, n = unit_vectors(90.0)
    np.testing.assert_allclose(d, [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(n, [-1.0, 0.0], atol=1e-12)
    assert px_angle_to_utm(30.0) == pytest.approx(150.0)
    assert px_angle_to_utm(0.0) == pytest.approx(0.0)


def test_mask_points_are_pixel_centres() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 3] = mask[2, 0] = True
    np.testing.assert_array_equal(mask_points(mask), np.array([[3.5, 1.5], [0.5, 2.5]], dtype=np.float32))


def test_subsample_is_deterministic_ordered_and_noop_when_small() -> None:
    pts = np.arange(200, dtype=np.float32).reshape(100, 2)
    assert subsample(pts, 100, 1) is pts
    a, b = subsample(pts, 10, 7), subsample(pts, 10, 7)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 10 and np.all(np.diff(a[:, 0]) > 0)
    assert not np.array_equal(a, subsample(pts, 10, 8))


@pytest.mark.parametrize("angle", [30.0, 0.0, 171.0])
def test_dominant_angle_of_stripes(angle: float, det: RowsDetectConfig) -> None:
    mask = striped_mask(SHAPE, angle_deg=angle, spacing_px=SPACING_PX, width_px=WIDTH_PX)
    found, _, _ = _analyse(mask, det)
    assert 0.0 <= found < 180.0
    assert _axial_err(found, angle) <= 0.5


def test_dominant_angle_needs_two_points() -> None:
    with pytest.raises(ValueError, match=">= 2 points"):
        dominant_angle_px(np.zeros((1, 2), np.float32), coarse_step_deg=2.0, fine_step_deg=0.5, bin_px=4.0)


def test_stripes_spacing_snr_and_peak_count(det: RowsDetectConfig) -> None:
    mask = striped_mask(SHAPE, angle_deg=30.0, spacing_px=SPACING_PX, width_px=WIDTH_PX)
    _, prof, sp = _analyse(mask, det)
    assert sp.ok and not sp.harmonic_flag
    assert sp.spacing_m == pytest.approx(2.5, abs=0.05)
    assert sp.snr >= 10.0
    peaks = find_row_offsets(prof, sp, det)
    assert len(peaks) == len(row_axes(SHAPE, angle_deg=30.0, spacing_px=SPACING_PX))
    assert all(p.on_lattice for p in peaks)


def test_random_noise_has_low_snr(det: RowsDetectConfig) -> None:
    noise = np.random.default_rng(0).random(SHAPE) < 0.2
    _, _, sp = _analyse(noise, det)
    assert sp.snr < det.periodicity_min_snr


def test_orchard_period_folds_to_harmonic(det: RowsDetectConfig) -> None:
    orchard = striped_mask(SHAPE, angle_deg=0.0, spacing_px=200.0, width_px=40.0)  # rows 5 m apart
    _, _, sp = _analyse(orchard, det)
    assert sp.harmonic_flag
    assert det.spacing_min_m <= sp.spacing_m <= det.spacing_max_m


def test_orchard_crowns_flag_harmonic(det: RowsDetectConfig) -> None:
    centres = [(u, v) for v in np.arange(100.0, 1024.0, 200.0) for u in np.arange(60.0, 1024.0, 200.0)]
    _, _, sp = _analyse(blobs_mask(SHAPE, centres, 60.0), det)
    assert sp.harmonic_flag


def test_off_lattice_peak_is_flagged(det: RowsDetectConfig) -> None:
    mask = striped_mask(SHAPE, angle_deg=0.0, spacing_px=SPACING_PX, width_px=WIDTH_PX)
    mask[492:609, :] = False  # drop the rows at v=500 and v=600 ...
    mask[545:555, :] = True  # ... and put a spurious row half-way between them
    _, prof, sp = _analyse(mask, det)
    peaks = find_row_offsets(prof, sp, det)
    spurious = [p for p in peaks if abs(p.offset_px - 550.0) < 10.0]
    assert len(spurious) == 1
    assert spurious[0].lattice_dev_frac >= det.onlattice_tol_factor and not spurious[0].on_lattice
    assert sum(p.on_lattice for p in peaks) == len(peaks) - 1


def test_offset_profile_counts_weights_and_errors() -> None:
    pts = np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 1.6], [3.0, 10.0]], dtype=np.float32)
    prof = offset_profile(pts, 0.0, 1.0, 0.0)
    assert prof.counts.sum() == pytest.approx(4.0)
    np.testing.assert_array_equal(prof.counts, prof.smooth)
    assert prof.bin_m == pytest.approx(GSD_M)
    assert float(prof.centre_px(0)) == pytest.approx(prof.origin_px + 0.5)
    weighted = offset_profile(pts, 0.0, 1.0, 1.0, weights=np.array([1, 2, 2, 0], dtype=np.float32))
    assert weighted.counts.sum() == pytest.approx(5.0)
    with pytest.raises(ValueError, match="at least one point"):
        offset_profile(np.zeros((0, 2), np.float32), 0.0, 1.0, 0.0)


def test_find_row_offsets_on_empty_profile(det: RowsDetectConfig) -> None:
    prof = Profile(angle_px_deg=0.0, origin_px=0.0, bin_px=2.0, counts=np.zeros(10), smooth=np.zeros(10))
    sp = SpacingEstimate(spacing_m=2.5, snr=0.0, harmonic_flag=False, ok=False)
    assert find_row_offsets(prof, sp, det) == ()


def test_lattice_deviation_values() -> None:
    assert lattice_deviation(np.zeros(0), np.zeros(0), 2.5).shape == (0,)
    dev = lattice_deviation(np.array([0.0, 2.5, 5.0, 6.25]), np.array([1.0, 1.0, 1.0, 0.1]), 2.5)
    np.testing.assert_allclose(dev[:3], 0.0, atol=0.02)
    assert dev[3] == pytest.approx(0.5, abs=0.02)


def test_spectral_snr_of_flat_profile_is_zero(det: RowsDetectConfig) -> None:
    prof = Profile(angle_px_deg=0.0, origin_px=0.0, bin_px=2.0, counts=np.ones(400), smooth=np.ones(400))
    snr, _ = spectral_snr(prof, 2.5, det)
    assert snr == 0.0
