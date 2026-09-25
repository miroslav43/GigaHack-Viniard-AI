from __future__ import annotations

import numpy as np
import pytest

from vineyard.perception.waste.lattice import (
    Lattice,
    LatticeParams,
    axis_projection,
    fit_lattice,
    is_periodic,
    phase_offset,
)

P = LatticeParams(pitch_range_m=(0.8, 2.0), min_points=4, bin_m=0.05)


def test_fit_recovers_pitch_and_phase() -> None:
    t = 0.3 + 1.2 * np.arange(10)
    lat = fit_lattice(t, P)
    assert lat is not None
    assert lat.pitch_m == pytest.approx(1.2, abs=0.05)
    assert lat.phase_m == pytest.approx(0.3, abs=0.05)
    assert lat.n_points == 10


def test_periodic_test_uses_quarter_pitch() -> None:
    lat = Lattice(pitch_m=1.2, phase_m=0.3, n_points=10)
    assert is_periodic(3.9, lat, 0.25)
    assert not is_periodic(4.5, lat, 0.25)
    assert is_periodic(3.9 + 0.29, lat, 0.25)
    assert not is_periodic(3.9 + 0.31, lat, 0.25)


def test_phase_offset_is_signed_distance_to_nearest_plant() -> None:
    lat = Lattice(pitch_m=1.2, phase_m=0.3, n_points=10)
    assert phase_offset(3.9, lat) == pytest.approx(0.0, abs=1e-9)
    assert phase_offset(4.0, lat) == pytest.approx(0.1)
    assert phase_offset(3.8, lat) == pytest.approx(-0.1)


def test_too_few_points_gives_none() -> None:
    assert fit_lattice(np.array([0.3, 1.5, 2.7]), P) is None


def test_no_difference_in_range_gives_none() -> None:
    assert fit_lattice(np.array([0.0, 0.1, 0.2, 0.3, 10.0]), P) is None


def test_missing_plants_keep_the_fundamental() -> None:
    k = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])
    lat = fit_lattice(0.5 + 0.9 * k, P)
    assert lat is not None
    assert lat.pitch_m == pytest.approx(0.9, abs=0.05)


def test_sub_harmonic_is_preferred_when_well_supported() -> None:
    t = np.sort(np.concatenate([0.2 + 1.8 * np.arange(10), 1.1 + 1.8 * np.array([0, 3, 6])]))
    lat = fit_lattice(t, P)
    assert lat is not None
    assert lat.pitch_m == pytest.approx(0.9, abs=0.05)


def test_jittered_positions() -> None:
    rng = np.random.default_rng(0)
    t = 0.7 + 1.1 * np.arange(20) + rng.normal(0.0, 0.03, 20)
    lat = fit_lattice(t, P)
    assert lat is not None
    assert lat.pitch_m == pytest.approx(1.1, abs=0.03)
    assert is_periodic(0.7 + 1.1 * 25, lat, 0.25)


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        LatticeParams(pitch_range_m=(2.0, 0.8), min_points=4, bin_m=0.05)
    with pytest.raises(ValueError):
        LatticeParams(pitch_range_m=(0.8, 2.0), min_points=1, bin_m=0.05)
    with pytest.raises(ValueError):
        LatticeParams(pitch_range_m=(0.8, 2.0), min_points=4, bin_m=0.0)


def test_axis_projection_along_and_distance() -> None:
    axis = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    pts = np.array([[10.0, 5.0], [110.0, 50.0], [-10.0, 0.0]])
    along, dist = axis_projection(axis, pts)
    assert along == pytest.approx([10.0, 150.0, 0.0])
    assert dist == pytest.approx([5.0, 10.0, 10.0])
