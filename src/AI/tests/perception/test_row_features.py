"""perception.row_features: occupancy, gaps, widths, periodicity, vine score and reject reasons."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from shapely.geometry import LineString

from tests.helpers.synth import blobs_mask
from vineyard.config import AppConfig, load_config
from vineyard.geo.tiling import GSD_M, TILE_PX
from vineyard.perception.profile import SpacingEstimate
from vineyard.perception.row_features import (
    FLAG_CURVED,
    FLAG_HARMONIC,
    FLAG_OFFLATTICE,
    FLAG_ORCHARD_PERIOD,
    FLAG_ORCHARD_RATIO,
    FLAG_WIDE,
    REASON_ANGLE_GATE,
    REASON_LOW_OCCUPANCY,
    REASON_LOW_SNR,
    REASON_LOW_VINE_SCORE,
    REASON_NO_PERIODICITY,
    REASON_SHORT,
    REASON_TOO_WIDE,
    RowFeatures,
    along_occupancy,
    along_periodicity,
    compute_row_features,
    corridor_mean,
    cross_contrast,
    final_reason,
    hard_reject_reason,
    is_hard_reject,
    occupancy_gaps,
    perpendicular_widths,
    soft_flags,
    soft_reject_reason,
    stations,
)

SHAPE = (TILE_PX, TILE_PX)
ROW_V = 1024
LINE = LineString([(24.0, ROW_V), (2024.0, ROW_V)])  # 50 m
SPACING_M = 2.5
M = 1.0 / GSD_M  # px per metre
GOOD = SpacingEstimate(spacing_m=SPACING_M, snr=20.0, harmonic_flag=False, ok=True)


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config()


def _features(cfg: AppConfig, veg: np.ndarray, line: LineString = LINE, **kw: object) -> RowFeatures:
    return compute_row_features(veg, line, spacing_m=SPACING_M, cfg=cfg.rows,
                                corridor_half_m=cfg.canopy.corridor_half_m, occ_bin_m=cfg.row_structure.occ_bin_m,
                                **kw)  # type: ignore[arg-type]


def _band(half_px: int, u_ranges: list[tuple[float, float]] | None = None) -> np.ndarray:
    veg = np.zeros(SHAPE, dtype=bool)
    for u0, u1 in u_ranges or [(0.0, float(TILE_PX))]:
        veg[ROW_V - half_px : ROW_V + half_px, int(u0) : int(u1)] = True
    return veg


def _hard(cfg: AppConfig, f: RowFeatures, sp: SpacingEstimate = GOOD, prob_present: bool = False) -> str | None:
    return hard_reject_reason(f, cfg.rows, prob_present=prob_present, sp=sp, orchard=cfg.orchard)


def test_stations_spacing_and_normals() -> None:
    st = stations(LINE, 1.0)
    assert len(st.points) == 50
    assert st.step_m == pytest.approx(1.0)
    np.testing.assert_allclose(st.points[0], [24.0 + 20.0, ROW_V])
    np.testing.assert_allclose(np.abs(st.normals), np.tile([0.0, 1.0], (50, 1)), atol=1e-9)


def test_continuous_half_metre_row(cfg: AppConfig) -> None:
    f = _features(cfg, _band(10))
    assert f.width_p80_m == pytest.approx(0.5, abs=0.05)
    assert f.width_med_m == pytest.approx(0.5, abs=0.05)
    assert f.along_duty > 0.6 and f.support_frac == pytest.approx(1.0)
    assert f.gaps == () and f.length_m == pytest.approx(50.0)
    assert f.width_spacing_ratio == pytest.approx(0.2, abs=0.02)
    assert f.rel_contrast == pytest.approx(1.0)
    assert math.isnan(f.vine_score)
    assert _hard(cfg, f) is None
    assert soft_flags(f, GOOD, cfg.rows, cfg.orchard, on_lattice=True, is_curved=False) == ()


def test_round_crowns_every_five_metres_are_too_wide(cfg: AppConfig) -> None:
    crowns = blobs_mask(SHAPE, [(u, float(ROW_V)) for u in np.arange(60.0, 2048.0, 5.0 * M)], 1.5 * M)
    f = _features(cfg, crowns)
    assert f.along_period_m == pytest.approx(5.0, abs=0.3)
    assert f.along_duty < 0.6
    assert f.width_p80_m >= cfg.rows.filter.width_p80_max_m
    assert _hard(cfg, f) == REASON_TOO_WIDE
    flags = soft_flags(f, GOOD, cfg.rows, cfg.orchard, on_lattice=True, is_curved=False)
    assert FLAG_ORCHARD_PERIOD in flags and FLAG_ORCHARD_RATIO in flags and FLAG_WIDE in flags


def test_wide_grassy_row_with_interrow_contrast_is_only_flagged(cfg: AppConfig) -> None:
    veg = _band(28)  # 1.4 m wide, bare interrow around it
    f = _features(cfg, veg)
    assert f.width_p80_m >= cfg.rows.filter.width_p80_max_m
    assert f.rel_contrast > 0.5
    assert _hard(cfg, f) is None
    harmonic = SpacingEstimate(spacing_m=SPACING_M, snr=20.0, harmonic_flag=True, ok=True)
    assert _hard(cfg, f, harmonic) == REASON_TOO_WIDE


def test_closed_canopy_without_contrast_is_too_wide(cfg: AppConfig) -> None:
    f = _features(cfg, np.ones(SHAPE, dtype=bool))
    assert f.rel_contrast == pytest.approx(0.0)
    assert _hard(cfg, f) == REASON_TOO_WIDE


def test_low_occupancy_and_short(cfg: AppConfig) -> None:
    sparse = _band(10, [(24.0 + k * 10 * M, 24.0 + k * 10 * M + 0.5 * M) for k in range(5)])
    f = _features(cfg, sparse)
    assert f.support_frac < cfg.rows.filter.occupancy_min
    assert _hard(cfg, f) == REASON_LOW_OCCUPANCY
    short = _features(cfg, _band(10), LineString([(1000.0, ROW_V), (1020.0, ROW_V)]))
    assert _hard(cfg, short) == REASON_SHORT


def test_gaps_are_recorded_from_three_metres(cfg: AppConfig) -> None:
    u0 = 24.0
    canopies = [(0.0, 2.0), (3.0, 4.0), (10.0, 12.0)]
    veg = _band(10, [(u0 + a * M, u0 + b * M) for a, b in canopies])
    line = LineString([(u0, ROW_V), (u0 + 12.0 * M, ROW_V)])
    f = _features(cfg, veg, line)
    assert len(f.gaps) == 1
    start, end = f.gaps[0]
    assert start == pytest.approx(4.0, abs=cfg.row_structure.occ_bin_m)
    assert end == pytest.approx(10.0, abs=cfg.row_structure.occ_bin_m)
    assert json.loads(f.gaps_json()) == [[round(start, 3), round(end, 3)]]


def test_occupancy_gaps_on_arrays() -> None:
    occ = np.array([1, 0, 0, 0, 1, np.nan, 0, 0, 0, 0, 0, 1], dtype=np.float64)
    assert occupancy_gaps(occ, 1.0, 3.0) == ((1.0, 4.0), (6.0, 11.0))
    assert occupancy_gaps(occ, 1.0, 4.0) == ((6.0, 11.0),)
    assert occupancy_gaps(np.zeros(0), 1.0, 1.0) == ()


def test_unknown_stations_are_nan() -> None:
    valid = np.ones(SHAPE, dtype=bool)
    valid[:, :1024] = False
    occ = along_occupancy(_band(10), LINE, 0.3, 0.5, valid)
    assert np.isnan(occ[:10]).all() and (occ[-10:] == 1.0).all()


def test_perpendicular_widths_seed_and_cap() -> None:
    widths = perpendicular_widths(_band(10), LINE, station_m=0.5, seed_half_m=0.3, max_half_m=1.0)
    assert len(widths) == 100 and np.allclose(widths, 0.5, atol=0.05)
    offset = np.roll(_band(4), 16, axis=0)  # 0.2 m wide run 0.4 m off the axis: beyond the seed window
    assert len(perpendicular_widths(offset, LINE, station_m=0.5, seed_half_m=0.3, max_half_m=1.0)) == 0
    capped = perpendicular_widths(np.ones(SHAPE, bool), LINE, station_m=0.5, seed_half_m=0.3, max_half_m=1.0)
    assert np.allclose(capped, 2.0 + GSD_M, atol=GSD_M)


def test_cross_contrast_values() -> None:
    assert math.isnan(cross_contrast(np.zeros((0, 0), bool), 0.3))
    assert math.isnan(cross_contrast(np.zeros((3, 81), bool), 0.3))


def test_along_periodicity_degenerate_cases() -> None:
    assert all(math.isnan(v) for v in along_periodicity(np.full(10, np.nan), 0.1, (3.0, 6.0)))
    period, duty = along_periodicity(np.ones(500), 0.1, (3.0, 6.0))
    assert math.isnan(period) and duty == 1.0
    period, _ = along_periodicity(np.tile([1.0, 0.0], 5), 0.1, (3.0, 6.0))  # 1 m long: band too short
    assert math.isnan(period)


def test_vine_score_from_prob_and_low_vine_score(cfg: AppConfig) -> None:
    prob = np.full(SHAPE, 0.1, dtype=np.float32)
    f = _features(cfg, _band(10), prob=prob)
    assert f.vine_score == pytest.approx(0.1, abs=1e-6)
    assert _hard(cfg, f, prob_present=True) == REASON_LOW_VINE_SCORE
    assert _hard(cfg, f, prob_present=False) is None
    assert math.isnan(corridor_mean(prob, LineString([(-500.0, -500.0), (-400.0, -500.0)]), 0.3, 0.5))


def test_soft_reject_and_flags(cfg: AppConfig) -> None:
    assert soft_reject_reason(SpacingEstimate(2.5, 50.0, False, False), cfg.rows) == REASON_NO_PERIODICITY
    assert soft_reject_reason(SpacingEstimate(2.5, 1.0, False, True), cfg.rows) == REASON_LOW_SNR
    assert soft_reject_reason(GOOD, cfg.rows) is None
    f = _features(cfg, _band(10))
    harmonic = SpacingEstimate(spacing_m=SPACING_M, snr=20.0, harmonic_flag=True, ok=True)
    flags = soft_flags(f, harmonic, cfg.rows, cfg.orchard, on_lattice=False, is_curved=True)
    assert flags == (FLAG_OFFLATTICE, FLAG_HARMONIC, FLAG_CURVED)


def test_final_reason_precedence() -> None:
    assert final_reason(REASON_ANGLE_GATE, REASON_SHORT, REASON_LOW_SNR) == REASON_ANGLE_GATE
    assert final_reason(None, REASON_SHORT, REASON_LOW_SNR) == REASON_SHORT
    assert final_reason(None, None, REASON_LOW_SNR) == REASON_LOW_SNR
    assert final_reason(None, None, None) is None
    assert is_hard_reject(REASON_TOO_WIDE) and not is_hard_reject(REASON_LOW_SNR) and not is_hard_reject(None)
