"""perception.rows_guided + the rows_link helper: neighbour-guided recovery of missed rows."""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from tests.helpers.synth import row_axes, striped_mask
from vineyard.config import AppConfig, load_config
from vineyard.config.sections_perception import RowsGuidedConfig
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_ref
from vineyard.perception.row_features import RowFeatures
from vineyard.perception.rows_detect import DetectParams
from vineyard.perception.rows_guided import (
    FLAG_GUIDED,
    GuidedInputs,
    LatticePrior,
    axial_median_deg,
    cluster_priors,
    guided_detect,
    guided_reject,
    lattice_phase_m,
    lattice_prior,
    line_offset_m,
    merge_priors,
    neighbour_tile_ids,
    phase_dev_frac,
)
from vineyard.pipeline.stages._rows_guided_io import target_tiles, tile_priors

TILE_ID = "siret3_r021_c012"
TILE = tile_ref(TILE_ID)
SHAPE = (TILE_PX, TILE_PX)
FRAME = box(0.0, 0.0, float(TILE_PX), float(TILE_PX))
ANGLE_PX = 30.0
SPACING_PX = 100.0  # 2.5 m
ANGLE_UTM = (-ANGLE_PX) % 180.0
SPACING_M = SPACING_PX * GSD_M
RANGE_M = (1.8, 3.8)


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture(scope="module")
def g(cfg: AppConfig) -> RowsGuidedConfig:
    return cfg.rows_guided


@pytest.fixture(scope="module")
def params(cfg: AppConfig) -> DetectParams:
    return DetectParams.from_config(cfg)


def _true_lines_utm() -> list[LineString]:
    axes = row_axes(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX)
    return [LineString(px_to_utm(TILE, np.asarray(a, dtype=np.float64))) for a in axes]


def _true_prior(g: RowsGuidedConfig, *, shift_m: float = 0.0, angle_utm: float = ANGLE_UTM) -> LatticePrior:
    lines = _true_lines_utm()
    prior = lattice_prior("siret3_r020_c012", lines, np.full(len(lines), ANGLE_UTM), np.full(len(lines), SPACING_M),
                          g, RANGE_M)
    assert prior is not None
    return LatticePrior(angle_utm_deg=angle_utm, spacing_m=prior.spacing_m,
                        phase_m=(prior.phase_m + shift_m) % prior.spacing_m, n_rows=prior.n_rows,
                        source_tile=prior.source_tile)


def _features(**kw: float) -> RowFeatures:
    base = dict(length_m=40.0, support_frac=0.8, width_med_m=0.5, width_p80_m=0.7, along_period_m=float("nan"),
                along_duty=0.8, gaps=(), vine_score=float("nan"), width_spacing_ratio=0.25, rel_contrast=0.9)
    return RowFeatures(**{**base, **kw})


NEIGHBOURS = tuple(_true_lines_utm())


def test_guided_needs_continuation_of_a_neighbour_row(stripes: np.ndarray, params: DetectParams,
                                                       g: RowsGuidedConfig) -> None:
    prior = _true_prior(g)
    alone = guided_detect(GuidedInputs(veg=stripes, clip_px=FRAME, tile=TILE), (prior,), [], params, g)
    assert alone.candidates == ()
    few = GuidedInputs(veg=stripes, clip_px=FRAME, tile=TILE, neighbour_utm=NEIGHBOURS[:6])
    assert len(guided_detect(few, (prior,), [], params, g).candidates) <= 6


def test_guided_rows_stop_at_a_bare_band(params: DetectParams, g: RowsGuidedConfig) -> None:
    """Rows cut by a road: only the piece continuing the neighbour rows (left of the road) is kept."""
    veg = striped_mask(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX, width_px=16.0)
    veg[:, 1000:1240] = False  # 6 m bare band across every row
    left = box(*px_to_utm(TILE, np.array([[0.0, 2048.0], [900.0, 0.0]])).ravel())
    neighbours = tuple(ln.intersection(left) for ln in NEIGHBOURS if ln.intersects(left))
    inp = GuidedInputs(veg=veg, clip_px=FRAME, tile=TILE, neighbour_utm=neighbours)
    res = guided_detect(inp, (_true_prior(g),), [], params, g)
    assert len(res.candidates) >= g.min_rows
    assert max(float(np.asarray(c.line_px.coords)[:, 0].max()) for c in res.candidates) < 1010.0


@pytest.fixture(scope="module")
def stripes() -> np.ndarray:
    return striped_mask(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX, width_px=16.0)


# ------------------------------------------------------------------ helpers


def test_neighbour_tile_ids_interior_and_edge() -> None:
    ids = neighbour_tile_ids("siret3_r021_c012")
    assert len(ids) == 8 and "siret3_r020_c011" in ids and "siret3_r022_c013" in ids
    assert TILE_ID not in ids
    assert len(neighbour_tile_ids("siret3_r000_c000")) == 3


def test_axial_median_wraps_around_180() -> None:
    med = axial_median_deg(np.array([179.0, 1.0, 0.5]))
    assert min(med, 180.0 - med) < 1.0


def test_lattice_phase_and_deviation() -> None:
    offsets = 1.0 + 2.5 * np.arange(-3, 5)
    phase = lattice_phase_m(offsets, 2.5)
    assert phase == pytest.approx(1.0, abs=1e-9)
    prior = LatticePrior(angle_utm_deg=0.0, spacing_m=2.5, phase_m=phase, n_rows=8, source_tile="t")
    assert phase_dev_frac(3.5, prior) == pytest.approx(0.0, abs=1e-9)
    assert phase_dev_frac(2.25, prior) == pytest.approx(0.5, abs=1e-9)


def test_lattice_prior_needs_rows_and_vine_spacing(g: RowsGuidedConfig) -> None:
    lines = _true_lines_utm()
    few = lines[: g.prior_min_rows - 1]
    assert lattice_prior("t", few, np.full(len(few), ANGLE_UTM), np.full(len(few), SPACING_M), g, RANGE_M) is None
    wide = np.full(len(lines), 6.0)
    assert lattice_prior("t", lines, np.full(len(lines), ANGLE_UTM), wide, g, RANGE_M) is None
    prior = _true_prior(g)
    assert prior.angle_utm_deg == pytest.approx(ANGLE_UTM)
    assert prior.spacing_m == pytest.approx(SPACING_M)
    assert all(phase_dev_frac(line_offset_m(ln, ANGLE_UTM), prior) < 0.01 for ln in lines)


def test_merge_priors_keeps_most_rows_per_angle() -> None:
    a = LatticePrior(128.0, 2.7, 0.1, 20, "siret3_r001_c001")
    b = LatticePrior(126.0, 2.7, 0.2, 25, "siret3_r001_c002")
    c = LatticePrior(50.0, 2.5, 0.3, 6, "siret3_r002_c001")
    assert merge_priors([a, b, c], 5.0) == (b, c)
    assert merge_priors([c, b, a], 5.0) == (b, c)


def test_cluster_priors_groups_by_angle_and_caps_members() -> None:
    a = LatticePrior(128.0, 2.7, 0.1, 20, "siret3_r001_c001")
    b = LatticePrior(126.0, 2.3, 0.2, 25, "siret3_r001_c002")
    c = LatticePrior(50.0, 2.5, 0.3, 6, "siret3_r002_c001")
    d = LatticePrior(127.0, 2.6, 0.4, 10, "siret3_r002_c002")
    assert cluster_priors([a, b, c, d], 5.0, 3) == ((b, a, d), (c,))
    assert cluster_priors([d, c, b, a], 5.0, 2) == ((b, a), (c,))


def test_guided_tries_every_prior_of_a_cluster(stripes: np.ndarray, params: DetectParams,
                                               g: RowsGuidedConfig) -> None:
    good = _true_prior(g)
    # the strongest neighbour has the wrong phase; the weaker one on the true lattice must still win
    bad = LatticePrior(ANGLE_UTM, good.spacing_m, (good.phase_m + SPACING_M / 2.0) % SPACING_M, good.n_rows + 10,
                       "siret3_r020_c011")
    res = guided_detect(GuidedInputs(veg=stripes, clip_px=FRAME, tile=TILE, neighbour_utm=NEIGHBOURS), (bad, good), [], params, g)
    assert len(res.candidates) >= 15 and len(res.orientations) == 1


def test_guided_reject_gates(g: RowsGuidedConfig) -> None:
    duty_max = 0.6
    assert guided_reject(_features(), g, duty_max) is None
    assert guided_reject(_features(rel_contrast=0.05), g, duty_max) == "low_contrast"
    assert guided_reject(_features(support_frac=0.1), g, duty_max) == "low_occupancy"
    assert guided_reject(_features(length_m=2.0), g, duty_max) == "short"
    # weedy but clearly structured row: kept; wide and only mildly structured: too_wide
    assert guided_reject(_features(width_p80_m=2.5, rel_contrast=0.7), g, duty_max) is None
    assert guided_reject(_features(width_p80_m=2.5, rel_contrast=0.3), g, duty_max) == "too_wide"
    # discrete crowns (orchard) vs young vines at a 3.0 m plant period
    assert guided_reject(_features(along_period_m=4.5, along_duty=0.4), g, duty_max) == "orchard_period"
    assert guided_reject(_features(along_period_m=3.0, along_duty=0.4), g, duty_max) is None


# ------------------------------------------------------------------ guided detection


def test_guided_finds_rows_on_the_neighbour_lattice(stripes: np.ndarray, params: DetectParams,
                                                    g: RowsGuidedConfig) -> None:
    prior = _true_prior(g)
    res = guided_detect(GuidedInputs(veg=stripes, clip_px=FRAME, tile=TILE, neighbour_utm=NEIGHBOURS), (prior,), [], params, g)
    assert len(res.candidates) >= 15
    assert all(c.rejected_reason is None and c.soft_flags == (FLAG_GUIDED,) for c in res.candidates)
    for c in res.candidates:
        line = LineString(px_to_utm(TILE, np.asarray(c.line_px.coords, dtype=np.float64)))
        assert phase_dev_frac(line_offset_m(line, ANGLE_UTM), prior) < 0.05
        assert abs(((c.angle_px_deg - ANGLE_PX + 90.0) % 180.0) - 90.0) < 1.0


def test_guided_rejects_off_phase_lattice(stripes: np.ndarray, params: DetectParams, g: RowsGuidedConfig) -> None:
    prior = _true_prior(g, shift_m=SPACING_M / 2.0)
    res = guided_detect(GuidedInputs(veg=stripes, clip_px=FRAME, tile=TILE, neighbour_utm=NEIGHBOURS), (prior,), [], params, g)
    assert res.candidates == ()


def test_guided_only_searches_near_the_prior_angle(stripes: np.ndarray, params: DetectParams,
                                                   g: RowsGuidedConfig) -> None:
    prior = _true_prior(g, angle_utm=(ANGLE_UTM + 40.0) % 180.0)
    res = guided_detect(GuidedInputs(veg=stripes, clip_px=FRAME, tile=TILE, neighbour_utm=NEIGHBOURS), (prior,), [], params, g)
    assert len(res.candidates) < g.min_rows


def test_guided_skips_rows_already_accepted(stripes: np.ndarray, params: DetectParams, g: RowsGuidedConfig) -> None:
    prior = _true_prior(g)
    inp = GuidedInputs(veg=stripes, clip_px=FRAME, tile=TILE, neighbour_utm=NEIGHBOURS)
    full = guided_detect(inp, (prior,), [], params, g)
    existing = _true_lines_utm()[:5]
    partial = guided_detect(inp, (prior,), existing, params, g)
    assert len(partial.candidates) <= len(full.candidates) - 4


def test_guided_empty_mask_or_no_prior(params: DetectParams, g: RowsGuidedConfig) -> None:
    empty = np.zeros(SHAPE, dtype=bool)
    assert guided_detect(GuidedInputs(veg=empty, clip_px=FRAME, tile=TILE, neighbour_utm=NEIGHBOURS), (_true_prior(g),), [], params,
                         g).candidates == ()
    stripes = striped_mask(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX, width_px=16.0)
    assert guided_detect(GuidedInputs(veg=stripes, clip_px=FRAME, tile=TILE, neighbour_utm=NEIGHBOURS), (), [], params, g).candidates == ()


def test_grass_field_gets_no_guided_rows(params: DetectParams, g: RowsGuidedConfig) -> None:
    rng = np.random.default_rng(0)
    grass = rng.random(SHAPE) < 0.6
    res = guided_detect(GuidedInputs(veg=grass, clip_px=FRAME, tile=TILE, neighbour_utm=NEIGHBOURS), (_true_prior(g),), [], params, g)
    assert res.candidates == ()


# ------------------------------------------------------------------ rows_link helper


def _cands(rows: list[tuple[str, float, float, str | None]]) -> gpd.GeoDataFrame:
    lines = _true_lines_utm()
    recs = [{"cand_id": f"{t}:K{i + 1:02d}", "tile_id": t, "angle_deg": a, "local_spacing_m": s,
             "rejected_reason": r, "geometry": lines[i % len(lines)]} for i, (t, a, s, r) in enumerate(rows)]
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=CRS_EPSG)


def test_tile_priors_and_targets(cfg: AppConfig) -> None:
    n = cfg.rows_guided.prior_min_rows
    rows = [("siret3_r020_c012", ANGLE_UTM, SPACING_M, None)] * n            # prior tile
    rows += [("siret3_r020_c020", ANGLE_UTM, SPACING_M, None)] * (n - 1)     # too few rows for a prior
    rows += [("siret3_r021_c012", ANGLE_UTM, SPACING_M, "too_wide")] * 3     # target (0 accepted)
    cands = _cands(rows)
    priors = tile_priors(cands[cands.rejected_reason.isna()], cfg)
    assert set(priors) == {"siret3_r020_c012"}
    tiles = ["siret3_r021_c012", "siret3_r020_c012", "siret3_r021_c020", "siret3_r030_c030"]
    targets = target_tiles(cands, tiles, priors, cfg.rows_guided.target_max_accepted)
    # the prior tile itself has > target_max_accepted rows; r021_c020's neighbour has no prior; r030 is isolated
    assert [t for t, _ in targets] == ["siret3_r021_c012"]
    assert targets[0][1][0].source_tile == "siret3_r020_c012"
    assert math.isclose(targets[0][1][0].spacing_m, SPACING_M)
