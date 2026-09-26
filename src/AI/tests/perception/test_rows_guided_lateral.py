"""perception.rows_guided: partial-block completion (own-lattice prior, lateral anchoring, rounds)."""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from tests.helpers.synth import row_axes, striped_mask
from vineyard.config import AppConfig, load_config
from vineyard.config.sections_perception import RowsGuidedConfig
from vineyard.contracts.ordering import canonical_normal
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_ref
from vineyard.perception.rows_detect import DetectParams
from vineyard.perception.rows_guided import (
    GuidedInputs,
    LatticePrior,
    anchor_score,
    guided_detect,
    lattice_prior,
    line_offset_m,
    parallel_rows,
)
from vineyard.pipeline.stages._rows_guided_io import neighbourhood, target_tiles, tile_priors

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


@pytest.fixture(scope="module")
def stripes() -> np.ndarray:
    return striped_mask(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX, width_px=16.0)


def _lines_utm(min_len_m: float = 10.0) -> list[LineString]:
    """True row axes (>= min_len_m), ordered by their offset along the row normal."""
    axes = row_axes(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX)
    lines = [LineString(px_to_utm(TILE, np.asarray(a, dtype=np.float64))) for a in axes]
    long = [ln for ln in lines if ln.length >= min_len_m]
    return sorted(long, key=lambda ln: line_offset_m(ln, ANGLE_UTM))


def _self_prior(g: RowsGuidedConfig) -> LatticePrior:
    lines = _lines_utm()
    prior = lattice_prior(TILE_ID, lines, np.full(len(lines), ANGLE_UTM), np.full(len(lines), SPACING_M), g, RANGE_M)
    assert prior is not None
    return prior


def _offsets(res_cands: tuple) -> list[float]:
    return [line_offset_m(LineString(px_to_utm(TILE, np.asarray(c.line_px.coords, dtype=np.float64))), ANGLE_UTM)
            for c in res_cands]


def test_config_enables_block_completion(g: RowsGuidedConfig) -> None:
    assert g.self_prior and g.lateral_enabled and g.rounds >= 2
    assert 1.0 < g.lateral_max_factor < 2.0  # the next row of the block, never the one after
    assert g.lateral_min_rows <= g.min_rows


def test_anchor_score_continuation_and_lateral(g: RowsGuidedConfig) -> None:
    piece = LineString([(0.0, 0.0), (20.0, 0.0)])
    beside = LineString([(0.0, SPACING_M), (20.0, SPACING_M)])
    two_away = LineString([(0.0, 2.0 * SPACING_M), (20.0, 2.0 * SPACING_M)])
    continued = LineString([(21.0, 0.0), (40.0, 0.0)])
    assert anchor_score(piece, (continued,), [], SPACING_M, g) <= 1.0
    assert anchor_score(piece, (), [beside], SPACING_M, g) <= 1.0
    assert anchor_score(piece, (), [two_away], SPACING_M, g) > 1.0
    assert anchor_score(piece, (), [], SPACING_M, g) == math.inf
    off = g.model_copy(update={"lateral_enabled": False})
    assert anchor_score(piece, (), [beside], SPACING_M, off) == math.inf


def test_parallel_rows_drops_other_orientations() -> None:
    a = LineString([(0.0, 0.0), (10.0, 0.0)])
    b = LineString([(0.0, 0.0), (0.0, 10.0)])
    c = LineString([(10.0, 0.2), (0.0, 0.0)])  # reversed, ~1 deg
    assert parallel_rows([a, b, c], 0.0, 5.0) == [a, c]
    assert parallel_rows([a, b], 90.0, 5.0) == [b]


def test_lateral_growth_completes_a_partial_block(stripes: np.ndarray, params: DetectParams,
                                                  g: RowsGuidedConfig) -> None:
    """A tile holding 5 rows of its block (no neighbour rows): the other rows are added beside them."""
    lines = _lines_utm()
    existing = lines[5:10]
    inp = GuidedInputs(veg=stripes, clip_px=FRAME, tile=TILE)
    res = guided_detect(inp, (_self_prior(g),), existing, params, g)
    assert len(res.candidates) >= len(lines) - len(existing) - 3
    off = g.model_copy(update={"lateral_enabled": False})
    assert guided_detect(inp, (_self_prior(g),), existing, params, off).candidates == ()


def test_single_missing_row_of_an_existing_block_is_added(stripes: np.ndarray, params: DetectParams,
                                                          g: RowsGuidedConfig) -> None:
    lines = _lines_utm(0.0)
    missing = _lines_utm()[8]
    existing = [ln for ln in lines if not ln.equals(missing)]
    res = guided_detect(GuidedInputs(veg=stripes, clip_px=FRAME, tile=TILE), (_self_prior(g),), existing, params, g)
    assert len(res.candidates) == 1
    assert _offsets(res.candidates)[0] == pytest.approx(line_offset_m(missing, ANGLE_UTM), abs=0.3)


def test_lateral_growth_stops_at_a_bare_gap(params: DetectParams, g: RowsGuidedConfig) -> None:
    """Two blocks on one lattice separated by 3 empty row slots: growth from block A never reaches block B."""
    veg = striped_mask(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX, width_px=16.0)
    lines = _lines_utm()
    cut_lo, cut_hi = (line_offset_m(lines[k], ANGLE_UTM) for k in (9, 12))
    v, u = np.mgrid[0:TILE_PX, 0:TILE_PX]
    utm = px_to_utm(TILE, np.stack([u.ravel() + 0.5, v.ravel() + 0.5], axis=1).astype(np.float64))
    off = (utm @ np.asarray(canonical_normal(ANGLE_UTM), dtype=np.float64)).reshape(SHAPE)
    veg[(off > cut_lo - SPACING_M / 2) & (off < cut_hi - SPACING_M / 2)] = False
    existing = lines[3:7]
    res = guided_detect(GuidedInputs(veg=veg, clip_px=FRAME, tile=TILE), (_self_prior(g),), existing, params, g)
    offs = _offsets(res.candidates)
    assert offs, "block A must still be completed"
    assert max(offs) < cut_lo
    assert min(offs) < line_offset_m(existing[0], ANGLE_UTM)


def test_grass_beside_a_block_gets_no_lateral_rows(params: DetectParams, g: RowsGuidedConfig) -> None:
    """Rows on one half, dense grass on the other: growth stops at the grass (flat density, no contrast)."""
    veg = striped_mask(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX, width_px=16.0)
    rng = np.random.default_rng(1)
    veg[:, 1024:] = rng.random((TILE_PX, TILE_PX - 1024)) < 0.6
    left = box(*px_to_utm(TILE, np.array([[0.0, 2048.0], [900.0, 0.0]])).ravel())
    existing = [ln for ln in _lines_utm() if ln.within(left)][:5]
    assert len(existing) >= 3
    res = guided_detect(GuidedInputs(veg=veg, clip_px=FRAME, tile=TILE), (_self_prior(g),), existing, params, g)
    for c in res.candidates:
        xs = np.asarray(c.line_px.coords)[:, 0]
        assert float(xs.min()) < 1100.0  # any added row lies (mostly) in the striped half


# ------------------------------------------------------------------ rows_link helper


def _cands(rows: list[tuple[str, str | None]]) -> gpd.GeoDataFrame:
    lines = _lines_utm()
    recs = [{"cand_id": f"{t}:K{i + 1:02d}", "tile_id": t, "angle_deg": ANGLE_UTM, "local_spacing_m": SPACING_M,
             "rejected_reason": r, "geometry": lines[i % len(lines)]} for i, (t, r) in enumerate(rows)]
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=CRS_EPSG)


def test_target_tiles_self_prior(cfg: AppConfig) -> None:
    n = cfg.rows_guided.prior_min_rows
    cands = _cands([(TILE_ID, None)] * n)
    priors = tile_priors(cands[cands.rejected_reason.isna()], cfg)
    assert list(priors) == [TILE_ID]
    assert target_tiles(cands, [TILE_ID], priors, 999) == []
    with_self = target_tiles(cands, [TILE_ID], priors, 999, self_prior=True)
    assert [t for t, _ in with_self] == [TILE_ID]
    assert with_self[0][1][0].source_tile == TILE_ID
    assert target_tiles(cands, [TILE_ID], priors, n - 1, self_prior=True) == []


def test_neighbourhood_of_gained_tiles() -> None:
    near = neighbourhood(["siret3_r021_c012"])
    assert len(near) == 9 and "siret3_r021_c012" in near and "siret3_r022_c013" in near
    assert neighbourhood([]) == set()
