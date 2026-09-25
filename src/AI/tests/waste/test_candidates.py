from __future__ import annotations

import numpy as np
import pytest

from tests.waste.synth_waste import paint, soil
from vineyard.perception.waste.candidates import CandidateParams, find_candidates
from vineyard.perception.waste.types import ColourClass

BLUE = (20, 60, 230)
WHITE = (250, 250, 250)
GREEN = (40, 200, 40)
TILE = "siret3_r021_c012"


@pytest.fixture
def params(cfg) -> CandidateParams:
    return CandidateParams.from_config(cfg.waste, cfg.grid.gsd_m)


def test_params_from_config_px_thresholds(params: CandidateParams) -> None:
    assert params.min_area_px == 24
    assert params.open_px == 1


def test_blue_5x5_is_one_candidate(params: CandidateParams) -> None:
    img = paint(soil(), BLUE, slice(50, 55), slice(60, 65))
    cands = find_candidates(img, np.ones(img.shape[:2], bool), TILE, params)
    assert len(cands) == 1
    c = cands[0]
    assert c.colour_class == ColourClass.VIVID and not c.is_white
    assert c.area_px == 25
    assert c.centroid_px == pytest.approx((62.5, 52.5))
    assert c.area_m2 == pytest.approx(25 * 0.000625)


def test_blob_below_min_area_is_dropped(params: CandidateParams) -> None:
    img = paint(soil(), BLUE, slice(50, 54), slice(60, 65))
    assert find_candidates(img, np.ones(img.shape[:2], bool), TILE, params) == ()


def test_white_blob_is_bright(params: CandidateParams) -> None:
    img = paint(soil(), WHITE, slice(10, 20), slice(10, 20))
    (c,) = find_candidates(img, np.ones(img.shape[:2], bool), TILE, params)
    assert c.is_white and c.colour_class == ColourClass.BRIGHT
    assert c.aspect == pytest.approx(1.0)


def test_green_blob_is_not_a_candidate(params: CandidateParams) -> None:
    img = paint(soil(), GREEN, slice(10, 30), slice(10, 30))
    assert find_candidates(img, np.ones(img.shape[:2], bool), TILE, params) == ()


def test_box_corner_convention_and_padding(params: CandidateParams) -> None:
    img = paint(soil(), BLUE, slice(100, 120), slice(100, 120))
    (c,) = find_candidates(img, np.ones(img.shape[:2], bool), TILE, params)
    assert c.box.as_tuple() == pytest.approx((99.0, 99.0, 121.0, 121.0))
    unpadded = CandidateParams(**{**vars(params), "box_pad": 0.0})
    (d,) = find_candidates(img, np.ones(img.shape[:2], bool), TILE, unpadded)
    assert d.box.as_tuple() == pytest.approx((100.0, 100.0, 120.0, 120.0))


def test_box_is_clipped_to_image(params: CandidateParams) -> None:
    img = paint(soil(), BLUE, slice(0, 20), slice(0, 20))
    (c,) = find_candidates(img, np.ones(img.shape[:2], bool), TILE, params)
    assert c.box.xtl == 0.0 and c.box.ytl == 0.0


def test_nodata_region_gives_no_candidates(params: CandidateParams) -> None:
    img = paint(soil(), BLUE, slice(50, 60), slice(50, 60))
    valid = np.ones(img.shape[:2], bool)
    valid[40:70, 40:70] = False
    assert find_candidates(img, valid, TILE, params) == ()


def test_large_blob_is_kept_for_the_vehicle_filter(params: CandidateParams) -> None:
    img = paint(soil(512), BLUE, slice(100, 200), slice(100, 200))
    (c,) = find_candidates(img, np.ones(img.shape[:2], bool), TILE, params)
    assert c.area_px == 10_000


def test_elongated_blob_shape_stats(params: CandidateParams) -> None:
    img = paint(soil(), WHITE, slice(100, 104), slice(50, 90))
    (c,) = find_candidates(img, np.ones(img.shape[:2], bool), TILE, params)
    assert c.aspect == pytest.approx(10.0)
    assert c.length_m == pytest.approx(40 * 0.025)
    assert c.width_m == pytest.approx(4 * 0.025)


def test_opening_removes_one_px_lines(params: CandidateParams) -> None:
    img = paint(soil(), WHITE, slice(100, 101), slice(20, 200))
    assert find_candidates(img, np.ones(img.shape[:2], bool), TILE, params) == ()
    no_open = CandidateParams(**{**vars(params), "open_px": 0})
    assert len(find_candidates(img, np.ones(img.shape[:2], bool), TILE, no_open)) == 1


def test_output_sorted_and_keys_unique(params: CandidateParams) -> None:
    img = paint(paint(soil(), BLUE, slice(200, 210), slice(10, 20)), WHITE, slice(20, 30), slice(200, 210))
    cands = find_candidates(img, np.ones(img.shape[:2], bool), TILE, params)
    assert [c.centroid_px[1] for c in cands] == sorted(c.centroid_px[1] for c in cands)
    assert len({c.cand_key for c in cands}) == 2
    assert cands[0].cand_key == f"{TILE}@0205_0025"


def test_rejects_bad_inputs(params: CandidateParams) -> None:
    with pytest.raises(ValueError, match="RGB"):
        find_candidates(np.zeros((10, 10), np.uint8), np.ones((10, 10), bool), TILE, params)
    with pytest.raises(ValueError, match="valid"):
        find_candidates(soil(), np.ones((10, 10), bool), TILE, params)
