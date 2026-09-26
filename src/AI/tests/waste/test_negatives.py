"""Probe negatives (design 03 W4): rule candidates on the example tiles, white rejects elsewhere, random crops."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.waste.synth_waste import make_candidate
from vineyard.perception.waste.crop_store import read_crop_store
from vineyard.perception.waste.crops import crop_window
from vineyard.perception.waste.negatives import (
    SOURCE_EXAMPLE,
    SOURCE_RANDOM,
    SOURCE_TILE,
    NegativeError,
    NegativeParams,
    TileSources,
    build_negative_store,
    candidate_negatives,
    plan_tile,
    random_negatives,
    run_tile_ids,
    tile_rng,
)
from vineyard.perception.waste.types import Candidate, ColourClass, RejectReason

EXAMPLE = "siret3_r021_c012"
OTHER = "siret3_r010_c010"
EXTENT = 512


def params(**over: object) -> NegativeParams:
    base = NegativeParams(
        example_tiles=(EXAMPLE, "siret3_r006_c004"),
        other_reasons=frozenset({"near_axis", "too_small_white", "bright_soil", "tube_shape"}),
        max_per_other_tile=10,
        random_per_example_tile=5,
        random_per_other_tile=2,
        random_box_px=(5.0, 40.0),
        max_random_attempts=200,
        context=0.5,
        min_px=64,
        out_px=32,
        extent_px=EXTENT,
        seed=0,
        background_tiles=8,
    )
    return replace(base, **over)


def cand(tile: str, x: float, reason: RejectReason | None, white: bool = True) -> Candidate:
    c = make_candidate(tile_id=tile, centroid=(x, 100.0), is_white=white)
    return replace(c, reject_reason=reason)


def _mixed(tile: str) -> list[Candidate]:
    return [
        cand(tile, 100, None),
        cand(tile, 150, RejectReason.NEAR_AXIS),
        cand(tile, 200, RejectReason.NEAR_AXIS),
        cand(tile, 250, RejectReason.TOO_SMALL_WHITE),
        cand(tile, 300, RejectReason.FORBIDDEN),
        cand(tile, 350, RejectReason.TUBE_SHAPE, white=False),
    ]


def test_example_tile_uses_every_candidate() -> None:
    specs = candidate_negatives(_mixed(EXAMPLE), EXAMPLE, params(), tile_rng(0, EXAMPLE))
    assert len(specs) == 6
    assert all(s.source == SOURCE_EXAMPLE and s.is_example for s in specs)
    assert [s.survives for s in specs] == [True, False, False, False, False, False]
    assert specs[0].reason == "kept" and specs[1].reason == "near_axis"


def test_other_tile_uses_only_white_rejects_with_allowed_reasons() -> None:
    specs = candidate_negatives(_mixed(OTHER), OTHER, params(), tile_rng(0, OTHER))
    assert sorted(s.reason for s in specs) == ["near_axis", "near_axis", "too_small_white"]
    assert all(s.source == SOURCE_TILE and not s.is_example and not s.survives for s in specs)


def test_other_tile_cap_is_stratified_by_reason() -> None:
    specs = candidate_negatives(_mixed(OTHER), OTHER, params(max_per_other_tile=2), tile_rng(0, OTHER))
    assert Counter(s.reason for s in specs) == Counter({"near_axis": 1, "too_small_white": 1})


def test_candidate_negatives_reject_foreign_tile() -> None:
    with pytest.raises(NegativeError, match="tile"):
        candidate_negatives(_mixed(OTHER), EXAMPLE, params(), tile_rng(0, EXAMPLE))


def _windows_valid(specs, valid: np.ndarray, p: NegativeParams) -> bool:
    for s in specs:
        w = crop_window(s.box, p.context, p.min_px, p.extent_px)
        if not valid[w.y0 : w.y0 + w.side, w.x0 : w.x0 + w.side].all():
            return False
    return True


def test_random_negatives_stay_in_valid_area_and_are_deterministic() -> None:
    valid = np.zeros((EXTENT, EXTENT), np.uint8)
    valid[:, : EXTENT // 2] = 1
    p = params()
    a = random_negatives(valid, OTHER, 6, (), p, tile_rng(1, OTHER))
    b = random_negatives(valid, OTHER, 6, (), p, tile_rng(1, OTHER))
    assert a == b and len(a) == 6
    assert _windows_valid(a, valid, p)
    assert all(s.source == SOURCE_RANDOM and s.reason == "random" for s in a)
    sides = [max(s.box.width, s.box.height) for s in a]
    assert all(5.0 <= v <= 40.0 for v in sides)


def test_random_negatives_avoid_given_boxes_and_give_up_when_impossible() -> None:
    valid = np.ones((EXTENT, EXTENT), np.uint8)
    avoid = (make_candidate(tile_id=OTHER, centroid=(256.0, 256.0), half=250.0).box,)
    assert random_negatives(valid, OTHER, 3, avoid, params(max_random_attempts=50), tile_rng(0, OTHER)) == ()
    assert (
        random_negatives(
            np.zeros_like(valid), OTHER, 3, (), params(max_random_attempts=20), tile_rng(0, OTHER)
        )
        == ()
    )


def test_plan_tile_other_avoids_kept_candidates() -> None:
    valid = np.ones((EXTENT, EXTENT), np.uint8)
    kept = [replace(make_candidate(tile_id=OTHER, centroid=(256.0, 256.0), half=250.0), reject_reason=None)]
    specs = plan_tile(OTHER, kept, valid, params())
    assert [s for s in specs if s.source == SOURCE_RANDOM] == []
    example_specs = plan_tile(EXAMPLE, [], valid, params())
    assert len(example_specs) == 5


def _fake_sources(tiles: dict[str, list[Candidate]]) -> TileSources:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(EXTENT, EXTENT, 3), dtype=np.uint8)
    return TileSources(
        load_candidates=lambda t: tuple(tiles[t]),
        load_rgb=lambda t: rgb,
        load_valid=lambda t: np.ones((EXTENT, EXTENT), np.uint8),
    )


def test_build_negative_store(tmp_path: Path) -> None:
    tiles = {EXAMPLE: _mixed(EXAMPLE), OTHER: _mixed(OTHER)}
    out = build_negative_store(tuple(tiles), _fake_sources(tiles), params(), tmp_path / "neg")
    store = read_crop_store(out)
    n_expected = 6 + 5 + 3 + 2
    assert store.crops.shape == (n_expected, 32, 32, 3)
    assert {r.group for r in store.records} == {EXAMPLE, OTHER}
    assert all(r.label == 0 for r in store.records)
    assert store.meta["kind"] == "negatives" and store.meta["counts"][SOURCE_EXAMPLE] == 6
    ex = [r for r in store.records if r.meta["is_example"]]
    assert len(ex) == 11 and sum(r.meta["survives"] for r in ex) == 1


def test_build_negative_store_needs_tiles(tmp_path: Path) -> None:
    with pytest.raises(NegativeError, match="no tiles"):
        build_negative_store((), _fake_sources({}), params(), tmp_path / "neg")


def test_run_tile_ids(tmp_path: Path) -> None:
    layers = tmp_path / "layers"
    layers.mkdir()
    pd.DataFrame({"tile_id": [OTHER, EXAMPLE, OTHER]}).to_parquet(layers / "tile_status.parquet")
    assert run_tile_ids(tmp_path) == (OTHER, EXAMPLE)  # sorted, unique
    with pytest.raises(NegativeError, match="tile_status"):
        run_tile_ids(tmp_path / "missing")


def test_params_from_config() -> None:
    from vineyard.config import load_config

    cfg = load_config()
    p = NegativeParams.from_config(cfg)
    assert p.example_tiles == tuple(cfg.eval.example_tiles)
    assert p.random_per_example_tile == cfg.waste.probe.negatives.random_per_tile
    assert p.out_px == 224 and p.extent_px == 2048 and p.min_px == 64


def test_colour_class_of_vivid_candidates_is_skipped_on_other_tiles() -> None:
    vivid = [replace(cand(OTHER, 100, RejectReason.NEAR_AXIS, white=False), colour_class=ColourClass.VIVID)]
    assert candidate_negatives(vivid, OTHER, params(), tile_rng(0, OTHER)) == ()


def test_work_sources_read_the_work_dir_layout(tmp_path: Path) -> None:
    from tests.conftest import write_geotiff
    from vineyard.geo.raster import write_mask_png
    from vineyard.perception.waste.negatives import work_sources
    from vineyard.perception.waste.types import candidate_to_record

    cands = _mixed(OTHER)
    (tmp_path / "cache" / "waste").mkdir(parents=True)
    pd.DataFrame([candidate_to_record(c) for c in cands]).to_parquet(
        tmp_path / "cache" / "waste" / f"{OTHER}.parquet"
    )
    write_mask_png(tmp_path / "cache" / "valid" / f"{OTHER}.png", np.ones((EXTENT, EXTENT), bool))
    write_geotiff(tmp_path / "tiles" / f"{OTHER}.tif", size=EXTENT)
    src = work_sources(tmp_path)
    assert src.load_candidates(OTHER) == tuple(cands)
    assert src.load_rgb(OTHER).shape == (EXTENT, EXTENT, 3)
    assert src.load_valid(OTHER).all()
    with pytest.raises(NegativeError, match="cache missing"):
        src.load_candidates(EXAMPLE)


def _const_sources(
    values: dict[str, int], valid: dict[str, np.ndarray], cands: dict[str, list[Candidate]]
) -> TileSources:
    def rgb(t: str) -> np.ndarray:
        img = np.empty((EXTENT, EXTENT, 3), np.uint8)
        img[...] = values[t]
        return img

    return TileSources(
        load_candidates=lambda t: tuple(cands.get(t, [])), load_rgb=rgb, load_valid=lambda t: valid[t]
    )


def test_background_sampler_uses_free_windows_of_non_example_tiles() -> None:
    from vineyard.perception.waste.negatives import background_sampler

    full = np.ones((EXTENT, EXTENT), bool)
    half = full.copy()
    half[:, : EXTENT // 2] = False
    kept = [replace(make_candidate(tile_id=OTHER, centroid=(400.0, 100.0), half=20.0), reject_reason=None)]
    src = _const_sources({EXAMPLE: 10, OTHER: 200}, {EXAMPLE: full, OTHER: half}, {OTHER: kept})
    sample = background_sampler((EXAMPLE, OTHER), src, params(), n_tiles=4)
    for _ in range(20):
        w = sample(64)
        assert w.shape == (64, 64, 3) and (w == 200).all()
    again = background_sampler((EXAMPLE, OTHER), src, params(), n_tiles=4)
    np.testing.assert_array_equal(
        again(32), background_sampler((EXAMPLE, OTHER), src, params(), n_tiles=4)(32)
    )


def test_background_sampler_errors() -> None:
    from vineyard.perception.waste.negatives import background_sampler

    full = np.ones((EXTENT, EXTENT), bool)
    src = _const_sources({EXAMPLE: 10, OTHER: 200}, {EXAMPLE: full, OTHER: np.zeros_like(full)}, {})
    with pytest.raises(NegativeError, match="non-example"):
        background_sampler((EXAMPLE,), src, params())
    sample = background_sampler((OTHER,), src, params(max_random_attempts=5))
    with pytest.raises(NegativeError, match="background window"):
        sample(64)
