"""Stage `tile_prep`: valid / veg / vis masks + TileStats per tile, static tile_valid layer."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.synth import rgb_from_mask, striped_mask, with_nodata_corner
from tests.ingest.synth_data import TILE_PX, build_data_root, make_ctx, uniform_rgb
from vineyard.contracts.enums import TileStatus
from vineyard.geo.tiling import TILE_M
from vineyard.geo.vector_io import read_layer
from vineyard.perception.types import VIS_NODATA
from vineyard.pipeline.context import RunContext
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.runner import run_stages
from vineyard.pipeline.stages.tile_prep import tile_prep_outputs
from vineyard.pipeline.tile_cache import (
    load_tile_stats,
    load_valid_mask,
    load_veg_mask,
    load_vis,
    tile_prep_key,
)
from vineyard.pipeline.tile_index import read_tile_index

STRIPED = "siret3_r006_c004"
BLACK = "siret3_r018_c010"
CORNER_PX = 512
STRIPE_SPACING_PX = 100.0
STRIPE_WIDTH_PX = 30.0


@pytest.fixture
def synth_root(tmp_path: Path, data_root: Path, make_geotiff: Callable[..., Path]) -> Path:
    stripes = striped_mask((TILE_PX, TILE_PX), angle_deg=90.0, spacing_px=STRIPE_SPACING_PX,
                           width_px=STRIPE_WIDTH_PX)
    striped = with_nodata_corner(rgb_from_mask(stripes), CORNER_PX, "tl")
    black = uniform_rgb((0, 0, 0))
    tifs = [make_geotiff(tmp_path / "src" / f"{STRIPED}.tif", rgb=striped, compress="DEFLATE"),
            make_geotiff(tmp_path / "src" / f"{BLACK}.tif", rgb=black, compress="DEFLATE")]
    return build_data_root(tmp_path / "data", data_root, tifs)


def _ctx(root: Path, tmp_path: Path, **kwargs: object) -> RunContext:
    return make_ctx(root, tmp_path / "work", 2, **kwargs)


def test_stage_spec_and_no_torch() -> None:
    spec = load_stage("tile_prep")
    assert (spec.name, spec.scope) == ("tile_prep", "tile")
    assert "veg" in spec.cfg_keys and "nodata" in spec.cfg_keys
    assert "torch" not in sys.modules


def test_tile_prep_masks_stats_and_tile_valid(synth_root: Path, tmp_path: Path) -> None:
    ctx = _ctx(synth_root, tmp_path)
    report = run_stages(ctx, ("ingest", "tile_prep"))
    assert report.exit_code == 0
    prep = report.results[1]
    assert (prep.n_items, prep.n_failed) == (2, 0)
    cache = ctx.paths.cache_dir

    stats = load_tile_stats(cache, STRIPED)
    corner_frac = CORNER_PX**2 / TILE_PX**2
    assert stats.status == TileStatus.OK
    assert stats.nodata_frac == pytest.approx(corner_frac, abs=0.01)
    assert stats.valid_frac == pytest.approx(1 - corner_frac, abs=0.01)
    assert stats.veg_frac == pytest.approx(STRIPE_WIDTH_PX / STRIPE_SPACING_PX, abs=0.02)
    valid = load_valid_mask(cache, STRIPED)
    assert not valid[:CORNER_PX, :CORNER_PX].any() and valid[-1, -1]
    veg = load_veg_mask(cache, STRIPED)
    assert veg.shape == (TILE_PX, TILE_PX) and not veg[:CORNER_PX, :CORNER_PX].any()
    vis = load_vis(cache, STRIPED)
    assert (vis[:CORNER_PX, :CORNER_PX] == VIS_NODATA).all()
    assert len(tile_prep_key(cache, STRIPED)) == 40

    black = load_tile_stats(cache, BLACK)
    assert black.status == TileStatus.EMPTY_NODATA and black.valid_frac < 0.02

    tv = read_layer(ctx.paths.tile_valid, "tile_valid").set_index("tile_id")
    assert list(tv.index) == [STRIPED, BLACK] or sorted(tv.index) == sorted([STRIPED, BLACK])
    assert tv.loc[STRIPED, "valid_frac"] == pytest.approx(stats.valid_frac, abs=1e-6)
    expected_area = TILE_M**2 * (1 - corner_frac)
    assert tv.geometry.loc[STRIPED].area == pytest.approx(expected_area, rel=0.01)
    assert tv.geometry.loc[BLACK].is_empty or tv.geometry.loc[BLACK].area < 0.02 * TILE_M**2

    index = read_tile_index(ctx).set_index("tile_id")
    assert index.loc[STRIPED, "nodata_frac"] == pytest.approx(stats.nodata_frac, abs=1e-6)
    assert index.loc[STRIPED, "valid_area_m2"] == pytest.approx(expected_area, rel=0.01)


def test_rerun_is_cached_and_key_follows_veg_threshold(synth_root: Path, tmp_path: Path) -> None:
    ctx = _ctx(synth_root, tmp_path)
    run_stages(ctx, ("ingest", "tile_prep"))
    key = tile_prep_key(ctx.paths.cache_dir, STRIPED)
    again = run_stages(_ctx(synth_root, tmp_path, run_id="again"), ("tile_prep",))
    assert again.results[0].n_cached == 2
    changed = _ctx(synth_root, tmp_path, run_id="thr", overrides=["veg.threshold=6.0"])
    result = run_stages(changed, ("tile_prep",)).results[0]
    assert result.n_cached == 0
    assert tile_prep_key(ctx.paths.cache_dir, STRIPED) != key


def test_tile_filter_keeps_other_tiles_in_tile_valid(synth_root: Path, tmp_path: Path) -> None:
    run_stages(_ctx(synth_root, tmp_path), ("ingest", "tile_prep"))
    subset = _ctx(synth_root, tmp_path, run_id="subset", tiles=(STRIPED,), force=("tile_prep",))
    result = run_stages(subset, ("tile_prep",)).results[0]
    assert (result.n_items, result.n_cached) == (1, 0)
    tv = read_layer(subset.paths.tile_valid, "tile_valid")
    assert sorted(tv["tile_id"]) == sorted([STRIPED, BLACK])


def test_missing_tif_fails_only_that_tile(synth_root: Path, tmp_path: Path) -> None:
    ctx = _ctx(synth_root, tmp_path, allow_failures=True)
    run_stages(ctx, ("ingest",))
    (ctx.paths.tiles_dir / f"{BLACK}.tif").unlink()
    result = run_stages(_ctx(synth_root, tmp_path, run_id="missing"), ("tile_prep",))
    assert result.exit_code == 1
    assert result.results[0].failed == (BLACK,)


def test_outputs_follow_tile_cache_layout(tmp_path: Path) -> None:
    from vineyard.pipeline.tile_cache import stats_path, valid_mask_path, veg_mask_path, vis_path

    outs = tile_prep_outputs(tmp_path, STRIPED)
    assert outs["valid"] == valid_mask_path(tmp_path, STRIPED)
    assert outs["veg"] == veg_mask_path(tmp_path, STRIPED)
    assert outs["vis"] == vis_path(tmp_path, STRIPED)
    assert outs["stats"] == stats_path(tmp_path, STRIPED)
    assert np.all([p.parent.parent == tmp_path for p in outs.values()])


@pytest.mark.examples
def test_example_tile_through_the_worker(tmp_path: Path, data_root: Path, example_tif: Callable[[str], Path]
                                         ) -> None:
    root = build_data_root(tmp_path / "data", data_root, [example_tif("siret3_r006_c004")], n_zips=1)
    ctx = make_ctx(root, tmp_path / "work", 1, workers=2)
    report = run_stages(ctx, ("ingest", "tile_prep"))
    assert report.exit_code == 0
    stats = load_tile_stats(ctx.paths.cache_dir, "siret3_r006_c004")
    assert stats.status == TileStatus.OK and 0.05 < stats.veg_frac < 0.6
