"""The 2 example tiles hold 0 waste and hundreds of white tubes: nothing may be exported and only a
handful of candidates may reach the review list (reference axes from annotations.xml)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.helpers.examples import load_examples, row_lines_utm
from vineyard.config import AppConfig
from vineyard.geo.raster import read_tile, valid_mask
from vineyard.geo.tiling import GSD_M, tile_ref
from vineyard.perception.waste.decide import DecideParams
from vineyard.perception.waste.filters import reject_counts
from vineyard.perception.waste.verify import load_verifier, rank_and_select
from vineyard.pipeline.stages.waste import filter_tile

EXAMPLES = ("siret3_r021_c012", "siret3_r006_c004")
MAX_SURVIVORS_PER_TILE = 5  # "few": the review list of an example tile must stay tiny


@pytest.mark.examples
@pytest.mark.parametrize("tile_id", EXAMPLES)
def test_examples_have_no_auto_and_few_survivors(
    tile_id: str, cfg: AppConfig, examples_xml: bytes, example_tif: Callable[[str], Path]
) -> None:
    img = load_examples(examples_xml)[tile_id]
    rgb = read_tile(example_tif(tile_id))
    nd = cfg.nodata
    valid = valid_mask(
        rgb,
        max_rgb=nd.max_rgb,
        min_area_px=nd.min_area_m2 / GSD_M**2,
        close_px=nd.close_px,
        dilate_px=nd.dilate_px,
    )
    cands = filter_tile(rgb, valid, tile_ref(tile_id), row_lines_utm(img), None, cfg)
    counts = reject_counts(cands)
    assert len(cands) > 50  # the tubes are found ...
    assert counts.get("near_axis", 0) > 10  # ... and rejected on the axes
    assert counts.get("kept", 0) <= MAX_SURVIVORS_PER_TILE
    verifier, status = load_verifier(cfg.waste, GSD_M)
    ranked = rank_and_select(
        cands, verifier, status, DecideParams.from_config(cfg.waste.decide), cfg.waste.nms_iou
    )
    assert not any(d.auto for d in ranked.decisions.values())
