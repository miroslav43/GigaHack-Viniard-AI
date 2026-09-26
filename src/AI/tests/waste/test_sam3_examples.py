"""Real SAM 3 on the example tile siret3_r006_c004 (slow + examples; skipped without weights or data).

Pins two measured facts: the 3 filter survivors of the example tile score 0 (no false positive), and
3 tube negative boxes suppress the "trash" detections SAM 3 otherwise makes on the other white tubes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vineyard.perception.waste.crops import box_to_crop, sam_window
from vineyard.perception.waste.sam3_adapter import (
    NEGATIVE_REASONS,
    Sam3Params,
    Sam3Verifier,
    has_transformers_weights,
    list_files,
    load_sam3,
    mask_overlap,
    select_negative_boxes,
)
from vineyard.perception.waste.types import Candidate, candidate_from_record

pytestmark = [pytest.mark.slow, pytest.mark.examples]

TILE = "siret3_r006_c004"
TUBE_CROP_KEY = f"{TILE}@0874_1084"  # 8 tubes in its 512 px window; 5 "trash" hits without negatives
TILE_PX = 2048


@pytest.fixture(scope="module")
def verifier(cfg) -> Sam3Verifier:
    p = Sam3Params.from_config(cfg.waste.sam3, cfg.paths.models_dir)
    local = [s for s in p.sources if Path(s).is_dir() and has_transformers_weights(list_files(s))]
    if not local:
        pytest.skip("local SAM 3 transformers weights not present")
    v, status = load_sam3(replace(p, sources=(local[0],)))
    assert v is not None, status
    return v


@pytest.fixture(scope="module")
def tile(cfg) -> tuple[np.ndarray, tuple[Candidate, ...]]:
    work = Path(cfg.paths.work_dir)
    tif, cache = work / "tiles" / f"{TILE}.tif", work / "cache" / "waste" / f"{TILE}.parquet"
    if not (tif.is_file() and cache.is_file()):
        pytest.skip("example tile or its waste candidate cache not present in the work dir")
    from vineyard.geo.raster import read_tile

    cands = tuple(candidate_from_record(r) for r in pd.read_parquet(cache).to_dict("records"))
    return read_tile(tif), cands


def _window(rgb: np.ndarray, c: Candidate, crop_px: int):
    w = sam_window(c.centroid_px, crop_px, TILE_PX)
    return w, np.ascontiguousarray(rgb[w.y0 : w.y0 + w.side, w.x0 : w.x0 + w.side])


def test_example_survivors_score_zero(verifier: Sam3Verifier, tile) -> None:
    rgb, cands = tile
    for c in (c for c in cands if not c.rejected):
        w, crop = _window(rgb, c, verifier.params.crop_px)
        negs = select_negative_boxes(c, cands, w, verifier.params.max_negative_boxes)
        assert verifier.verify(crop, box_to_crop(c.box.as_tuple(), w), negs).score == 0.0, c.cand_key


def test_tube_negatives_suppress_tube_detections(verifier: Sam3Verifier, tile) -> None:
    rgb, cands = tile
    tubes = [c for c in cands if c.is_white and c.reject_reason in NEGATIVE_REASONS]
    centre = next(c for c in tubes if c.cand_key == TUBE_CROP_KEY)
    w, crop = _window(rgb, centre, verifier.params.crop_px)
    negs = select_negative_boxes(centre, tubes, w, verifier.params.max_negative_boxes)
    in_window = [box_to_crop(t.box.as_tuple(), w) for t in tubes if w.x0 <= t.box.centre[0] < w.x0 + w.side]
    held_out = [b for b in in_window if b not in set(negs) and b[1] >= 0 and b[3] <= w.side]

    def hits(negatives) -> int:
        out = verifier.backend.detect(crop, verifier.params.text_prompts, negatives)
        return sum(
            any(mask_overlap(i.mask, b) >= verifier.params.min_overlap_frac for b in held_out)
            for insts in out.per_prompt
            for i in insts
        )

    assert len(negs) == verifier.params.max_negative_boxes
    assert hits(()) > 0  # SAM 3 alone calls white vine tubes "trash"
    assert hits(negs) == 0
