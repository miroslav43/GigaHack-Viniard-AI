from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import box

from vineyard.config import load_config
from vineyard.nn.probs import prob_png_path, write_prob_png
from vineyard.perception.row_seeds import RowSeed
from vineyard.perception.rows_detect import DetectParams
from vineyard.pipeline.stages._rows_seeded_io import SeededJob, nn_in_seed_zones, usable_job_seeds

TILE = "siret3_r021_c012"


def job(tmp_path: Path, kind: str, nn_version: str = "v1") -> SeededJob:
    cfg = load_config(environ={})
    seed = RowSeed(TILE, "yes", "medium", box(0, 0, 1024, 2048), 127.0, 2.7, kind, "", 2)
    return SeededJob(cache_dir=tmp_path, tile_id=TILE, seeds=(seed,), clip_px=box(0, 0, 2048, 2048), forbidden=(),
                     existing=(), params=DetectParams.from_config(cfg),
                     s=cfg.rows_seeded.model_copy(update={"nn_kinds": ("cadastre_auto",)}), nn_version=nn_version)


def test_nn_mask_replaces_veg_inside_the_seed_only(tmp_path: Path) -> None:
    prob = np.zeros((1024, 1024), dtype=np.float32)
    prob[:, :256] = 1.0          # NN: vines in the left quarter
    write_prob_png(prob_png_path(tmp_path, "v1", "canopy_prob", TILE), prob)
    veg = np.ones((2048, 2048), dtype=bool)   # veg mask: everything green (grass too)
    out = nn_in_seed_zones(veg, job(tmp_path, "cadastre_auto"))
    assert out[:, :500].all()            # inside the seed, NN says vine
    assert not out[:, 600:1000].any()    # inside the seed, NN says no vine: grass removed
    assert out[:, 1100:].all()           # outside the seed: the veg mask is untouched


def test_other_kinds_or_no_raster_keep_the_veg_mask(tmp_path: Path) -> None:
    veg = np.zeros((2048, 2048), dtype=bool)
    assert nn_in_seed_zones(veg, job(tmp_path, "manual_review")) is veg
    assert nn_in_seed_zones(veg, job(tmp_path, "cadastre_auto")) is veg  # no probability raster
    assert nn_in_seed_zones(veg, job(tmp_path, "cadastre_auto", nn_version="")) is veg


def test_nn_seeds_are_skipped_without_a_raster(tmp_path: Path) -> None:
    assert usable_job_seeds(job(tmp_path, "cadastre_auto")) == ()          # no raster: never on the veg mask
    assert len(usable_job_seeds(job(tmp_path, "manual_review"))) == 1     # other kinds are untouched
    prob = np.zeros((1024, 1024), dtype=np.float32)
    write_prob_png(prob_png_path(tmp_path, "v1", "canopy_prob", TILE), prob)
    assert len(usable_job_seeds(job(tmp_path, "cadastre_auto"))) == 1     # raster present: kept
