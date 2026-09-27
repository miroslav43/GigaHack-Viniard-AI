"""vineyard.demo.export / status: web files of a demo job (contract: src/Web/docs/design/2026-09-27-analiza-tif-spec.md)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from vineyard.demo.export import MaskColour, mask_rgba_png, model_version, review_waste, tile_corners
from vineyard.demo.status import StageProgress, write_status
from vineyard.geo.tiling import CRS_EPSG, tile_ref
from vineyard.logging_setup import get_logger, log_event


def test_mask_colour_from_hex() -> None:
    assert MaskColour.from_hex("#D946EF", 0.55) == MaskColour((0xD9, 0x46, 0xEF), 0.55)
    with pytest.raises(ValueError):
        MaskColour.from_hex("purple", 0.5)
    with pytest.raises(ValueError):
        MaskColour.from_hex("#D946EF", 1.5)


def test_mask_png_is_coloured_on_vegetation_and_transparent_elsewhere() -> None:
    mask = np.zeros((2048, 2048), bool)
    mask[:2, :2] = True  # one web pixel (max-pool 2 x 2)
    img = cv2.imdecode(np.frombuffer(mask_rgba_png(mask, MaskColour((255, 0, 0), 0.5)), np.uint8),
                       cv2.IMREAD_UNCHANGED)
    assert img.shape == (1024, 1024, 4)
    assert tuple(img[0, 0]) == (0, 0, 255, 128)  # BGRA
    assert img[1:, 1:, 3].max() == 0


def test_tile_corners_follow_maplibre_order() -> None:
    tl, tr, br, bl = tile_corners(tile_ref("siret3_r021_c012"))
    assert tl[0] < tr[0] and bl[0] < br[0]  # west to east
    assert tl[1] > bl[1] and tr[1] > br[1]  # north to south


def _candidates(review: list[bool | None], rank: list[float]) -> gpd.GeoDataFrame:
    n = len(review)
    return gpd.GeoDataFrame(
        {"waste_id": [f"W{i}" for i in range(n)], "rank_score": rank, "probe_p": rank, "area_m2": [1.0] * n,
         "category": ["unknown"] * n, "review": review, "model_version": ["pipe;waste=probe+clip@1"] * n},
        geometry=[box(i, 0, i + 1, 1) for i in range(n)], crs=CRS_EPSG,
    )


def test_review_waste_keeps_flagged_candidates_best_first() -> None:
    got = review_waste(_candidates([True, False, None, True], [0.4, 0.9, 0.8, 0.7]))
    assert list(got["waste_id"]) == ["W3", "W0"]
    assert "review" not in got.columns


def test_review_waste_without_candidates_is_empty() -> None:
    assert len(review_waste(None)) == 0


def test_model_version_prefers_the_waste_stage() -> None:
    assert model_version({"model_version": "run"}, _candidates([True], [0.5])) == "pipe;waste=probe+clip@1"
    assert model_version({"model_version": "run"}, None) == "run"


def test_write_status_error(tmp_path: Path) -> None:
    write_status(tmp_path, "error", n_stages=11, error=("wrong_crs", "bad"))
    doc = json.loads((tmp_path / "status.json").read_text())
    assert doc["state"] == "error" and doc["error"] == {"code": "wrong_crs", "message": "bad"}


def test_stage_progress_follows_stage_start(tmp_path: Path) -> None:
    logger = get_logger("pipeline.test_demo")
    logger.setLevel(logging.INFO)
    with StageProgress(tmp_path, ("ingest", "canopy", "assemble")):
        log_event(logger, "stage.start", stage="canopy")
        log_event(logger, "stage.end", stage="assemble")  # not a start: ignored
    doc = json.loads((tmp_path / "status.json").read_text())
    assert (doc["state"], doc["stage"], doc["stage_index"], doc["n_stages"]) == ("running", "canopy", 2, 3)
