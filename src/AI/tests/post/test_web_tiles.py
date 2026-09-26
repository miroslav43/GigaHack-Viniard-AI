"""Web bundle tile layer: tiles.geojson footprints / status / counts, tile_review.csv, vegetation masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from tests.post.factories import (
    build_annset,
    canopy_record,
    interrow_piece_record,
    row_piece_record,
    waste_record,
)
from tests.post.web_oracle import check_tiles
from vineyard.config import load_config
from vineyard.contracts.ids import tile_grid_ids
from vineyard.errors import ConfigError, SchemaError, StageError
from vineyard.geo.raster import read_mask_png, write_mask_png
from vineyard.perception.types import TileStats, TileStatus
from vineyard.pipeline.stages.web_bundle import read_tile_status
from vineyard.pipeline.tile_cache import veg_mask_path, write_tile_stats
from vineyard.web.bundle import WebInputs, build_web_bundle, bundle_dir, web_params
from vineyard.web.masks_export import MASK_SOURCE, downsample_mask, mask_png_bytes
from vineyard.web.tile_review import TileReview, read_tile_review
from vineyard.web.tiles_export import (
    TileFacts,
    facts_from_status,
    merge_facts,
    tile_counts,
    tile_features,
    tile_footprint,
)

T_A: Final = "siret3_r006_c004"
T_B: Final = "siret3_r021_c012"
T_C: Final = "siret3_r018_c011"
GEN: Final = "2026-09-26T03:10:00+03:00"
HEADER: Final = "tile_id,status,note\n"


def _annset() -> Any:
    x0, y0 = 629196.8 + 5.0, 5220915.2 - 5.0  # inside T_A
    return build_annset(
        canopies=[canopy_record(f"{T_A}:C0001", T_A, "V01", box(x0, y0, x0 + 2, y0 + 1), row_id="V01-R001")],
        row_pieces=[row_piece_record("V01-R001", "V01", T_A, LineString([(x0, y0 + 0.5), (x0 + 10, y0 + 0.5)])),
                    row_piece_record("V01-R002", "V01", T_A, LineString([(x0, y0 + 3.0), (x0 + 10, y0 + 3.0)]))],
        interrow_pieces=[interrow_piece_record(f"{T_A}:I001", T_A, "V01", box(x0, y0 + 1, x0 + 10, y0 + 2.5))],
        waste=[waste_record(f"{T_C}:W001", T_C, (629560.0, 5220290.0), 1.0)])


# ------------------------------------------------------------------ footprints, status, counts


def test_footprint_is_the_ccw_grid_square() -> None:
    poly = tile_footprint(T_A)
    assert poly.bounds == pytest.approx((629196.8, 5220915.2 - 51.2, 629196.8 + 51.2, 5220915.2))
    assert poly.area == pytest.approx(51.2 * 51.2)
    assert poly.exterior.is_ccw


def test_counts_per_tile_from_the_annset() -> None:
    counts = tile_counts(_annset())
    assert counts[T_A] == {"n_rows": 2, "n_canopies": 1, "n_interrows": 1, "n_waste": 0}
    assert counts[T_C] == {"n_rows": 0, "n_canopies": 0, "n_interrows": 0, "n_waste": 1}


def test_tile_features_status_review_and_facts() -> None:
    review = {T_B: TileReview(status="missed", note="vine rows at the top")}
    facts = {T_A: TileFacts(veg_frac=0.1234, nodata_frac=0.5, review_priority=2)}
    frame = tile_features((T_A, T_B, T_C), tile_counts(_annset()), facts, review, frozenset({T_A}))
    rows = {r["tile"]: r for r in frame.drop(columns="geometry").to_dict("records")}
    assert list(rows) == [T_A, T_B, T_C]
    assert rows[T_A] =={"tile": T_A, "status": "vineyard", "n_rows": 2, "n_canopies": 1, "n_interrows": 1,
                              "n_waste": 0, "veg_frac": 0.1234, "nodata_frac": 0.5, "review_priority": 2,
                              "review_note": None, "review_status": None, "has_mask": True}
    assert rows[T_B]["status"] == "no_vineyard" and rows[T_B]["review_status"] == "missed"
    assert rows[T_B]["review_note"] == "vine rows at the top" and rows[T_B]["veg_frac"] is None
    assert rows[T_C]["status"] == "no_vineyard" and rows[T_C]["n_waste"] == 1 and rows[T_C]["has_mask"] is False
    assert type(rows[T_A]["n_rows"]) is int and type(rows[T_A]["review_priority"]) is int


def test_facts_from_the_model_tile_status() -> None:
    status = pd.DataFrame({"tile_id": [T_A, T_B], "veg_frac": np.array([0.25, np.nan], dtype=np.float32),
                           "review_priority": np.array([1, 3], dtype=np.int8)})
    facts = facts_from_status(status)
    assert facts[T_A] == TileFacts(veg_frac=0.25, nodata_frac=None, review_priority=1)
    assert facts[T_B] == TileFacts(veg_frac=None, nodata_frac=None, review_priority=3)
    assert facts_from_status(None) == {}


# ------------------------------------------------------------------ tile_review.csv


def _review(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "tile_review.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_review_file_reads_status_and_note(tmp_path: Path) -> None:
    path = _review(tmp_path, f"{T_A},missed,rows on the left\n{T_B},verify,\"garden, maybe\"\n")
    assert read_tile_review(path) == {T_A: TileReview("missed", "rows on the left"),
                                      T_B: TileReview("verify", "garden, maybe")}
    assert read_tile_review(None) == {}


@pytest.mark.parametrize(("body", "line", "fragment"), [
    ("siret3_r000_c000,missed,x\n", 2, "unknown tile"),
    (f"{T_A},missed?,x\n", 2, "status"),
    (f"{T_A},missed,x\n{T_A},partial,y\n", 3, "duplicate"),
    (f"{T_A},missed\n", 2, "3 columns"),
])
def test_review_file_errors_name_file_and_line(tmp_path: Path, body: str, line: int, fragment: str) -> None:
    path = _review(tmp_path, body)
    with pytest.raises(ConfigError, match=fragment) as err:
        read_tile_review(path)
    assert err.value.context["line"] == line and err.value.context["path"] == str(path)


def test_review_file_header_and_missing_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("tile,status,note\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="header"):
        read_tile_review(bad)
    with pytest.raises(ConfigError, match="not found"):
        read_tile_review(tmp_path / "absent.csv")


def test_committed_review_file_is_valid() -> None:
    cfg = load_config(environ={})
    review = read_tile_review(cfg.web.tile_review)
    assert len(review) >= 30 and {r.status for r in review.values()} <= {"missed", "partial", "verify"}


# ------------------------------------------------------------------ masks


def test_downsample_is_any_vegetation_in_the_block() -> None:
    src = np.zeros((8, 8), dtype=bool)
    src[0, 1] = True  # block (0, 0)
    src[7, 7] = True  # block (3, 3)
    out = downsample_mask(src, 4)
    expected = np.zeros((4, 4), dtype=bool)
    expected[0, 0] = expected[3, 3] = True
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(downsample_mask(src, 8), src)


def test_downsample_rejects_non_integer_factors() -> None:
    with pytest.raises(SchemaError):
        downsample_mask(np.zeros((10, 10), dtype=bool), 4)
    with pytest.raises(SchemaError):
        downsample_mask(np.zeros((4, 4), dtype=bool), 8)


def test_mask_png_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    mask = rng.random((64, 64)) > 0.7
    path = tmp_path / "m.png"
    path.write_bytes(mask_png_bytes(mask))
    np.testing.assert_array_equal(read_mask_png(path), mask)


# ------------------------------------------------------------------ bundle


def _cache(tmp_path: Path, px: int) -> Path:
    cache = tmp_path / "cache"
    mask = np.zeros((px, px), dtype=bool)
    mask[:2, :2] = True
    path = veg_mask_path(cache, T_A)
    path.parent.mkdir(parents=True)
    write_mask_png(path, mask)
    write_tile_stats(cache, TileStats(tile_id=T_A, valid_frac=0.75, nodata_frac=0.25, veg_frac=0.4,
                                      status=TileStatus.OK, veg_threshold=4.0, veg_method="lab_a"))
    return cache


def test_bundle_writes_tiles_masks_and_manifest_extras(tmp_path: Path) -> None:
    cfg = load_config(overrides=("web.mask_px=4",), environ={})
    params = web_params(cfg)
    (tmp_path / "cache" / "stats").mkdir(parents=True)
    inputs = WebInputs(annset=_annset(), cache_dir=_cache(tmp_path, 8),
                       tile_review={T_B: TileReview("partial", "left half")})
    out = bundle_dir(tmp_path / "web", "siret3")
    stale = out / "masks" / "siret3_r099_c099.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"x")
    result = build_web_bundle(inputs, out, params, generated_at=GEN, pipeline_version="abc", run_id="r")
    tiles = check_tiles(out)
    assert len(tiles) == len(tile_grid_ids()) == 311
    by_tile = {t["tile"]: t for t in tiles}
    assert by_tile[T_A]["has_mask"] is True and by_tile[T_A]["veg_frac"] == 0.4
    assert by_tile[T_A]["nodata_frac"] == 0.25 and by_tile[T_A]["status"] == "vineyard"
    assert by_tile[T_B]["review_status"] == "partial" and by_tile[T_B]["has_mask"] is False
    assert sorted(p.name for p in (out / "masks").iterdir()) == [f"{T_A}.png"]
    mask = read_mask_png(out / "masks" / f"{T_A}.png")
    assert mask.shape == (4, 4) and mask[0, 0] and mask.sum() == 1
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["tiles"] == 311
    assert manifest["masks"] == {"dir": "masks", "px": 4, "n": 1, "source": MASK_SOURCE}
    assert result.masks == (out / "masks" / f"{T_A}.png",)
    assert stale in result.removed


def test_bundle_without_cache_has_no_masks(tmp_path: Path) -> None:
    params = web_params(load_config(environ={}))
    out = tmp_path / "b"
    build_web_bundle(WebInputs(annset=_annset()), out, params, generated_at=GEN, pipeline_version="abc", run_id="r")
    tiles = check_tiles(out)
    assert not any(t["has_mask"] for t in tiles) and all(t["veg_frac"] is None for t in tiles)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["masks"]["n"] == 0 and manifest["masks"]["px"] == 1024


def test_stats_cache_wins_over_tile_status_field_by_field() -> None:
    primary = {T_A: TileFacts(veg_frac=0.4, nodata_frac=0.25)}
    fallback = {T_A: TileFacts(veg_frac=0.3, review_priority=2), T_B: TileFacts(review_priority=1)}
    assert merge_facts(primary, fallback) == {T_A: TileFacts(0.4, 0.25, 2), T_B: TileFacts(review_priority=1)}


def test_read_tile_status_keeps_the_used_columns(tmp_path: Path) -> None:
    layers = tmp_path / "layers"
    layers.mkdir()
    assert read_tile_status(tmp_path) is None
    pd.DataFrame({"tile_id": [T_A], "veg_frac": [0.5], "review_priority": [1], "issues": ["x"]}).to_parquet(
        layers / "tile_status.parquet")
    assert list(read_tile_status(tmp_path).columns) == ["tile_id", "veg_frac", "review_priority"]
    pd.DataFrame({"tile_id": [T_A]}).to_parquet(layers / "tile_status.parquet")
    with pytest.raises(StageError, match="lacks columns"):
        read_tile_status(tmp_path)
