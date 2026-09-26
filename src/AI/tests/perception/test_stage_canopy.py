"""Stages `canopy` and `interrow` on a synthetic one-tile work dir (inline, workers=1)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString

from tests.conftest import write_geotiff
from vineyard.config import load_config
from vineyard.contracts.enums import Source, TileStatus
from vineyard.errors import StageError
from vineyard.geo.raster import write_mask_png, write_u8_png
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_box, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.perception.types import VIS_OK, TileStats
from vineyard.pipeline.cache import read_key, write_key
from vineyard.pipeline.context import RunContext, ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages import canopy as canopy_stage
from vineyard.pipeline.stages import interrow as interrow_stage
from vineyard.pipeline.stages.canopy import geometry_digest, tile_clip
from vineyard.pipeline.tile_cache import (
    stats_path,
    valid_mask_path,
    veg_mask_path,
    vis_path,
    write_tile_stats,
)
from vineyard.pipeline.tile_index import tile_path

TILE_ID = "siret3_r021_c012"
EMPTY_TILE = "siret3_r021_c013"
TILE = tile_ref(TILE_ID)
ROW_VS = (500.0, 620.0, 740.0)
VINE_RGB = (60, 130, 40)  # ExG 160
SHADOW_RGB = (8, 18, 5)  # green hue (a* keeps it) but ExG 23 < the ExG threshold
SOIL_RGB = (150, 115, 90)  # ExG -10


def _write_tif(ctx: RunContext, tile_id: str, layers: list[tuple[np.ndarray, tuple[int, int, int]]]) -> None:
    rgb = np.empty((TILE_PX, TILE_PX, 3), dtype=np.uint8)
    rgb[...] = SOIL_RGB
    for mask, colour in layers:
        rgb[mask] = colour
    write_geotiff(tile_path(ctx, tile_id), tile_id=tile_id, size=TILE_PX, rgb=rgb, compress="DEFLATE")


def _tile_index(tile_ids: list[str]) -> gpd.GeoDataFrame:
    refs = [tile_ref(t) for t in tile_ids]
    return gpd.GeoDataFrame(
        {"tile_id": tile_ids, "file_name": [f"{t}.tif" for t in tile_ids], "grid_row": [r.grid_row for r in refs],
         "grid_col": [r.grid_col for r in refs], "x0": [r.x0 for r in refs], "y0": [r.y0 for r in refs],
         "x1": [r.bounds[2] for r in refs], "y1": [r.bounds[1] for r in refs], "gsd_m": GSD_M,
         "width_px": TILE_PX, "height_px": TILE_PX, "src_zip": "z.zip", "path": "p", "sha256": "0" * 64,
         "file_size": 1, "nodata_frac": 0.0, "valid_area_m2": 2621.44},
        geometry=[tile_box(r) for r in refs], crs=f"EPSG:{CRS_EPSG}",
    )


def _tile_prep(ctx: RunContext, tile_id: str, veg: np.ndarray) -> None:
    cache = ctx.paths.cache_dir
    for path in (veg_mask_path(cache, tile_id), valid_mask_path(cache, tile_id), vis_path(cache, tile_id)):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_mask_png(veg_mask_path(cache, tile_id), veg)
    write_mask_png(valid_mask_path(cache, tile_id), np.ones_like(veg))
    write_u8_png(vis_path(cache, tile_id), np.full(veg.shape, VIS_OK, dtype=np.uint8))
    stats = TileStats(tile_id, 1.0, 0.0, float(veg.mean()), TileStatus.OK, 4.0, "lab_a")
    stats_path(cache, tile_id).parent.mkdir(parents=True, exist_ok=True)
    write_tile_stats(cache, stats)
    write_key(stats_path(cache, tile_id), f"prep-{tile_id}")


def _rows() -> gpd.GeoDataFrame:
    lines = [LineString(px_to_utm(TILE, np.array([[-400.0, v], [1900.0, v]]))) for v in ROW_VS]
    n = len(lines)
    return gpd.GeoDataFrame(
        {"row_id": [f"V01-R{k + 1:03d}" for k in range(n)], "vineyard_id": "V01", "row_index": range(1, n + 1),
         "length_m": [g.length for g in lines], "extent_m": [g.length for g in lines], "n_pieces": 1,
         "tile_ids": TILE_ID, "angle_deg": 0.0, "spacing_prev_m": 3.0, "spacing_next_m": 3.0,
         "max_gap_m": np.nan, "n_gaps_ge5": None, "structure_any": None, "source": Source.MODEL.value,
         "run_id": "test-run", "model_version": "pipe@test", "confidence": 1.0, "qa_flags": ""},
        geometry=lines, crs=f"EPSG:{CRS_EPSG}",
    )


def _pairs(rows: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ids = list(rows["row_id"])
    conn = [LineString([rows.geometry.iloc[k].centroid, rows.geometry.iloc[k + 1].centroid]) for k in range(len(ids) - 1)]
    return gpd.GeoDataFrame(
        {"vineyard_id": "V01", "row_a": ids[:-1], "row_b": ids[1:], "k_a": range(1, len(ids)), "spacing_m": 3.0,
         "overlap_from_m": 0.0, "overlap_to_m": 60.0, "angle_diff_deg": 0.0, "source": Source.MODEL.value,
         "run_id": "test-run", "model_version": "pipe@test", "confidence": 1.0,
         "qa_flags": ["", "missing_row_suspect"]},
        geometry=conn, crs=f"EPSG:{CRS_EPSG}",
    )


@pytest.fixture
def ctx(tmp_work: Path) -> RunContext:
    cfg = load_config()
    context = new_run_context(cfg, source=Source.MODEL, run_id="test-run", workers=1)
    ensure_run_dirs(context.paths)
    tiles = [TILE_ID, EMPTY_TILE]
    write_layer(_tile_index(tiles), "tile_index", context.paths.tile_index)
    valid = gpd.GeoDataFrame({"tile_id": tiles, "valid_frac": 1.0},
                             geometry=[tile_box(tile_ref(t)) for t in tiles], crs=f"EPSG:{CRS_EPSG}")
    write_layer(valid, "tile_valid", context.paths.tile_valid)
    veg = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    for v in ROW_VS:
        for u0 in range(100, 1900, 150):
            veg[int(v) - 10 : int(v) + 10, u0 : u0 + 80] = True
    _tile_prep(context, TILE_ID, veg)
    _tile_prep(context, EMPTY_TILE, np.zeros_like(veg))
    _write_tif(context, TILE_ID, [(veg, VINE_RGB)])
    _write_tif(context, EMPTY_TILE, [])
    rows = _rows()
    write_layer(rows, "rows", context.paths.layers_dir / "rows.parquet")
    write_layer(_pairs(rows), "row_pairs", context.paths.layers_dir / "row_pairs.parquet")
    return context


def test_registry_loads_both_stages() -> None:
    assert load_stage("canopy") is canopy_stage.STAGE
    assert load_stage("interrow") is interrow_stage.STAGE


def test_canopy_stage_writes_cache_and_reuses_it(ctx: RunContext) -> None:
    res = canopy_stage.run(ctx)
    assert res.n_items == 2 and res.n_failed == 0 and res.n_cached == 0
    out = ctx.paths.tile_cache("canopy", TILE_ID, "parquet")
    can = read_layer(out, "canopies")
    assert len(can) == 3 * 12
    assert can["source"].unique().tolist() == ["model"] and can["run_id"].unique().tolist() == ["test-run"]
    stats = json.loads(ctx.paths.tile_cache("canopy", TILE_ID, "json").read_text())
    assert stats["n_canopies"] == 36 and stats["fusion_variant"] == "A" and stats["n_pieces"] == 3
    assert len(read_layer(ctx.paths.tile_cache("canopy", EMPTY_TILE, "parquet"), "canopies")) == 0
    assert gpd.read_parquet(canopy_stage.evidence_path(ctx.paths, TILE_ID)).empty
    assert gpd.read_parquet(canopy_stage.evidence_path(ctx.paths, EMPTY_TILE)).empty
    key = read_key(out)
    again = canopy_stage.run(ctx)
    assert again.n_cached == 2 and read_key(out) == key


def test_canopy_stage_exg_drops_shadow_and_keeps_gap_evidence(ctx: RunContext) -> None:
    veg = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    vine, shadow = veg.copy(), veg.copy()
    vine[490:510, 100:180] = True
    vine[493:508, 300:313] = True  # 13 x 15 px: a plant below the 0.19 m2 canopy minimum
    shadow[610:630, 100:180] = True  # row 2: a* sees a plant, ExG does not
    _tile_prep(ctx, TILE_ID, vine | shadow)
    _write_tif(ctx, TILE_ID, [(vine, VINE_RGB), (shadow, SHADOW_RGB)])
    canopy_stage.run(ctx)
    can = read_layer(ctx.paths.tile_cache("canopy", TILE_ID, "parquet"), "canopies")
    assert can["row_id"].tolist() == ["V01-R001"]
    evidence = gpd.read_parquet(canopy_stage.evidence_path(ctx.paths, TILE_ID))
    assert evidence["row_id"].tolist() == ["V01-R001"]
    assert ctx.cfg.canopy.gap_evidence_min_area_m2 <= evidence["area_m2"].iloc[0] < ctx.cfg.canopy.min_area_m2
    stats = json.loads(ctx.paths.tile_cache("canopy", TILE_ID, "json").read_text())
    assert (stats["mask_method"], stats["n_evidence"]) == ("exg", 1)
    veg_cfg = ctx.cfg.model_copy(update={"canopy": ctx.cfg.canopy.model_copy(update={"mask_method": "veg"})})
    canopy_stage.run(replace(ctx, cfg=veg_cfg))
    can_veg = read_layer(ctx.paths.tile_cache("canopy", TILE_ID, "parquet"), "canopies")
    assert can_veg["row_id"].tolist() == ["V01-R001", "V01-R002"]


def test_canopy_key_changes_only_for_touched_tiles(ctx: RunContext) -> None:
    canopy_stage.run(ctx)
    key = read_key(ctx.paths.tile_cache("canopy", TILE_ID, "parquet"))
    rows = _rows()
    moved = rows.assign(geometry=[g if k else LineString(px_to_utm(TILE, np.array([[-400.0, 505.0], [1900.0, 505.0]])))
                                  for k, g in enumerate(rows.geometry)])
    write_layer(moved, "rows", ctx.paths.layers_dir / "rows.parquet")
    res = canopy_stage.run(ctx)
    assert res.n_cached == 1
    assert read_key(ctx.paths.tile_cache("canopy", TILE_ID, "parquet")) != key


def test_interrow_stage_bands_and_pieces(ctx: RunContext) -> None:
    res = interrow_stage.run(ctx)
    assert res.n_failed == 0
    bands_path = ctx.paths.layers_dir / "interrows.parquet"
    assert res.outputs[0] == bands_path
    bands = read_layer(bands_path, "interrows")
    assert bands["interrow_id"].tolist() == ["V01-I001", "V01-I002"]
    assert bands["qa_flags"].tolist() == ["", "missing_row_suspect"]
    pieces = read_layer(ctx.paths.tile_cache("interrow", TILE_ID, "parquet"), "interrow_pieces")
    assert pieces["piece_id"].tolist() == [f"V01-I001@{TILE_ID}", f"V01-I002@{TILE_ID}"]
    assert set(pieces["interrow_cover"]) == {"bare_soil"}
    assert pieces["qa_flags"].tolist() == ["", "missing_row_suspect"]
    assert len(read_layer(ctx.paths.tile_cache("interrow", EMPTY_TILE, "parquet"), "interrow_pieces")) == 0


def test_interrow_bands_reach_the_data_boundary_where_rows_stop(ctx: RunContext) -> None:
    slanted = [LineString(px_to_utm(TILE, np.array([[0.0, v], [1900.0, v + 400.0]]))) for v in ROW_VS]
    rows = _rows().set_geometry(gpd.GeoSeries(slanted, crs=f"EPSG:{CRS_EPSG}"))
    write_layer(rows, "rows", ctx.paths.layers_dir / "rows.parquet")
    write_layer(_pairs(rows), "row_pairs", ctx.paths.layers_dir / "row_pairs.parquet")
    interrow_stage.run(ctx)
    pieces = read_layer(ctx.paths.tile_cache("interrow", TILE_ID, "parquet"), "interrow_pieces")
    west_edge = LineString(px_to_utm(TILE, np.array([[0.0, 0.0], [0.0, float(TILE_PX)]])))
    # rows stop at the western data edge: each band must cover the edge across its width (~2.4 m), not a corner
    assert [round(g.intersection(west_edge).length, 1) for g in pieces.geometry] == [2.4, 2.4]
    assert read_layer(ctx.paths.layers_dir / "rows.parquet", "rows").geometry.iloc[0].equals(slanted[0])


def test_clips_boundary_ignores_float_seams_between_tiles() -> None:
    # adjacent tile clips whose shared edge differs by 1e-10 m (px -> UTM rounding) must merge
    a = shapely.box(629145.6, 5220864.0, 629196.7999999999, 5220915.2)
    b = shapely.box(629196.8, 5220864.0, 629248.0, 5220915.2)
    assert shapely.union_all([a, b]).geom_type == "MultiPolygon"
    boundary = interrow_stage.clips_boundary([a, b])
    assert boundary.length == pytest.approx(2 * (102.4 + 51.2), abs=1e-6)
    assert not boundary.intersects(shapely.Point(629196.8, 5220890.0).buffer(0.001))
    assert interrow_stage.clips_boundary([]) is None


def test_stages_fail_loudly_without_inputs(ctx: RunContext) -> None:
    (ctx.paths.layers_dir / "rows.parquet").unlink()
    with pytest.raises(StageError, match="rows layer missing"):
        canopy_stage.run(ctx)
    with pytest.raises(StageError, match="input layer missing"):
        interrow_stage.run(ctx)
    ctx.paths.tile_valid.unlink()
    with pytest.raises(StageError, match="tile_valid"):
        interrow_stage.run(ctx)


def test_helpers_digest_and_clip(ctx: RunContext) -> None:
    rows = _rows()
    assert geometry_digest(rows, ("row_id",)) == geometry_digest(rows.copy(), ("row_id",))
    assert geometry_digest(rows, ("row_id",)) != geometry_digest(rows.iloc[:2], ("row_id",))
    assert tile_clip(ctx.paths.tile_valid, TILE).equals(tile_box(TILE))
    with pytest.raises(StageError, match="missing from tile_valid"):
        tile_clip(ctx.paths.tile_valid, tile_ref("siret3_r005_c000"))
    frame = pd.DataFrame()
    assert frame.empty
