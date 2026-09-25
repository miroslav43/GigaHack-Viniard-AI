"""Stages `rows_link` and `blocks` on a synthetic two-tile work dir (global stages, no tile pool)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.errors import StageError
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_M, TILE_PX, tile_box, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.perception.overrides import EMPTY_OVERRIDES
from vineyard.pipeline.cache import write_key
from vineyard.pipeline.context import RunContext, ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.runner import run_stages
from vineyard.pipeline.stages import blocks as blocks_stage
from vineyard.pipeline.stages import rows_link as link_stage

TILES = ["siret3_r011_c005", "siret3_r011_c006"]
MISSING = "siret3_r011_c007"
T0 = tile_ref(TILES[0])
Y_ROWS = [T0.y0 - 10.0 - 2.5 * k for k in range(5)]


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


def _candidates(tile_id: str, reason: str | None = None) -> gpd.GeoDataFrame:
    t = tile_ref(tile_id)
    lines = [LineString([(t.x0 + 0.5, y), (t.x0 + TILE_M - 0.5, y)]) for y in Y_ROWS]
    n = len(lines)
    return gpd.GeoDataFrame(
        {"cand_id": [f"{tile_id}:K{k + 1:02d}" for k in range(n)], "tile_id": tile_id, "angle_deg": 0.0,
         "offset_m": 0.0, "length_m": [g.length for g in lines], "support_frac": 0.9, "width_med_m": 0.5,
         "local_spacing_m": 2.5, "vine_score": float("nan"), "rejected_reason": reason, "gaps_json": "[]",
         "width_p80_m": 0.55, "along_duty": 0.9, "along_period_m": float("nan"), "harmonic_flag": False,
         "is_curved": False, "source": Source.MODEL.value, "run_id": "r", "model_version": "m", "confidence": 0.9,
         "qa_flags": ""},
        geometry=lines, crs=f"EPSG:{CRS_EPSG}",
    )


def _write_candidates(ctx: RunContext, tile_id: str, frame: gpd.GeoDataFrame) -> Path:
    path = link_stage.candidates_path(ctx, tile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_layer(frame, "row_candidates", path)
    write_key(path, f"detect-{tile_id}")
    return path


def _passages(ctx: RunContext, geom: object) -> None:
    frame = gpd.GeoDataFrame({"fid": [1], "type": ["road"], "name": [None], "source": [None]}, geometry=[geom],
                             crs=f"EPSG:{CRS_EPSG}")
    write_layer(frame, "in_passages", ctx.paths.static_layers_dir / "in_passages.parquet")


@pytest.fixture
def ctx(tmp_work: Path) -> RunContext:
    context = new_run_context(load_config(), source=Source.MODEL, run_id="test-run", workers=1)
    ensure_run_dirs(context.paths)
    tiles = [*TILES, MISSING]
    write_layer(_tile_index(tiles), "tile_index", context.paths.tile_index)
    valid = gpd.GeoDataFrame({"tile_id": tiles, "valid_frac": 1.0}, geometry=[tile_box(tile_ref(t)) for t in tiles],
                             crs=f"EPSG:{CRS_EPSG}")
    write_layer(valid, "tile_valid", context.paths.tile_valid)
    for t in TILES:
        _write_candidates(context, t, _candidates(t))
    return context


def test_stage_specs_registered() -> None:
    assert load_stage("rows_link") is link_stage.STAGE and link_stage.STAGE.scope == "global"
    assert load_stage("blocks") is blocks_stage.STAGE and blocks_stage.STAGE.requires == ("rows_link",)


def test_rows_link_then_blocks(ctx: RunContext) -> None:
    res = link_stage.run(ctx)
    assert res.n_items == 3 and res.metrics["n_chains"] == 5.0
    rows_raw = read_layer(ctx.paths.layers_dir / "rows_raw.parquet", "rows_raw")
    assert list(rows_raw.n_tiles) == [2] * 5 and set(rows_raw.run_id) == {"test-run"}
    cands = read_layer(ctx.paths.layers_dir / "row_candidates.parquet", "row_candidates")
    assert len(cands) == 10 and set(cands.link_decision) == {"linked"}
    qa = read_layer(ctx.paths.qa_dir / "qa_rows_link.parquet", "qa_issues")
    assert list(qa.code) == ["tile_failed"] and list(qa.tile_id) == [MISSING]
    out = blocks_stage.run(ctx)
    assert out.metrics["n_blocks"] == 1.0 and out.metrics["n_rows"] == 5.0
    rows = read_layer(ctx.paths.layers_dir / "rows.parquet", "rows")
    assert list(rows.row_id) == [f"V01-R{k:03d}" for k in range(1, 6)]
    assert rows.length_m.iloc[0] == pytest.approx(2 * TILE_M, abs=0.01)
    pairs = read_layer(ctx.paths.layers_dir / "row_pairs.parquet", "row_pairs")
    assert len(pairs) == 4
    assert read_layer(ctx.paths.layers_dir / "rows_rejected.parquet", "rows_rejected").empty
    assert read_layer(ctx.paths.qa_dir / "qa_blocks.parquet", "qa_issues").empty


def test_run_stages_is_deterministic(ctx: RunContext) -> None:
    report = run_stages(ctx, ["rows_link", "blocks"])
    assert report.exit_code == 0
    first = read_layer(ctx.paths.layers_dir / "rows.parquet", "rows")
    run_stages(ctx, ["rows_link", "blocks"])
    second = read_layer(ctx.paths.layers_dir / "rows.parquet", "rows")
    assert first.drop(columns="geometry").equals(second.drop(columns="geometry"))
    assert [g.wkb for g in first.geometry] == [g.wkb for g in second.geometry]


def test_passage_splits_the_block(ctx: RunContext) -> None:
    mid = (Y_ROWS[1] + Y_ROWS[2]) / 2.0
    _passages(ctx, box(T0.x0 - 5, mid - 1.5, T0.x0 + 2 * TILE_M + 5, mid + 1.5))
    link_stage.run(ctx)
    blocks_stage.run(ctx)
    blocks = read_layer(ctx.paths.layers_dir / "blocks.parquet", "blocks")
    rejected = read_layer(ctx.paths.layers_dir / "rows_rejected.parquet", "rows_rejected")
    assert len(blocks) == 1 and len(rejected) == 2
    qa = read_layer(ctx.paths.qa_dir / "qa_blocks.parquet", "qa_issues")
    assert list(qa.code) == ["block_too_few_rows"]


def test_overrides_are_applied(ctx: RunContext, tmp_path: Path) -> None:
    text = (f"version: 1\nforce_empty_tiles: [{TILES[1]}]\n"
            f"add_rows: [{{id: A001, wkt: 'LINESTRING ({T0.x0 + 1} {Y_ROWS[-1] - 2.5}, "
            f"{T0.x0 + 40} {Y_ROWS[-1] - 2.5})'}}]\n")
    path = tmp_path / "ov.yaml"
    path.write_text(text, encoding="utf-8")
    cfg = load_config(overrides=[f"paths.overrides={path}"])
    context = new_run_context(cfg, source=Source.MODEL, run_id="test-run", workers=1)
    link_stage.run(context)
    rows_raw = read_layer(context.paths.layers_dir / "rows_raw.parquet", "rows_raw")
    assert all(TILES[1] not in str(m) for m in rows_raw.member_cand_ids)
    blocks_stage.run(context)
    rows = read_layer(context.paths.layers_dir / "rows.parquet", "rows")
    assert len(rows) == 6 and rows.qa_flags.str.contains("override:A001").sum() == 1
    codes = list(read_layer(context.paths.qa_dir / "qa_blocks.parquet", "qa_issues").code)
    assert codes == ["override_applied"]


def test_missing_inputs_raise(tmp_work: Path) -> None:
    context = new_run_context(load_config(), source=Source.MODEL, run_id="bare", workers=1)
    ensure_run_dirs(context.paths)
    with pytest.raises(StageError):
        blocks_stage.run(context)
    with pytest.raises(StageError):
        link_stage.load_clips(context)


def test_missing_overrides_file_is_empty(ctx: RunContext, tmp_path: Path) -> None:
    cfg = load_config(overrides=[f"paths.overrides={tmp_path / 'absent.yaml'}"])
    context = new_run_context(cfg, source=Source.MODEL, run_id="test-run", workers=1)
    assert link_stage.load_overrides_cfg(context) == EMPTY_OVERRIDES
    assert link_stage.load_passages(context) is None
