"""Stage `waste` on a synthetic one-tile work dir (inline, workers=1)."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from vineyard.config import load_config
from vineyard.contracts.enums import Source, TileStatus
from vineyard.errors import StageError
from vineyard.geo.raster import write_mask_png
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_box, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.perception.types import TileStats
from vineyard.pipeline.cache import write_key
from vineyard.pipeline.context import RunContext, ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages import waste as waste_stage
from vineyard.pipeline.tile_cache import stats_path, valid_mask_path, write_tile_stats
from vineyard.pipeline.tile_index import tile_path

TILE_ID = "siret3_r021_c012"
TILE = tile_ref(TILE_ID)
AXIS_V = 500.0
SOIL = (120, 100, 80)
BLUE = (20, 60, 230)
WHITE = (250, 250, 250)
CONFIRM_HEADER = "tile_id,xtl,ytl,xbr,ybr,decision,category,reviewer,note\n"
RULE_ONLY = ("waste.probe.enabled=false", "waste.sam3.enabled=false")  # no CLIP / SAM 3 weights in unit tests


def _image() -> np.ndarray:
    img = np.empty((TILE_PX, TILE_PX, 3), np.uint8)
    img[...] = SOIL
    for k in range(20):  # white tubes on the axis every 1.2 m (48 px)
        u = 40 + 48 * k
        img[int(AXIS_V) - 6 : int(AXIS_V) + 6, u - 6 : u + 6] = WHITE
    img[1400:1430, 1400:1440] = BLUE  # a blue bag far from the rows
    img[1000:1030, 300:330] = BLUE  # a bag with a white label: vivid ring + bright centre (NMS pair)
    img[1005:1025, 305:325] = WHITE
    img[1800:1802, 200:400] = WHITE  # a 2 px hose line, 5 m long
    img[100:110, 1500:1512] = (250, 235, 215)  # pale warm patch: bright soil
    return img


def _tile_index() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"tile_id": [TILE_ID], "file_name": [f"{TILE_ID}.tif"], "grid_row": [TILE.grid_row],
         "grid_col": [TILE.grid_col], "x0": [TILE.x0], "y0": [TILE.y0], "x1": [TILE.bounds[2]],
         "y1": [TILE.bounds[1]], "gsd_m": GSD_M, "width_px": TILE_PX, "height_px": TILE_PX, "src_zip": "z.zip",
         "path": "p", "sha256": "0" * 64, "file_size": 1, "nodata_frac": 0.0, "valid_area_m2": 2621.44},
        geometry=[tile_box(TILE)], crs=f"EPSG:{CRS_EPSG}",
    )  # fmt: skip


def _rows() -> gpd.GeoDataFrame:
    line = LineString(px_to_utm(TILE, np.array([[-200.0, AXIS_V], [2300.0, AXIS_V]])))
    return gpd.GeoDataFrame(
        {"row_id": ["V01-R001"], "vineyard_id": "V01", "row_index": [1], "length_m": [line.length],
         "extent_m": [line.length], "n_pieces": 1, "tile_ids": TILE_ID, "angle_deg": 0.0, "spacing_prev_m": 3.0,
         "spacing_next_m": 3.0, "max_gap_m": np.nan, "n_gaps_ge5": None, "structure_any": None,
         "source": Source.MODEL.value, "run_id": "test-run", "model_version": "pipe@test", "confidence": 1.0,
         "qa_flags": ""},
        geometry=[line], crs=f"EPSG:{CRS_EPSG}",
    )  # fmt: skip


def _blocks() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"vineyard_id": ["V01"], "n_rows": [1], "n_row_pieces": [1], "n_canopies": [None], "n_tiles": [1],
         "row_length_m": [51.2], "canopy_area_m2": [0.0], "interrow_area_m2": [0.0], "outline_area_m2": [2621.44],
         "angle_deg": [0.0], "spacing_med_m": [3.0], "is_garden": [False], "source": Source.MODEL.value,
         "run_id": "test-run", "model_version": "pipe@test", "confidence": 1.0, "qa_flags": ""},
        geometry=[tile_box(TILE)], crs=f"EPSG:{CRS_EPSG}",
    )  # fmt: skip


def _context(tmp_work: Path, *sets: str) -> RunContext:
    confirmed = tmp_work / "waste_confirmed.csv"
    cfg = load_config(overrides=(f"paths.waste_confirmed={json.dumps(str(confirmed))}", *RULE_ONLY, *sets))
    return new_run_context(cfg, source=Source.MODEL, run_id="test-run", workers=1)


@pytest.fixture
def ctx(tmp_work: Path, make_geotiff) -> RunContext:
    context = _context(tmp_work)
    ensure_run_dirs(context.paths)
    (tmp_work / "waste_confirmed.csv").write_text(CONFIRM_HEADER, encoding="utf-8")
    write_layer(_tile_index(), "tile_index", context.paths.tile_index)
    make_geotiff(tile_path(context, TILE_ID), tile_id=TILE_ID, size=TILE_PX, rgb=_image(), compress="DEFLATE")
    cache = context.paths.cache_dir
    write_mask_png(valid_mask_path(cache, TILE_ID), np.ones((TILE_PX, TILE_PX), bool))
    stats_path(cache, TILE_ID).parent.mkdir(parents=True, exist_ok=True)
    write_tile_stats(cache, TileStats(TILE_ID, 1.0, 0.0, 0.1, TileStatus.OK, 4.0, "lab_a"))
    write_key(stats_path(cache, TILE_ID), "prep-key")
    write_layer(_rows(), "rows", context.paths.layers_dir / "rows.parquet")
    write_layer(_blocks(), "blocks", context.paths.layers_dir / "blocks.parquet")
    return context


def _reasons(ctx: RunContext) -> dict[str, str | None]:
    frame = pd.read_parquet(ctx.paths.tile_cache("waste", TILE_ID, "parquet"))
    return {
        k: (r if isinstance(r, str) else None)
        for k, r in zip(frame["cand_key"], frame["reject_reason"], strict=True)
    }


def test_registry_loads_the_stage() -> None:
    assert load_stage("waste") is waste_stage.STAGE


def test_stage_rejects_tubes_keeps_the_bag_and_exports_nothing(ctx: RunContext) -> None:
    res = waste_stage.run(ctx)
    assert (res.n_items, res.n_failed, res.n_cached) == (1, 0, 0)
    reasons = _reasons(ctx)
    cache = pd.read_parquet(ctx.paths.tile_cache("waste", TILE_ID, "parquet"))
    tubes = cache[(cache["cy"] - AXIS_V).abs() <= 1]
    assert len(tubes) == 20 and set(tubes["reject_reason"]) == {"near_axis"}
    assert reasons[f"{TILE_ID}@1420_1415"] is None
    assert not any(k.endswith("_1801") for k in reasons)  # 2 px line removed by the opening
    assert reasons[f"{TILE_ID}@1506_0105"] == "bright_soil"
    layer = read_layer(ctx.paths.layers_dir / "waste_candidates.parquet", "waste_candidates")
    live = layer[layer["reject_reason"].isna()]
    assert sorted(live["cand_key"])[1] == f"{TILE_ID}@1420_1415"
    assert sorted(live["cand_key"])[0].startswith(f"{TILE_ID}@0315_1015")
    assert (layer["reject_reason"] == "nms").sum() == 1
    assert set(live["vineyard_id"]) == {"V01"} and not layer["exported"].any() and not layer["auto"].any()
    waste = read_layer(ctx.paths.layers_dir / "waste.parquet", "waste")
    assert waste.empty
    assert res.metrics["n_exported"] == 0 and res.metrics["n_auto"] == 0
    assert res.metrics["degradation_level"] == 3 and res.metrics["n_review"] == 2
    csv_rows = pd.read_csv(ctx.paths.qa_dir / "waste_candidates.csv")
    assert list(csv_rows["rank"]) == [1, 2]
    assert all((ctx.paths.qa_dir / c).is_file() for c in csv_rows["crop"])
    html = (ctx.paths.qa_dir / "waste_review.html").read_text(encoding="utf-8")
    assert html.count("<img") == 2 and "L3" in html
    status = json.loads((ctx.paths.metrics_dir / "waste.json").read_text())
    assert status["level"] == "L3" and status["per_tile"][TILE_ID]["near_axis"] == 20


def test_second_run_hits_the_cache_and_confirmed_rows_are_exported(ctx: RunContext, tmp_work: Path) -> None:
    waste_stage.run(ctx)
    bag = pd.read_csv(ctx.paths.qa_dir / "waste_candidates.csv").iloc[0]
    (tmp_work / "waste_confirmed.csv").write_text(
        CONFIRM_HEADER
        + bag["confirm_line"].replace(",unknown,", ",bag,")
        + "\n"
        + f"{TILE_ID},10,10,40,40,add,tyre,qa,\n",
        encoding="utf-8",
    )
    again = waste_stage.run(ctx)
    assert again.n_cached == 1
    waste = read_layer(ctx.paths.layers_dir / "waste.parquet", "waste")
    assert list(waste["waste_id"]) == ["W0001", "W0002"]
    assert sorted(waste["category"]) == ["bag", "tyre"] and set(waste["vineyard_id"]) == {"V01"}
    assert again.metrics["n_exported"] == 2


def test_hose_filter_when_the_opening_is_off(tmp_work: Path, ctx: RunContext) -> None:
    no_open = _context(tmp_work, "waste.colour.open_px=0")
    waste_stage.run(no_open)
    reasons = _reasons(no_open)
    assert [r for k, r in reasons.items() if k.endswith("_1801")] == ["hose"]


def test_missing_rows_layer_is_an_error(ctx: RunContext) -> None:
    (ctx.paths.layers_dir / "rows.parquet").unlink()
    with pytest.raises(StageError, match="rows layer missing"):
        waste_stage.run(ctx)


def test_missing_blocks_gives_empty_vineyard_ids(ctx: RunContext) -> None:
    (ctx.paths.layers_dir / "blocks.parquet").unlink()
    waste_stage.run(ctx)
    layer = read_layer(ctx.paths.layers_dir / "waste_candidates.parquet", "waste_candidates")
    assert set(layer.loc[layer["reject_reason"].isna(), "vineyard_id"]) == {""}


def test_local_forbidden_ignores_a_zone_that_only_touches_the_tile() -> None:
    from shapely.geometry import box as sbox

    x0, y0, x1, y1 = TILE.bounds
    touching = gpd.GeoDataFrame({"fid": [1]}, geometry=[sbox(x1, y0, x1 + 10, y1)], crs=f"EPSG:{CRS_EPSG}")
    assert waste_stage.local_forbidden(touching, TILE) is None
    inside = gpd.GeoDataFrame({"fid": [1]}, geometry=[sbox(x1 - 5, y0, x1 + 10, y1)], crs=f"EPSG:{CRS_EPSG}")
    zone = waste_stage.local_forbidden(inside, TILE)
    assert zone is not None and abs(zone.area - 5 * 51.2) < 1e-6
    assert waste_stage.local_forbidden(None, TILE) is None


def test_forbidden_zone_rejects_candidates(ctx: RunContext) -> None:
    from shapely.geometry import box as sbox

    x0, y0, x1, y1 = TILE.bounds
    zone = gpd.GeoDataFrame(
        {"fid": [1], "type": ["forbidden"], "name": [None], "source": ["test"]},
        geometry=[sbox(x0, y0, x1, y1 - 30.0)],
        crs=f"EPSG:{CRS_EPSG}",
    )
    write_layer(zone, "in_forbidden", ctx.paths.static_layers_dir / "in_forbidden.parquet")
    waste_stage.run(ctx)
    assert _reasons(ctx)[f"{TILE_ID}@1420_1415"] == "forbidden"


def test_stage_ranks_with_clip_when_the_probe_is_missing(
    tmp_work: Path, ctx: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from functools import partial

    from tests.waste.test_verify import FakeEmbedder
    from vineyard.perception.waste import verify_setup

    ml = _context(tmp_work, "waste.probe.enabled=true", f"paths.models_dir={json.dumps(str(tmp_work / 'models'))}")
    loader = partial(verify_setup.setup_verifier, clip_loader=lambda _cfg: FakeEmbedder())
    monkeypatch.setattr(waste_stage, "setup_verifier", loader)
    res = waste_stage.run(ml)
    assert res.metrics["degradation_level"] == 2 and res.metrics["n_auto"] == 0
    status = json.loads((ml.paths.metrics_dir / "waste.json").read_text())
    assert status["level"] == "L2" and "probe waste-probe@v1 missing" in status["reason"]
    layer = read_layer(ml.paths.layers_dir / "waste_candidates.parquet", "waste_candidates")
    live = layer[layer["reject_reason"].isna()]
    assert live["clip_pos_p"].notna().all() and live["probe_p"].isna().all()
