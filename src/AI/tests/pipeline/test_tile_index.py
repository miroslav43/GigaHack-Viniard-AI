"""pipeline.tile_index: read the ingested tile index, join tile_valid, tile refs and paths."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, box

from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.errors import StageError
from vineyard.geo.tiling import CRS_EPSG, TILE_M, tile_box, tile_ref
from vineyard.geo.vector_io import write_layer
from vineyard.pipeline.context import RunContext, new_run_context
from vineyard.pipeline.tile_index import indexed_tile_ids, read_tile_index, tile_path, tile_refs

IDS = ("siret3_r021_c012", "siret3_r006_c004")


@pytest.fixture
def ctx(tmp_work: Path) -> RunContext:
    return new_run_context(load_config(), source=Source.MODEL, run_id="t", workers=1)


def _write_index(ctx: RunContext) -> None:
    rows = []
    for t in IDS:
        ref = tile_ref(t)
        rows.append({"tile_id": t, "file_name": f"{t}.tif", "grid_row": ref.grid_row, "grid_col": ref.grid_col,
                     "x0": ref.x0, "y0": ref.y0, "x1": ref.x0 + TILE_M, "y1": ref.y0 - TILE_M, "gsd_m": 0.025,
                     "width_px": 2048, "height_px": 2048, "src_zip": "z.zip", "path": f"/x/{t}.tif",
                     "sha256": "0" * 64, "file_size": 1, "nodata_frac": np.nan, "valid_area_m2": np.nan})
    frame = gpd.GeoDataFrame(rows, geometry=[tile_box(tile_ref(t)) for t in IDS], crs=CRS_EPSG)
    write_layer(frame, "tile_index", ctx.paths.tile_index)


def test_missing_index_asks_for_ingest(ctx: RunContext) -> None:
    with pytest.raises(StageError, match="vineyard ingest"):
        read_tile_index(ctx)
    with pytest.raises(StageError, match="vineyard ingest"):
        indexed_tile_ids(ctx.paths)


def test_index_sorted_and_refs(ctx: RunContext) -> None:
    _write_index(ctx)
    assert indexed_tile_ids(ctx) == tuple(sorted(IDS))
    index = read_tile_index(ctx)
    assert list(index["tile_id"]) == sorted(IDS)
    assert index["nodata_frac"].isna().all()
    refs = tile_refs(ctx.paths)
    assert list(refs) == sorted(IDS) and refs["siret3_r006_c004"] == tile_ref("siret3_r006_c004")
    assert tile_path(ctx, "siret3_r006_c004") == ctx.paths.tiles_dir / "siret3_r006_c004.tif"


def test_join_tile_valid(ctx: RunContext) -> None:
    _write_index(ctx)
    t = tile_ref("siret3_r006_c004")
    half = box(t.x0, t.y0 - TILE_M / 2, t.x0 + TILE_M, t.y0)
    valid = gpd.GeoDataFrame({"tile_id": ["siret3_r006_c004"], "valid_frac": [0.5]},
                             geometry=[MultiPolygon([half])], crs=CRS_EPSG)
    write_layer(valid, "tile_valid", ctx.paths.tile_valid)
    index = read_tile_index(ctx).set_index("tile_id")
    assert index.loc["siret3_r006_c004", "nodata_frac"] == pytest.approx(0.5)
    assert index.loc["siret3_r006_c004", "valid_area_m2"] == pytest.approx(TILE_M * TILE_M / 2)
    assert np.isnan(index.loc["siret3_r021_c012", "nodata_frac"])
