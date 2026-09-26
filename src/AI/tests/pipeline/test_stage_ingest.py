"""Stage `ingest`: tiles -> work/tiles + tile_index, 02_route -> work/layers/in_*."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.conftest import tile_zip_paths
from tests.ingest.synth_data import build_data_root, make_ctx
from vineyard.contracts.ids import tile_grid_ids
from vineyard.geo.vector_io import read_layer
from vineyard.ingest.tiles import scan_zips
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.runner import run_stages
from vineyard.pipeline.tile_index import indexed_tile_ids, read_tile_index, tile_path, tile_refs

IDS = ("siret3_r006_c004", "siret3_r018_c010", "siret3_r021_c012")


@pytest.fixture
def synth_root(tmp_path: Path, data_root: Path, make_geotiff: Callable[..., Path]) -> Path:
    tifs = [make_geotiff(tmp_path / "src" / f"{t}.tif", size=2048) for t in IDS]
    return build_data_root(tmp_path / "data", data_root, tifs)


def test_stage_spec() -> None:
    spec = load_stage("ingest")
    assert (spec.name, spec.scope) == ("ingest", "global")
    code = "import sys; from vineyard.pipeline.registry import load_stage; load_stage('ingest'); print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_ingest_stage_writes_index_and_route_layers(synth_root: Path, tmp_path: Path) -> None:
    ctx = make_ctx(synth_root, tmp_path / "work", len(IDS))
    report = run_stages(ctx, ("ingest",))
    assert report.exit_code == 0
    result = report.results[0]
    assert (result.n_items, result.n_cached, result.n_failed) == (3, 0, 0)
    assert indexed_tile_ids(ctx) == IDS
    index = read_tile_index(ctx)
    assert list(index["tile_id"]) == list(IDS)
    assert set(tile_refs(ctx)) == set(IDS)
    assert tile_path(ctx, IDS[0]).is_file()
    for name in ("in_passages", "in_forbidden", "in_study_area", "in_start"):
        assert len(read_layer(ctx.paths.static_layers_dir / f"{name}.parquet", name)) == 1
    rerun = run_stages(make_ctx(synth_root, tmp_path / "work", len(IDS), run_id="rerun"), ("ingest",))
    assert rerun.results[0].n_cached == 3


def test_ingest_stage_count_mismatch_fails(synth_root: Path, tmp_path: Path) -> None:
    report = run_stages(make_ctx(synth_root, tmp_path / "work", 5), ("ingest",))
    assert report.exit_code == 1 and report.results == ()


@pytest.mark.needs_tiles
def test_real_zips_hold_exactly_the_tile_grid(data_root: Path) -> None:
    members = scan_zips(tile_zip_paths(data_root))
    assert tuple(m.tile_id for m in members) == tuple(sorted(tile_grid_ids()))
