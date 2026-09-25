"""Gate G0b: cross-package consistency between config, contracts, geo, pipeline, nn and the real data."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path
from typing import get_args

import pytest

from vineyard.config import AppConfig, load_config
from vineyard.config.sections_nn import FusionName
from vineyard.contracts import CONTRACT_VERSION
from vineyard.contracts.enums import FusionVariant, Severity
from vineyard.contracts.ids import IdKind, file_name_from_tile_id, is_valid_id, tile_id_from_file_name
from vineyard.contracts.qa import QA_LAYER, QaIssue, issues_to_gdf
from vineyard.contracts.schemas import LAYER_SCHEMAS, column_dtype, empty_layer, validate_layer
from vineyard.geo import tiling
from vineyard.geo.vector_io import read_layer, write_layer

N_TILE_ZIPS = 5

TORCH_FREE_MODULES = (
    "vineyard",
    "vineyard.errors",
    "vineyard.config",
    "vineyard.contracts.enums",
    "vineyard.contracts.ids",
    "vineyard.contracts.ordering",
    "vineyard.contracts.qa",
    "vineyard.contracts.schemas",
    "vineyard.annset.model",
    "vineyard.annset.io",
    "vineyard.cvat.model",
    "vineyard.cvat.template",
    "vineyard.geo.tiling",
    "vineyard.geo.raster",
    "vineyard.geo.vector_io",
    "vineyard.perception.types",
    "vineyard.perception.vegmask",
    "vineyard.pipeline.atomic",
    "vineyard.pipeline.context",
    "vineyard.pipeline.registry",
    "vineyard.pipeline.tile_cache",
    "vineyard.nn.probs",
    "vineyard.nn.fusion",
    "vineyard.doctor",
    "vineyard.cli",
    "vineyard.cli_post",
)


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config()


# ---- (a) config <-> contract constants ----------------------------------------------------


def test_config_grid_equals_tiling_constants(cfg: AppConfig) -> None:
    grid = cfg.grid
    assert grid.gsd_m == tiling.GSD_M
    assert grid.tile_px == tiling.TILE_PX
    assert grid.tile_m == tiling.TILE_M
    assert grid.origin_x == tiling.GRID_ORIGIN_X
    assert grid.origin_y == tiling.GRID_ORIGIN_Y
    assert grid.tiepoint_tol_m == tiling.TIEPOINT_TOL_M
    assert grid.expected_tiles == len(tiling.existing_tile_ids())


def test_config_crs_and_contract_version_match_contracts(cfg: AppConfig) -> None:
    assert cfg.project.crs == f"EPSG:{tiling.CRS_EPSG}"
    assert cfg.contract_version == CONTRACT_VERSION


def test_config_fusion_names_equal_fusion_variant_enum(cfg: AppConfig) -> None:
    variants = {v.value for v in FusionVariant}
    assert set(get_args(FusionName)) == variants
    assert cfg.nn.fusion in variants
    assert set(cfg.nn.ablation.promotable) <= variants


def test_config_tile_lists_name_existing_tiles(cfg: AppConfig) -> None:
    existing = tiling.existing_tile_ids()
    for tile_id in (*cfg.nn.holdout_tiles, *cfg.eval.example_tiles):
        assert is_valid_id(IdKind.TILE, tile_id), tile_id
        assert tile_id in existing, tile_id


# ---- (b) import hygiene ---------------------------------------------------------------------


def test_phase0_imports_never_load_torch(project_root: Path) -> None:
    code = "import sys\n" + "".join(f"import {m}\n" for m in TORCH_FREE_MODULES)
    code += "print(sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.')))\n"
    out = subprocess.run([sys.executable, "-c", code], cwd=project_root, capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


# ---- (c) tile grid <-> the organizers' ZIPs (via the config paths) --------------------------


@pytest.mark.needs_tiles
def test_existing_tile_ids_equal_zip_names(cfg: AppConfig) -> None:
    zips = sorted(cfg.paths.data_root.glob(cfg.paths.tiles_zip_glob))
    assert len(zips) == N_TILE_ZIPS
    names: list[str] = []
    for zpath in zips:
        with zipfile.ZipFile(zpath) as zf:
            names.extend(zf.namelist())
    ids = [tile_id_from_file_name(n) for n in names]
    assert len(ids) == len(set(ids)) == cfg.grid.expected_tiles
    assert set(ids) == tiling.existing_tile_ids()
    assert sorted(names) == sorted(file_name_from_tile_id(t) for t in ids)


# ---- (d) every contract layer survives empty -> write -> read -------------------------------


def test_qa_issues_with_and_without_location_round_trip(tmp_path: Path) -> None:
    x0, y0 = tiling.tile_ref("siret3_r021_c012").x0, tiling.tile_ref("siret3_r021_c012").y0
    issues = [
        QaIssue(Severity.WARNING, "row_interpolated", "siret3_r021_c012", "V01-R003", "gap filled", x0 + 1.0, y0 - 1.0),
        QaIssue(Severity.ERROR, "tile_failed", "siret3_r006_c004", "", "decode error"),
    ]
    gdf = issues_to_gdf(issues, run_id="20260925T1200-model-abc123", model_version="pipe@0.1.0")
    back = read_layer(write_layer(gdf, QA_LAYER, tmp_path / "qa_issues.parquet"), QA_LAYER)
    assert list(back["issue_id"]) == ["Q00001", "Q00002"]
    assert list(back["code"]) == ["tile_failed", "row_interpolated"]
    assert back.geometry.iloc[0].is_empty
    assert back.geometry.iloc[1].equals(gdf.geometry.iloc[1])


@pytest.mark.parametrize("name", sorted(LAYER_SCHEMAS))
def test_empty_layer_parquet_round_trip(name: str, tmp_path: Path) -> None:
    empty = empty_layer(name)
    path = write_layer(empty, name, tmp_path / f"{name}.parquet")
    back = read_layer(path, name)
    schema = LAYER_SCHEMAS[name]
    assert list(back.columns) == [*schema.column_names, "geometry"]
    assert len(back) == 0
    assert back.crs is not None and back.crs.to_epsg() == tiling.CRS_EPSG
    for col in schema.all_columns:
        assert back[col.name].dtype == column_dtype(col), (name, col.name, back[col.name].dtype)
    validate_layer(back, name)
