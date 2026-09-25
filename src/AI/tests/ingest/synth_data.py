"""Synthetic data root for ingest / tile_prep stage tests: tile ZIPs + a copy of the real 02_route."""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import Source
from vineyard.pipeline.context import RunContext, new_run_context

TILE_PX = 2048
ROUTE_SUBDIR = Path("02_route")
ZIP_NAME = "siret3_challenge_tiles_part{k}of5.zip"
TILES_SUBDIR = Path("01_tiles")


def build_data_root(root: Path, real_data_root: Path, tifs: Sequence[Path], *, n_zips: int = 2) -> Path:
    """<root>/01_tiles/siret3_challenge_tiles_part{k}of5.zip (tifs spread round-robin) + <root>/02_route."""
    route = real_data_root / ROUTE_SUBDIR
    if not route.is_dir():
        pytest.skip(f"route inputs missing under {route}")
    shutil.copytree(route, root / ROUTE_SUBDIR)
    zip_dir = root / TILES_SUBDIR
    zip_dir.mkdir(parents=True)
    for k in range(n_zips):
        with zipfile.ZipFile(zip_dir / ZIP_NAME.format(k=k + 1), "w", compression=zipfile.ZIP_STORED) as zf:
            for tif in tifs[k::n_zips]:
                zf.write(tif, Path(tif).name)
    return root


def make_ctx(data_root: Path, work: Path, n_tiles: int, *, overrides: Sequence[str] = (),
             **kwargs: object) -> RunContext:
    """RunContext on the synthetic data root with grid.expected_tiles = n_tiles."""
    cfg = config_for(data_root, work, n_tiles, overrides)
    opts: Mapping[str, object] = {"source": Source.MODEL, "run_id": "test-run", "workers": 1} | dict(kwargs)
    return new_run_context(cfg, **opts)  # type: ignore[arg-type]


def config_for(data_root: Path, work: Path, n_tiles: int, overrides: Sequence[str] = ()) -> AppConfig:
    base = [f'paths.data_root="{data_root}"', f'paths.work_dir="{work}"', f"grid.expected_tiles={n_tiles}"]
    return load_config(overrides=[*base, *overrides])


def uniform_rgb(value: tuple[int, int, int]) -> np.ndarray:
    img = np.empty((TILE_PX, TILE_PX, 3), np.uint8)
    img[...] = np.asarray(value, np.uint8)
    return img
