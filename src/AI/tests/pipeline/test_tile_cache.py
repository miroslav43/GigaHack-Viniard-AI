"""tile_prep cache layout and accessors (work/cache/{valid,veg,vis,stats}/<tile>.*)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vineyard.contracts.enums import TileStatus
from vineyard.errors import StageError
from vineyard.geo.raster import write_mask_png, write_u8_png
from vineyard.perception.types import TileStats
from vineyard.pipeline.tile_cache import (
    key_path,
    load_tile_stats,
    load_valid_mask,
    load_veg_mask,
    load_vis,
    stats_path,
    tile_prep_key,
    valid_mask_path,
    veg_mask_path,
    vis_path,
    write_tile_stats,
)

TILE = "siret3_r021_c012"


def _stats() -> TileStats:
    return TileStats(TILE, 0.98, 0.02, 0.11, TileStatus.OK, 4.0, "lab_a")


def test_paths_layout(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    assert veg_mask_path(cache, TILE) == cache / "veg" / f"{TILE}.png"
    assert valid_mask_path(cache, TILE) == cache / "valid" / f"{TILE}.png"
    assert vis_path(cache, TILE) == cache / "vis" / f"{TILE}.png"
    assert stats_path(cache, TILE) == cache / "stats" / f"{TILE}.json"
    assert key_path(stats_path(cache, TILE)) == cache / "stats" / f"{TILE}.json.key"


def test_paths_reject_bad_tile_id(tmp_path: Path) -> None:
    with pytest.raises(StageError):
        veg_mask_path(tmp_path, "../etc/passwd")


def test_masks_roundtrip(tmp_path: Path) -> None:
    veg = np.zeros((16, 16), bool)
    veg[3:7, 2:9] = True
    vis = np.zeros((16, 16), np.uint8)
    vis[0] = 3
    write_mask_png(veg_mask_path(tmp_path, TILE), veg)
    write_mask_png(valid_mask_path(tmp_path, TILE), ~veg)
    write_u8_png(vis_path(tmp_path, TILE), vis)
    np.testing.assert_array_equal(load_veg_mask(tmp_path, TILE), veg)
    np.testing.assert_array_equal(load_valid_mask(tmp_path, TILE), ~veg)
    np.testing.assert_array_equal(load_vis(tmp_path, TILE), vis)


def test_stats_roundtrip(tmp_path: Path) -> None:
    path = write_tile_stats(tmp_path, _stats())
    assert path == stats_path(tmp_path, TILE)
    assert load_tile_stats(tmp_path, TILE) == _stats()


def test_stats_tile_mismatch(tmp_path: Path) -> None:
    write_tile_stats(tmp_path, _stats())
    other = "siret3_r006_c004"
    stats_path(tmp_path, other).write_bytes(stats_path(tmp_path, TILE).read_bytes())
    with pytest.raises(StageError, match="tile_id"):
        load_tile_stats(tmp_path, other)


def test_stats_corrupt(tmp_path: Path) -> None:
    path = stats_path(tmp_path, TILE)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    with pytest.raises(StageError, match="stats"):
        load_tile_stats(tmp_path, TILE)
    path.write_text('{"tile_id": "x"}')
    with pytest.raises(StageError, match="stats"):
        load_tile_stats(tmp_path, TILE)


def test_missing_artifacts_raise_stage_error(tmp_path: Path) -> None:
    for loader in (load_veg_mask, load_valid_mask, load_vis, load_tile_stats, tile_prep_key):
        with pytest.raises(StageError, match=TILE):
            loader(tmp_path, TILE)


def test_tile_prep_key(tmp_path: Path) -> None:
    write_tile_stats(tmp_path, _stats())
    key_path(stats_path(tmp_path, TILE)).write_text("abc123\n")
    assert tile_prep_key(tmp_path, TILE) == "abc123"
    key_path(stats_path(tmp_path, TILE)).write_text("  \n")
    with pytest.raises(StageError, match="empty"):
        tile_prep_key(tmp_path, TILE)
