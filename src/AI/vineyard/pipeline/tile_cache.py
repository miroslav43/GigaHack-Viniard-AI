"""tile_prep cache layout and accessors (plan R10; contract §3.4 paths).

    <cache_dir>/valid/<tile>.png   1-bit, True = image
    <cache_dir>/veg/<tile>.png     1-bit vegetation mask
    <cache_dir>/vis/<tile>.png     uint8 VIS_* codes
    <cache_dir>/stats/<tile>.json  TileStats; its "<artifact>.key" is the tile_prep cache key

tile_prep writes the stats JSON last and then its key, so the key marks a complete tile.
Consumers must use these accessors instead of building paths themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np

from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.errors import StageError
from vineyard.geo.raster import read_mask_png, read_u8_png
from vineyard.perception.types import TileStats
from vineyard.pipeline.atomic import atomic_write_json

STAGE: Final = "tile_prep"
VALID_DIR: Final = "valid"
VEG_DIR: Final = "veg"
VIS_DIR: Final = "vis"
STATS_DIR: Final = "stats"
PNG_EXT: Final = ".png"
JSON_EXT: Final = ".json"
KEY_SUFFIX: Final = ".key"


def _artifact(cache_dir: Path, subdir: str, tile_id: str, ext: str) -> Path:
    if not is_valid_id(IdKind.TILE, tile_id):
        raise StageError("invalid tile id for the tile cache", stage=STAGE, tile_id=tile_id)
    return Path(cache_dir) / subdir / f"{tile_id}{ext}"


def key_path(artifact: Path) -> Path:
    """"<artifact>.key" (same convention as pipeline.cache.write_key)."""
    return artifact.with_name(artifact.name + KEY_SUFFIX)


def veg_mask_path(cache_dir: Path, tile_id: str) -> Path:
    return _artifact(cache_dir, VEG_DIR, tile_id, PNG_EXT)


def valid_mask_path(cache_dir: Path, tile_id: str) -> Path:
    return _artifact(cache_dir, VALID_DIR, tile_id, PNG_EXT)


def vis_path(cache_dir: Path, tile_id: str) -> Path:
    return _artifact(cache_dir, VIS_DIR, tile_id, PNG_EXT)


def stats_path(cache_dir: Path, tile_id: str) -> Path:
    return _artifact(cache_dir, STATS_DIR, tile_id, JSON_EXT)


def _require(path: Path, tile_id: str, what: str) -> Path:
    if not path.is_file():
        raise StageError(f"tile_prep {what} missing; run tile_prep first", stage=STAGE, tile_id=tile_id, path=str(path))
    return path


def load_veg_mask(cache_dir: Path, tile_id: str) -> np.ndarray:
    """Bool vegetation mask of the tile."""
    return read_mask_png(_require(veg_mask_path(cache_dir, tile_id), tile_id, "veg mask"))


def load_valid_mask(cache_dir: Path, tile_id: str) -> np.ndarray:
    """Bool valid-image mask of the tile (True = image, False = nodata)."""
    return read_mask_png(_require(valid_mask_path(cache_dir, tile_id), tile_id, "valid mask"))


def load_vis(cache_dir: Path, tile_id: str) -> np.ndarray:
    """uint8 visibility codes (perception.types.VIS_*)."""
    return read_u8_png(_require(vis_path(cache_dir, tile_id), tile_id, "vis codes"))


def write_tile_stats(cache_dir: Path, stats: TileStats) -> Path:
    """Atomically write the stats JSON (the key file is written separately, after it)."""
    return atomic_write_json(stats_path(cache_dir, stats.tile_id), stats.to_dict())


def load_tile_stats(cache_dir: Path, tile_id: str) -> TileStats:
    """TileStats of the tile; the stored tile_id must match."""
    path = _require(stats_path(cache_dir, tile_id), tile_id, "stats")
    try:
        stats = TileStats.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except ValueError as exc:  # JSONDecodeError is a ValueError
        raise StageError(f"unreadable tile_prep stats: {exc}", stage=STAGE, tile_id=tile_id, path=str(path)) from exc
    if stats.tile_id != tile_id:
        raise StageError("tile_prep stats belong to another tile_id", stage=STAGE, tile_id=tile_id, found=stats.tile_id)
    return stats


def tile_prep_key(cache_dir: Path, tile_id: str) -> str:
    """Cache key of the tile's tile_prep outputs, for downstream cache keys (contract §3.2)."""
    path = _require(key_path(stats_path(cache_dir, tile_id)), tile_id, "cache key")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise StageError("tile_prep cache key file is empty", stage=STAGE, tile_id=tile_id, path=str(path))
    return key
