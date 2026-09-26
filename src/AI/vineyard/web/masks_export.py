"""Vegetation masks of the web bundle: `masks/<tile>.png` (web contract §6.3).

Source: tile_prep's cache/veg/<tile>.png (tile_px square, 1 = vegetation, nodata = 0). Downsampled by an integer
factor with an any-vegetation max-pool (a web pixel is vegetation when any of its source pixels is: thin canopy
lines stay visible at 1024 px), written as a 1-bit PNG (255 = vegetation when decoded as 8-bit). Pixel (0, 0) is
the tile's north-west corner, the same footprint as the tiles.geojson feature.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np

from vineyard.errors import SchemaError
from vineyard.geo.raster import encode_mask_png
from vineyard.pipeline.tile_cache import load_veg_mask, veg_mask_path

MASKS_DIRNAME: Final = "masks"
MASK_EXT: Final = ".png"
MASK_SOURCE: Final = "tile_prep vegetation mask (Lab a*)"


def downsample_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """`px` x `px` bool mask; True where any source pixel of the block is True (source side = k * px)."""
    h, w = mask.shape
    if h != w or px <= 0 or h % px:
        raise SchemaError("mask side must be a positive multiple of the web mask side", shape=[h, w], px=px)
    k = h // px
    return np.asarray(mask, dtype=bool).reshape(px, k, px, k).any(axis=(1, 3))


def mask_png_bytes(mask: np.ndarray) -> bytes:
    return encode_mask_png(np.asarray(mask, dtype=bool))


def mask_pngs(cache_dir: Path | None, tile_ids: Iterable[str], px: int) -> Mapping[str, bytes]:
    """tile -> PNG bytes for every tile whose tile_prep veg mask exists (none without a cache dir)."""
    if cache_dir is None:
        return MappingProxyType({})
    return MappingProxyType({t: mask_png_bytes(downsample_mask(load_veg_mask(cache_dir, t), px))
                             for t in tile_ids if veg_mask_path(cache_dir, t).is_file()})


def mask_manifest(px: int, n: int) -> dict[str, object]:
    return {"dir": MASKS_DIRNAME, "px": px, "n": n, "source": MASK_SOURCE}
