"""Study-area overview: every tile's preview thumbnail on the grid, framed and labelled by its block."""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from typing import Final

import cv2
import numpy as np

from vineyard.config import QaConfig
from vineyard.contracts.ids import parse_tile_id
from vineyard.perception.types import U8

BACKGROUND_BGR: Final = (0, 0, 0)  # grid cells that are not tiles of the set
MISSING_BGR: Final = (64, 64, 64)  # tiles of the set without a thumbnail
BORDER_PX: Final = 3
LABEL_FONT_SCALE: Final = 0.4
LABEL_POS: Final = (6, 16)
# Distinct, bright BGR colours; a block keeps its colour across runs (crc32 of its id).
PALETTE: Final[tuple[tuple[int, int, int], ...]] = (
    (60, 180, 75), (48, 130, 245), (25, 225, 255), (230, 50, 240), (240, 240, 70), (180, 30, 145),
    (0, 128, 255), (75, 25, 230), (200, 130, 0), (128, 128, 0), (255, 190, 220), (40, 110, 170),
)


def block_colour(vineyard_id: str) -> tuple[int, int, int]:
    return PALETTE[zlib.crc32(vineyard_id.encode("utf-8")) % len(PALETTE)]


def _cell(thumb: U8 | None, block: str | None, px: int) -> np.ndarray:
    if thumb is None:
        return np.full((px, px, 3), MISSING_BGR, dtype=np.uint8)
    cell = cv2.resize(thumb, (px, px), interpolation=cv2.INTER_AREA)
    if block:
        colour = block_colour(block)
        cv2.rectangle(cell, (0, 0), (px - 1, px - 1), colour, BORDER_PX)
        cv2.putText(cell, block, LABEL_POS, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, colour, 1, cv2.LINE_AA)
    return cell


def build_overview(thumbs: Mapping[str, U8], tile_blocks: Mapping[str, str], cfg: QaConfig, *,
                   tile_ids: Sequence[str] | None = None) -> U8:
    """BGR mosaic of `tile_ids` (default: the thumbnails' tiles), qa.overview_px_per_tile per tile.

    tile_blocks: tile -> its dominant vineyard_id (tiles without a block get no frame).
    """
    ids = sorted(set(tile_ids if tile_ids is not None else thumbs))
    if not ids:
        raise ValueError("build_overview needs at least one tile")
    cells = {t: parse_tile_id(t) for t in ids}
    r0, c0 = min(r for r, _ in cells.values()), min(c for _, c in cells.values())
    r1, c1 = max(r for r, _ in cells.values()), max(c for _, c in cells.values())
    px = cfg.overview_px_per_tile
    mosaic = np.full(((r1 - r0 + 1) * px, (c1 - c0 + 1) * px, 3), BACKGROUND_BGR, dtype=np.uint8)
    for tile_id, (r, c) in cells.items():
        y, x = (r - r0) * px, (c - c0) * px
        mosaic[y : y + px, x : x + px] = _cell(thumbs.get(tile_id), tile_blocks.get(tile_id), px)
    return mosaic
