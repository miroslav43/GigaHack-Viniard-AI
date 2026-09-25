"""CVAT `<mask>` run-length encoding -> polygon rings (import only, contract §4.3).

CVAT counts alternate background/foreground starting with background, row-major over the mask box
(`left`, `top`, `width`, `height`). Rings follow pixel edges, so a ring's area equals its pixel count.
"""

from typing import Final

import numpy as np
from rasterio.features import shapes
from rasterio.transform import Affine

from vineyard.errors import CvatFormatError

Ring = tuple[tuple[float, float], ...]
RLE_SEP: Final = ","
CONNECTIVITY: Final = 4  # 4-connected parts never self-touch, so every ring is a valid polygon
_FOREGROUND: Final = 1


def parse_rle(text: str) -> tuple[int, ...]:
    """'2, 8, 2' -> (2, 8, 2); counts must be non-negative integers."""
    parts = [p.strip() for p in (text or "").split(RLE_SEP)]
    if not parts or any(p == "" for p in parts):
        raise CvatFormatError("empty or malformed mask rle", rle=text[:40] if text else "")
    try:
        counts = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise CvatFormatError("non-integer mask rle count", rle=text[:40]) from exc
    if any(c < 0 for c in counts):
        raise CvatFormatError("negative mask rle count", rle=text[:40])
    return counts


def rle_to_mask(counts: tuple[int, ...], *, width: int, height: int) -> np.ndarray:
    """Decode counts into a (height, width) bool mask."""
    if width <= 0 or height <= 0:
        raise CvatFormatError("mask box must have a positive size", width=width, height=height)
    if sum(counts) != width * height:
        raise CvatFormatError("mask rle does not cover the box", total=sum(counts), expected=width * height)
    values = np.arange(len(counts)) % 2 == _FOREGROUND
    return np.repeat(values, counts).reshape(height, width)


def _exterior(geojson: dict) -> Ring:
    coords = geojson["coordinates"][0]
    return tuple((float(u), float(v)) for u, v in coords[:-1])


def rle_to_rings(rle: str, *, left: int, top: int, width: int, height: int) -> tuple[Ring, ...]:
    """Exterior rings (continuous px, no closing vertex) of each 4-connected part; holes are dropped.

    Sorted by (top, left) of the ring bounds so the output is deterministic.
    """
    mask = rle_to_mask(parse_rle(rle), width=width, height=height)
    if not mask.any():
        return ()
    transform = Affine(1.0, 0.0, float(left), 0.0, 1.0, float(top))
    parts = shapes(mask.astype(np.uint8), mask=mask, connectivity=CONNECTIVITY, transform=transform)
    rings = [_exterior(geom) for geom, _ in parts]
    return tuple(sorted(rings, key=lambda r: (min(v for _, v in r), min(u for u, _ in r))))
