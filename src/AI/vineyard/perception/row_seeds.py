"""Reviewed row seeds (configs/row_seeds.csv): strict parsing and the seed angle convention.

One line per reviewed region of a tile: `tile_id,vines,confidence,polygon_px,angle_deg,spacing_m,kind,note`.
polygon_px = "x1 y1;x2 y2;..." in 2048-px tile coordinates (x right, y down). angle_deg = row direction in
image coordinates, measured from +x counter-clockwise towards the image top, in [0, 180) — i.e. the same
number as the UTM axial angle (UTM y points north = image up). Every line is validated; only lines with
vines=yes and a configured confidence are used, the others document the review.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from shapely.geometry import Polygon

from vineyard.errors import SchemaError
from vineyard.geo.tiling import TILE_PX, parse_tile_id

SEED_COLUMNS: Final = ("tile_id", "vines", "confidence", "polygon_px", "angle_deg", "spacing_m", "kind", "note")
VINES_VALUES: Final = ("yes", "no", "maybe")
CONFIDENCE_VALUES: Final = ("high", "medium", "low")
VINES_YES: Final = "yes"
AXIAL_DEG: Final = 180.0
MIN_VERTICES: Final = 3
VERTEX_SEP: Final = ";"


class SeedFileError(ValueError):
    """A malformed seed file (message names the file and line)."""


@dataclass(frozen=True, eq=False)
class RowSeed:
    tile_id: str
    vines: str
    confidence: str
    polygon_px: Polygon
    angle_deg: float
    spacing_m: float | None
    kind: str
    note: str
    line_no: int


def seed_angle_to_px(angle_deg: float) -> float:
    """Pixel angle (from +u towards +v, v down) of a seed angle (from +x towards the image top)."""
    return (-float(angle_deg)) % AXIAL_DEG


def px_angle_to_seed(angle_px_deg: float) -> float:
    """Inverse of seed_angle_to_px."""
    return (-float(angle_px_deg)) % AXIAL_DEG


def parse_polygon(text: str) -> Polygon:
    """"x1 y1;x2 y2;..." -> valid Polygon inside [0, TILE_PX]^2 (ValueError otherwise)."""
    parts = [p.strip() for p in text.strip().split(VERTEX_SEP) if p.strip()]
    if len(parts) < MIN_VERTICES:
        raise ValueError(f"polygon_px needs >= {MIN_VERTICES} vertices, got {len(parts)}")
    coords = []
    for part in parts:
        xy = part.split()
        if len(xy) != 2:
            raise ValueError(f"polygon_px vertex {part!r} is not 'x y'")
        x, y = float(xy[0]), float(xy[1])
        if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= TILE_PX and 0.0 <= y <= TILE_PX):
            raise ValueError(f"polygon_px vertex {part!r} outside the tile [0, {TILE_PX}]")
        coords.append((x, y))
    poly = Polygon(coords)
    if not poly.is_valid or poly.area <= 0.0:
        raise ValueError("polygon_px is not a valid polygon with positive area (self-intersecting?)")
    return poly


def _angle(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and 0.0 <= value < AXIAL_DEG):
        raise ValueError(f"angle_deg {text!r} outside [0, 180)")
    return value


def _spacing(text: str) -> float | None:
    if not text.strip():
        return None
    value = float(text)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"spacing_m {text!r} must be a positive number or empty")
    return value


def _choice(name: str, text: str, allowed: tuple[str, ...]) -> str:
    value = text.strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} {text!r} not in {allowed}")
    return value


def _tile(text: str, known_tiles: Collection[str] | None) -> str:
    tile_id = text.strip()
    try:
        parse_tile_id(tile_id)
    except SchemaError as exc:
        raise ValueError(f"invalid tile id {tile_id!r}") from exc
    if known_tiles is not None and tile_id not in known_tiles:
        raise ValueError(f"unknown tile {tile_id!r}")
    return tile_id


def _seed(row: dict[str, str], line_no: int, known_tiles: Collection[str] | None) -> RowSeed:
    return RowSeed(
        tile_id=_tile(row["tile_id"], known_tiles), vines=_choice("vines", row["vines"], VINES_VALUES),
        confidence=_choice("confidence", row["confidence"], CONFIDENCE_VALUES),
        polygon_px=parse_polygon(row["polygon_px"]), angle_deg=_angle(row["angle_deg"]),
        spacing_m=_spacing(row["spacing_m"]), kind=row["kind"].strip(), note=row["note"].strip(), line_no=line_no)


def load_row_seeds(path: Path, known_tiles: Collection[str] | None = None) -> tuple[RowSeed, ...]:
    """Every seed line of the file (header must equal SEED_COLUMNS); SeedFileError names file:line."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None or tuple(h.strip() for h in header) != SEED_COLUMNS:
            raise SeedFileError(f"{path}:1: header must be {','.join(SEED_COLUMNS)}, got {header}")
        seeds = []
        for values in reader:
            line_no = reader.line_num
            if not any(v.strip() for v in values):
                continue
            if len(values) != len(SEED_COLUMNS):
                raise SeedFileError(f"{path}:{line_no}: expected {len(SEED_COLUMNS)} fields, got {len(values)}")
            try:
                seeds.append(_seed(dict(zip(SEED_COLUMNS, values, strict=True)), line_no, known_tiles))
            except ValueError as exc:
                raise SeedFileError(f"{path}:{line_no}: {exc}") from exc
    return tuple(seeds)


def usable_seeds(seeds: Iterable[RowSeed], confidences: Collection[str]) -> tuple[RowSeed, ...]:
    """The seeds that drive detection: vines=yes and confidence in `confidences` (file order kept)."""
    return tuple(s for s in seeds if s.vines == VINES_YES and s.confidence in confidences)
