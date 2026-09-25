"""Per-tile QA preview: the tile downscaled to qa.preview_px with the AnnSet drawn on top.

Canopies: translucent green fill + outline; interrows: cyan outline; waste: box; rejected row
candidates: thin grey lines (possible missed rows); row axes coloured by row_structure (regular /
disrupted / unassessable) with short row labels; a header bar with the tile id, status, review
priority, counts and issue codes. Colours come from qa.colors_bgr.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.annset.model import AnnSet
from vineyard.config import QaConfig
from vineyard.geo.ops import split_multi
from vineyard.geo.tiling import TILE_PX, TileRef, utm_to_px
from vineyard.perception.types import U8
from vineyard.pipeline.atomic import atomic_write_bytes

SHIFT: Final = 4  # cv2 sub-pixel drawing: coordinates x 16
_SCALE_FIX: Final = 1 << SHIFT
HEADER_PX: Final = 22
HEADER_ALPHA: Final = 0.65
FONT: Final = cv2.FONT_HERSHEY_SIMPLEX
HEADER_FONT_SCALE: Final = 0.45
LABEL_FONT_SCALE: Final = 0.35
LABEL_AT: Final = 0.15  # row labels sit 15 % along the axis, clear of the midpoint
ROW_THICKNESS: Final = 2
OUTLINE_THICKNESS: Final = 1
TEXT_BGR: Final = (255, 255, 255)
SHADOW_BGR: Final = (0, 0, 0)
STRUCTURE_COLOUR_KEY: Final[Mapping[str, str]] = {
    "regular": "row", "disrupted": "disrupted", "unassessable": "unassessable",
}
_HEADER_MARGIN_PX: Final = 6
_HEADER_BASELINE_PX: Final = 15
_MAX_HEADER_FLAGS: Final = 4


@dataclass(frozen=True)
class PreviewHeader:
    tile_id: str
    status: str = ""
    counts: Mapping[str, int] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    priority: int | None = None

    def text(self) -> str:
        parts = [self.tile_id]
        parts += [self.status] if self.status else []
        parts += [f"P{self.priority}"] if self.priority is not None else []
        parts += [" ".join(f"{k}{v}" for k, v in self.counts.items())] if self.counts else []
        shown = list(self.flags[:_MAX_HEADER_FLAGS]) + (["..."] if len(self.flags) > _MAX_HEADER_FLAGS else [])
        parts += [",".join(shown)] if shown else []
        return " | ".join(parts)


@dataclass(frozen=True, eq=False)
class TileObjects:
    canopies: tuple[BaseGeometry, ...] = ()
    rows: tuple[tuple[LineString, str, str], ...] = ()  # (axis, row_id, row_structure)
    interrows: tuple[BaseGeometry, ...] = ()
    waste: tuple[BaseGeometry, ...] = ()
    rejected: tuple[LineString, ...] = ()  # rejected row candidates (rows_detect)

    @classmethod
    def from_annset(cls, annset: AnnSet, tile_id: str) -> TileObjects:
        def of(name: str):
            frame = annset.layer(name)
            return frame[(frame["tile_id"] == tile_id).to_numpy()]

        rows = of("row_pieces")
        return cls(
            canopies=tuple(of("canopies").geometry),
            rows=tuple(zip(rows.geometry, rows["row_id"].astype(str), rows["row_structure"].astype(str), strict=True)),
            interrows=tuple(of("interrow_pieces").geometry),
            waste=tuple(of("waste").geometry),
        )


def _fixed(tile: TileRef, coords: np.ndarray, scale: float) -> np.ndarray:
    uv = utm_to_px(tile, np.asarray(coords, dtype=np.float64)[:, :2]) * scale
    return np.round(uv * _SCALE_FIX).astype(np.int32).reshape(-1, 1, 2)


def _rings(tile: TileRef, geoms: Sequence[BaseGeometry], scale: float) -> list[np.ndarray]:
    polys = [p for g in geoms if g is not None for p in split_multi(g) if isinstance(p, Polygon)]
    return [_fixed(tile, np.asarray(p.exterior.coords), scale) for p in polys]


def _colour(cfg: QaConfig, key: str) -> tuple[int, int, int]:
    b, g, r = cfg.colors_bgr[key]
    return (int(b), int(g), int(r))


def _draw_canopies(img: np.ndarray, rings: list[np.ndarray], cfg: QaConfig) -> np.ndarray:
    if not rings:
        return img
    colour = _colour(cfg, "canopy")
    fill = img.copy()
    cv2.fillPoly(fill, rings, colour, lineType=cv2.LINE_8, shift=SHIFT)
    out = cv2.addWeighted(fill, cfg.canopy_fill_alpha, img, 1.0 - cfg.canopy_fill_alpha, 0.0)
    cv2.polylines(out, rings, True, colour, OUTLINE_THICKNESS, cv2.LINE_8, SHIFT)
    return out


def _label(img: np.ndarray, text: str, at: tuple[int, int], colour: tuple[int, int, int], scale: float) -> None:
    cv2.putText(img, text, at, FONT, scale, SHADOW_BGR, 2, cv2.LINE_AA)
    cv2.putText(img, text, at, FONT, scale, colour, 1, cv2.LINE_AA)


def _draw_rejected(img: np.ndarray, tile: TileRef, lines: Sequence[LineString], scale: float, cfg: QaConfig) -> None:
    pts = [_fixed(tile, np.asarray(g.coords), scale) for g in lines if g is not None and not g.is_empty]
    if pts:
        cv2.polylines(img, pts, False, _colour(cfg, "rejected"), OUTLINE_THICKNESS, cv2.LINE_8, SHIFT)


def _draw_rows(img: np.ndarray, tile: TileRef, rows: Sequence[tuple[LineString, str, str]], scale: float,
               cfg: QaConfig) -> None:
    for line, row_id, structure in rows:
        if line is None or line.is_empty:
            continue
        colour = _colour(cfg, STRUCTURE_COLOUR_KEY.get(structure, "row"))
        pts = _fixed(tile, np.asarray(line.coords), scale)
        cv2.polylines(img, [pts], False, colour, ROW_THICKNESS, cv2.LINE_8, SHIFT)
        anchor = _fixed(tile, np.asarray(line.interpolate(LABEL_AT, normalized=True).coords), scale)[0, 0]
        x, y = (int(v) >> SHIFT for v in anchor)
        _label(img, row_id.rsplit("-", 1)[-1], (x + 3, y - 3), colour, LABEL_FONT_SCALE)


def _draw_header(img: np.ndarray, header: PreviewHeader) -> np.ndarray:
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (img.shape[1] - 1, HEADER_PX), SHADOW_BGR, thickness=-1)
    out = cv2.addWeighted(bar, HEADER_ALPHA, img, 1.0 - HEADER_ALPHA, 0.0)
    cv2.putText(out, header.text(), (_HEADER_MARGIN_PX, _HEADER_BASELINE_PX), FONT, HEADER_FONT_SCALE, TEXT_BGR, 1,
                cv2.LINE_AA)
    return out


def render_preview(rgb: U8, tile: TileRef, objs: TileObjects, header: PreviewHeader, cfg: QaConfig) -> U8:
    """New BGR uint8 image of qa.preview_px² (the input array is not modified)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"render_preview expects (H, W, 3) uint8 RGB, got {rgb.shape} {rgb.dtype}")
    size = cfg.preview_px
    scale = size / TILE_PX
    base = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (size, size), interpolation=cv2.INTER_AREA)
    _draw_rejected(base, tile, objs.rejected, scale, cfg)
    img = _draw_canopies(base, _rings(tile, objs.canopies, scale), cfg)
    cv2.polylines(img, _rings(tile, objs.interrows, scale), True, _colour(cfg, "interrow"), OUTLINE_THICKNESS,
                  cv2.LINE_8, SHIFT)
    cv2.polylines(img, _rings(tile, objs.waste, scale), True, _colour(cfg, "waste"), ROW_THICKNESS, cv2.LINE_8,
                  SHIFT)
    _draw_rows(img, tile, objs.rows, scale, cfg)
    return _draw_header(img, header)


def encode_jpeg(img: U8, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError(f"JPEG encoding failed for an image of shape {img.shape}")
    return buf.tobytes()


def write_preview(path: Path, img: U8, quality: int) -> Path:
    """Write a BGR image as JPEG, atomically."""
    return atomic_write_bytes(Path(path), encode_jpeg(img, quality))
