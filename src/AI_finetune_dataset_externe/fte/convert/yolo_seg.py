"""YOLO-seg label files -> per-instance polygons (pixel coords) -> uint16 instance map.

One line per instance: ``<class> x1 y1 x2 y2 ...`` with coordinates normalised to [0, 1] by the image
width / height. Instance ids in the map are 1..N in file order (0 = background); where polygons
overlap, the later polygon wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_POINTS = 3
MAX_INSTANCES = int(np.iinfo(np.uint16).max)


@dataclass(frozen=True, eq=False)
class YoloPolygon:
    cls: int
    xy: np.ndarray  # (N, 2) float64 pixel coords (x, y), continuous convention


def parse_yolo_seg(text: str, width: int, height: int) -> tuple[YoloPolygon, ...]:
    """Parse the text of one YOLO-seg label file; malformed lines raise ValueError with the line number."""
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    out = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < 1 + 2 * MIN_POINTS or (len(parts) - 1) % 2:
            raise ValueError(f"line {lineno}: expected '<cls> x1 y1 ... xn yn' with n >= 3, got {len(parts)} fields")
        try:
            cls = int(float(parts[0]))
            coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float64).reshape(-1, 2)
        except ValueError as exc:
            raise ValueError(f"line {lineno}: non-numeric value ({exc})") from exc
        xy = np.clip(coords, 0.0, 1.0) * np.array([width, height], dtype=np.float64)
        out.append(YoloPolygon(cls, xy))
    return tuple(out)


def read_yolo_seg(path: Path, width: int, height: int) -> tuple[YoloPolygon, ...]:
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise ValueError(f"cannot read YOLO label file {path}: {exc}") from exc
    try:
        return parse_yolo_seg(text, width, height)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def rasterize_instances(polys: tuple[YoloPolygon, ...], shape_hw: tuple[int, int]) -> np.ndarray:
    """uint16 instance map of ``shape_hw``: polygon k (0-based) gets id k + 1."""
    if len(polys) > MAX_INSTANCES:
        raise ValueError(f"{len(polys)} instances exceed the uint16 id range")
    # cv2.fillPoly has no uint16 support for all builds -> burn into int32 and cast.
    out = np.zeros(shape_hw, dtype=np.int32)
    for k, poly in enumerate(polys):
        # pixel centres at +0.5: shift continuous coords to index space before rounding
        ring = np.round(poly.xy - 0.5).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(out, [ring], k + 1)
    return out.astype(np.uint16)
