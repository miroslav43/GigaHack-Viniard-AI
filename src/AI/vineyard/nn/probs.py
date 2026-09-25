"""NN probability rasters (contract §6), torch-free.

Per tile: single-channel uint8 PNG (p*255), 1024², aligned to the UL corner; 1024-px k covers
CVAT u ∈ [2k, 2k+2). Upsampling uses cv2.INTER_LINEAR, whose half-pixel-centre convention keeps
integer factors aligned.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from vineyard.errors import SchemaError
from vineyard.pipeline.atomic import atomic_write_bytes

PROB_PX: Final = 1024
PROB_SCALE: Final = 255.0
CHANNELS: Final = ("canopy_prob", "axis_prob")
NN_CACHE_DIR: Final = "nn"
PROB_P95: Final = 95.0

__all__ = [
    "CHANNELS",
    "PROB_PX",
    "ProbStats",
    "corridor_prob_stats",
    "load_canopy_prob",
    "prob_png_path",
    "read_prob_png",
    "upsample_prob",
    "write_prob_png",
]


@dataclass(frozen=True)
class ProbStats:
    mean: float
    p95: float
    n_px: int


def prob_png_path(cache_root: Path, nn_version: str, channel: str, tile_id: str) -> Path:
    """`<cache_root>/nn/<nn_version>/<channel>/<tile_id>.png` with cache_root = work/cache."""
    if channel not in CHANNELS:
        raise SchemaError("unknown NN channel", channel=channel, tile_id=tile_id)
    return Path(cache_root) / NN_CACHE_DIR / nn_version / channel / f"{tile_id}.png"


def write_prob_png(path: Path, prob: np.ndarray) -> None:
    """float probabilities (1024, 1024) in [0, 1] → uint8 rint(p*255) PNG, written atomically."""
    arr = np.asarray(prob, dtype=np.float32)
    if arr.shape != (PROB_PX, PROB_PX):
        raise SchemaError("probability raster has wrong shape", path=str(path), shape=arr.shape)
    if not np.isfinite(arr).all() or arr.min(initial=0.0) < 0.0 or arr.max(initial=0.0) > 1.0:
        raise SchemaError("probabilities must be finite and in [0, 1]", path=str(path))
    ok, encoded = cv2.imencode(".png", np.rint(arr * PROB_SCALE).astype(np.uint8))
    if not ok:
        raise SchemaError("PNG encoding failed", path=str(path))
    atomic_write_bytes(Path(path), encoded.tobytes())


def read_prob_png(path: Path) -> np.ndarray:
    """uint8 (1024, 1024) probability raster; SchemaError on unreadable data or wrong shape/dtype."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    arr = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED) if raw.size else None
    if arr is None:
        raise SchemaError("unreadable probability PNG", path=str(path))
    if arr.dtype != np.uint8 or arr.shape != (PROB_PX, PROB_PX):
        raise SchemaError("probability PNG has wrong shape/dtype", path=str(path), shape=arr.shape,
                          dtype=str(arr.dtype))
    return arr


def load_canopy_prob(path: Path) -> np.ndarray | None:
    """uint8 canopy probability, or None when the file is absent (NN disabled or no weights)."""
    return read_prob_png(path) if Path(path).is_file() else None


def upsample_prob(prob_1024: np.ndarray | None, size_px: int) -> np.ndarray | None:
    """uint8 square raster → float32 (size_px, size_px) in [0, 1]; None passes through."""
    if prob_1024 is None:
        return None
    if prob_1024.dtype != np.uint8 or prob_1024.ndim != 2 or prob_1024.shape[0] != prob_1024.shape[1]:
        raise SchemaError("upsample_prob expects a square uint8 raster", shape=prob_1024.shape,
                          dtype=str(prob_1024.dtype))
    scaled = prob_1024.astype(np.float32) / np.float32(PROB_SCALE)
    # Interpolate in float so the upsample adds no second uint8 quantisation.
    return cv2.resize(scaled, (size_px, size_px), interpolation=cv2.INTER_LINEAR)


def corridor_prob_stats(prob: np.ndarray, corridor: np.ndarray) -> ProbStats:
    """Mean (rows_detect vine_score) and p95 (tile_status vineyard_score) of prob inside the corridor."""
    if prob.shape != corridor.shape:
        raise SchemaError("prob and corridor shapes differ", prob=prob.shape, corridor=corridor.shape)
    values = np.asarray(prob, dtype=np.float64)[np.asarray(corridor, dtype=bool)]
    if values.size == 0:
        return ProbStats(mean=float("nan"), p95=float("nan"), n_px=0)
    return ProbStats(mean=float(values.mean()), p95=float(np.percentile(values, PROB_P95)), n_px=int(values.size))
