"""Cross-package perception types (tile_prep cache, vegetation masks) and array aliases.

Types used inside a single package stay in that package; only shared ones live here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from vineyard.contracts.enums import TileStatus

type BoolMask = NDArray[np.bool_]
type U8 = NDArray[np.uint8]
type F32 = NDArray[np.float32]
type I32 = NDArray[np.int32]

# Visibility codes stored in cache/vis/<tile>.png (uint8).
VIS_OK: Final = 0
VIS_SHADOW: Final = 1
VIS_OVEREXPOSED: Final = 2
VIS_NODATA: Final = 3


def readonly(arr: np.ndarray) -> np.ndarray:
    """Read-only view of `arr`; the input keeps its own flags."""
    view = arr.view()
    view.flags.writeable = False
    return view


def _check_frac(name: str, value: float) -> None:
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be a fraction in [0, 1], got {value}")


@dataclass(frozen=True, eq=False)
class TileMasks:
    """Per-tile rasters from tile_prep. veg: bool; vis: uint8 VIS_* codes; arrays are read-only."""

    veg: BoolMask
    vis: U8
    method: str
    threshold: float
    veg_frac: float  # veg pixels / valid pixels

    def __post_init__(self) -> None:
        if self.veg.dtype != np.bool_ or self.veg.ndim != 2:
            raise ValueError(f"veg must be a 2-D bool array, got {self.veg.shape} {self.veg.dtype}")
        if self.vis.dtype != np.uint8 or self.vis.shape != self.veg.shape:
            raise ValueError(f"vis must be uint8 with shape {self.veg.shape}, got {self.vis.shape} {self.vis.dtype}")
        _check_frac("veg_frac", self.veg_frac)
        object.__setattr__(self, "veg", readonly(self.veg))
        object.__setattr__(self, "vis", readonly(self.vis))


@dataclass(frozen=True)
class TileStats:
    """Scalar tile_prep results, stored as cache/stats/<tile>.json."""

    tile_id: str
    valid_frac: float
    nodata_frac: float
    veg_frac: float
    status: TileStatus
    veg_threshold: float
    veg_method: str

    def __post_init__(self) -> None:
        if not self.tile_id:
            raise ValueError("tile_id must be non-empty")
        for name in ("valid_frac", "nodata_frac", "veg_frac"):
            _check_frac(name, getattr(self, name))
        if not isinstance(self.status, TileStatus):
            raise ValueError(f"status must be a TileStatus, got {self.status!r}")
        if not math.isfinite(self.veg_threshold):
            raise ValueError(f"veg_threshold must be finite, got {self.veg_threshold}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping (status as its string value)."""
        return {**asdict(self), "status": self.status.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TileStats:
        """Inverse of to_dict; the key set must match exactly."""
        expected = {f.name for f in fields(cls)}
        missing, extra = expected - set(data), set(data) - expected
        if missing:
            raise ValueError(f"TileStats: missing keys {sorted(missing)}")
        if extra:
            raise ValueError(f"TileStats: unexpected keys {sorted(extra)}")
        try:
            status = TileStatus(data["status"])
        except ValueError as exc:
            raise ValueError(f"TileStats: invalid status {data['status']!r}") from exc
        return cls(
            tile_id=str(data["tile_id"]),
            valid_frac=float(data["valid_frac"]),
            nodata_frac=float(data["nodata_frac"]),
            veg_frac=float(data["veg_frac"]),
            status=status,
            veg_threshold=float(data["veg_threshold"]),
            veg_method=str(data["veg_method"]),
        )
