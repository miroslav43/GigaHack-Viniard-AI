"""Waste value types (design 03 §2): boxes in CVAT continuous px, candidates, scores, decisions.

All types are frozen; filters and decisions return new instances (dataclasses.replace).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from shapely.geometry import Polygon, box

TILE_EXTENT_PX: Final = 2048.0


class ColourClass(StrEnum):
    VIVID = "vivid"  # saturated, non-vegetation hue
    BRIGHT = "bright"  # very bright, unsaturated (white)


class RejectReason(StrEnum):
    NEAR_AXIS = "near_axis"
    PERIODIC_TUBE = "periodic_tube"
    TOO_SMALL_WHITE = "too_small_white"
    BRIGHT_SOIL = "bright_soil"  # pale soil / dust highlight: warm hue, saturation near the bright limit
    TUBE_SHAPE = "tube_shape"
    HOSE = "hose"
    FORBIDDEN = "forbidden"
    VEHICLE = "vehicle"
    NMS = "nms"


class Category(StrEnum):
    BAG = "bag"
    BOTTLE = "bottle"
    TYRE = "tyre"
    DEBRIS = "debris"
    HEAP = "heap"
    UNKNOWN = "unknown"


class Detector(StrEnum):
    SAM3 = "sam3"
    NN = "nn"
    RULE = "rule"
    MANUAL = "manual"


Decision_ = Literal["accept", "reject", "add"]
DECISIONS: Final[tuple[str, ...]] = ("accept", "reject", "add")


@dataclass(frozen=True)
class BoxPx:
    """Axis-aligned box in CVAT continuous tile px; xtl < xbr, ytl < ybr, inside [0, 2048]."""

    xtl: float
    ytl: float
    xbr: float
    ybr: float

    def __post_init__(self) -> None:
        values = (self.xtl, self.ytl, self.xbr, self.ybr)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"box has non-finite coordinates: {values}")
        if not (self.xtl < self.xbr and self.ytl < self.ybr):
            raise ValueError(f"box must have xtl < xbr and ytl < ybr: {values}")
        if min(values) < 0.0 or max(values) > TILE_EXTENT_PX:
            raise ValueError(f"box outside [0, {TILE_EXTENT_PX}]: {values}")

    @property
    def width(self) -> float:
        return self.xbr - self.xtl

    @property
    def height(self) -> float:
        return self.ybr - self.ytl

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.xtl + self.xbr) / 2.0, (self.ytl + self.ybr) / 2.0)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.xtl, self.ytl, self.xbr, self.ybr)

    def polygon(self) -> Polygon:
        return box(self.xtl, self.ytl, self.xbr, self.ybr)

    def iou(self, other: BoxPx) -> float:
        ix = max(0.0, min(self.xbr, other.xbr) - max(self.xtl, other.xtl))
        iy = max(0.0, min(self.ybr, other.ybr) - max(self.ytl, other.ytl))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Candidate:
    tile_id: str
    cand_key: str
    box: BoxPx
    centroid_px: tuple[float, float]
    area_px: int
    area_m2: float
    colour_class: ColourClass
    is_white: bool
    aspect: float
    length_m: float
    width_m: float
    mean_hsv: tuple[float, float, float]
    reject_reason: RejectReason | None = None

    @property
    def rejected(self) -> bool:
        return self.reject_reason is not None


@dataclass(frozen=True)
class CandidateScores:
    cand_key: str
    probe_p: float | None
    clip_pos_p: float | None
    clip_margin: float | None
    sam_score: float | None
    category: Category
    rule_score: float | None = None


@dataclass(frozen=True)
class Decision:
    cand_key: str
    auto: bool
    review: bool
    rank_score: float
    detector: Detector


@dataclass(frozen=True)
class Confirmation:
    line_no: int
    tile_id: str
    box: BoxPx
    decision: Decision_
    category: Category
    reviewer: str
    note: str


@dataclass(frozen=True)
class BlockAssignment:
    vineyard_id: str
    dist_block_m: float


# ------------------------------------------------------------------ flat records (tile cache rows)

RECORD_COLUMNS: Final[tuple[str, ...]] = (
    "cand_key",
    "tile_id",
    "xtl",
    "ytl",
    "xbr",
    "ybr",
    "cx",
    "cy",
    "area_px",
    "area_m2",
    "colour_class",
    "is_white",
    "aspect",
    "length_m",
    "width_m",
    "mean_h",
    "mean_s",
    "mean_v",
    "reject_reason",
)


def candidate_to_record(c: Candidate) -> dict[str, object]:
    """Flat, parquet-friendly dict of a candidate (column order = RECORD_COLUMNS)."""
    return {
        "cand_key": c.cand_key,
        "tile_id": c.tile_id,
        "xtl": c.box.xtl,
        "ytl": c.box.ytl,
        "xbr": c.box.xbr,
        "ybr": c.box.ybr,
        "cx": c.centroid_px[0],
        "cy": c.centroid_px[1],
        "area_px": c.area_px,
        "area_m2": c.area_m2,
        "colour_class": c.colour_class.value,
        "is_white": c.is_white,
        "aspect": c.aspect,
        "length_m": c.length_m,
        "width_m": c.width_m,
        "mean_h": c.mean_hsv[0],
        "mean_s": c.mean_hsv[1],
        "mean_v": c.mean_hsv[2],
        "reject_reason": None if c.reject_reason is None else c.reject_reason.value,
    }


def candidate_from_record(r: dict[str, object]) -> Candidate:
    """Inverse of candidate_to_record (a missing / NaN / None reason means not rejected)."""
    reason = r.get("reject_reason")
    rejected = isinstance(reason, str) and reason != ""
    return Candidate(
        tile_id=str(r["tile_id"]),
        cand_key=str(r["cand_key"]),
        box=BoxPx(float(r["xtl"]), float(r["ytl"]), float(r["xbr"]), float(r["ybr"])),
        centroid_px=(float(r["cx"]), float(r["cy"])),
        area_px=int(r["area_px"]),
        area_m2=float(r["area_m2"]),
        colour_class=ColourClass(str(r["colour_class"])),
        is_white=bool(r["is_white"]),
        aspect=float(r["aspect"]),
        length_m=float(r["length_m"]),
        width_m=float(r["width_m"]),
        mean_hsv=(float(r["mean_h"]), float(r["mean_s"]), float(r["mean_v"])),
        reject_reason=RejectReason(reason) if rejected else None,
    )
