"""Geometric row order per block: R001 = max n·c with the canonical block normal (contract §1.5).

The block angle is the axial median of its row angles. Spacing to the neighbours is measured on the
rows' common extent, so rows of different lengths or slightly fanned rows are not biased.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from shapely.geometry import Point

from vineyard.annset.lines import along_range, axial_median_deg, direction_of_angle, points_at
from vineyard.annset.merge_rows import MergedRow
from vineyard.contracts.ordering import canonical_normal

SPACING_SAMPLES: Final = 3  # overlap start, middle and end
# n·c values closer than this are ties (broken by row_id), so float noise never decides the order.
ORDER_TIE_M: Final = 1e-3


@dataclass(frozen=True)
class BlockAxis:
    vineyard_id: str
    angle_deg: float
    normal: tuple[float, float]

    @property
    def direction(self) -> np.ndarray:
        return direction_of_angle(self.angle_deg)


@dataclass(frozen=True)
class OrderedRow:
    row: MergedRow
    row_index: int
    spacing_prev_m: float  # NaN on the first row of the block
    spacing_next_m: float  # NaN on the last row of the block


def block_axes(rows: Sequence[MergedRow]) -> Mapping[str, BlockAxis]:
    """vineyard_id -> axis (axial median of the block's row angles), sorted by vineyard_id."""
    angles: dict[str, list[float]] = {}
    for row in rows:
        angles.setdefault(row.vineyard_id, []).append(row.angle_deg)
    axes = {}
    for vid in sorted(angles):
        theta = axial_median_deg(angles[vid])
        axes[vid] = BlockAxis(vid, theta, canonical_normal(theta))
    return MappingProxyType(axes)


def _centroid(row: MergedRow) -> np.ndarray:
    c = row.line.centroid
    return np.array([c.x, c.y])


def row_spacing(a: MergedRow, b: MergedRow, axis: BlockAxis) -> float:
    """Median distance from `a` to `b` over their common extent; |n·Δc| when they do not overlap."""
    (a0, a1), (b0, b1) = along_range(a.line, axis.direction), along_range(b.line, axis.direction)
    lo, hi = max(a0, b0), min(a1, b1)
    if hi <= lo:
        return abs(float(np.dot(_centroid(a) - _centroid(b), axis.normal)))
    samples = points_at(a.line, axis.direction, np.linspace(lo, hi, SPACING_SAMPLES))
    return float(np.median([b.line.distance(Point(p)) for p in samples]))


def _block_order(members: Sequence[MergedRow], axis: BlockAxis) -> list[MergedRow]:
    def key(row: MergedRow) -> tuple[int, str]:
        return (-round(float(np.dot(_centroid(row), axis.normal)) / ORDER_TIE_M), row.row_id)

    return sorted(members, key=key)


def order_rows(rows: Sequence[MergedRow]) -> tuple[OrderedRow, ...]:
    """Rows with row_index 1..n per block and neighbour spacings, sorted by (vineyard_id, row_index)."""
    out: list[OrderedRow] = []
    for vid, axis in block_axes(rows).items():
        seq = _block_order([r for r in rows if r.vineyard_id == vid], axis)
        gaps = [row_spacing(a, b, axis) for a, b in pairwise(seq)]
        for k, row in enumerate(seq):
            prev = gaps[k - 1] if k > 0 else math.nan
            nxt = gaps[k] if k < len(gaps) else math.nan
            out.append(OrderedRow(row, k + 1, prev, nxt))
    return tuple(out)


def row_record(ordered: OrderedRow, *, plant_count: int = 0) -> dict[str, Any]:
    """`rows` layer record (contract §2.5.5; max_gap_m / n_gaps_ge5 come from the gap engine, not here)."""
    row = ordered.row
    return {
        "row_id": row.row_id, "vineyard_id": row.vineyard_id, "row_index": ordered.row_index,
        "length_m": row.length_m, "extent_m": row.extent_m, "n_pieces": row.n_pieces,
        "tile_ids": ",".join(row.tile_ids), "angle_deg": row.angle_deg,
        "spacing_prev_m": ordered.spacing_prev_m, "spacing_next_m": ordered.spacing_next_m,
        "max_gap_m": math.nan, "n_gaps_ge5": None, "structure_any": row.structure_any,
        "tile_structures": json.dumps(dict(row.tile_structures), sort_keys=True),
        "plant_count": int(plant_count), "geometry": row.line,
    }
