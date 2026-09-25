"""Test-only helpers for the targets / measure / publish tests (POST-TGT).

- `reference_gap_fn`: a pure-python gap engine (canopies clipped to the flat-capped corridor, vertices
  projected on the axis, intervals merged). It stands in for `vineyard.perception.attrs` in tests and
  reproduces the documented reference gaps (r006: R06 5.054; R08 7.09/9.25/7.745; R09 12.683/7.812/12.903;
  R15 5.009).
- `rows_from_pieces`: a minimal `rows` layer (one merged line per row_id, row_index by n·c) so the
  targets tests do not depend on the derive stage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from tests.post.factories import Prov, layer_frame
from vineyard.annset.model import AnnSet
from vineyard.contracts.ordering import angle_deg_utm, mean_axial_angle_deg, order_by_normal
from vineyard.geo.ops import drop_consecutive_duplicates
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.route.target_gaps import RowGap, along_interval, merge_intervals, unknown_intervals

CORRIDOR_HALF_M: Final = 0.30
EPS_M: Final = 1e-9


def _occupied(axis: LineString, canopies: Sequence[Polygon], half_m: float) -> tuple[tuple[float, float], ...]:
    corridor = axis.buffer(half_m, cap_style="flat")
    spans = [along_interval(axis, c.intersection(corridor)) for c in canopies if c.intersects(corridor)]
    return merge_intervals([s for s in spans if s is not None])


def _free_intervals(length: float, blocked: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    free, cursor = [], 0.0
    for lo, hi in merge_intervals(blocked):
        if lo > cursor:
            free.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < length:
        free.append((cursor, length))
    return free


def _kind(lo: float, hi: float, length: float) -> str:
    """By position, like the pixel engine: a run touching the axis start is head, the end tail, both full."""
    head, tail = lo <= EPS_M, hi >= length - EPS_M
    if head and tail:
        return "full"
    return "head" if head else "tail" if tail else "interior"


def _touches(lo: float, hi: float, spans: Sequence[tuple[float, float]]) -> bool:
    return any(abs(lo - s_hi) < 1e-6 or abs(hi - s_lo) < 1e-6 for s_lo, s_hi in spans)


def make_reference_gap_fn(half_m: float = CORRIDOR_HALF_M):
    def gap_fn(axis: LineString, canopies: Sequence[Polygon], unknown: BaseGeometry | None,
               min_length_m: float) -> tuple[RowGap, ...]:
        occupied = _occupied(axis, canopies, half_m)
        unknown_iv = unknown_intervals(axis, unknown)
        gaps = []
        for lo, hi in _free_intervals(axis.length, [*occupied, *unknown_iv]):
            if hi - lo >= min_length_m and hi > lo:
                gaps.append(RowGap(lo, hi, _kind(lo, hi, axis.length), _touches(lo, hi, unknown_iv)))  # type: ignore[arg-type]
        return tuple(gaps)

    return gap_fn


reference_gap_fn = make_reference_gap_fn()


# ------------------------------------------------------------------ rows layer from pieces


def _merged_line(lines: Sequence[LineString]) -> LineString:
    coords = np.vstack([np.asarray(line.coords) for line in lines])
    centred = coords - coords.mean(axis=0)
    direction = np.linalg.svd(centred, full_matrices=False)[2][0]
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction
    order = np.argsort(centred @ direction, kind="stable")
    return LineString(drop_consecutive_duplicates(coords[order], closed=False))


def _row_record(row_id: str, pieces: gpd.GeoDataFrame) -> dict[str, Any]:
    line = _merged_line(list(pieces.geometry))
    coords = np.asarray(line.coords)
    return {"row_id": row_id, "vineyard_id": pieces["vineyard_id"].mode().sort_values().iloc[0],
            "length_m": float(pieces.geometry.length.sum()), "extent_m": line.length,
            "n_pieces": len(pieces), "tile_ids": ",".join(sorted(pieces["tile_id"].unique())),
            "angle_deg": angle_deg_utm(tuple(coords[0]), tuple(coords[-1])), "spacing_prev_m": math.nan,
            "spacing_next_m": math.nan, "max_gap_m": math.nan, "geometry": line}


def _with_row_index(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for vid in sorted({r["vineyard_id"] for r in records}):
        block = [r for r in records if r["vineyard_id"] == vid]
        angle = mean_axial_angle_deg([r["angle_deg"] for r in block])
        centroids = np.array([[r["geometry"].centroid.x, r["geometry"].centroid.y] for r in block])
        for rank, i in enumerate(order_by_normal(angle, centroids), start=1):
            out.append(block[int(i)] | {"row_index": rank})
    return out


def rows_from_pieces(annset: AnnSet, prov: Prov | None = None) -> gpd.GeoDataFrame:
    """Contract `rows` layer: one merged line per row_id, row_index by n·c descending per block."""
    pieces = annset.row_pieces
    records = [_row_record(rid, grp) for rid, grp in pieces.groupby("row_id", sort=True)]
    used = prov or Prov(annset.meta.source, annset.meta.run_id, annset.meta.model_version)
    return layer_frame("rows", _with_row_index(records), used)


def tile_coverage(annset: AnnSet) -> BaseGeometry:
    """Union of the AnnSet's tile squares (no nodata)."""
    return shapely.union_all([tile_box(tile_ref(t)) for t in sorted(annset.tile_ids())])


def tile_valid_frame(tile_ids: Sequence[str], holes: dict[str, BaseGeometry] | None = None) -> gpd.GeoDataFrame:
    """A `tile_valid` layer: full tile squares minus optional nodata `holes` per tile."""
    geoms = [tile_box(tile_ref(t)).difference((holes or {}).get(t, Polygon())) for t in tile_ids]
    frame = gpd.GeoDataFrame({"tile_id": list(tile_ids), "valid_frac": np.ones(len(tile_ids), np.float32)},
                             geometry=geoms, crs="EPSG:32635")
    return frame.assign(valid_frac=pd.Series([g.area / tile_box(tile_ref(t)).area for t, g in
                                              zip(tile_ids, geoms, strict=True)], dtype=np.float32))
