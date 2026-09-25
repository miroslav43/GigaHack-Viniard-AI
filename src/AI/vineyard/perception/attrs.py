"""Row structure: THE single gap engine (plan S2), shared by row_attrs, targets and imports.

Gaps are measured on canopy PIXEL sets (index-convention rasterization, the exact inverse of raw
findContours) projected on the row axis in 1-px bins inside the +-half corridor, piece ends included.
Bins over nodata are `unknown`: they are neither canopy nor gap, and the gaps next to them are
`censored`. This reproduces all 51 reference row_structure labels (R15 4.975 m, R06 5.025 m).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from vineyard.config import RowStructureConfig
from vineyard.contracts.enums import RowStructure, Source
from vineyard.contracts.ids import format_row_piece_id
from vineyard.contracts.schemas import coerce_layer, empty_layer
from vineyard.geo.ops import drop_consecutive_duplicates
from vineyard.geo.raster import rasterize_utm
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TileRef, utm_to_px
from vineyard.perception.types import F32, U8, VIS_NODATA, VIS_OK, BoolMask, readonly

BIN_PX: Final = 1.0  # S2: 1-px bins on the canopy pixel set
BORDERLINE_CODE: Final = "structure_borderline"
QA_FLAG_SEP: Final = ";"
ROW_PIECES_LAYER: Final = "row_pieces"
DEFAULT_CONFIDENCE: Final = 1.0
_ACROSS_STEP_PX: Final = 1.0
_REQUIRED_PIECE_COLUMNS: Final = ("row_id", "vineyard_id")
GAP_COLUMNS: Final = ("piece_id", "row_id", "vineyard_id", "tile_id", "start_m", "end_m", "length_m", "kind",
                      "censored")


class GapKind(StrEnum):
    INTERIOR = "interior"
    HEAD = "head"  # from the piece start to the first canopy
    TAIL = "tail"  # from the last canopy to the piece end
    FULL = "full"  # the piece has no canopy at all


@dataclass(frozen=True)
class Gap:
    """A run of empty known bins, from `start_m` to `end_m` along the axis (UTM metres)."""

    start_m: float
    end_m: float
    kind: GapKind
    censored: bool  # next to an unknown (nodata) span, so the true gap may be longer

    def __post_init__(self) -> None:
        if not (0.0 <= self.start_m <= self.end_m):
            raise ValueError(f"invalid gap interval start={self.start_m} end={self.end_m}")
        object.__setattr__(self, "kind", GapKind(self.kind))

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


@dataclass(frozen=True, eq=False)
class OccupancyProfile:
    """Per-bin state along one axis: canopy cover fraction, occupied, unknown (nodata), hidden."""

    bin_m: float
    length_m: float
    cover: F32
    occupied: BoolMask
    unknown: BoolMask
    hidden: BoolMask

    def __post_init__(self) -> None:
        n = len(self.cover)
        if n == 0 or any(len(a) != n for a in (self.occupied, self.unknown, self.hidden)):
            raise ValueError(f"profile arrays must be non-empty and equally long, got {n}")
        for name in ("cover", "occupied", "unknown", "hidden"):
            object.__setattr__(self, name, readonly(np.asarray(getattr(self, name))))

    @property
    def n_bins(self) -> int:
        return len(self.cover)

    @property
    def visible_frac(self) -> float:
        return float(1.0 - self.hidden.mean())


@dataclass(frozen=True)
class StructureResult:
    row_structure: RowStructure
    max_gap_m: float
    visible_frac: float
    gaps: tuple[Gap, ...]
    flags: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class RowAttrsResult:
    row_pieces: gpd.GeoDataFrame  # contract `row_pieces`
    gaps: gpd.GeoDataFrame  # internal: GAP_COLUMNS + LineString along the axis


@dataclass(frozen=True, eq=False)
class _AxisFrame:
    starts: np.ndarray  # (S, 2) segment start points, px
    dirs: np.ndarray  # (S, 2) unit directions
    lengths: np.ndarray  # (S,)
    cum: np.ndarray  # (S,) along position of each segment start

    @property
    def total(self) -> float:
        return float(self.cum[-1] + self.lengths[-1])


# ------------------------------------------------------------------ geometry inputs


def axis_to_px(axis_utm: LineString, tile: TileRef) -> np.ndarray:
    """(N, 2) continuous px vertices of a UTM axis, without repeated vertices."""
    coords = np.asarray(axis_utm.coords, dtype=np.float64)[:, :2] if not axis_utm.is_empty else np.zeros((0, 2))
    pts = drop_consecutive_duplicates(utm_to_px(tile, coords), closed=False) if len(coords) else coords
    if len(pts) < 2:
        raise ValueError(f"row axis needs >= 2 distinct vertices, tile={tile.tile_id}")
    return pts


def canopy_pixels(canopies: Sequence[BaseGeometry], tile: TileRef) -> np.ndarray:
    """(M, 2) float (i, j) indices of the canopy pixel set (index convention, boundary included)."""
    polys = [g for g in canopies if g is not None and not g.is_empty]
    if not polys:
        return np.zeros((0, 2), dtype=np.float64)
    ys, xs = np.nonzero(rasterize_utm(polys, tile, convention="index"))
    return np.column_stack((xs, ys)).astype(np.float64)


def _axis_frame(axis_px: np.ndarray) -> _AxisFrame:
    deltas = np.diff(axis_px, axis=0)
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    keep = lengths > 0
    if not keep.any():
        raise ValueError("row axis has zero length")
    lengths, deltas, starts = lengths[keep], deltas[keep], axis_px[:-1][keep]
    cum = np.concatenate(([0.0], np.cumsum(lengths)[:-1]))
    return _AxisFrame(starts, deltas / lengths[:, None], lengths, cum)


def _normal(dirs: np.ndarray) -> np.ndarray:
    return np.column_stack((-dirs[:, 1], dirs[:, 0]))


def _project(frame: _AxisFrame, pts: np.ndarray, half_px: float) -> np.ndarray:
    """Along positions of the points inside the flat-capped corridor of any segment."""
    if len(pts) == 0:
        return np.zeros(0)
    lo = np.minimum(frame.starts.min(axis=0), (frame.starts + frame.dirs * frame.lengths[:, None]).min(axis=0))
    hi = np.maximum(frame.starts.max(axis=0), (frame.starts + frame.dirs * frame.lengths[:, None]).max(axis=0))
    near = pts[np.all((pts >= lo - half_px) & (pts <= hi + half_px), axis=1)]
    normals = _normal(frame.dirs)
    alongs = []
    for s in range(len(frame.lengths)):
        rel = near - frame.starts[s]
        t, o = rel @ frame.dirs[s], rel @ normals[s]
        sel = (np.abs(o) < half_px) & (t >= 0.0) & (t <= frame.lengths[s])
        alongs.append(frame.cum[s] + t[sel])
    return np.concatenate(alongs)


def _points_at(frame: _AxisFrame, along: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg = np.clip(np.searchsorted(frame.cum, along, side="right") - 1, 0, len(frame.cum) - 1)
    base = frame.starts[seg] + frame.dirs[seg] * (along - frame.cum[seg])[:, None]
    return base, _normal(frame.dirs)[seg]


def _bin_vis_fractions(frame: _AxisFrame, n_bins: int, bin_px: float, half_px: float,
                       vis: U8) -> tuple[np.ndarray, np.ndarray]:
    """(nodata fraction, hidden fraction) of each bin's corridor strip, sampled every 1 px across."""
    centres = np.minimum((np.arange(n_bins) + 0.5) * bin_px, frame.total)
    offsets = np.arange(-half_px + _ACROSS_STEP_PX / 2, half_px, _ACROSS_STEP_PX)
    base, normals = _points_at(frame, centres)
    pts = base[:, None, :] + offsets[None, :, None] * normals[:, None, :]
    h, w = vis.shape
    i = np.clip(np.rint(pts[..., 0]), 0, w - 1).astype(np.intp)
    j = np.clip(np.rint(pts[..., 1]), 0, h - 1).astype(np.intp)
    codes = vis[j, i]
    return (codes == VIS_NODATA).mean(axis=1), (codes != VIS_OK).mean(axis=1)


# ------------------------------------------------------------------ profile and gaps


def occupancy_profile(axis_px: np.ndarray, canopy_ij: np.ndarray, *, half_px: float, vis: U8 | None = None,
                      max_hidden: float = 0.5, bin_px: float = BIN_PX) -> OccupancyProfile:
    """Bin the canopy pixels inside the corridor along the axis; ends included (int(L)+1 bins)."""
    frame = _axis_frame(np.asarray(axis_px, dtype=np.float64))
    n_bins = int(frame.total / bin_px) + 1
    along = _project(frame, np.asarray(canopy_ij, dtype=np.float64).reshape(-1, 2), half_px)
    idx = np.clip((along / bin_px).astype(np.intp), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    occupied = counts > 0
    cover = np.minimum(counts / (2.0 * half_px * bin_px), 1.0).astype(np.float32)
    if vis is None:
        unknown = hidden = np.zeros(n_bins, dtype=bool)
    else:
        nodata_frac, hidden_frac = _bin_vis_fractions(frame, n_bins, bin_px, half_px, vis)
        unknown = (nodata_frac > max_hidden) & ~occupied
        hidden = (hidden_frac > max_hidden) & ~occupied
    return OccupancyProfile(bin_px * GSD_M, frame.total * GSD_M, cover, occupied, unknown, hidden)


def smooth_profile(profile: OccupancyProfile, cfg: RowStructureConfig) -> OccupancyProfile:
    """Optional model-canopy smoothing: window mean >= occ_min, then a 1-D closing of occ_close_m."""
    window = max(1, round(cfg.occ_window_m / profile.bin_m))
    occupied = uniform_filter1d(profile.cover.astype(np.float64), window, mode="nearest") >= cfg.occ_min
    close = round(cfg.occ_close_m / profile.bin_m)
    if close > 1:
        occupied = minimum_filter1d(maximum_filter1d(occupied, close, mode="nearest"), close, mode="nearest")
    occupied = occupied & ~profile.unknown
    return OccupancyProfile(profile.bin_m, profile.length_m, profile.cover, occupied, profile.unknown,
                            profile.hidden & ~occupied)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def _gap_kind(start: int, end: int, n_bins: int) -> GapKind:
    head, tail = start == 0, end == n_bins
    if head and tail:
        return GapKind.FULL
    return GapKind.HEAD if head else GapKind.TAIL if tail else GapKind.INTERIOR


def find_gaps(profile: OccupancyProfile, *, include_ends: bool = True, min_length_m: float = 0.0) -> tuple[Gap, ...]:
    """Every run of empty known bins, in along order; `include_ends=False` keeps interior gaps only."""
    empty = ~profile.occupied & ~profile.unknown
    n, unknown = profile.n_bins, profile.unknown
    gaps = []
    for s, e in _runs(empty):
        kind = _gap_kind(s, e, n)
        gap = Gap(s * profile.bin_m, min(e * profile.bin_m, profile.length_m), kind,
                  bool((s > 0 and unknown[s - 1]) or (e < n and unknown[e])))
        if (include_ends or kind == GapKind.INTERIOR) and gap.length_m >= min_length_m:
            gaps.append(gap)
    return tuple(gaps)


def classify_row_structure(profile: OccupancyProfile, cfg: RowStructureConfig) -> StructureResult:
    """disrupted iff the max gap >= gap_disrupted_m; unassessable when too little of the piece is visible."""
    gaps = find_gaps(profile)
    counted = [g for g in gaps if cfg.include_end_gaps or g.kind == GapKind.INTERIOR]
    max_gap = max((g.length_m for g in counted), default=0.0)
    visible = profile.visible_frac
    if visible < cfg.unassessable_visible_min:
        structure = RowStructure.UNASSESSABLE
    else:
        structure = RowStructure.DISRUPTED if max_gap >= cfg.gap_disrupted_m else RowStructure.REGULAR
    lo, hi = cfg.borderline_m
    flags = (BORDERLINE_CODE,) if lo <= max_gap <= hi else ()
    return StructureResult(structure, float(max_gap), visible, gaps, flags)


def measure_piece(axis_utm: LineString, canopy_ij: np.ndarray, tile: TileRef, cfg: RowStructureConfig, *,
                  half_m: float, vis: U8 | None = None) -> StructureResult:
    """Structure of one row piece of `tile` from the tile's canopy pixel set (see canopy_pixels)."""
    profile = occupancy_profile(axis_to_px(axis_utm, tile), canopy_ij, half_px=half_m / GSD_M, vis=vis,
                                max_hidden=cfg.visible_bin_max_hidden)
    if cfg.occ_smoothing_enabled:
        profile = smooth_profile(profile, cfg)
    return classify_row_structure(profile, cfg)


def gap_centre(axis_utm: LineString, gap: Gap) -> Point:
    return axis_utm.interpolate((gap.start_m + gap.end_m) / 2.0)


def gap_line(axis_utm: LineString, gap: Gap) -> BaseGeometry:
    return substring(axis_utm, gap.start_m, gap.end_m)


# ------------------------------------------------------------------ per-tile frames


def merge_flags(*groups: object) -> str:
    """';'-joined unique qa codes in first-seen order; strings are split on ';', nulls are skipped."""
    codes: list[str] = []
    for group in groups:
        if isinstance(group, str):
            codes.extend(group.split(QA_FLAG_SEP))
        elif isinstance(group, list | tuple):
            codes.extend(str(c) for c in group)
    return QA_FLAG_SEP.join(dict.fromkeys(c.strip() for c in codes if c.strip()))


def _confidence(value: object) -> float:
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    return conf if np.isfinite(conf) else DEFAULT_CONFIDENCE


def _check_pieces(pieces: gpd.GeoDataFrame, tile: TileRef) -> None:
    missing = [c for c in _REQUIRED_PIECE_COLUMNS if c not in pieces.columns]
    if missing:
        raise ValueError(f"row pieces of {tile.tile_id} lack columns {missing} (need row_id, vineyard_id)")


def _kept_pieces(pieces: gpd.GeoDataFrame, min_piece_m: float) -> pd.DataFrame:
    geoms = pieces.geometry
    ok = geoms.notna() & ~geoms.is_empty & (geoms.geom_type == "LineString") & (geoms.length >= min_piece_m)
    return pieces[ok.to_numpy()].reset_index(drop=True)


def _piece_ids(kept: pd.DataFrame, tile: TileRef) -> list[str]:
    counts: dict[str, int] = {}
    ids = []
    existing = kept["piece_id"] if "piece_id" in kept.columns else pd.Series([None] * len(kept))
    for row_id, given in zip(kept["row_id"], existing, strict=True):
        counts[row_id] = counts.get(row_id, 0) + 1
        ids.append(given if isinstance(given, str) and given else format_row_piece_id(row_id, tile.tile_id,
                                                                                    counts[row_id]))
    return ids


def _gap_records(piece_id: str, row: pd.Series, tile: TileRef, res: StructureResult) -> list[dict]:
    return [
        {"piece_id": piece_id, "row_id": row["row_id"], "vineyard_id": row["vineyard_id"], "tile_id": tile.tile_id,
         "start_m": g.start_m, "end_m": g.end_m, "length_m": g.length_m, "kind": g.kind.value,
         "censored": g.censored, "geometry": gap_line(row.geometry, g)}
        for g in res.gaps
    ]


def _gaps_frame(records: list[dict]) -> gpd.GeoDataFrame:
    if not records:
        data = {c: pd.Series([], dtype=object) for c in GAP_COLUMNS}
        return gpd.GeoDataFrame(data, geometry=gpd.GeoSeries([], crs=CRS_EPSG), crs=CRS_EPSG)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_EPSG)


def row_pieces_attributes(pieces: gpd.GeoDataFrame, canopies: gpd.GeoDataFrame, tile: TileRef,
                          cfg: RowStructureConfig, *, half_m: float, min_piece_m: float, vis: U8 | None = None,
                          source: Source = Source.MODEL, run_id: str = "",
                          model_version: str = "") -> RowAttrsResult:
    """AnnSet `row_pieces` of one tile (+ internal gaps) from its clipped axes and its canopies."""
    _check_pieces(pieces, tile)
    kept = _kept_pieces(pieces, min_piece_m)
    if kept.empty:
        return RowAttrsResult(empty_layer(ROW_PIECES_LAYER), _gaps_frame([]))
    canopy_ij = canopy_pixels(list(canopies.geometry), tile)
    ids = _piece_ids(kept, tile)
    records, gap_records = [], []
    for piece_id, (_, row) in zip(ids, kept.iterrows(), strict=True):
        res = measure_piece(row.geometry, canopy_ij, tile, cfg, half_m=half_m, vis=vis)
        records.append({
            "piece_id": piece_id, "row_id": row["row_id"], "vineyard_id": row["vineyard_id"],
            "tile_id": tile.tile_id, "row_structure": res.row_structure.value, "length_m": row.geometry.length,
            "max_gap_m": res.max_gap_m, "n_vertices": len(row.geometry.coords), "source": Source(source).value,
            "run_id": run_id, "model_version": model_version,
            "confidence": _confidence(row.get("confidence")),
            "qa_flags": merge_flags(row.get("qa_flags"), res.flags), "geometry": row.geometry,
        })
        gap_records.extend(_gap_records(piece_id, row, tile, res))
    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_EPSG)
    return RowAttrsResult(coerce_layer(frame, ROW_PIECES_LAYER), _gaps_frame(gap_records))
