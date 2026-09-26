"""Cross-paths: tracks / tractor lanes running across the rows of one vineyard block.

A track through a block leaves a gap in every row it crosses, at nearly the same place. Seen row by row
those gaps look like missing vines (row_gap / missing_plant targets lined up across the block); seen
together they are one walkable strip. Detection works on the row-gap engine's output (`RowContext.gaps`):

1. per block, a block frame: u = mean row direction, v = its normal. Every interior gap of at least
   `min_gap_m` (at most `max_gap_m`) becomes an interval [s0, s1] on u at the offset t (on v) of its centre;
2. two gaps are linked when they lie one row apart (`link_min_m` <= dt <= `link_max_m`: neighbouring physical
   rows; row_index is not used, blocks with fragmented rows interleave short and long row ids) and their
   u-intervals overlap (grown by `overlap_tol_m`). The links form a DAG in t order; the longest chain (ties:
   the straightest) is taken, its gaps removed, and so on while chains cross >= `min_rows` rows;
3. straightness: Douglas-Peucker on the gap centres (t, s) with tolerance `max_residual_m`. Tracks bend, so a
   path may be a polyline, but every straight piece must cross >= `min_segment_rows` rows (shorter wiggles
   split the chain; each remaining piece needs `min_rows` rows). A scatter of missing vines never lines up
   like that. The median crossing gap may not exceed `max_width_m` (wider: a patch of missing vines);
4. geometry: the simplified polyline, extended `end_extend_spacing` row spacings past the first and the last
   crossed row (into the outer interrows), buffered (flat caps) by half the median gap width across the path.

Pure: row contexts in, `CrossPath`s out. `subtract_paths` removes the crossed stretches from row gaps so the
targets rules no longer see the track as missing vines.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.geo.tiling import CRS_EPSG
from vineyard.route.target_gaps import RowGap, unknown_intervals
from vineyard.route.target_rules import RowContext

if TYPE_CHECKING:
    from vineyard.config.sections_post import CrossPathsConfig

PATH_ID_FMT: Final = "XP{:03d}"
LAYER_NAME: Final = "cross_paths"
_TOL_M: Final = 1e-9


@dataclass(frozen=True)
class CrossPathParams:
    min_gap_m: float
    max_gap_m: float
    max_width_m: float
    min_rows: int
    min_segment_rows: int
    link_min_m: float
    link_max_m: float
    overlap_tol_m: float
    max_residual_m: float
    end_extend_spacing: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 < self.min_gap_m <= self.max_gap_m:
            raise ValueError(f"cross_paths: need 0 < min_gap_m <= max_gap_m, got {self.min_gap_m}/{self.max_gap_m}")
        if self.max_width_m <= 0.0:
            raise ValueError(f"cross_paths.max_width_m must be > 0, got {self.max_width_m}")
        if not 2 <= self.min_segment_rows <= self.min_rows:
            raise ValueError(f"cross_paths: need 2 <= min_segment_rows <= min_rows, got "
                             f"{self.min_segment_rows}/{self.min_rows}")
        if not 0.0 < self.link_min_m < self.link_max_m:
            raise ValueError(f"cross_paths: need 0 < link_min_m < link_max_m, got {self.link_min_m}/{self.link_max_m}")
        if self.end_extend_spacing <= 0.0:
            raise ValueError(f"cross_paths.end_extend_spacing must be > 0, got {self.end_extend_spacing}")
        if min(self.overlap_tol_m, self.max_residual_m) < 0.0:
            raise ValueError("cross_paths: overlap_tol_m and max_residual_m must be >= 0")

    @classmethod
    def from_config(cls, cfg: CrossPathsConfig) -> CrossPathParams:
        return cls(min_gap_m=cfg.min_gap_m, max_gap_m=cfg.max_gap_m, max_width_m=cfg.max_width_m, min_rows=cfg.min_rows,
                   min_segment_rows=cfg.min_segment_rows, link_min_m=cfg.link_min_m, link_max_m=cfg.link_max_m,
                   overlap_tol_m=cfg.overlap_tol_m, max_residual_m=cfg.max_residual_m,
                   end_extend_spacing=cfg.end_extend_spacing)


@dataclass(frozen=True)
class Crossing:
    """The gap one row leaves where the path crosses it (metres along the row axis)."""

    row_id: str
    start_m: float
    end_m: float


@dataclass(frozen=True, eq=False)
class CrossPath:
    path_id: str
    vineyard_id: str
    crossings: tuple[Crossing, ...]
    centerline: LineString
    polygon: Polygon
    width_m: float
    residual_m: float

    @property
    def n_rows(self) -> int:
        return len(self.crossings)

    @property
    def length_m(self) -> float:
        return float(self.centerline.length)

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(c.row_id for c in self.crossings)


# ------------------------------------------------------------------ block frame


@dataclass(frozen=True)
class _Frame:
    u: np.ndarray   # unit row direction
    v: np.ndarray   # unit normal

    def st(self, xy: np.ndarray) -> np.ndarray:
        return np.column_stack([xy @ self.u, xy @ self.v])

    def xy(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.outer(s, self.u) + np.outer(t, self.v)


def _direction(axis: LineString) -> np.ndarray:
    coords = np.asarray(axis.coords)[:, :2]
    d = coords[-1] - coords[0]
    n = float(np.hypot(*d))
    return d / n if n > 0.0 else np.array([1.0, 0.0])


def block_frame(axes: Sequence[LineString]) -> _Frame:
    """Length-weighted mean axial direction (signs aligned to the first row) and its normal."""
    ref = _direction(axes[0])
    total = np.zeros(2)
    for axis in axes:
        d = _direction(axis)
        total += (d if d @ ref >= 0.0 else -d) * axis.length
    norm = float(np.hypot(*total))
    u = total / norm if norm > 0.0 else ref
    return _Frame(u=u, v=np.array([-u[1], u[0]]))


# ------------------------------------------------------------------ candidates


@dataclass(frozen=True)
class _Gap:
    row: RowContext
    gap: RowGap
    s0: float
    s1: float
    t: float

    @property
    def s_mid(self) -> float:
        return (self.s0 + self.s1) / 2.0


def _gap_items(block: Sequence[RowContext], frame: _Frame, params: CrossPathParams) -> list[_Gap]:
    out: list[_Gap] = []
    for row in block:
        for gap in row.gaps:
            if gap.kind != "interior" or not params.min_gap_m - _TOL_M <= gap.length_m <= params.max_gap_m:
                continue
            ends = np.asarray([row.axis.interpolate(gap.start_m).coords[0][:2],
                               row.axis.interpolate(gap.end_m).coords[0][:2]], dtype=np.float64)
            st = frame.st(ends)
            out.append(_Gap(row, gap, float(st[:, 0].min()), float(st[:, 0].max()), float(st[:, 1].mean())))
    return sorted(out, key=lambda g: (g.t, g.s0, g.row.row_id))


def _linked(a: _Gap, b: _Gap, params: CrossPathParams) -> bool:
    dt = abs(b.t - a.t)
    tol = params.overlap_tol_m
    return (params.link_min_m <= dt <= params.link_max_m
            and min(a.s1, b.s1) + tol >= max(a.s0, b.s0) - tol)


def _predecessors(items: Sequence[_Gap], params: CrossPathParams) -> list[list[int]]:
    """Per gap (items sorted by t), the earlier gaps it links to."""
    preds: list[list[int]] = [[] for _ in items]
    for i, a in enumerate(items):
        for j in range(i + 1, len(items)):
            b = items[j]
            if b.t - a.t > params.link_max_m:
                break
            if _linked(a, b, params):
                preds[j].append(i)
    return preds


def _longest_chain(items: Sequence[_Gap], preds: Sequence[Sequence[int]], used: set[int]) -> list[int]:
    """Longest linked chain among unused gaps (DAG in t order); ties: smaller total |ds| (straighter)."""
    best: list[tuple[int, float]] = [(0, 0.0)] * len(items)
    back = [-1] * len(items)
    for j in range(len(items)):
        if j in used:
            continue
        score = (1, 0.0)
        for i in preds[j]:
            if i in used:
                continue
            n, bend = best[i]
            cand = (n + 1, bend + abs(items[j].s_mid - items[i].s_mid))
            if cand[0] > score[0] or (cand[0] == score[0] and cand[1] < score[1]):
                score, back[j] = cand, i
        best[j] = score
    free = [k for k in range(len(items)) if k not in used]
    if not free:
        return []
    end = max(free, key=lambda k: (best[k][0], -best[k][1], -k))
    chain = []
    while end >= 0:
        chain.append(end)
        end = back[end]
    return chain[::-1]


def _chains(items: Sequence[_Gap], params: CrossPathParams) -> list[list[_Gap]]:
    """Disjoint chains of >= min_rows linked gaps, longest first."""
    preds = _predecessors(items, params)
    used: set[int] = set()
    out = []
    while True:
        chain = _longest_chain(items, preds, used)
        if len(chain) < params.min_rows:
            return out
        used.update(chain)
        out.append([items[k] for k in chain])


def _dp(points: np.ndarray, tol: float) -> list[int]:
    """Douglas-Peucker vertex indices of a (t, s) polyline; distance measured along s (across the rows)."""
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        (t0, s0), (t1, s1) = points[i], points[j]
        mid = points[i + 1:j]
        frac = (mid[:, 0] - t0) / (t1 - t0) if t1 != t0 else np.zeros(len(mid))
        dev = np.abs(mid[:, 1] - (s0 + frac * (s1 - s0)))
        k = int(np.argmax(dev))
        if dev[k] > tol:
            keep.add(i + 1 + k)
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return sorted(keep)


@dataclass(frozen=True)
class _Fit:
    chain: tuple[_Gap, ...]
    vertices: tuple[int, ...]
    residual_m: float


def _residual(points: np.ndarray, vertices: Sequence[int]) -> float:
    t, s = points[:, 0], points[:, 1]
    fitted = np.interp(t, t[list(vertices)], s[list(vertices)])
    return float(np.max(np.abs(s - fitted)))


def _pieces(vertices: Sequence[int], min_segment_rows: int) -> list[tuple[int, int]]:
    """Runs of consecutive DP segments crossing >= min_segment_rows rows each, as (first, last) vertex."""
    runs: list[list[int]] = []
    for a, b in zip(vertices[:-1], vertices[1:], strict=True):
        if b - a + 1 < min_segment_rows:
            continue
        if runs and runs[-1][1] == a:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    return [(a, b) for a, b in runs]


def _straight(chain: Sequence[_Gap], params: CrossPathParams) -> list[_Fit]:
    """The straight-enough pieces of a chain: DP polyline pieces whose every segment crosses
    >= min_segment_rows rows and that cross >= min_rows rows in all (a wiggle splits the chain)."""
    points = np.asarray([(g.t, g.s_mid) for g in chain], dtype=np.float64)
    vertices = _dp(points, params.max_residual_m)
    fits = []
    for a, b in _pieces(vertices, params.min_segment_rows):
        if b - a + 1 < params.min_rows:
            continue
        if float(np.median([g.gap.length_m for g in chain[a:b + 1]])) > params.max_width_m:
            continue  # wider than any track: a patch of missing vines, not a lane
        sub = points[a:b + 1]
        verts = tuple(v - a for v in vertices if a <= v <= b)
        fits.append(_Fit(tuple(chain[a:b + 1]), verts, _residual(sub, verts)))
    return fits


def _geometry(fit: _Fit, frame: _Frame, extend_spacing: float) -> tuple[LineString, Polygon, float]:
    chain = fit.chain
    t = np.asarray([chain[k].t for k in fit.vertices])
    s = np.asarray([chain[k].s_mid for k in fit.vertices])
    all_t = np.asarray([g.t for g in chain])
    half = float(np.median(np.diff(all_t))) * extend_spacing
    head = (s[0] - (s[1] - s[0]) * half / (t[1] - t[0]), t[0] - half)
    tail = (s[-1] + (s[-1] - s[-2]) * half / (t[-1] - t[-2]), t[-1] + half)
    ss = np.concatenate([[head[0]], s, [tail[0]]])
    tt = np.concatenate([[head[1]], t, [tail[1]]])
    line = LineString(frame.xy(ss, tt))
    slope = (s[-1] - s[0]) / (t[-1] - t[0])
    # gap lengths run along u; across the (possibly oblique) path the strip is cos(angle) as wide
    width = float(np.median([g.gap.length_m for g in chain])) / math.sqrt(1.0 + slope * slope)
    polygon = line.buffer(width / 2.0, cap_style="flat", join_style="mitre")
    return line, polygon, width


def _block_fits(block: Sequence[RowContext], params: CrossPathParams) -> list[tuple[_Fit, _Frame]]:
    frame = block_frame([r.axis for r in block])
    items = _gap_items(block, frame, params)
    fits = []
    for chain in _chains(items, params):
        fits.extend((fit, frame) for fit in _straight(chain, params))
    return fits


def detect_cross_paths(contexts: Sequence[RowContext], params: CrossPathParams) -> tuple[CrossPath, ...]:
    """Every cross-path of every block; ids XP001.. ordered by (vineyard_id, first crossing)."""
    blocks: dict[str, list[RowContext]] = defaultdict(list)
    for ctx in contexts:
        blocks[ctx.vineyard_id].append(ctx)
    found = []
    for vid in sorted(blocks):
        block = sorted(blocks[vid], key=lambda c: c.row_id)
        for fit, frame in _block_fits(block, params):
            line, polygon, width = _geometry(fit, frame, params.end_extend_spacing)
            first = fit.chain[0]
            found.append(((vid, round(first.t, 3), round(first.s_mid, 3)), vid, fit, line, polygon, width))
    out = []
    for k, (_, vid, fit, line, polygon, width) in enumerate(sorted(found, key=lambda f: f[0]), start=1):
        crossings = tuple(Crossing(g.row.row_id, g.gap.start_m, g.gap.end_m) for g in fit.chain)
        out.append(CrossPath(PATH_ID_FMT.format(k), vid, crossings, line, polygon, width, fit.residual_m))
    return tuple(out)


# ------------------------------------------------------------------ targets: gaps minus paths


def _minus(gap: RowGap, spans: Sequence[tuple[float, float]]) -> list[RowGap]:
    parts, cursor = [], gap.start_m
    for lo, hi in spans:
        if hi <= cursor or lo >= gap.end_m:
            continue
        if lo > cursor:
            parts.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < gap.end_m:
        parts.append((cursor, gap.end_m))
    return [RowGap(lo, hi, "interior", gap.censored) for lo, hi in parts if hi - lo > _TOL_M]


def subtract_paths(ctx: RowContext, paths: BaseGeometry | None) -> tuple[RowContext, float]:
    """`ctx` with the stretches of its axis inside `paths` cut out of its interior gaps; the metres removed.

    Only interior gaps change (a row end is never a crossing); what is left of a gap longer than the track
    stays a gap, so missing vines next to a track are still reported.
    """
    if paths is None or paths.is_empty or not ctx.axis.intersects(paths):
        return ctx, 0.0
    spans = unknown_intervals(ctx.axis, paths)
    if not spans:
        return ctx, 0.0
    gaps: list[RowGap] = []
    for gap in ctx.gaps:
        gaps.extend(_minus(gap, spans) if gap.kind == "interior" else [gap])
    removed = sum(g.length_m for g in ctx.gaps) - sum(g.length_m for g in gaps)
    return replace(ctx, gaps=tuple(sorted(gaps, key=lambda g: (g.start_m, g.end_m, g.kind)))), float(removed)


# ------------------------------------------------------------------ layer


def paths_frame(paths: Sequence[CrossPath], provenance: Mapping[str, Any]) -> gpd.GeoDataFrame:
    """The `cross_paths` layer: one polygon per path (EPSG:32635) + its centreline as WKT-free columns."""
    data: dict[str, list[Any]] = {
        "path_id": [p.path_id for p in paths], "vineyard_id": [p.vineyard_id for p in paths],
        "n_rows": [p.n_rows for p in paths], "width_m": [p.width_m for p in paths],
        "length_m": [p.length_m for p in paths], "residual_m": [p.residual_m for p in paths],
        "row_ids": [",".join(p.row_ids) for p in paths],
        **{k: [v] * len(paths) for k, v in provenance.items()},
    }
    return gpd.GeoDataFrame(pd.DataFrame(data), geometry=gpd.GeoSeries([p.polygon for p in paths], crs=CRS_EPSG),
                            crs=CRS_EPSG)


__all__ = [
    "LAYER_NAME", "CrossPath", "CrossPathParams", "Crossing", "block_frame", "detect_cross_paths", "paths_frame",
    "subtract_paths",
]
