"""Target rules on the global rows (arch §4.10, contract §2.5.11, design 04 §3.5) → `TargetDraft`s.

Kinds produced here: row_gap (GAP), missing_plant (MSP), sparse (SPR), row_end_short (END: (a) the row
line runs past its last canopy, (b) both neighbours run past its end) and missing_row (MRW). Waste
targets come from `target_waste`. Every rule is pure: rows in, drafts out; ids are assigned later.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.ops import substring

from vineyard.config import TargetsConfig
from vineyard.contracts.enums import TargetKind
from vineyard.route.target_gaps import RowGap

# CONFIG-REQUEST: targets.spacing_sample_step_m = 2.0
SPACING_SAMPLE_STEP_M: Final = 2.0
PRIORITY_HIGH: Final = 1
PRIORITY_NORMAL: Final = 2
PRIORITY_LOW: Final = 3
_LOW_PRIORITY_KINDS: Final = frozenset({TargetKind.MISSING_PLANT, TargetKind.SPARSE, TargetKind.OTHER})
_GAP_TOL_M: Final = 1e-9
_END_KINDS: Final = frozenset({"head", "tail"})


@dataclass(frozen=True)
class RowContext:
    """One global row with its gaps (engine output) and coverage facts."""

    row_id: str
    vineyard_id: str
    row_index: int | None
    axis: LineString
    gaps: tuple[RowGap, ...]
    unknown: tuple[tuple[float, float], ...]
    head_on_boundary: bool
    tail_on_boundary: bool


@dataclass(frozen=True)
class TargetDraft:
    """A target before dedupe, id assignment and reachability."""

    kind: TargetKind
    x: float
    y: float
    vineyard_id: str
    priority: int
    reason: str
    row_id: str | None = None
    along_m: float = math.nan
    gap_length_m: float = math.nan
    extent: LineString | None = None
    interrow_id: str | None = None
    waste_id: str | None = None
    sort_ref: str = ""
    pair: tuple[str, str] | None = None


def priority_for(kind: TargetKind, gap_length_m: float, cfg: TargetsConfig) -> int:
    """1 = waste and gaps > priority_long_gap_m; 2 = other GAP, END, MRW; 3 = MSP, SPR, other."""
    if kind is TargetKind.WASTE or (kind is TargetKind.ROW_GAP and gap_length_m > cfg.priority_long_gap_m):
        return PRIORITY_HIGH
    return PRIORITY_LOW if kind in _LOW_PRIORITY_KINDS else PRIORITY_NORMAL


def segment_samples(start_m: float, end_m: float, long_m: float, step_m: float) -> tuple[tuple[float, float], ...]:
    """One sub-segment when the length is <= long_m, else ceil(L / step_m) equal sub-segments."""
    if step_m <= 0 or long_m <= 0:
        raise ValueError(f"segment_samples needs positive long_m/step_m, got {long_m}/{step_m}")
    length = end_m - start_m
    n = 1 if length <= long_m else math.ceil(length / step_m)
    edges = np.linspace(start_m, end_m, n + 1)
    return tuple((float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True))


def _line_drafts(line: LineString, lo: float, hi: float, kind: TargetKind, base: TargetDraft,
                 cfg: TargetsConfig, along_offset: float = 0.0, along_sign: float = 1.0) -> tuple[TargetDraft, ...]:
    """Drafts at the sub-segment centres of [lo, hi] along `line` (extent = the sub-segment)."""
    drafts = []
    for a, b in segment_samples(lo, hi, cfg.long_gap_m, cfg.gap_step_m):
        centre = line.interpolate((a + b) / 2.0)
        drafts.append(TargetDraft(
            kind=kind, x=centre.x, y=centre.y, vineyard_id=base.vineyard_id,
            priority=priority_for(kind, base.gap_length_m, cfg), reason=base.reason, row_id=base.row_id,
            along_m=along_offset + along_sign * (a + b) / 2.0, gap_length_m=base.gap_length_m,
            extent=substring(line, a, b), sort_ref=base.sort_ref, pair=base.pair))
    return tuple(drafts)


def _row_base(ctx: RowContext, gap_length_m: float, reason: str) -> TargetDraft:
    return TargetDraft(kind=TargetKind.OTHER, x=math.nan, y=math.nan, vineyard_id=ctx.vineyard_id, priority=0,
                       reason=reason, row_id=ctx.row_id, gap_length_m=gap_length_m, sort_ref=ctx.row_id)


def _gap_reason(prefix: str, gap: RowGap) -> str:
    return f"{prefix} {gap.length_m:.2f} m" + (" (censored)" if gap.censored else "")


def _interior(ctx: RowContext, lo_m: float, hi_m: float) -> tuple[RowGap, ...]:
    return tuple(g for g in ctx.gaps if g.kind == "interior" and lo_m - _GAP_TOL_M <= g.length_m < hi_m)


def gap_drafts(ctx: RowContext, cfg: TargetsConfig) -> tuple[TargetDraft, ...]:
    """GAP: interior gaps >= gap_min_m, long gaps sampled every gap_step_m."""
    return tuple(d for g in _interior(ctx, cfg.gap_min_m, math.inf)
                 for d in _line_drafts(ctx.axis, g.start_m, g.end_m, TargetKind.ROW_GAP,
                                       _row_base(ctx, g.length_m, _gap_reason("gap", g)), cfg))


def missing_plant_drafts(ctx: RowContext, cfg: TargetsConfig) -> tuple[TargetDraft, ...]:
    """MSP: interior gaps in [missing_min_m, gap_min_m)."""
    return tuple(d for g in _interior(ctx, cfg.missing_min_m, cfg.gap_min_m - _GAP_TOL_M)
                 for d in _line_drafts(ctx.axis, g.start_m, g.end_m, TargetKind.MISSING_PLANT,
                                       _row_base(ctx, g.length_m, _gap_reason("missing plant", g)), cfg))


# ------------------------------------------------------------------ SPR


def _overlap(lo: float, hi: float, spans: Sequence[tuple[float, float]]) -> float:
    return float(sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in spans))


def _window_is_sparse(ctx: RowContext, lo: float, hi: float, cfg: TargetsConfig) -> bool:
    width = hi - lo
    known = width - _overlap(lo, hi, ctx.unknown)
    if known <= 0 or known / width < cfg.sparse_known_min:
        return False
    spans = [(g.start_m, g.end_m) for g in ctx.gaps]
    if any(g.length_m >= cfg.gap_min_m and g.start_m < hi and g.end_m > lo for g in ctx.gaps):
        return False
    occupied = known - _overlap(lo, hi, spans)
    return occupied / known < cfg.sparse_occ_max


def _sparse_runs(ctx: RowContext, cfg: TargetsConfig) -> list[tuple[float, float]]:
    runs: list[list[float]] = []
    length, start = ctx.axis.length, 0.0
    while start + cfg.sparse_window_m <= length + _GAP_TOL_M:
        end = min(start + cfg.sparse_window_m, length)
        if _window_is_sparse(ctx, start, end, cfg):
            if runs and start <= runs[-1][1]:
                runs[-1][1] = end
            else:
                runs.append([start, end])
        start += cfg.sparse_step_m
    return [(a, b) for a, b in runs]


def sparse_drafts(ctx: RowContext, cfg: TargetsConfig) -> tuple[TargetDraft, ...]:
    """SPR: merged runs of windows with occupancy < sparse_occ_max, mostly known, without a gap >= gap_min_m."""
    return tuple(d for lo, hi in _sparse_runs(ctx, cfg)
                 for d in _line_drafts(ctx.axis, lo, hi, TargetKind.SPARSE,
                                       _row_base(ctx, math.nan, f"sparse {hi - lo:.1f} m"), cfg))


# ------------------------------------------------------------------ END


def end_gap_drafts(ctx: RowContext, cfg: TargetsConfig) -> tuple[TargetDraft, ...]:
    """END (a): the row line runs >= end_short_min_m past its first/last canopy, end inside coverage."""
    drafts: list[TargetDraft] = []
    for gap in ctx.gaps:
        if gap.kind not in _END_KINDS:  # interior gaps are GAP/MSP; a "full" row has no canopy to be short of
            continue
        on_boundary = ctx.head_on_boundary if gap.kind == "head" else ctx.tail_on_boundary
        if on_boundary or gap.length_m < cfg.end_short_min_m:
            continue
        base = _row_base(ctx, gap.length_m, _gap_reason(f"row end ({gap.kind}) without vines", gap))
        drafts += _line_drafts(ctx.axis, gap.start_m, gap.end_m, TargetKind.ROW_END_SHORT, base, cfg)
    return tuple(drafts)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.hypot(*v))
    if norm == 0.0:
        raise ValueError("zero-length direction on a row axis")
    return v / norm


def _neighbour_extension(axis: LineString, other: LineString, at_head: bool) -> float:
    coords = np.asarray(axis.coords)
    u = _unit(coords[-1] - coords[0])
    t = (np.asarray(other.coords) - coords[0]) @ u
    return float(-t.min()) if at_head else float(t.max() - (coords[-1] - coords[0]) @ u)


def _extrapolated(axis: LineString, at_head: bool, length_m: float) -> LineString:
    coords = np.asarray(axis.coords)
    end, prev = (coords[0], coords[1]) if at_head else (coords[-1], coords[-2])
    return LineString([end, end + _unit(end - prev) * length_m])


def end_extension_drafts(ctx: RowContext, neighbours: Sequence[RowContext],
                         cfg: TargetsConfig) -> tuple[TargetDraft, ...]:
    """END (b): both neighbours (row_index ±1) run >= end_short_min_m past this row's end."""
    if len(neighbours) != 2:
        return ()
    drafts: list[TargetDraft] = []
    for at_head, on_boundary in ((True, ctx.head_on_boundary), (False, ctx.tail_on_boundary)):
        ext = min(_neighbour_extension(ctx.axis, n.axis, at_head) for n in neighbours)
        if on_boundary or ext < cfg.end_short_min_m:
            continue
        line = _extrapolated(ctx.axis, at_head, ext)
        base = _row_base(ctx, ext, f"row shorter than both neighbours by {ext:.2f} m")
        offset, sign = (0.0, -1.0) if at_head else (ctx.axis.length, 1.0)
        drafts += _line_drafts(line, 0.0, ext, TargetKind.ROW_END_SHORT, base, cfg, offset, sign)
    return tuple(drafts)


# ------------------------------------------------------------------ MRW


def _pair_samples(a: LineString, b: LineString) -> tuple[np.ndarray, np.ndarray] | None:
    """Points of `a` over its overlap with `b` (every SPACING_SAMPLE_STEP_M) and their nearest points on b."""
    ends = shapely.points(np.asarray(b.coords)[[0, -1]])
    lo, hi = sorted(float(t) for t in shapely.line_locate_point(a, ends))
    if hi - lo < SPACING_SAMPLE_STEP_M:
        return None
    along = np.linspace(lo, hi, math.ceil((hi - lo) / SPACING_SAMPLE_STEP_M) + 1)
    pa = shapely.line_interpolate_point(a, along)
    pb = shapely.line_interpolate_point(b, shapely.line_locate_point(b, pa))
    return shapely.get_coordinates(pa), shapely.get_coordinates(pb)


def _virtual_axis(pa: np.ndarray, pb: np.ndarray, frac: float) -> LineString:
    return LineString(pa + frac * (pb - pa))


def _pair_drafts(a: RowContext, b: RowContext, samples: tuple[np.ndarray, np.ndarray], spacing: float,
                 median: float, cfg: TargetsConfig) -> tuple[TargetDraft, ...]:
    n_missing = max(1, round(spacing / median) - 1)
    reason = f"missing row between {a.row_id} and {b.row_id}: spacing {spacing:.2f} m (median {median:.2f} m)"
    drafts: list[TargetDraft] = []
    for j in range(1, n_missing + 1):
        base = TargetDraft(kind=TargetKind.MISSING_ROW, x=math.nan, y=math.nan, vineyard_id=a.vineyard_id,
                           priority=0, reason=reason, sort_ref=f"{a.row_id}+{j}", pair=(a.row_id, b.row_id))
        line = _virtual_axis(*samples, j / (n_missing + 1))
        drafts += _line_drafts(line, 0.0, line.length, TargetKind.MISSING_ROW, base, cfg)
    return tuple(drafts)


def missing_row_drafts(block: Sequence[RowContext], cfg: TargetsConfig) -> tuple[TargetDraft, ...]:
    """MRW: consecutive rows (by row_index) spaced > missing_row_spacing_factor × the block median."""
    rows = sorted((r for r in block if r.row_index is not None), key=lambda r: (r.row_index, r.row_id))
    pairs = []
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        samples = _pair_samples(a.axis, b.axis)
        if samples is not None:
            pairs.append((a, b, samples, float(np.median(np.hypot(*(samples[1] - samples[0]).T)))))
    if not pairs:
        return ()
    median = float(np.median([p[3] for p in pairs]))
    if median <= 0.0:
        return ()
    return tuple(d for a, b, samples, spacing in pairs if spacing > cfg.missing_row_spacing_factor * median
                 for d in _pair_drafts(a, b, samples, spacing, median, cfg))
