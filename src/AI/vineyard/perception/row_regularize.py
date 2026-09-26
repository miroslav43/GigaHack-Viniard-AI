"""Row-frame regularisation of a block (RC8): headland lines, spacing lattice, stray edge rows.

Real vineyards are regular in the row frame (u along the rows, v across): parallel rows on a constant
spacing lattice and row ends on straight or piecewise-straight headland lines. Per block:

1. the rows are grouped into lattice lines (collinear pieces of one row share a line);
2. off-lattice lines (normal offset off the local lattice v0 + k·s by more than lattice_tol_frac·s) are
   removed when short or weakly supported (tillage, scrub), else flagged;
3. short edge rows without along-overlap with their neighbour line are removed (strays);
4. the start and end profiles u(v) get a robust piecewise-linear headland fit (1..max_segments independent
   Huber lines, so a field-edge step between segments is allowed; breakpoints by penalised search);
5. an end overshooting its headland by more than over_m is trimmed back to the line + margin when the
   evidence beyond it is weak (row band barely greener than the inter-row bands), else flagged; an end short
   of the line by more than under_m is flagged (extended only when enabled and the evidence clearly shows a
   row there).

Pure functions; evidence comes in through `Evidence` (veg fraction of a band around a line, None when
unknown). Every change is a `RowChange`; the caller applies them and emits the QA issues.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring

from vineyard.contracts.ordering import mean_axial_angle_deg

if TYPE_CHECKING:
    from vineyard.config.sections_perception import RowRegularizeConfig

ACTION_TRIM: Final = "row_trimmed_headland"
ACTION_OFF_LATTICE: Final = "row_off_lattice_removed"
ACTION_STRAY: Final = "row_stray_removed"
ACTION_UNDERSHOOT: Final = "row_undershoot"
ACTION_EXTEND: Final = "row_extended_headland"
FLAG_OFF_LATTICE: Final = "row_offlattice"
FLAG_OVERSHOOT: Final = "row_overshoot_suspect"
REMOVALS: Final = frozenset({ACTION_OFF_LATTICE, ACTION_STRAY})
SPAN_ACTIONS: Final = frozenset({ACTION_TRIM, ACTION_EXTEND})
_EPS: Final = 1e-9
_HUBER_ITERS: Final = 4
_MEDFILT: Final = 5
_LOCAL_S_RANGE: Final = (0.8, 1.25)  # a local spacing outside this x block spacing falls back to the block's

# (line, lateral offset m) -> veg fraction in the evidence band around the shifted line, None when unknown
Evidence = Callable[[LineString, float], "float | None"]


@dataclass(frozen=True)
class RegularizeOptions:
    enabled: bool
    min_block_rows: int
    collinear_tol_m: float
    max_segments: int
    min_segment_rows: int
    segment_penalty_m: float
    huber_delta_m: float
    residual_cap_m: float
    max_breakpoints: int
    over_m: float
    trim_margin_m: float
    trim_max_frac: float
    trim_max_contrast: float
    under_m: float
    extend_enabled: bool
    extend_min_contrast: float
    lattice_enabled: bool
    lattice_tol_frac: float
    lattice_window: int
    off_lattice_max_len_m: float
    off_lattice_min_contrast: float
    off_lattice_short_max_contrast: float
    stray_enabled: bool
    stray_min_len_m: float
    min_evidence_len_m: float

    @classmethod
    def from_config(cls, cfg: RowRegularizeConfig) -> RegularizeOptions:
        return cls(**{f: getattr(cfg, f) for f in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RowChange:
    """One change (or flag) on row `index` of the block's `lines`; lo/hi: new along span for trims/extends."""

    index: int
    action: str
    metres: float
    message: str
    x: float
    y: float
    lo: float = 0.0
    hi: float = 0.0


@dataclass(frozen=True, eq=False)
class RowFrame:
    origin: np.ndarray
    direction: np.ndarray

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def uv(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(pts, dtype=np.float64)[:, :2] - self.origin
        return d @ self.direction, d @ self.normal

    def xy(self, u: float, v: float) -> tuple[float, float]:
        p = self.origin + u * self.direction + v * self.normal
        return float(p[0]), float(p[1])


@dataclass(frozen=True)
class LatticeLine:
    """Rows of one lattice position: members sorted by start; start/end = extreme u of the members."""

    members: tuple[int, ...]
    v: float
    start: float
    end: float
    start_member: int
    end_member: int
    length: float


@dataclass(frozen=True)
class Headland:
    """Piecewise-linear headland: predicted u per lattice line, and the segment breakpoints (line index)."""

    pred: np.ndarray
    breaks: tuple[int, ...]


# ---------------------------------------------------------------- frame and lattice


def row_frame(lines: Sequence[LineString]) -> RowFrame:
    angles = []
    for ln in lines:
        c = np.asarray(ln.coords, dtype=np.float64)
        d = c[-1] - c[0]
        angles.append(math.degrees(math.atan2(d[1], d[0])) % 180.0)
    a = math.radians(mean_axial_angle_deg(angles))
    centre = np.mean([np.asarray(ln.centroid.coords[0]) for ln in lines], axis=0)
    return RowFrame(centre, np.array([math.cos(a), math.sin(a)]))


def lattice_lines(lines: Sequence[LineString], fr: RowFrame, tol_m: float) -> list[LatticeLine]:
    """Rows grouped into lattice lines (|dv| <= tol_m between consecutive rows), sorted by v."""
    recs = []
    for i, ln in enumerate(lines):
        u, v = fr.uv(np.asarray(ln.coords))
        recs.append((float(np.mean(v)), float(u.min()), float(u.max()), i, float(ln.length)))
    recs.sort()
    groups: list[list[tuple[float, float, float, int, float]]] = []
    for r in recs:
        if groups and r[0] - groups[-1][-1][0] <= tol_m:
            groups[-1].append(r)
        else:
            groups.append([r])
    out = []
    for g in groups:
        w = np.array([r[4] for r in g]) + _EPS
        s_mem = min(g, key=lambda r: r[1])
        e_mem = max(g, key=lambda r: r[2])
        out.append(LatticeLine(tuple(r[3] for r in sorted(g, key=lambda r: r[1])),
                               float(np.average([r[0] for r in g], weights=w)), s_mem[1], e_mem[2], s_mem[3],
                               e_mem[3], float(w.sum())))
    return out


def lattice_spacing(v: np.ndarray) -> float:
    """Robust row spacing of sorted lattice offsets (median step, refined over skip-one multiples)."""
    dv = np.diff(np.asarray(v, dtype=np.float64))
    dv = dv[dv > _EPS]
    if dv.size == 0:
        return math.nan
    s = float(np.median(dv))
    k = np.maximum(np.round(dv / s), 1.0)
    keep = np.abs(dv / k - s) <= 0.35 * s
    return float(np.median(dv[keep] / k[keep])) if keep.any() else s


def lattice_residuals(v: np.ndarray, s: float, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per line: (offset from the local lattice of its +-window neighbours, local spacing).

    The local lattice excludes the line itself: its spacing is `lattice_spacing` of the neighbours (a gap
    across the line is a multiple of it) and its phase the circular mean of their offsets."""
    v = np.asarray(v, dtype=np.float64)
    res, loc = np.zeros(len(v)), np.full(len(v), s)
    if not math.isfinite(s) or s <= 0 or len(v) < 3:
        return res, loc
    for i in range(len(v)):
        nb = [j for j in range(max(0, i - window), min(len(v), i + window + 1)) if j != i]
        s_loc = lattice_spacing(v[nb])
        s_loc = s_loc if math.isfinite(s_loc) and _LOCAL_S_RANGE[0] * s <= s_loc <= _LOCAL_S_RANGE[1] * s else s
        ang = 2.0 * math.pi * v / s_loc
        phase = math.atan2(float(np.sin(ang[nb]).mean()), float(np.cos(ang[nb]).mean()))
        d = (ang[i] - phase + math.pi) % (2.0 * math.pi) - math.pi
        res[i], loc[i] = d * s_loc / (2.0 * math.pi), s_loc
    return res, loc


# ---------------------------------------------------------------- headland fit


def huber_line(x: np.ndarray, y: np.ndarray, delta: float) -> tuple[float, float]:
    """(intercept, slope) of a Huber IRLS line, started from a median-filtered least-squares fit."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) == 1 or np.ptp(x) < _EPS:
        return float(np.median(y)), 0.0
    half = _MEDFILT // 2
    ym = np.array([np.median(y[max(0, i - half):i + half + 1]) for i in range(len(y))])
    w = np.ones_like(x)
    target = ym
    for _ in range(_HUBER_ITERS):
        a = np.vstack([np.ones_like(x), x]).T * w[:, None]
        coef, *_ = np.linalg.lstsq(a, target * w, rcond=None)
        r = np.abs(y - (coef[0] + coef[1] * x))
        w = np.sqrt(np.minimum(1.0, delta / np.maximum(r, _EPS)))
        target = y
    return float(coef[0]), float(coef[1])


def _segment_cost(x: np.ndarray, y: np.ndarray, o: RegularizeOptions) -> tuple[float, np.ndarray]:
    b0, b1 = huber_line(x, y, o.huber_delta_m)
    pred = b0 + b1 * x
    return float(np.minimum(np.abs(y - pred), o.residual_cap_m).sum()), pred


def _candidate_breaks(n: int, o: RegularizeOptions) -> list[int]:
    cand = list(range(o.min_segment_rows, n - o.min_segment_rows + 1))
    if len(cand) > o.max_breakpoints:
        idx = np.linspace(0, len(cand) - 1, o.max_breakpoints).round().astype(int)
        cand = sorted({cand[k] for k in idx})
    return cand


def _partitions(n: int, o: RegularizeOptions) -> list[tuple[int, ...]]:
    cand = _candidate_breaks(n, o)
    parts: list[tuple[int, ...]] = [()]
    if o.max_segments >= 2:
        parts += [(b,) for b in cand]
    if o.max_segments >= 3:
        parts += [(a, b) for a in cand for b in cand if b - a >= o.min_segment_rows]
    return parts


def fit_headland(v: np.ndarray, u: np.ndarray, o: RegularizeOptions) -> Headland:
    """Penalised piecewise-linear robust fit of the end profile u(v) (lines sorted by v)."""
    v, u = np.asarray(v, dtype=np.float64), np.asarray(u, dtype=np.float64)
    n = len(v)
    cache: dict[tuple[int, int], tuple[float, np.ndarray]] = {}

    def seg(a: int, b: int) -> tuple[float, np.ndarray]:
        if (a, b) not in cache:
            cache[(a, b)] = _segment_cost(v[a:b], u[a:b], o)
        return cache[(a, b)]

    best: tuple[float, tuple[int, ...]] | None = None
    for brk in _partitions(n, o):
        bounds = (0, *brk, n)
        cost = sum(seg(a, b)[0] for a, b in zip(bounds[:-1], bounds[1:], strict=True))
        cost += o.segment_penalty_m * len(brk)
        if best is None or cost < best[0] - _EPS:
            best = (cost, brk)
    brk = best[1] if best else ()
    bounds = (0, *brk, n)
    pred = np.concatenate([seg(a, b)[1] for a, b in zip(bounds[:-1], bounds[1:], strict=True)])
    return Headland(pred, tuple(brk))


# ---------------------------------------------------------------- evidence


def _shifted(line: LineString, offset_m: float) -> LineString:
    return line if abs(offset_m) < _EPS else line.offset_curve(offset_m)


def row_contrast(evidence: Evidence, line: LineString, spacing_m: float) -> float | None:
    """Veg fraction on the row band minus the mean of the two inter-row bands (+-s/2); None when unknown."""
    on = evidence(line, 0.0)
    offs = [evidence(line, sgn * spacing_m / 2.0) for sgn in (-1.0, 1.0)]
    offs = [f for f in offs if f is not None]
    if on is None or not offs:
        return None
    return float(on - np.mean(offs))


def span_on_line(line: LineString, fr: RowFrame, u_lo: float, u_hi: float) -> tuple[float, float]:
    """Along-line (from, to) of the part of `line` whose u lies in [u_lo, u_hi] (u monotonic along it)."""
    c = np.asarray(line.coords, dtype=np.float64)[:, :2]
    u, _ = fr.uv(c)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))])
    if u[-1] < u[0]:
        u, d = u[::-1], d[::-1]
    a = float(np.interp(u_lo, u, d))
    b = float(np.interp(u_hi, u, d))
    return min(a, b), max(a, b)


def _piece(line: LineString, lo: float, hi: float) -> LineString:
    return substring(line, lo, hi)


# ---------------------------------------------------------------- per-block steps


def _flagged_line(i: int, action: str, metres: float, msg: str, line: LineString, **kw: float) -> RowChange:
    p = line.interpolate(0.5, normalized=True)
    return RowChange(i, action, metres, msg, float(p.x), float(p.y), **kw)


def lattice_changes(lines: Sequence[LineString], groups: Sequence[LatticeLine], s: float, o: RegularizeOptions,
                    evidence: Evidence) -> list[RowChange]:
    """Off-lattice lines: removed when short or weakly supported, else flagged."""
    if not o.lattice_enabled or len(groups) < o.min_block_rows:
        return []
    res, loc = lattice_residuals(np.array([g.v for g in groups]), s, o.lattice_window)
    out = []
    for g, r, s_loc in zip(groups, res, loc, strict=True):
        if abs(r) <= o.lattice_tol_frac * s_loc:
            continue
        for i in g.members:
            ev = row_contrast(evidence, lines[i], s_loc)
            weak = ev is not None and ev < o.off_lattice_min_contrast
            short = ev is not None and g.length < o.off_lattice_max_len_m and ev < o.off_lattice_short_max_contrast
            ev_txt = "n/a" if ev is None else f"{ev:.2f}"
            msg = f"offset {r:+.2f} m off the {s_loc:.2f} m lattice, line {g.length:.1f} m, contrast {ev_txt}"
            action = ACTION_OFF_LATTICE if (short or weak) else FLAG_OFF_LATTICE
            out.append(_flagged_line(i, action, float(lines[i].length), msg, lines[i]))
    return out


def stray_changes(lines: Sequence[LineString], groups: Sequence[LatticeLine], o: RegularizeOptions
                  ) -> list[RowChange]:
    """Short outermost lines without along-overlap with their neighbour line."""
    if not o.stray_enabled or len(groups) < o.min_block_rows:
        return []
    out = []
    for k, nb in ((0, 1), (len(groups) - 1, len(groups) - 2)):
        g, h = groups[k], groups[nb]
        overlap = min(g.end, h.end) - max(g.start, h.start)
        if g.length < o.stray_min_len_m and overlap <= 0.0:
            out += [_flagged_line(i, ACTION_STRAY, float(lines[i].length),
                                  f"edge row {g.length:.1f} m, no overlap with its neighbour ({overlap:.1f} m)",
                                  lines[i]) for i in g.members]
    return out


def _end_change(i: int, line: LineString, fr: RowFrame, g: LatticeLine, pred: float, is_end: bool, s: float,
                o: RegularizeOptions, evidence: Evidence) -> RowChange | None:
    """Trim / flag / extend one end of line i against the headland prediction `pred` (u)."""
    cur = g.end if is_end else g.start
    outward = (cur - pred) if is_end else (pred - cur)
    side = "end" if is_end else "start"
    if outward > o.over_m:
        target = pred + o.trim_margin_m if is_end else pred - o.trim_margin_m
        u_lo, u_hi = (min(cur, target), max(cur, target))
        c_lo, c_hi = span_on_line(line, fr, u_lo, u_hi)
        beyond = _piece(line, c_lo, c_hi)
        ev = row_contrast(evidence, beyond, s) if beyond.length >= o.min_evidence_len_m else None
        keep_lo, keep_hi = (0.0, c_lo) if _cut_is_tail(line, fr, is_end) else (c_hi, float(line.length))
        cut = float(line.length) - (keep_hi - keep_lo)
        ev_txt = "n/a" if ev is None else f"{ev:.2f}"
        x, y = fr.xy(cur, g.v)
        if ev is not None and ev < o.trim_max_contrast and cut <= o.trim_max_frac * line.length:
            return RowChange(i, ACTION_TRIM, cut, f"{side} {outward:.1f} m past the headland, trimmed {cut:.1f} m "
                             f"(contrast {ev_txt})", x, y, keep_lo, keep_hi)
        return RowChange(i, FLAG_OVERSHOOT, outward, f"{side} {outward:.1f} m past the headland, kept "
                         f"(contrast {ev_txt})", x, y)
    if -outward > o.under_m:
        x, y = fr.xy(cur, g.v)
        return _undershoot(i, line, fr, g, pred, is_end, -outward, s, o, evidence, (x, y))
    return None


def _cut_is_tail(line: LineString, fr: RowFrame, is_end: bool) -> bool:
    """True when the trimmed end (max u for is_end, min u otherwise) is the line's last vertex."""
    c = np.asarray(line.coords, dtype=np.float64)
    u, _ = fr.uv(c[[0, -1]])
    return bool((u[1] > u[0]) == is_end)


def _undershoot(i: int, line: LineString, fr: RowFrame, g: LatticeLine, pred: float, is_end: bool, gap: float,
                s: float, o: RegularizeOptions, evidence: Evidence, at: tuple[float, float]) -> RowChange:
    side = "end" if is_end else "start"
    msg = f"{side} {gap:.1f} m short of the headland"
    if not o.extend_enabled:
        return RowChange(i, ACTION_UNDERSHOOT, gap, msg, *at)
    cur = g.end if is_end else g.start
    ext = LineString([fr.xy(cur, g.v), fr.xy(pred, g.v)])
    ev = row_contrast(evidence, ext, s) if ext.length >= o.min_evidence_len_m else None
    if ev is None or ev < o.extend_min_contrast:
        return RowChange(i, ACTION_UNDERSHOOT, gap, f"{msg} (contrast {'n/a' if ev is None else f'{ev:.2f}'})", *at)
    tail = _cut_is_tail(line, fr, is_end)
    return RowChange(i, ACTION_EXTEND, gap, f"{msg}, extended (contrast {ev:.2f})", *at,
                     lo=0.0 if tail else -gap, hi=float(line.length) + gap if tail else float(line.length))


def headland_changes(lines: Sequence[LineString], fr: RowFrame, groups: Sequence[LatticeLine], s: float,
                     o: RegularizeOptions, evidence: Evidence) -> list[RowChange]:
    """Overshoot trims / flags and undershoot flags against both fitted headlands."""
    if len(groups) < o.min_block_rows:
        return []
    v = np.array([g.v for g in groups])
    out: list[RowChange] = []
    for is_end in (False, True):
        prof = np.array([g.end if is_end else g.start for g in groups])
        head = fit_headland(v, prof, o)
        for g, pred in zip(groups, head.pred, strict=True):
            i = g.end_member if is_end else g.start_member
            ch = _end_change(i, lines[i], fr, g, float(pred), is_end, s, o, evidence)
            if ch is not None:
                out.append(ch)
    return _merge_trims(out)


def _merge_trims(changes: Sequence[RowChange]) -> list[RowChange]:
    """A row trimmed at both ends keeps the intersection of the two kept spans (one change per row)."""
    trims: dict[int, RowChange] = {}
    rest = []
    for c in changes:
        if c.action != ACTION_TRIM:
            rest.append(c)
        elif c.index in trims:
            t = trims[c.index]
            trims[c.index] = RowChange(c.index, ACTION_TRIM, t.metres + c.metres, f"{t.message}; {c.message}", t.x,
                                       t.y, max(t.lo, c.lo), min(t.hi, c.hi))
        else:
            trims[c.index] = c
    return [*rest, *trims.values()]


def regularize_block(lines: Sequence[LineString], o: RegularizeOptions, evidence: Evidence) -> tuple[RowChange, ...]:
    """All changes / flags of one block's rows (indexes into `lines`), deterministic order."""
    if not o.enabled or len(lines) < o.min_block_rows:
        return ()
    fr = row_frame(lines)
    groups = lattice_lines(lines, fr, o.collinear_tol_m)
    s = lattice_spacing(np.array([g.v for g in groups]))
    if not math.isfinite(s):
        return ()
    changes = lattice_changes(lines, groups, s, o, evidence)
    removed = {c.index for c in changes if c.action in REMOVALS}
    kept = [g for g in groups if not set(g.members) & removed]
    strays = stray_changes(lines, kept, o)
    changes += strays
    removed |= {c.index for c in strays}
    kept = [g for g in kept if not set(g.members) & removed]
    changes += headland_changes(lines, fr, kept, s, o, evidence)
    return tuple(sorted(changes, key=lambda c: (c.index, c.action)))


# ---------------------------------------------------------------- applying changes


def apply_span(line: LineString, lo: float, hi: float) -> LineString:
    """`line` restricted to the along span [lo, hi]; lo < 0 / hi > length extrapolate the end segments."""
    c = np.asarray(line.coords, dtype=np.float64)[:, :2]
    pre, post = max(0.0, -lo), max(0.0, hi - float(line.length))
    if pre > 0.0 or post > 0.0:
        d0 = (c[0] - c[1]) / max(float(np.hypot(*(c[0] - c[1]))), _EPS)
        d1 = (c[-1] - c[-2]) / max(float(np.hypot(*(c[-1] - c[-2]))), _EPS)
        pts = ([c[0] + d0 * pre] if pre > 0.0 else []) + list(c) + ([c[-1] + d1 * post] if post > 0.0 else [])
        line, lo, hi = LineString(pts), lo + pre, hi + pre
    return _piece(line, max(0.0, lo), min(float(line.length), hi))


def point_of(change: RowChange) -> Point:
    return Point(change.x, change.y)
