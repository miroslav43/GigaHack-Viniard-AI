"""Global row linking (02 §3.5, plan S4): support-line greedy union-find across tiles in UTM.

Pieces = per-tile row candidates. The longer piece of a pair is the reference: |Δθ| <= angle max when both
are long, the shorter piece's ends within offset max of the reference support line, along gap <= N tiles,
connector outside the eroded passages. Unions that put two overlapping pieces of one tile into a chain are
refused. Chains are refit (TLS line or DP polyline), split at unsupported holes, and end-snapped to the clip.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from vineyard.config import AppConfig
from vineyard.contracts.enums import Severity, Source
from vineyard.contracts.ids import format_chain_id, parse_tile_id
from vineyard.contracts.ordering import angle_deg_utm
from vineyard.contracts.qa import QaIssue
from vineyard.errors import StageError
from vineyard.geo.tiling import CRS_EPSG, TILE_M, tile_box, tile_of_point, tile_ref, tiles_for_bounds
from vineyard.perception.blocks_graph import LineFrame, axial_diff_deg, line_coords, line_frame
from vineyard.perception.row_features import SOFT_REJECT_REASONS
from vineyard.perception.rows_seams import duplicate_pairs, groups_of, join_plan

PIECE_ATTRS: Final = ("width_p80_m", "along_duty", "along_period_m", "harmonic_flag", "is_curved")
FLAG_RESCUED: Final = "row_rescued_low_snr"
FLAG_INTERPOLATED: Final = "row_interpolated"
FLAG_CURVED: Final = "curved_row"
FLAG_SEP: Final = ";"
LIST_SEP: Final = ","
DECISION_HARD: Final = "hard_rejected"
DECISION_SOFT_DROPPED: Final = "soft_dropped"
DECISION_DUPLICATE: Final = "duplicate"
DECISION_SOFT_CHAIN: Final = "soft_only_chain"
DECISION_LINKED: Final = "linked"
DECISION_RESCUED: Final = "rescued"
EPS_M: Final = 1e-6
SNAP_PROBE_M: Final = 1e-3
JSON_DECIMALS: Final = 3


@dataclass(frozen=True)
class LinkSettings:
    angle_max_deg: float
    offset_max_m: float
    gap_max_m: float
    neighbour_tiles: int
    angle_min_len_m: float
    passage_erode_m: float
    dup_overlap_m: float
    rescue_soft: bool
    residual_split_m: float
    dp_tolerance_m: float
    vertex_m: float
    snap_m: float
    gap_record_min_m: float
    hole_split_m: float
    neighbour_max_m: float
    parallel_max_deg: float
    interp_min_m: float
    hole_support_min_frac: float
    refit_sample_m: float
    seam_join_max_m: float = 0.0
    seam_join_angle_max_deg: float = 3.0
    seam_align_min_m: float = 0.0
    dup_chain_max_m: float = 0.0
    dup_chain_min_frac: float = 1.0

    @classmethod
    def from_config(cls, cfg: AppConfig) -> LinkSettings:
        lk, det, blk = cfg.rows.link, cfg.rows.detect, cfg.blocks
        return cls(
            angle_max_deg=lk.link_angle_max_deg, offset_max_m=lk.link_offset_max_m,
            gap_max_m=lk.link_gap_max_tiles * TILE_M, neighbour_tiles=1 + lk.link_gap_max_tiles,
            angle_min_len_m=lk.angle_min_len_m, passage_erode_m=lk.passage_erode_m, dup_overlap_m=lk.dup_overlap_m,
            rescue_soft=lk.rescue_soft_rejects, residual_split_m=det.residual_split_m,
            dp_tolerance_m=det.dp_tolerance_m, vertex_m=det.track_vertex_m, snap_m=det.snap_to_edge_m,
            gap_record_min_m=det.gap_record_min_m, hole_split_m=blk.collinear_gap_max_m,
            neighbour_max_m=blk.neighbour_max_m, parallel_max_deg=blk.parallel_max_deg,
            interp_min_m=cfg.export.min_row_piece_m, hole_support_min_frac=lk.hole_support_min_frac,
            refit_sample_m=lk.refit_sample_m, seam_join_max_m=lk.seam_join_max_m,
            seam_join_angle_max_deg=lk.seam_join_angle_max_deg, seam_align_min_m=lk.seam_align_min_m,
            dup_chain_max_m=lk.dup_chain_max_m,
            dup_chain_min_frac=lk.dup_chain_min_frac,
        )


@dataclass(frozen=True, eq=False)
class Piece:
    idx: int
    cand_id: str
    tile_id: str
    grid: tuple[int, int]
    line: LineString
    frame: LineFrame
    soft: bool
    gaps: tuple[tuple[float, float], ...]
    attrs: Mapping[str, float]

    @property
    def length(self) -> float:
        return float(self.line.length)


@dataclass(frozen=True)
class LinkResult:
    rows_raw: gpd.GeoDataFrame
    decisions: pd.DataFrame
    issues: tuple[QaIssue, ...]


# ---------------------------------------------------------------- candidates -> pieces


def _status(reason: object) -> str:
    if reason is None or (isinstance(reason, float) and math.isnan(reason)) or str(reason).strip() == "":
        return "accepted"
    return "soft" if str(reason) in SOFT_REJECT_REASONS else "hard"


def parse_gaps(text: object, cand_id: str) -> tuple[tuple[float, float], ...]:
    """gaps_json: [[start_m, end_m], ...] or [{"start_m":..,"end_m":..}, ...] along the line; blank = none."""
    if text is None or (isinstance(text, float) and math.isnan(text)) or str(text).strip() == "":
        return ()
    try:
        items = json.loads(str(text))
        pairs = [(float(g["start_m"]), float(g["end_m"])) if isinstance(g, dict) else (float(g[0]), float(g[1]))
                 for g in items]
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise StageError(f"unreadable gaps_json: {exc}", cand_id=cand_id, text=str(text)[:80]) from exc
    return tuple(sorted((min(a, b), max(a, b)) for a, b in pairs))


def _attr(row: pd.Series, name: str) -> float:
    value = row.get(name, math.nan)
    return math.nan if value is None or pd.isna(value) else float(value)


def _make_pieces(cands: gpd.GeoDataFrame, s: LinkSettings) -> tuple[list[Piece], list[tuple[str, str]]]:
    pieces, decisions = [], []
    for _, row in cands.sort_values("cand_id", kind="stable").iterrows():
        status = _status(row.get("rejected_reason"))
        if status == "hard" or (status == "soft" and not s.rescue_soft):
            decisions.append((row.cand_id, DECISION_HARD if status == "hard" else DECISION_SOFT_DROPPED))
            continue
        line = LineString(line_coords(row.geometry))
        attrs = {k: _attr(row, k) for k in PIECE_ATTRS}
        pieces.append(Piece(len(pieces), str(row.cand_id), str(row.tile_id), parse_tile_id(str(row.tile_id)), line,
                            line_frame(line), status == "soft", parse_gaps(row.get("gaps_json"), row.cand_id), attrs))
    return pieces, decisions


# ---------------------------------------------------------------- link test


def _extended_offset(ref: Piece, pts: np.ndarray) -> float:
    """Max distance of `pts` to ref's support line (its end segments extended beyond the ends)."""
    coords = line_coords(ref.line)
    if len(coords) == 2:
        return float(np.abs(ref.frame.across(pts)).max())
    out = []
    for p, t in zip(pts, ref.frame.along(pts), strict=True):
        seg = coords[:2] if t < 0 else coords[-2:] if t > ref.frame.length else None
        out.append(float(Point(p).distance(ref.line)) if seg is None else
                   abs(float(line_frame(LineString(seg)).across(p[None, :])[0])))
    return max(out)


def _interval(ref: Piece, other: Piece) -> tuple[float, float]:
    t = np.sort(ref.frame.along(line_coords(other.line)[[0, -1]]))
    return float(t[0]), float(t[1])


def _overlap_gap(ref: Piece, other: Piece) -> tuple[float, float]:
    t0, t1 = _interval(ref, other)
    lo, hi = max(0.0, t0), min(ref.frame.length, t1)
    return max(0.0, hi - lo), max(0.0, lo - hi)


def _connector(ref: Piece, other: Piece) -> LineString:
    a, b = line_coords(ref.line)[[0, -1]], line_coords(other.line)[[0, -1]]
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
    return LineString([a[i], b[j]])


def link_cost(ref: Piece, other: Piece, s: LinkSettings, passages: BaseGeometry | None) -> tuple[float, float] | None:
    """(offset, gap) if `other` (the shorter piece) links onto `ref`, else None."""
    if min(ref.length, other.length) >= s.angle_min_len_m and \
            axial_diff_deg(ref.frame.angle_deg, other.frame.angle_deg) > s.angle_max_deg:
        return None
    offset = _extended_offset(ref, line_coords(other.line)[[0, -1]])
    overlap, gap = _overlap_gap(ref, other)
    if offset > s.offset_max_m + EPS_M or gap > s.gap_max_m + EPS_M:
        return None
    if ref.tile_id == other.tile_id and overlap > s.dup_overlap_m:
        return None
    if gap > EPS_M and passages is not None and _connector(ref, other).intersects(passages):
        return None
    return offset, gap


def _dedupe(pieces: Sequence[Piece], s: LinkSettings) -> tuple[list[Piece], list[tuple[str, str]]]:
    """Same-tile duplicates (multi-orientation): the shorter of two overlapping collinear pieces is dropped."""
    kept: list[Piece] = []
    dups = []
    for p in sorted(pieces, key=lambda q: (-q.length, q.cand_id)):
        same = [k for k in kept if k.tile_id == p.tile_id]
        if any(_extended_offset(k, line_coords(p.line)[[0, -1]]) <= s.offset_max_m and
               _overlap_gap(k, p)[0] > s.dup_overlap_m for k in same):
            dups.append((p.cand_id, DECISION_DUPLICATE))
        else:
            kept.append(p)
    ordered = sorted(kept, key=lambda q: q.cand_id)
    return [replace(p, idx=i) for i, p in enumerate(ordered)], dups


def _neighbours(pieces: Sequence[Piece], reach: int) -> dict[int, list[int]]:
    by_tile: dict[tuple[int, int], list[int]] = {}
    for p in pieces:
        by_tile.setdefault(p.grid, []).append(p.idx)
    out = {}
    for p in pieces:
        r, c = p.grid
        near = [j for dr in range(-reach, reach + 1) for dc in range(-reach, reach + 1)
                for j in by_tile.get((r + dr, c + dc), [])]
        out[p.idx] = sorted(j for j in near if j > p.idx)
    return out




@dataclass(frozen=True, eq=False)
class _Arrays:
    """Chord-frame arrays of all pieces, for the vectorised pair prefilter."""

    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    chord: np.ndarray
    length: np.ndarray
    angle: np.ndarray
    dev: np.ndarray

    @classmethod
    def of(cls, pieces: Sequence[Piece]) -> _Arrays:
        ends = np.array([line_coords(p.line)[[0, -1]] for p in pieces]).reshape(-1, 2, 2)
        dev = [p.line.hausdorff_distance(LineString(line_coords(p.line)[[0, -1]])) for p in pieces]
        return cls(ends[:, 0], ends[:, 1], np.array([p.frame.direction for p in pieces]).reshape(-1, 2),
                   np.array([p.frame.length for p in pieces]), np.array([p.length for p in pieces]),
                   np.array([p.frame.angle_deg for p in pieces]), np.asarray(dev, dtype=np.float64))


def _frame_stats(origin: np.ndarray, d: np.ndarray, chord: np.ndarray, p0: np.ndarray, p1: np.ndarray):
    """Per pair: max |across| of (p0, p1) and their along gap to [0, chord], in the (origin, d) frames."""
    n = np.column_stack((-d[:, 1], d[:, 0]))
    t = np.column_stack((((p0 - origin) * d).sum(1), ((p1 - origin) * d).sum(1)))
    o = np.column_stack((((p0 - origin) * n).sum(1), ((p1 - origin) * n).sum(1)))
    lo, hi = np.maximum(0.0, t.min(1)), np.minimum(chord, t.max(1))
    return np.abs(o).max(1), np.maximum(0.0, lo - hi)


def _prefilter(i: int, js: np.ndarray, arr: _Arrays, s: LinkSettings) -> np.ndarray:
    k = len(js)
    ref_i = arr.length[i] >= arr.length[js]
    rep = np.repeat
    origin = np.where(ref_i[:, None], rep(arr.start[i][None], k, 0), arr.start[js])
    d = np.where(ref_i[:, None], rep(arr.direction[i][None], k, 0), arr.direction[js])
    chord = np.where(ref_i, arr.chord[i], arr.chord[js])
    p0 = np.where(ref_i[:, None], arr.start[js], rep(arr.start[i][None], k, 0))
    p1 = np.where(ref_i[:, None], arr.end[js], rep(arr.end[i][None], k, 0))
    offset, gap = _frame_stats(origin, d, chord, p0, p1)
    dev = np.where(ref_i, arr.dev[i], arr.dev[js])
    diff = np.abs(arr.angle[i] - arr.angle[js]) % 180.0
    short = np.minimum(arr.length[i], arr.length[js]) < s.angle_min_len_m
    angle_ok = short | (np.minimum(diff, 180.0 - diff) <= s.angle_max_deg + EPS_M)
    return angle_ok & (offset <= s.offset_max_m + dev + EPS_M) & (gap <= s.gap_max_m + EPS_M)


def link_pairs(pieces: Sequence[Piece], s: LinkSettings, passages: BaseGeometry | None) -> list[tuple]:
    """Sorted (offset, gap, cand_a, cand_b, i, j) for every linkable pair (i < j)."""
    if len(pieces) < 2:
        return []
    arr, pairs = _Arrays.of(pieces), []
    for i, js in _neighbours(pieces, s.neighbour_tiles).items():
        if not js:
            continue
        cand = np.asarray(js, dtype=np.intp)
        for j in cand[_prefilter(i, cand, arr, s)]:
            a, b = pieces[i], pieces[int(j)]
            ref, oth = (a, b) if a.length >= b.length else (b, a)
            cost = link_cost(ref, oth, s, passages)
            if cost is not None:
                pairs.append((round(cost[0], 9), round(cost[1], 9), a.cand_id, b.cand_id, i, int(j)))
    return sorted(pairs)


# ---------------------------------------------------------------- greedy constrained union


def _conflict(a: Sequence[int], b: Sequence[int], pieces: Sequence[Piece], s: LinkSettings) -> bool:
    for i in a:
        for j in b:
            p, q = pieces[i], pieces[j]
            if p.tile_id == q.tile_id and _overlap_gap(p, q)[0] > s.dup_overlap_m:
                return True
    return False


def greedy_chains(pieces: Sequence[Piece], pairs: Sequence[tuple], s: LinkSettings) -> list[tuple[int, ...]]:
    """Union pieces in ascending (offset, gap) order, refusing same-tile overlaps; chains sorted by first member."""
    root = list(range(len(pieces)))
    members = {i: [i] for i in range(len(pieces))}

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for *_, i, j in pairs:
        ra, rb = find(i), find(j)
        if ra == rb or _conflict(members[ra], members[rb], pieces, s):
            continue
        keep, drop = min(ra, rb), max(ra, rb)
        root[drop] = keep
        members[keep] = sorted(members[keep] + members.pop(drop))
    return sorted(tuple(m) for m in members.values())


# ---------------------------------------------------------------- refit, snap, holes


def _samples(line: LineString, step: float) -> np.ndarray:
    n = max(1, math.ceil(line.length / step))
    return np.array([line.interpolate(k / n, normalized=True).coords[0] for k in range(n + 1)])


def _canonical_direction(d: np.ndarray) -> np.ndarray:
    return -d if (d[0] < -EPS_M or (abs(d[0]) <= EPS_M and d[1] < 0)) else d


def refit_chain(lines: Sequence[LineString], s: LinkSettings) -> LineString:
    """TLS 2-point line when every sample is within residual_split_m, else a binned DP-simplified polyline."""
    pts = np.vstack([_samples(ln, s.refit_sample_m) for ln in lines])
    c = pts.mean(axis=0)
    d = _canonical_direction(np.linalg.svd(pts - c, full_matrices=False)[2][0])
    n = np.array([-d[1], d[0]])
    t, o = (pts - c) @ d, (pts - c) @ n
    if np.abs(o).max() <= s.residual_split_m:
        return LineString([c + t.min() * d, c + t.max() * d])
    edges = np.arange(t.min(), t.max() + s.vertex_m, s.vertex_m)
    bins = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 1)
    verts = [(t.min(), o[np.argmin(t)])] + [(t[bins == b].mean(), o[bins == b].mean()) for b in np.unique(bins)]
    verts.append((t.max(), o[np.argmax(t)]))
    tv = np.array(sorted(verts))
    tv = tv[np.concatenate(([True], np.diff(tv[:, 0]) > EPS_M))]
    xy = c + tv[:, :1] * d + tv[:, 1:] * n
    return LineString(xy).simplify(s.dp_tolerance_m, preserve_topology=False)


def _clip_for(p: np.ndarray, clips: Mapping[str, BaseGeometry]) -> BaseGeometry | None:
    try:
        tile_id = tile_of_point(float(p[0]), float(p[1])).tile_id
    except ValueError:  # outside the grid: nothing to snap to
        return None
    return clips.get(tile_id)


def _snapped_end(end: np.ndarray, inner: np.ndarray, clips: Mapping[str, BaseGeometry], max_m: float) -> np.ndarray:
    u = (end - inner) / np.hypot(*(end - inner))
    clip = _clip_for(end - u * SNAP_PROBE_M, clips)
    if clip is None or max_m <= 0.0:
        return end
    ray = LineString([end, end + u * max_m])
    parts = [g for g in getattr(ray.intersection(clip), "geoms", [ray.intersection(clip)]) if not g.is_empty]
    start = Point(end)
    touching = [g for g in parts if g.distance(start) <= EPS_M and g.geom_type == "LineString"]
    if not touching:
        return end
    far = max((np.asarray(c) for g in touching for c in g.coords), key=lambda c: np.hypot(*(c - end)))
    return far if np.hypot(*(far - end)) < max_m - EPS_M else end


def extend_chain_ends(line: LineString, clips: Mapping[str, BaseGeometry], max_m: float) -> LineString:
    """Extend each end along its end segment to the boundary of its tile clip when that is <= max_m away."""
    xy = line_coords(line)
    first = _snapped_end(xy[0], xy[1], clips, max_m)
    last = _snapped_end(xy[-1], xy[-2], clips, max_m)
    return LineString(np.vstack([first, xy[1:-1], last]))


def merge_intervals(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in spans):
        if merged and a <= merged[-1][1] + EPS_M:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _coverage(line: LineString, members: Sequence[LineString]) -> list[tuple[float, float]]:
    return merge_intervals([(line.project(Point(m.coords[0])), line.project(Point(m.coords[-1]))) for m in members])


def holes(line: LineString, members: Sequence[LineString]) -> list[tuple[float, float]]:
    """Interior along intervals of `line` not covered by any member piece."""
    cov = _coverage(line, members)
    return [(a[1], b[0]) for a, b in zip(cov, cov[1:], strict=False) if b[0] - a[1] > EPS_M]


# ---------------------------------------------------------------- S4 split, chain records


@dataclass(frozen=True, eq=False)
class _Chain:
    members: tuple[int, ...]
    line: LineString


def _build_chain(members: Sequence[int], pieces: Sequence[Piece], s: LinkSettings) -> _Chain:
    line = refit_chain([pieces[i].line for i in members], s)
    order = sorted(members, key=lambda i: (line.project(pieces[i].line.centroid), pieces[i].cand_id))
    return _Chain(tuple(order), line)


def _member_lines(ch: _Chain, pieces: Sequence[Piece]) -> list[LineString]:
    return [pieces[i].line for i in ch.members]


def _neighbour_covers(line: LineString, hole: tuple[float, float], other: _Chain, pieces: Sequence[Piece]) -> float:
    a, b = line.interpolate(hole[0]), line.interpolate(hole[1])
    u0, u1 = sorted((other.line.project(a), other.line.project(b)))
    if u1 - u0 <= EPS_M:
        return 0.0
    cov = _coverage(other.line, _member_lines(other, pieces))
    return sum(max(0.0, min(y, u1) - max(x, u0)) for x, y in cov) / (u1 - u0)


def _supported(k: int, hole: tuple[float, float], chains: Sequence[_Chain], tree: shapely.STRtree,
               pieces: Sequence[Piece], s: LinkSettings) -> bool:
    line = chains[k].line
    mid = line.interpolate(0.5 * (hole[0] + hole[1]))
    angle = line_frame(line).angle_deg
    for m in sorted(int(v) for v in tree.query(mid, predicate="dwithin", distance=s.neighbour_max_m)):
        other = chains[m]
        if m == k or other.line.distance(mid) <= s.offset_max_m:
            continue
        if axial_diff_deg(angle, line_frame(other.line).angle_deg) > s.parallel_max_deg:
            continue
        if _neighbour_covers(line, hole, other, pieces) >= s.hole_support_min_frac:
            return True
    return False


def _partition(ch: _Chain, cuts: Sequence[float], pieces: Sequence[Piece]) -> list[tuple[int, ...]]:
    pos = [ch.line.project(pieces[i].line.centroid) for i in ch.members]
    groups: dict[int, list[int]] = {}
    for i, g in zip(ch.members, np.searchsorted(np.sort(cuts), pos), strict=True):
        groups.setdefault(int(g), []).append(i)
    return [tuple(sorted(v)) for _, v in sorted(groups.items())]


def split_unsupported(chains: Sequence[_Chain], pieces: Sequence[Piece], s: LinkSettings) -> list[_Chain]:
    """Split chains at holes >= hole_split_m unless a parallel neighbour row has pieces covering the hole."""
    if not chains:
        return []
    tree = shapely.STRtree([c.line for c in chains])
    out: list[_Chain] = []
    for k, ch in enumerate(chains):
        cuts = [0.5 * (h0 + h1) for h0, h1 in holes(ch.line, _member_lines(ch, pieces))
                if h1 - h0 >= s.hole_split_m and not _supported(k, (h0, h1), chains, tree, pieces, s)]
        out.extend([ch] if not cuts else [_build_chain(g, pieces, s) for g in _partition(ch, cuts, pieces)])
    return out


def tile_spans(line: LineString, clips: Mapping[str, BaseGeometry]) -> list[tuple[str, float, float]]:
    """(tile_id, along_from, along_to) of the line inside each tile clip (first to last vertex), axis order."""
    out = []
    for t in tiles_for_bounds(*line.bounds):
        clip = clips[t] if t in clips else tile_box(tile_ref(t))
        inter = line.intersection(clip)
        coords = [c for g in getattr(inter, "geoms", [inter]) if not g.is_empty for c in g.coords]
        if not coords:
            continue
        pos = [line.project(Point(c)) for c in coords]
        if max(pos) - min(pos) > EPS_M:
            out.append((t, float(min(pos)), float(max(pos))))
    return sorted(out, key=lambda x: (x[1], x[0]))


def _weighted(values: Sequence[float], weights: Sequence[float]) -> float:
    v, w = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(v)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else math.nan


def _chain_gaps(line: LineString, mem: Sequence[Piece], s: LinkSettings) -> str:
    spans = [h for h in holes(line, [p.line for p in mem]) if h[1] - h[0] >= s.gap_record_min_m]
    for p in mem:
        spans.extend((line.project(p.line.interpolate(a)), line.project(p.line.interpolate(b))) for a, b in p.gaps)
    merged = merge_intervals([sp for sp in spans if abs(sp[1] - sp[0]) > EPS_M])
    return json.dumps([[round(a, JSON_DECIMALS), round(b, JSON_DECIMALS)] for a, b in merged])


def _chain_record(chain_id: str, ch: _Chain, pieces: Sequence[Piece], clips: Mapping[str, BaseGeometry],
                  s: LinkSettings) -> dict[str, object]:
    line, mem = ch.line, [pieces[i] for i in ch.members]
    lengths = [p.length for p in mem]
    spans = tile_spans(line, clips)
    member_tiles = {p.tile_id for p in mem}
    interp = [t for t, lo, hi in spans if hi - lo >= s.interp_min_m and t not in member_tiles]
    rescued = sum(p.length for p in mem if p.soft) / sum(lengths)
    curved = len(line.coords) > 2 or any(p.attrs["is_curved"] == 1.0 for p in mem)
    support = min(1.0, sum(lengths) / line.length)
    flags = [f for f, on in ((FLAG_RESCUED, rescued > 0), (FLAG_INTERPOLATED, bool(interp)), (FLAG_CURVED, curved))
             if on]
    xy = line_coords(line)
    return {
        "chain_id": chain_id, "member_cand_ids": LIST_SEP.join(sorted(p.cand_id for p in mem)),
        "tile_ids": LIST_SEP.join(t for t, _, _ in spans), "n_tiles": len(spans),
        "angle_deg": angle_deg_utm(tuple(xy[0]), tuple(xy[-1])), "extent_m": float(line.length),
        "support_frac": support, "width_p80_m": _weighted([p.attrs["width_p80_m"] for p in mem], lengths),
        "along_duty": _weighted([p.attrs["along_duty"] for p in mem], lengths),
        "along_period_m": _weighted([p.attrs["along_period_m"] for p in mem], lengths),
        "harmonic_frac": _weighted([float(p.attrs["harmonic_flag"] == 1.0) for p in mem], lengths),
        "rescued_frac": rescued, "is_curved": curved, "gaps_json": _chain_gaps(line, mem, s),
        "interp_tile_ids": LIST_SEP.join(interp), "n_members": len(mem), "source": Source.MODEL.value,
        "run_id": "", "model_version": "", "confidence": support, "qa_flags": FLAG_SEP.join(flags), "geometry": line,
    }


# ---------------------------------------------------------------- orchestration


ROWS_RAW_COLUMNS: Final = (
    "chain_id", "member_cand_ids", "tile_ids", "n_tiles", "angle_deg", "extent_m", "support_frac", "width_p80_m",
    "along_duty", "along_period_m", "harmonic_frac", "rescued_frac", "is_curved", "gaps_json", "interp_tile_ids", "n_members",
    "source", "run_id", "model_version", "confidence", "qa_flags", "geometry",
)


def eroded(passages: BaseGeometry | None, erode_m: float) -> BaseGeometry | None:
    if passages is None or passages.is_empty:
        return None
    shrunk = passages.buffer(-erode_m) if erode_m > 0 else passages
    if shrunk.is_empty:
        return None
    shapely.prepare(shrunk)
    return shrunk


def _issues(frame: gpd.GeoDataFrame, chains: Sequence[_Chain], pieces: Sequence[Piece]) -> tuple[QaIssue, ...]:
    out = []
    for rec, ch in zip(frame.itertuples(index=False), chains, strict=True):
        soft = [pieces[i] for i in ch.members if pieces[i].soft]
        if soft:
            p = soft[0].line.interpolate(0.5, normalized=True)
            out.append(QaIssue(Severity.INFO, FLAG_RESCUED, soft[0].tile_id, rec.chain_id,
                               f"{len(soft)} low-SNR pieces rescued ({rec.rescued_frac:.2f} of length)", p.x, p.y))
        if rec.is_curved:
            p = rec.geometry.interpolate(0.5, normalized=True)
            out.append(QaIssue(Severity.INFO, FLAG_CURVED, pieces[ch.members[0]].tile_id, rec.chain_id,
                               "chain refit as a polyline", p.x, p.y))
    return tuple(out)


def _decisions(pre: Sequence[tuple[str, str]], chains: Sequence[_Chain], ids: Sequence[str],
               pieces: Sequence[Piece], dropped: Sequence[int]) -> pd.DataFrame:
    rows = [(c, d, None) for c, d in pre] + [(pieces[i].cand_id, DECISION_SOFT_CHAIN, None) for i in dropped]
    for cid, ch in zip(ids, chains, strict=True):
        rows.extend((pieces[i].cand_id, DECISION_RESCUED if pieces[i].soft else DECISION_LINKED, cid)
                    for i in ch.members)
    return pd.DataFrame(sorted(rows, key=lambda r: r[0]), columns=["cand_id", "link_decision", "chain_id"])


def merge_duplicate_chains(chains: Sequence[_Chain], pieces: Sequence[Piece], clips: Mapping[str, BaseGeometry],
                           s: LinkSettings) -> list[_Chain]:
    """Chains running within dup_chain_max_m of each other over >= dup_chain_min_frac of the shorter one
    (one row detected twice, e.g. by two orientations or a guided pass) become one refit chain."""
    if s.dup_chain_max_m <= 0.0 or len(chains) < 2:
        return list(chains)
    groups = groups_of(duplicate_pairs([c.line for c in chains], s.dup_chain_max_m, s.dup_chain_min_frac,
                                       s.seam_join_angle_max_deg), len(chains))
    grouped = {k for g in groups for k in g}
    built = [_build_chain(sorted(i for k in g for i in chains[k].members), pieces, s) for g in groups]
    merged = [replace(ch, line=extend_chain_ends(ch.line, clips, s.snap_m)) for ch in built]
    return [c for k, c in enumerate(chains) if k not in grouped] + merged


def join_chain_ends(chains: Sequence[_Chain], pieces: Sequence[Piece], s: LinkSettings) -> list[_Chain]:
    """Chains whose facing ends meet (<= seam_join_max_m, head to tail, <= seam_join_angle_max_deg) are one
    physical row cut at a tile edge: joined into one chain (seam vertex = midpoint of the two ends)."""
    if s.seam_join_max_m <= 0.0 or len(chains) < 2:
        return list(chains)
    plan = join_plan([c.line for c in chains], s.seam_join_max_m, s.seam_join_angle_max_deg, s.seam_align_min_m)
    joined = {k for path, _ in plan for k in path}
    out = [c for k, c in enumerate(chains) if k not in joined]
    for path, line in plan:
        members = [i for k in path for i in chains[k].members]
        order = sorted(members, key=lambda i: (line.project(pieces[i].line.centroid), pieces[i].cand_id))
        out.append(_Chain(tuple(order), line))
    return out


def link_candidates(cands: gpd.GeoDataFrame, passages: BaseGeometry | None, clips: Mapping[str, BaseGeometry],
                    s: LinkSettings) -> LinkResult:
    """row_candidates (UTM) -> rows_raw chains L00001.. (deterministic for any input order)."""
    pieces, pre = _make_pieces(cands, s)
    pieces, dups = _dedupe(pieces, s)
    groups = greedy_chains(pieces, link_pairs(pieces, s, eroded(passages, s.passage_erode_m)), s)
    chains = split_unsupported([_build_chain(g, pieces, s) for g in groups], pieces, s)
    keep = [ch for ch in chains if not all(pieces[i].soft for i in ch.members)]
    dropped = sorted(i for ch in chains if all(pieces[i].soft for i in ch.members) for i in ch.members)
    keep = [replace(ch, line=extend_chain_ends(ch.line, clips, s.snap_m)) for ch in keep]
    keep = join_chain_ends(merge_duplicate_chains(keep, pieces, clips, s), pieces, s)
    keep = sorted(keep, key=lambda ch: min(pieces[i].cand_id for i in ch.members))
    ids = [format_chain_id(k + 1) for k in range(len(keep))]
    records = [_chain_record(cid, ch, pieces, clips, s) for cid, ch in zip(ids, keep, strict=True)]
    frame = gpd.GeoDataFrame(records if records else None, columns=list(ROWS_RAW_COLUMNS), geometry="geometry",
                             crs=CRS_EPSG)
    return LinkResult(frame, _decisions([*pre, *dups], keep, ids, pieces, dropped), _issues(frame, keep, pieces))
