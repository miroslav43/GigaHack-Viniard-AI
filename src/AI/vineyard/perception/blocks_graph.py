"""Row graph for blocks (02 §3.6, plan S3): pair geometry, neighbour edges, adjacency, components, bands.

Rows are UTM LineStrings. Every function is pure: inputs are never modified, outputs are new objects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from vineyard.config import BlocksConfig, RowsDetectConfig, RowsLinkConfig
from vineyard.contracts.ordering import mean_axial_angle_deg, order_by_normal
from vineyard.errors import StageError
from vineyard.geo.ops import split_multi
from vineyard.perception.types import readonly

EDGE_PARALLEL: Final = "parallel"
EDGE_SKIP_ONE: Final = "skip_one"
EDGE_COLLINEAR: Final = "collinear"
ADJACENT_KINDS: Final = (EDGE_PARALLEL, EDGE_SKIP_ONE)
EDGE_COLUMNS: Final = ("a", "b", "kind", "spacing_m", "overlap_from_m", "overlap_to_m", "angle_diff_deg", "geometry")
AXIAL_PERIOD_DEG: Final = 180.0


@dataclass(frozen=True, eq=False)
class LineFrame:
    """Chord frame of a line: origin at the first vertex, unit direction towards the last one."""

    origin: np.ndarray
    direction: np.ndarray
    length: float

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def along(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, dtype=np.float64) - self.origin) @ self.direction

    def across(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, dtype=np.float64) - self.origin) @ self.normal

    def point_at(self, t: float, o: float = 0.0) -> np.ndarray:
        return self.origin + t * self.direction + o * self.normal

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.direction[1], self.direction[0])) % AXIAL_PERIOD_DEG


def line_coords(line: LineString) -> np.ndarray:
    return np.asarray(line.coords, dtype=np.float64)[:, :2]


def line_frame(line: LineString) -> LineFrame:
    coords = line_coords(line)
    chord = coords[-1] - coords[0]
    length = float(np.hypot(chord[0], chord[1]))
    if length <= 0.0:
        raise StageError("row line has zero chord length", wkt=line.wkt[:80])
    return LineFrame(readonly(coords[0].copy()), readonly(chord / length), length)


def axial_diff_deg(a: float, b: float) -> float:
    """Smallest difference between two axial angles, in [0, 90]."""
    d = abs(float(a) - float(b)) % AXIAL_PERIOD_DEG
    return min(d, AXIAL_PERIOD_DEG - d)


@dataclass(frozen=True)
class PairGeometry:
    """Geometry of `other` relative to the chord frame of `ref` (along values are metres on ref)."""

    angle_diff_deg: float
    lateral_m: float
    overlap_from_m: float
    overlap_to_m: float
    gap_m: float
    spacing_m: float
    connector: LineString | None

    @property
    def overlap_m(self) -> float:
        return max(0.0, self.overlap_to_m - self.overlap_from_m)


def _end_connector(ref: LineString, other: LineString) -> LineString | None:
    a, b = line_coords(ref)[[0, -1]], line_coords(other)[[0, -1]]
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
    return None if dist[i, j] <= 0.0 else LineString([a[i], b[j]])


def pair_geometry(ref: LineString, other: LineString) -> PairGeometry:
    fr, fo = line_frame(ref), line_frame(other)
    ends = line_coords(other)[[0, -1]]
    t = np.sort(fr.along(ends))
    lateral = float(np.abs(fr.across(ends)).max())
    lo, hi = max(0.0, float(t[0])), min(fr.length, float(t[1]))
    angle = axial_diff_deg(fr.angle_deg, fo.angle_deg)
    if hi <= lo:
        return PairGeometry(angle, lateral, lo, hi, lo - hi, math.nan, _end_connector(ref, other))
    p = ref.interpolate(0.5 * (lo + hi))
    q = nearest_points(other, p)[0]
    spacing = float(p.distance(other))
    connector = LineString([p, q]) if spacing > 0.0 else None
    return PairGeometry(angle, lateral, lo, hi, 0.0, spacing, connector)


@dataclass(frozen=True)
class GraphSettings:
    parallel_max_deg: float
    min_overlap_frac: float
    spacing_min_m: float
    spacing_max_m: float
    collinear_lateral_m: float
    collinear_gap_max_m: float
    skip_one_enabled: bool
    skip_one_factor: tuple[float, float]
    band_min_m: float
    band_min_rows: int
    band_full_width: bool
    band_bin_m: float
    band_pad_m: float

    @classmethod
    def from_config(cls, blocks: BlocksConfig, detect: RowsDetectConfig, link: RowsLinkConfig) -> GraphSettings:
        return cls(
            parallel_max_deg=blocks.parallel_max_deg, min_overlap_frac=blocks.min_overlap_frac,
            spacing_min_m=detect.spacing_min_m, spacing_max_m=blocks.neighbour_max_m,
            collinear_lateral_m=link.link_offset_max_m, collinear_gap_max_m=blocks.collinear_gap_max_m,
            skip_one_enabled=blocks.skip_one_enabled, skip_one_factor=tuple(blocks.skip_one_spacing_factor),
            band_min_m=blocks.transverse_band_min_m, band_min_rows=blocks.headland_min_rows,
            band_full_width=blocks.transverse_full_width, band_bin_m=blocks.band_bin_m, band_pad_m=blocks.band_pad_m,
        )

    @property
    def max_parallel_m(self) -> float:
        factor = self.skip_one_factor[1] if self.skip_one_enabled else 1.0
        return self.spacing_max_m * max(factor, 1.0)

    @property
    def query_distance_m(self) -> float:
        return max(self.max_parallel_m, self.collinear_gap_max_m + self.collinear_lateral_m)


def _edge(a: int, b: int, kind: str, spacing: float, g: PairGeometry) -> dict[str, object]:
    return {"a": min(a, b), "b": max(a, b), "kind": kind, "spacing_m": spacing, "overlap_from_m": g.overlap_from_m,
            "overlap_to_m": g.overlap_to_m, "angle_diff_deg": g.angle_diff_deg, "geometry": g.connector}


def _classify(i: int, j: int, lines: Sequence[LineString], s: GraphSettings) -> dict[str, object] | None:
    ref, other = (lines[i], lines[j]) if lines[i].length >= lines[j].length else (lines[j], lines[i])
    g = pair_geometry(ref, other)
    if g.angle_diff_deg > s.parallel_max_deg:
        return None
    if g.lateral_m <= s.collinear_lateral_m and g.gap_m <= s.collinear_gap_max_m:
        return _edge(i, j, EDGE_COLLINEAR, g.lateral_m, g)
    if g.overlap_m <= 0.0 or g.overlap_m < s.min_overlap_frac * other.length:
        return None
    if not (s.spacing_min_m <= g.spacing_m <= s.max_parallel_m):
        return None
    return _edge(i, j, EDGE_PARALLEL, g.spacing_m, g)


def candidate_pairs(lines: Sequence[LineString], distance: float) -> np.ndarray:
    """(M, 2) index pairs i < j whose lines are within `distance`, sorted."""
    if len(lines) < 2:
        return np.zeros((0, 2), dtype=np.intp)
    tree = shapely.STRtree(list(lines))
    left, right = tree.query(list(lines), predicate="dwithin", distance=distance)
    keep = left < right
    pairs = np.column_stack((left[keep], right[keep])).astype(np.intp)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def _empty_edges() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in EDGE_COLUMNS})


def _regular_spacings(edges: pd.DataFrame, s: GraphSettings) -> dict[int, list[tuple[int, float]]]:
    out: dict[int, list[tuple[int, float]]] = {}
    for k, e in edges.iterrows():
        if e.kind == EDGE_PARALLEL and e.spacing_m <= s.spacing_max_m:
            out.setdefault(int(e.a), []).append((int(k), float(e.spacing_m)))
            out.setdefault(int(e.b), []).append((int(k), float(e.spacing_m)))
    return out


def _local_spacing(k: int, a: int, b: int, regular: dict[int, list[tuple[int, float]]]) -> float:
    values = [sp for row in (a, b) for idx, sp in regular.get(row, []) if idx != k]
    return min(values) if values else math.nan


def _has_row_between(a: int, b: int, spacing: float, nbrs: dict[int, dict[int, float]]) -> bool:
    common = set(nbrs.get(a, {})) & set(nbrs.get(b, {}))
    return any(nbrs[a][c] < spacing and nbrs[b][c] < spacing for c in sorted(common))


def _neighbour_map(edges: pd.DataFrame, kinds: Sequence[str]) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for e in edges.itertuples(index=False):
        if e.kind in kinds:
            out.setdefault(int(e.a), {})[int(e.b)] = float(e.spacing_m)
            out.setdefault(int(e.b), {})[int(e.a)] = float(e.spacing_m)
    return out


def _resolve_kind(k: int, e: pd.Series, regular: dict, nbrs: dict, s: GraphSettings) -> str | None:
    is_regular = e.spacing_m <= s.spacing_max_m
    if e.kind != EDGE_PARALLEL or not s.skip_one_enabled:
        return e.kind if (e.kind != EDGE_PARALLEL or is_regular) else None
    s_loc = _local_spacing(k, int(e.a), int(e.b), regular)
    lo, hi = s.skip_one_factor
    skip_like = math.isfinite(s_loc) and lo * s_loc <= e.spacing_m <= hi * s_loc
    if skip_like and not _has_row_between(int(e.a), int(e.b), float(e.spacing_m), nbrs):
        return EDGE_SKIP_ONE
    return EDGE_PARALLEL if is_regular else None


def _resolve_skip_one(edges: pd.DataFrame, s: GraphSettings) -> pd.DataFrame:
    regular = _regular_spacings(edges, s)
    nbrs = _neighbour_map(edges[edges.spacing_m <= s.spacing_max_m], (EDGE_PARALLEL,))
    kinds = [_resolve_kind(int(k), e, regular, nbrs, s) for k, e in edges.iterrows()]
    out = edges.assign(kind=kinds)
    return out[out.kind.notna()].reset_index(drop=True)


def crosses_cut(connector: LineString | None, cut: BaseGeometry | None) -> bool:
    """True when the connector runs through `cut` with both ends outside it (a row lying inside an
    imprecise passage polygon, like r021 R25, keeps its edges)."""
    if connector is None or cut is None or cut.is_empty or not connector.intersects(cut):
        return False
    ends = line_coords(connector)[[0, -1]]
    return not any(cut.intersects(Point(e)) for e in ends)


def neighbour_pairs(lines: Sequence[LineString], s: GraphSettings, cut: BaseGeometry | None = None) -> pd.DataFrame:
    """Graph edges (parallel, skip_one, collinear) between row lines; connectors crossing `cut` are dropped."""
    records = []
    for i, j in candidate_pairs(lines, s.query_distance_m):
        rec = _classify(int(i), int(j), lines, s)
        if rec is None or crosses_cut(rec["geometry"], cut):
            continue
        records.append(rec)
    if not records:
        return _empty_edges()
    return _resolve_skip_one(pd.DataFrame.from_records(records, columns=list(EDGE_COLUMNS)), s)


def adjacent_pairs(edges: pd.DataFrame) -> pd.DataFrame:
    """Parallel / skip-one edges with no row lying between their two rows (interrow adjacency)."""
    par = edges[edges.kind.isin(ADJACENT_KINDS)]
    nbrs = _neighbour_map(par, ADJACENT_KINDS)
    keep = [not _has_row_between(int(e.a), int(e.b), float(e.spacing_m), nbrs) for e in par.itertuples(index=False)]
    return par[np.asarray(keep, dtype=bool)].reset_index(drop=True) if len(par) else par.reset_index(drop=True)


def components(n: int, edges: pd.DataFrame) -> tuple[tuple[int, ...], ...]:
    """Connected components over all edges; each sorted, ordered by smallest member."""
    if n == 0:
        return ()
    a = edges.a.to_numpy(dtype=np.intp) if len(edges) else np.zeros(0, np.intp)
    b = edges.b.to_numpy(dtype=np.intp) if len(edges) else np.zeros(0, np.intp)
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    groups: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(idx)
    return tuple(sorted((tuple(g) for g in groups.values()), key=lambda g: g[0]))


# ---------------------------------------------------------------- transverse bands


@dataclass(frozen=True)
class Band:
    """A full-width gap band across one component: along interval [t0, t1] on the component axis."""

    t0: float
    t1: float
    rows: tuple[int, ...]
    polygon: Polygon

    @property
    def width_m(self) -> float:
        return self.t1 - self.t0


def _component_frame(lines: Sequence[LineString]) -> LineFrame:
    angle = math.radians(mean_axial_angle_deg([line_frame(ln).angle_deg for ln in lines]))
    centre = np.mean([np.asarray(ln.centroid.coords[0]) for ln in lines], axis=0)
    return LineFrame(readonly(centre), readonly(np.array([math.cos(angle), math.sin(angle)])), 0.0)


def _row_interval(fr: LineFrame, line: LineString) -> tuple[float, float]:
    t = fr.along(line_coords(line))
    return float(t.min()), float(t.max())


def _gap_intervals(fr: LineFrame, line: LineString, gaps: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for g0, g1 in gaps:
        pts = np.array([line.interpolate(g0).coords[0], line.interpolate(g1).coords[0]])
        t = np.sort(fr.along(pts))
        out.append((float(t[0]), float(t[1])))
    return out


def _band_mask(span: np.ndarray, gap: np.ndarray, s: GraphSettings) -> np.ndarray:
    n_span = span.sum(axis=0)
    n_gap = (gap & span).sum(axis=0)
    if s.band_full_width:
        return (n_span >= s.band_min_rows) & (n_gap == n_span)
    return n_gap >= s.band_min_rows


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _band_polygon(fr: LineFrame, t0: float, t1: float, lines: Sequence[LineString], pad_m: float) -> Polygon:
    o = np.concatenate([fr.across(line_coords(ln)) for ln in lines])
    o0, o1 = float(o.min()) - pad_m, float(o.max()) + pad_m
    return Polygon([fr.point_at(t0, o0), fr.point_at(t1, o0), fr.point_at(t1, o1), fr.point_at(t0, o1)])


def transverse_bands(
    lines: Sequence[LineString], gaps: Sequence[Sequence[tuple[float, float]]], members: Sequence[int],
    s: GraphSettings,
) -> tuple[Band, ...]:
    """Bands >= band_min_m where every row spanning them has a gap (full width) inside one component."""
    if len(members) < s.band_min_rows:
        return ()
    sub = [lines[i] for i in members]
    fr = _component_frame(sub)
    spans = [_row_interval(fr, ln) for ln in sub]
    lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    bin_m = s.band_bin_m
    centres = lo + (np.arange(max(1, math.ceil((hi - lo) / bin_m))) + 0.5) * bin_m
    span = np.array([(centres >= a) & (centres <= b) for a, b in spans])
    gap = np.zeros_like(span)
    for r, i in enumerate(members):
        for g0, g1 in _gap_intervals(fr, lines[i], gaps[i]):
            gap[r] |= (centres >= g0) & (centres <= g1)
    bands = []
    for a, b in _runs(_band_mask(span, gap, s)):
        t0, t1 = float(centres[a] - 0.5 * bin_m), float(centres[b - 1] + 0.5 * bin_m)
        if t1 - t0 >= s.band_min_m:
            rows = tuple(members[r] for r in range(len(members)) if gap[r, a:b].any())
            bands.append(Band(t0, t1, rows, _band_polygon(fr, t0, t1, sub, s.band_pad_m)))
    return tuple(bands)


def midpoint(line: LineString) -> Point:
    return line.interpolate(0.5, normalized=True)


# ---------------------------------------------------------------- row units, band split


@dataclass(frozen=True, eq=False)
class RowUnit:
    """One row line of the graph: `src` indexes the source chain; gaps are along-intervals on `line` (m)."""

    src: int
    line: LineString
    gaps: tuple[tuple[float, float], ...]
    band_cut: bool = False


def _oriented_like(part: LineString, line: LineString) -> LineString:
    a, b = line.project(Point(part.coords[0])), line.project(Point(part.coords[-1]))
    return part if a <= b else LineString(line_coords(part)[::-1])


def _shifted_gaps(gaps: Sequence[tuple[float, float]], a0: float, a1: float) -> tuple[tuple[float, float], ...]:
    out = [(max(g0, a0) - a0, min(g1, a1) - a0) for g0, g1 in gaps if min(g1, a1) > max(g0, a0)]
    return tuple(out)


def split_unit(unit: RowUnit, cut: BaseGeometry, min_len_m: float) -> list[RowUnit]:
    """Parts of the unit outside `cut` (>= min_len_m), in line order, gaps re-based on each part."""
    parts = [_oriented_like(p, unit.line) for p in split_multi(unit.line.difference(cut))
             if isinstance(p, LineString) and p.length >= min_len_m]
    starts = [unit.line.project(Point(p.coords[0])) for p in parts]
    return [RowUnit(unit.src, p, _shifted_gaps(unit.gaps, a0, a0 + p.length), True)
            for a0, p in sorted(zip(starts, parts, strict=True), key=lambda x: x[0])]


def split_at_bands(units: Sequence[RowUnit], bands: Sequence[Band], min_len_m: float) -> list[RowUnit]:
    """Cut every row of a band at the band polygon (02 §3.6); other units are kept as they are."""
    cuts: dict[int, list[Polygon]] = {}
    for band in bands:
        for r in band.rows:
            cuts.setdefault(r, []).append(band.polygon)
    out: list[RowUnit] = []
    for k, unit in enumerate(units):
        out.extend([unit] if k not in cuts else split_unit(unit, shapely.union_all(cuts[k]), min_len_m))
    return out


# ---------------------------------------------------------------- spacing / phase step cut (flag, default off)


@dataclass(frozen=True)
class SpacingCut:
    enabled: bool
    jump_max_m: float
    phase_tol_factor: float
    side_rows: int


def order_rows(lines: Sequence[LineString], members: Sequence[int]) -> list[int]:
    """Members ordered by n·c descending (contract §1.5) on the component's mean axial angle."""
    sub = [lines[i] for i in members]
    angle = mean_axial_angle_deg([line_frame(ln).angle_deg for ln in sub])
    cents = np.array([ln.centroid.coords[0] for ln in sub], dtype=np.float64)
    return [int(members[k]) for k in order_by_normal(angle, cents)]


def _spacing_sequence(order: Sequence[int], edges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sp: dict[tuple[int, int], tuple[float, bool]] = {}
    for e in edges.itertuples(index=False):
        if e.kind in ADJACENT_KINDS:
            sp[(int(e.a), int(e.b))] = (float(e.spacing_m), e.kind == EDGE_SKIP_ONE)
    vals = [sp.get((min(a, b), max(a, b)), (math.nan, False)) for a, b in zip(order, order[1:], strict=False)]
    seq = np.array([v for v, _ in vals], dtype=np.float64)
    skip = np.array([k for _, k in vals], dtype=bool)
    return np.where(skip, math.nan, seq), skip


def spacing_step_positions(seq: np.ndarray, skip: np.ndarray, cut: SpacingCut) -> list[int]:
    """Indices k (a cut between ordered rows k and k+1) where the local spacing or phase steps (arch §4.4.2)."""
    out = []
    n = cut.side_rows
    for k in range(n, len(seq) - n):
        before, after = seq[k - n:k], seq[k + 1:k + 1 + n]
        if skip[k] or np.isnan(before).any() or np.isnan(after).any() or math.isnan(seq[k]):
            continue
        mb, ma = float(np.median(before)), float(np.median(after))
        ref = 0.5 * (mb + ma)
        if abs(mb - ma) > cut.jump_max_m or abs(seq[k] - ref) > cut.phase_tol_factor * ref:
            out.append(k)
    return out


def spacing_cut_edges(lines: Sequence[LineString], edges: pd.DataFrame, members: Sequence[int],
                      cut: SpacingCut) -> pd.DataFrame:
    """Edges of one component minus those joining rows on different sides of a spacing step."""
    if not cut.enabled or len(members) < 2 * cut.side_rows + 2:
        return edges
    order = order_rows(lines, members)
    seq, skip = _spacing_sequence(order, edges)
    steps = spacing_step_positions(seq, skip, cut)
    if not steps:
        return edges
    side = {r: sum(1 for k in steps if k < p) for p, r in enumerate(order)}
    keep = [side.get(int(e.a), -1) == side.get(int(e.b), -1) or int(e.a) not in side
            for e in edges.itertuples(index=False)]
    return edges[np.asarray(keep, dtype=bool)].reset_index(drop=True)
