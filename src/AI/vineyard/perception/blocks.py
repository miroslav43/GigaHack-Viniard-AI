"""Blocks (02 §3.6, plan S3): rows_raw chains -> row graph -> components -> V01.. blocks and R001.. rows.

Edges: parallel / skip-one (flag `missing_row_suspect`) / collinear, cut where the connector crosses a
passage; full-width transverse bands split rows; optional spacing-step cut and orchard rejection (flags);
components with fewer than `min_rows` rows go to `rows_rejected`. Every output is a new frame.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from vineyard.config import AppConfig
from vineyard.contracts.enums import Severity, Source
from vineyard.contracts.ids import format_row_id, format_vineyard_id
from vineyard.contracts.ordering import angle_deg_utm, mean_axial_angle_deg, order_blocks
from vineyard.contracts.qa import QaIssue
from vineyard.contracts.schema_defs import MIN_LINE_LENGTH_M
from vineyard.contracts.schemas import empty_layer
from vineyard.geo.ops import make_valid_polygonal, orient_ccw
from vineyard.geo.tiling import CRS_EPSG, tile_of_point
from vineyard.perception.blocks_graph import (
    EDGE_COLLINEAR,
    EDGE_SKIP_ONE,
    Band,
    GraphSettings,
    RowUnit,
    SpacingCut,
    adjacent_pairs,
    components,
    line_coords,
    neighbour_pairs,
    order_rows,
    spacing_cut_edges,
    split_at_bands,
    transverse_bands,
)
from vineyard.perception.road_split import RoadCut, RoadSplitOptions, road_cuts
from vineyard.perception.row_regularize import (
    ACTION_UNDERSHOOT,
    FLAG_OFF_LATTICE,
    FLAG_OVERSHOOT,
    REMOVALS,
    SPAN_ACTIONS,
    Evidence,
    RegularizeOptions,
    RowChange,
    apply_span,
    regularize_block,
)
from vineyard.perception.rows_link import (
    FLAG_INTERPOLATED,
    FLAG_SEP,
    LIST_SEP,
    eroded,
    parse_gaps,
    tile_spans,
)

REASON_TOO_FEW: Final = "block_too_few_rows"
REASON_ORCHARD: Final = "orchard_rejected"
CODE_MISSING_ROW: Final = "missing_row_suspect"
CODE_BAND_CUT: Final = "transverse_band_cut"
CODE_ROAD_CUT: Final = "row_split_at_road"
STUB_TOUCH_M: Final = 0.5  # a piece this close to a road cut is on its edge
FULL_CONFIDENCE: Final = 1.0


@dataclass(frozen=True)
class OrchardRule:
    enabled: bool
    width_spacing_ratio_max: float
    period_m: tuple[float, float]
    duty_max: float
    majority_frac: float


@dataclass(frozen=True)
class BlockSettings:
    graph: GraphSettings
    min_rows: int
    cut_by_passages: bool
    passage_erode_m: float
    outline_buffer_m: float
    corridor_half_m: float
    min_piece_m: float
    garden_max_row_length_m: float
    spacing_cut: SpacingCut
    orchard: OrchardRule
    too_few_issue_min_rows: int
    regularize: RegularizeOptions | None = None
    road_split: RoadSplitOptions | None = None

    @classmethod
    def from_config(cls, cfg: AppConfig) -> BlockSettings:
        blk, orc = cfg.blocks, cfg.orchard
        return cls(
            graph=GraphSettings.from_config(blk, cfg.rows.detect, cfg.rows.link), min_rows=blk.min_rows_per_block,
            cut_by_passages=blk.cut_by_passages, passage_erode_m=cfg.rows.link.passage_erode_m,
            outline_buffer_m=blk.outline_buffer_m, corridor_half_m=cfg.canopy.corridor_half_m,
            min_piece_m=cfg.export.min_row_piece_m, garden_max_row_length_m=blk.garden_max_row_length_m,
            spacing_cut=SpacingCut(blk.spacing_cut_enabled, blk.spacing_jump_max_m, blk.phase_tol_factor,
                                   blk.spacing_cut_side_rows),
            orchard=OrchardRule(orc.block_reject_enabled, orc.width_spacing_ratio_max, tuple(orc.along_period_m),
                                orc.along_duty_max, orc.block_majority_frac),
            too_few_issue_min_rows=blk.too_few_rows_issue_min,
            regularize=RegularizeOptions.from_config(blk.regularize),
            road_split=RoadSplitOptions.from_config(blk.road_split),
        )


@dataclass(frozen=True)
class BlockResult:
    rows: gpd.GeoDataFrame
    blocks: gpd.GeoDataFrame
    row_pairs: gpd.GeoDataFrame
    rows_rejected: gpd.GeoDataFrame
    issues: tuple[QaIssue, ...]


@dataclass(frozen=True, eq=False)
class _Graph:
    base: tuple[RowUnit, ...]
    units: tuple[RowUnit, ...]
    edges: pd.DataFrame
    comps: tuple[tuple[int, ...], ...]
    bands: tuple[Band, ...]
    cut: BaseGeometry | None = None  # passages + the barriers of the road cuts: no edge crosses it
    road_cuts: tuple[RoadCut, ...] = ()


@dataclass(frozen=True)
class _Block:
    members: tuple[int, ...]  # unit indexes in R001.. order
    vineyard_id: str
    outline: BaseGeometry


@dataclass(frozen=True)
class _Prov:
    run_id: str
    model_version: str


# ---------------------------------------------------------------- graph


def _units(chains: gpd.GeoDataFrame) -> list[RowUnit]:
    gaps = chains["gaps_json"] if "gaps_json" in chains.columns else pd.Series([""] * len(chains))
    return [RowUnit(k, LineString(line_coords(g)), parse_gaps(gj, str(cid)))
            for k, (cid, g, gj) in enumerate(zip(chains.chain_id, chains.geometry, gaps, strict=True))]


def _edges(units: Sequence[RowUnit], cut: BaseGeometry | None, s: BlockSettings) -> pd.DataFrame:
    return neighbour_pairs([u.line for u in units], s.graph, cut)


def _road_cuts(units: Sequence[RowUnit], edges0: pd.DataFrame, roads: Sequence[LineString],
               evidence: Evidence | None, s: BlockSettings) -> tuple[RoadCut, ...]:
    o = s.road_split
    if o is None or not o.enabled or evidence is None or not roads:
        return ()
    lines = [u.line for u in units]
    return tuple(c for comp in components(len(units), edges0) for c in road_cuts(lines, comp, roads, evidence, o))


def _with_barriers(cut: BaseGeometry | None, road: Sequence[RoadCut]) -> BaseGeometry | None:
    parts = ([cut] if cut is not None else []) + [c.barrier for c in road]
    return shapely.union_all(parts) if parts else None


def drop_road_stubs(units: Sequence[RowUnit], road: Sequence[RoadCut], min_side_m: float) -> tuple[RowUnit, ...]:
    """Without the pieces a road cut leaves shorter than `min_side_m` next to it: rows that overshot a road
    (into a yard, a field) end at the road instead of becoming a block of stubs on its far side."""
    if not road or min_side_m <= 0:
        return tuple(units)
    zone = shapely.union_all([c.polygon for c in road]).buffer(STUB_TOUCH_M)
    shapely.prepare(zone)
    return tuple(u for u in units if not (u.band_cut and u.line.length < min_side_m and zone.intersects(u.line)))


def build_graph(units: Sequence[RowUnit], cut: BaseGeometry | None, s: BlockSettings,
                roads: Sequence[LineString] = (), evidence: Evidence | None = None) -> _Graph:
    """Edges -> components -> full-width bands + road cuts (rows split) -> edges again -> optional spacing cut."""
    edges0 = _edges(units, cut, s)
    lines0, gaps0 = [u.line for u in units], [u.gaps for u in units]
    bands = tuple(b for comp in components(len(units), edges0) for b in transverse_bands(lines0, gaps0, comp, s.graph))
    road = _road_cuts(units, edges0, roads, evidence, s)
    cut = _with_barriers(cut, road)
    splits = bands + tuple(Band(0.0, 0.0, c.rows, c.polygon) for c in road)
    split = tuple(split_at_bands(units, splits, s.min_piece_m)) if splits else tuple(units)
    split = drop_road_stubs(split, road, s.road_split.min_side_m if s.road_split else 0.0)
    edges = _edges(split, cut, s) if splits else edges0
    if s.spacing_cut.enabled:
        lines = [u.line for u in split]
        for comp in components(len(split), edges):
            edges = spacing_cut_edges(lines, edges, comp, s.spacing_cut)
    return _Graph(tuple(units), split, edges, components(len(split), edges), bands, cut, road)


def distinct_rows(members: Sequence[int], edges: pd.DataFrame) -> int:
    """Rows of a component, counting collinear pieces of one row once."""
    col = edges[edges.kind == EDGE_COLLINEAR]
    index = {m: k for k, m in enumerate(members)}
    sub = col[col.a.isin(list(index)) & col.b.isin(list(index))]
    local = pd.DataFrame({"a": [index[int(a)] for a in sub.a], "b": [index[int(b)] for b in sub.b]})
    return len(components(len(members), local))


# ---------------------------------------------------------------- acceptance


def _median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else math.nan


def _spacing_med(members: Sequence[int], edges: pd.DataFrame) -> float:
    sel = edges[edges.a.isin(list(members)) & (edges.kind != EDGE_COLLINEAR) & (edges.kind != EDGE_SKIP_ONE)]
    return _median(adjacent_pairs(sel).spacing_m.astype(np.float64).tolist()) if len(sel) else math.nan


def _col(chains: gpd.GeoDataFrame, name: str, srcs: Sequence[int]) -> np.ndarray:
    if name not in chains.columns:
        return np.full(len(srcs), math.nan)
    return pd.to_numeric(chains[name].iloc[list(srcs)], errors="coerce").to_numpy(dtype=np.float64)


def orchard_reason(chains: gpd.GeoDataFrame, srcs: Sequence[int], spacing_med: float, rule: OrchardRule) -> bool:
    """Block-level orchard test (02 §3.6): wide canopies for the spacing, tree-like period, or harmonics."""
    if not rule.enabled:
        return False
    ratio = _median(_col(chains, "width_p80_m", srcs)) / spacing_med if spacing_med > 0 else math.nan
    period, duty = _col(chains, "along_period_m", srcs), _col(chains, "along_duty", srcs)
    lo, hi = rule.period_m
    tree_like = np.nan_to_num((period >= lo) & (period <= hi) & (duty < rule.duty_max), nan=0.0).astype(bool)
    harmonic = np.nan_to_num(_col(chains, "harmonic_frac", srcs), nan=0.0) > rule.majority_frac
    return bool((math.isfinite(ratio) and ratio > rule.width_spacing_ratio_max)
                or tree_like.mean() > rule.majority_frac or harmonic.mean() > rule.majority_frac)


def _outline(lines: Sequence[LineString], s: BlockSettings) -> BaseGeometry:
    corridors = [ln.buffer(s.corridor_half_m, cap_style="flat") for ln in lines]
    grown = shapely.union_all([c.buffer(s.outline_buffer_m) for c in corridors])
    parts = [orient_ccw(p) for p in make_valid_polygonal(grown.buffer(-s.outline_buffer_m))]
    parts = sorted(parts, key=lambda p: (-p.area, p.centroid.x, p.centroid.y))
    return parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)


def select_blocks(g: _Graph, chains: gpd.GeoDataFrame, s: BlockSettings
                  ) -> tuple[list[_Block], list[tuple[int, str]]]:
    """Kept components as V-numbered blocks (rows in n·c order) and (unit, reason) rejections."""
    kept, rejected = [], []
    lines = [u.line for u in g.units]
    for comp in g.comps:
        srcs = sorted({g.units[i].src for i in comp})
        if distinct_rows(comp, g.edges) < s.min_rows:
            rejected.extend((i, REASON_TOO_FEW) for i in comp)
        elif orchard_reason(chains, srcs, _spacing_med(comp, g.edges), s.orchard):
            rejected.extend((i, REASON_ORCHARD) for i in comp)
        else:
            kept.append(tuple(order_rows(lines, comp)))
    outlines = [_outline([lines[i] for i in comp], s) for comp in kept]
    reps = np.array([[o.centroid.x, o.centroid.y] for o in outlines]).reshape(-1, 2)
    order = order_blocks(reps)
    blocks = [_Block(kept[k], format_vineyard_id(n + 1, len(kept)), outlines[k]) for n, k in enumerate(order)]
    return blocks, sorted(rejected)


# ---------------------------------------------------------------- frames


def _flags(text: object) -> list[str]:
    raw = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    return [f for f in raw.split(FLAG_SEP) if f]


def _row_record(unit: RowUnit, chain: pd.Series, clips: Mapping[str, BaseGeometry]) -> dict[str, object]:
    spans = [sp for sp in tile_spans(unit.line, clips) if sp[2] - sp[1] >= MIN_LINE_LENGTH_M]
    tiles = [t for t, _, _ in spans]
    interp = [t for t in str(chain.get("interp_tile_ids") or "").split(LIST_SEP) if t in tiles]
    flags = [f for f in _flags(chain.get("qa_flags")) if f != FLAG_INTERPOLATED]
    flags += [FLAG_INTERPOLATED] if interp else []
    flags += [CODE_BAND_CUT] if unit.band_cut else []
    xy = line_coords(unit.line)
    conf = pd.to_numeric(pd.Series([chain.get("confidence", FULL_CONFIDENCE)]), errors="coerce").iloc[0]
    return {
        "length_m": float(sum(hi - lo for _, lo, hi in spans)), "extent_m": float(unit.line.length),
        "n_pieces": len(spans), "tile_ids": LIST_SEP.join(tiles),
        "angle_deg": angle_deg_utm(tuple(xy[0]), tuple(xy[-1])), "max_gap_m": math.nan, "n_gaps_ge5": None,
        "structure_any": None, "chain_id": str(chain.chain_id), "interp_tile_ids": LIST_SEP.join(interp),
        "gaps_json": "[" + ", ".join(f"[{a:.3f}, {b:.3f}]" for a, b in unit.gaps) + "]",
        "confidence": FULL_CONFIDENCE if pd.isna(conf) else float(conf),
        "qa_flags": FLAG_SEP.join(dict.fromkeys(flags)), "geometry": unit.line,
    }


def _neighbour_spacing(members: Sequence[int], adj: pd.DataFrame) -> tuple[list[float], list[float]]:
    pos = {m: k for k, m in enumerate(members)}
    prev, nxt = [math.inf] * len(members), [math.inf] * len(members)
    for e in adj.itertuples(index=False):
        a, b = sorted((pos[int(e.a)], pos[int(e.b)]))
        nxt[a], prev[b] = min(nxt[a], float(e.spacing_m)), min(prev[b], float(e.spacing_m))
    fix = [math.nan if math.isinf(v) else v for v in prev], [math.nan if math.isinf(v) else v for v in nxt]
    return fix


def rows_frame(blocks: Sequence[_Block], g: _Graph, chains: gpd.GeoDataFrame, adj: pd.DataFrame,
               clips: Mapping[str, BaseGeometry], prov: _Prov) -> gpd.GeoDataFrame:
    recs = []
    for blk in blocks:
        prev, nxt = _neighbour_spacing(blk.members, adj[adj.a.isin(list(blk.members))])
        for k, i in enumerate(blk.members):
            rec = _row_record(g.units[i], chains.iloc[g.units[i].src], clips)
            recs.append({"row_id": format_row_id(blk.vineyard_id, k + 1), "vineyard_id": blk.vineyard_id,
                         "row_index": k + 1, **rec, "spacing_prev_m": prev[k], "spacing_next_m": nxt[k],
                         "source": Source.MODEL.value, "run_id": prov.run_id, "model_version": prov.model_version})
    return _frame(recs, "rows")


def _row_ids(blocks: Sequence[_Block]) -> dict[int, tuple[str, str, int]]:
    return {i: (b.vineyard_id, format_row_id(b.vineyard_id, k + 1), k + 1)
            for b in blocks for k, i in enumerate(b.members)}


def pairs_frame(blocks: Sequence[_Block], adj: pd.DataFrame, prov: _Prov) -> gpd.GeoDataFrame:
    ids = _row_ids(blocks)
    recs = []
    for e in adj.itertuples(index=False):
        if int(e.a) not in ids or int(e.b) not in ids:
            continue
        (vid, ra, ka), (_, rb, kb) = sorted((ids[int(e.a)], ids[int(e.b)]), key=lambda t: t[2])
        recs.append({"vineyard_id": vid, "row_a": ra, "row_b": rb, "k_a": ka, "k_b": kb, "kind": e.kind,
                     "spacing_m": float(e.spacing_m), "overlap_from_m": float(e.overlap_from_m),
                     "overlap_to_m": float(e.overlap_to_m), "angle_diff_deg": float(e.angle_diff_deg),
                     "source": Source.MODEL.value, "run_id": prov.run_id, "model_version": prov.model_version,
                     "confidence": FULL_CONFIDENCE, "qa_flags": CODE_MISSING_ROW if e.kind == EDGE_SKIP_ONE else "",
                     "geometry": e.geometry})
    recs.sort(key=lambda r: (r["vineyard_id"], r["k_a"], r["k_b"]))
    return _frame(recs, "row_pairs")


def blocks_frame(blocks: Sequence[_Block], rows: gpd.GeoDataFrame, pairs: gpd.GeoDataFrame, g: _Graph,
                 s: BlockSettings, prov: _Prov) -> gpd.GeoDataFrame:
    recs = []
    for blk in blocks:
        r = rows[rows.vineyard_id == blk.vineyard_id]
        p = pairs[pairs.vineyard_id == blk.vineyard_id]
        tiles = sorted({t for ts in r.tile_ids for t in str(ts).split(LIST_SEP) if t})
        skips = int((p.kind == EDGE_SKIP_ONE).sum()) if len(p) else 0
        flags = [CODE_MISSING_ROW] * bool(skips) + [CODE_BAND_CUT] * any(g.units[i].band_cut for i in blk.members)
        row_len = float(r.length_m.sum())
        regular = p[p.kind != EDGE_SKIP_ONE].spacing_m if len(p) else pd.Series(dtype=float)
        recs.append({
            "vineyard_id": blk.vineyard_id, "n_rows": len(r), "n_row_pieces": int(r.n_pieces.sum()),
            "n_canopies": None, "n_tiles": len(tiles), "row_length_m": row_len, "canopy_area_m2": math.nan,
            "interrow_area_m2": math.nan, "outline_area_m2": float(blk.outline.area),
            "angle_deg": mean_axial_angle_deg(r.angle_deg.astype(float).tolist()),
            "spacing_med_m": _median(regular.astype(float).tolist()), "is_garden": row_len < s.garden_max_row_length_m,
            "tile_ids": LIST_SEP.join(tiles), "n_skip_one": skips, "source": Source.MODEL.value,
            "run_id": prov.run_id, "model_version": prov.model_version, "confidence": FULL_CONFIDENCE,
            "qa_flags": FLAG_SEP.join(flags), "geometry": blk.outline,
        })
    return _frame(recs, "blocks")


def rejected_frame(rejected: Sequence[tuple[int, str]], g: _Graph, chains: gpd.GeoDataFrame,
                   prov: _Prov) -> gpd.GeoDataFrame:
    first: dict[int, str] = {}
    for i, reason in rejected:
        first.setdefault(g.units[i].src, reason)
    recs = [{"chain_id": str(chains.chain_id.iloc[src]), "reason": reason, "source": Source.MODEL.value,
             "run_id": prov.run_id, "model_version": prov.model_version, "confidence": FULL_CONFIDENCE,
             "qa_flags": reason, "geometry": chains.geometry.iloc[src]} for src, reason in sorted(first.items())]
    return _frame(recs, "rows_rejected")


def _frame(recs: Sequence[Mapping[str, object]], layer: str) -> gpd.GeoDataFrame:
    if not recs:
        return empty_layer(layer)
    return gpd.GeoDataFrame(list(recs), geometry="geometry", crs=CRS_EPSG)


# ---------------------------------------------------------------- issues, orchestration


def _tile_at(p: Point) -> str:
    try:
        return tile_of_point(p.x, p.y).tile_id
    except ValueError:  # outside the grid: issue stays unlocated by tile
        return ""


def _issue(sev: Severity, code: str, object_id: str, message: str, where: BaseGeometry) -> QaIssue:
    p = where.representative_point() if where.geom_type != "Point" else where
    return QaIssue(sev, code, _tile_at(p), object_id, message, p.x, p.y)


def block_issues(g: _Graph, pairs: gpd.GeoDataFrame, rejected: Sequence[tuple[int, str]],
                 chains: gpd.GeoDataFrame, too_few_issue_min_rows: int) -> tuple[QaIssue, ...]:
    out = [_issue(Severity.WARNING, CODE_MISSING_ROW, f"{p.row_a}|{p.row_b}",
                  f"skip-one pair, spacing {p.spacing_m:.2f} m", p.geometry.interpolate(0.5, normalized=True))
           for p in pairs.itertuples(index=False) if p.kind == EDGE_SKIP_ONE]
    out += [_issue(Severity.INFO, CODE_BAND_CUT, str(chains.chain_id.iloc[g.base[b.rows[0]].src]),
                   f"full-width band {b.width_m:.1f} m across {len(b.rows)} rows", b.polygon) for b in g.bands]
    by_reason: dict[tuple[str, int], list[int]] = {}
    comp_of = {i: k for k, comp in enumerate(g.comps) for i in comp}
    for i, reason in rejected:
        by_reason.setdefault((reason, comp_of[i]), []).append(i)
    for (reason, _), members in sorted(by_reason.items()):
        if reason == REASON_TOO_FEW and len(members) < too_few_issue_min_rows:
            continue
        sev = Severity.WARNING if reason == REASON_ORCHARD else Severity.INFO
        geom = shapely.union_all([g.units[i].line for i in members])
        out.append(_issue(sev, reason, str(chains.chain_id.iloc[g.units[members[0]].src]),
                          f"{len(members)} rows rejected: {reason}", geom))
    return tuple(out)


# ---------------------------------------------------------------- row-frame regularisation (RC8)

_FLAG_CODES: frozenset[str] = frozenset({ACTION_UNDERSHOOT, FLAG_OFF_LATTICE, FLAG_OVERSHOOT})


def _changed_unit(unit: RowUnit, change: RowChange) -> RowUnit:
    line = apply_span(unit.line, change.lo, change.hi)
    lo = max(0.0, change.lo)
    gaps = tuple((max(0.0, a - lo), min(float(line.length), b - lo)) for a, b in unit.gaps
                 if min(float(line.length), b - lo) > max(0.0, a - lo))
    return RowUnit(unit.src, line, gaps, unit.band_cut)


def regularized_graph(g: _Graph, blocks: Sequence[_Block], cut: BaseGeometry | None, s: BlockSettings,
                      evidence: Evidence, chains: gpd.GeoDataFrame) -> tuple[_Graph, tuple[QaIssue, ...]]:
    """New graph with every kept block's rows regularised (perception.row_regularize) + one issue per change."""
    changes: dict[int, RowChange] = {}
    issues = []
    for blk in blocks:
        for c in regularize_block([g.units[i].line for i in blk.members], s.regularize, evidence):
            unit = blk.members[c.index]
            sev = Severity.WARNING if c.action in _FLAG_CODES else Severity.INFO
            issues.append(_issue(sev, c.action, str(chains.chain_id.iloc[g.units[unit].src]),
                                 f"{blk.vineyard_id}: {c.message}", Point(c.x, c.y)))
            if c.action in REMOVALS or c.action in SPAN_ACTIONS:
                changes[unit] = c
    if not changes:
        return g, tuple(issues)
    units = tuple(u if k not in changes else _changed_unit(u, changes[k]) for k, u in enumerate(g.units)
                  if changes.get(k) is None or changes[k].action not in REMOVALS)
    edges = _edges(units, cut, s)
    if s.spacing_cut.enabled:
        lines = [u.line for u in units]
        for comp in components(len(units), edges):
            edges = spacing_cut_edges(lines, edges, comp, s.spacing_cut)
    return _Graph(g.base, units, edges, components(len(units), edges), g.bands, cut, g.road_cuts), tuple(issues)


def road_issues(g: _Graph, chains: gpd.GeoDataFrame) -> tuple[QaIssue, ...]:
    """One info issue per road cut: the road crossing, how many rows it cut out of how many crossing it."""
    out = []
    for c in g.road_cuts:
        at = c.polygon.representative_point()
        ref = str(chains.chain_id.iloc[g.base[c.rows[0]].src])
        out.append(_issue(Severity.INFO, CODE_ROAD_CUT, ref,
                          f"road {c.road_index}: {len(c.rows)}/{c.n_crossing} crossing rows cut (vine-free stretch)",
                          at))
    return tuple(out)


def _sorted_chains(rows_raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return rows_raw.sort_values("chain_id", kind="stable").reset_index(drop=True)


def build_blocks(rows_raw: gpd.GeoDataFrame, passages: BaseGeometry | None, clips: Mapping[str, BaseGeometry],
                 s: BlockSettings, *, run_id: str = "", model_version: str = "",
                 evidence: Evidence | None = None, roads: Sequence[LineString] = ()) -> BlockResult:
    """rows_raw (UTM chains) -> rows / blocks / row_pairs / rows_rejected + QA issues (deterministic).

    With `evidence` and blocks.regularize.enabled, the kept blocks are regularised in the row frame and the
    graph is rebuilt from the changed rows (blocks re-selected and re-numbered)."""
    chains = _sorted_chains(rows_raw)
    cut = eroded(passages, s.passage_erode_m) if s.cut_by_passages else None
    g = build_graph(_units(chains), cut, s, roads, evidence)
    blocks, rejected = select_blocks(g, chains, s)
    reg_issues: tuple[QaIssue, ...] = road_issues(g, chains)
    if evidence is not None and s.regularize is not None and s.regularize.enabled and blocks:
        g, more = regularized_graph(g, blocks, g.cut, s, evidence, chains)
        reg_issues += more
        blocks, rejected = select_blocks(g, chains, s)
    prov = _Prov(run_id, model_version)
    adj = adjacent_pairs(g.edges)
    rows = rows_frame(blocks, g, chains, adj, clips, prov)
    pairs = pairs_frame(blocks, adj, prov)
    out_blocks = blocks_frame(blocks, rows, pairs, g, s, prov)
    return BlockResult(rows, out_blocks, pairs, rejected_frame(rejected, g, chains, prov),
                       block_issues(g, pairs, rejected, chains, s.too_few_issue_min_rows) + reg_issues)
