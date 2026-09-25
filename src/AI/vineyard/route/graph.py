"""Walking-graph assembly (arch §4.11.2, design 04 §3.7).

Every walkable polyline (interrow centerline piece, passage skeleton branch, connector) is cut into
edges at regular node positions (interrows every `centerline_node_step_m`, passages every
`passage_node_step_m`, connectors only at their ends) and at the attachment points where connectors
join it. Cut points closer than `NODE_MERGE_M` become one node (KD-tree + union-find); edge
geometry ends are snapped onto their node. Each edge gets its outside length against
`domain.inner` and a cost `length + penalty * outside`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import scipy.sparse as sp
import shapely
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import EdgeKind
from vineyard.geo.ops import drop_consecutive_duplicates
from vineyard.route.cells import CellIndex
from vineyard.route.graph_types import NodeKind, PolylineDraft, WalkGraph, connector_cost, make_walk_graph

if TYPE_CHECKING:
    from vineyard.config.sections_post import RouteGraphConfig

# Cut points (and so nodes) closer than this are one node; shorter edges vanish.
NODE_MERGE_M: Final = 0.01
# Cut parameters closer than this along one polyline are one cut (well inside NODE_MERGE_M).
PARAM_MERGE_M: Final = 1e-3
# CONFIG-REQUEST: route.graph.passage_node_step_m = 5.0
PASSAGE_NODE_STEP_M: Final = 5.0
START_REF: Final = "START"
# Lower value wins when several labels land on one node.
_PRIORITY: Final = {NodeKind.START: 0, NodeKind.ROW_END: 1, NodeKind.ATTACH: 2, NodeKind.JUNCTION: 3,
                    NodeKind.PASSAGE: 4, NodeKind.INTERROW: 5, NodeKind.TARGET: 6}
_JUNCTION_FROM: Final = frozenset({NodeKind.PASSAGE, NodeKind.INTERROW})
_JUNCTION_DEGREE: Final = 3


@dataclass(frozen=True)
class GraphParams:
    centerline_node_step_m: float
    passage_node_step_m: float
    outside_penalty: float

    def __post_init__(self) -> None:
        if min(self.centerline_node_step_m, self.passage_node_step_m) < 0.0:
            raise ValueError("node_step values must be >= 0 (0 = nodes at polyline ends only)")
        if self.outside_penalty < 0.0:
            raise ValueError(f"outside_penalty must be >= 0, got {self.outside_penalty}")

    @classmethod
    def from_config(cls, cfg: RouteGraphConfig) -> GraphParams:
        return cls(centerline_node_step_m=cfg.centerline_step_m, passage_node_step_m=PASSAGE_NODE_STEP_M,
                   outside_penalty=cfg.connector_outside_penalty)

    def node_step(self, kind: EdgeKind) -> float:
        if kind == EdgeKind.INTERROW_CENTERLINE:
            return self.centerline_node_step_m
        if kind == EdgeKind.PASSAGE_CENTERLINE:
            return self.passage_node_step_m
        return 0.0


@dataclass(frozen=True)
class Attachment:
    """A connector joins draft `draft` at `along_m` from its start: the draft must be cut there."""

    draft: int
    along_m: float
    ref: str


@dataclass(frozen=True)
class ComponentInfo:
    label: int
    has_start: bool
    n_nodes: int
    n_edges: int
    length_m: float
    x: float
    y: float
    interrow_ids: tuple[str, ...]


def cut_params(length: float, step: float, extra: np.ndarray) -> np.ndarray:
    """Sorted cut positions: 0, L, every ~`step` (evenly spread) and `extra`, merged within PARAM_MERGE_M."""
    n = max(1, math.ceil(length / step - 1e-9)) if step > 0.0 else 1
    params = np.sort(np.concatenate([np.linspace(0.0, length, n + 1), np.clip(extra, 0.0, length)]))
    keep = np.concatenate([[True], np.diff(params) > PARAM_MERGE_M])
    out = params[keep]
    out[-1] = length
    return out


def _cum(coords: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(coords, axis=0).T))])


def _at(coords: np.ndarray, cum: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(s, cum, coords[:, 0]), np.interp(s, cum, coords[:, 1])])


def _between(coords: np.ndarray, cum: np.ndarray, s0: float, s1: float) -> np.ndarray:
    return coords[(cum > s0) & (cum < s1)]


@dataclass(frozen=True, eq=False)
class _Cut:
    coords: np.ndarray
    cum: np.ndarray
    params: np.ndarray


def _cuts(drafts: Sequence[PolylineDraft], attachments: Sequence[Attachment], params: GraphParams) -> list[_Cut]:
    extra: list[list[float]] = [[] for _ in drafts]
    for a in attachments:
        if not 0 <= a.draft < len(drafts):
            raise ValueError(f"attachment {a} points at draft {a.draft}, but there are {len(drafts)} drafts")
        extra[a.draft].append(a.along_m)
    out: list[_Cut] = []
    for d, ex in zip(drafts, extra, strict=True):
        coords = drop_consecutive_duplicates(np.asarray(d.geom.coords)[:, :2], closed=False)
        cum = _cum(coords)
        out.append(_Cut(coords, cum, cut_params(float(cum[-1]), params.node_step(d.kind), np.asarray(ex, float))))
    return out


def _labels(draft: PolylineDraft, n: int) -> list[NodeKind]:
    if draft.kind == EdgeKind.INTERROW_CENTERLINE:
        return [NodeKind.ROW_END] + [NodeKind.INTERROW] * (n - 2) + [NodeKind.ROW_END]
    if draft.kind == EdgeKind.PASSAGE_CENTERLINE:
        return [NodeKind.PASSAGE] * n
    return [NodeKind.ATTACH] * n


@dataclass(frozen=True, eq=False)
class _Occurrences:
    xy: np.ndarray
    kind: tuple[NodeKind, ...]
    ref: tuple[str, ...]
    offsets: tuple[int, ...]   # first occurrence index of each draft's cut points


def _occurrences(drafts: Sequence[PolylineDraft], cuts: Sequence[_Cut], attachments: Sequence[Attachment],
                 start_xy: tuple[float, float]) -> _Occurrences:
    xy: list[np.ndarray] = [np.asarray([start_xy], float)]
    kinds: list[NodeKind] = [NodeKind.START]
    refs: list[str] = [START_REF]
    offsets: list[int] = []
    for d, c in zip(drafts, cuts, strict=True):
        offsets.append(len(kinds))
        xy.append(_at(c.coords, c.cum, c.params))
        kinds.extend(_labels(d, len(c.params)))
        refs.extend([d.ref] * len(c.params))
    for a in attachments:
        c = cuts[a.draft]
        xy.append(_at(c.coords, c.cum, np.asarray([min(max(a.along_m, 0.0), c.cum[-1])])))
        kinds.append(NodeKind.ATTACH)
        refs.append(a.ref)
    return _Occurrences(np.vstack(xy), tuple(kinds), tuple(refs), tuple(offsets))


def _merge(occ: _Occurrences) -> tuple[np.ndarray, np.ndarray]:
    """(node id per occurrence, representative occurrence per node); nodes in first-occurrence order."""
    n = len(occ.xy)
    pairs = cKDTree(occ.xy).query_pairs(NODE_MERGE_M, output_type="ndarray")
    adj = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) \
        else sp.coo_matrix((n, n))
    _, labels = connected_components(adj, directed=False)
    _, first = np.unique(labels, return_index=True)
    order = np.argsort(first, kind="stable")
    node_of_label = np.empty(len(order), dtype=np.int64)
    node_of_label[order] = np.arange(len(order))
    node = node_of_label[labels]
    prio = np.asarray([_PRIORITY[k] for k in occ.kind])
    ranked = np.lexsort((np.arange(n), prio, node))
    is_first = np.concatenate([[True], node[ranked][1:] != node[ranked][:-1]])
    return node, ranked[is_first]


@dataclass(frozen=True, eq=False)
class _Edges:
    uv: np.ndarray
    geoms: tuple[LineString, ...]
    kinds: tuple[str, ...]
    refs: tuple[str, ...]


def _edges(drafts: Sequence[PolylineDraft], cuts: Sequence[_Cut], offsets: Sequence[int], node: np.ndarray,
           node_xy: np.ndarray) -> _Edges:
    uv: list[tuple[int, int]] = []
    geoms: list[LineString] = []
    kinds: list[str] = []
    refs: list[str] = []
    for d, c, off in zip(drafts, cuts, offsets, strict=True):
        ids = node[off:off + len(c.params)]
        for k in range(len(c.params) - 1):
            u, v = int(ids[k]), int(ids[k + 1])
            if u == v:
                continue
            pts = np.vstack([node_xy[u], _between(c.coords, c.cum, c.params[k], c.params[k + 1]), node_xy[v]])
            pts = drop_consecutive_duplicates(pts, closed=False)
            if len(pts) < 2:
                continue
            uv.append((u, v))
            geoms.append(LineString(pts))
            kinds.append(str(d.kind))
            refs.append(d.ref)
    return _Edges(np.asarray(uv, dtype=np.int64).reshape(-1, 2), tuple(geoms), tuple(kinds), tuple(refs))


def outside_lengths(geoms: Sequence[BaseGeometry], inner: BaseGeometry, index: CellIndex) -> np.ndarray:
    """Length of each line outside `inner`: prepared `covered_by` first, exact local clip for the rest."""
    arr = np.asarray(list(geoms), dtype=object)
    out = np.zeros(len(arr), dtype=np.float64)
    if len(arr) == 0:
        return out
    uncovered = ~shapely.covered_by(arr, inner)
    if uncovered.any():
        lengths = shapely.length(arr[uncovered])
        out[uncovered] = np.maximum(lengths - index.inside_length(arr[uncovered]), 0.0)
    return out


def _final_kinds(rep_kinds: Sequence[NodeKind], uv: np.ndarray) -> list[str]:
    degree = np.bincount(uv.ravel(), minlength=len(rep_kinds)) if len(uv) else np.zeros(len(rep_kinds), int)
    return [str(NodeKind.JUNCTION) if k in _JUNCTION_FROM and degree[i] >= _JUNCTION_DEGREE else str(k)
            for i, k in enumerate(rep_kinds)]


def assemble_graph(
    drafts: Sequence[PolylineDraft],
    attachments: Sequence[Attachment],
    start_xy: tuple[float, float],
    inner: BaseGeometry,
    params: GraphParams,
    inner_index: CellIndex | None = None,
) -> WalkGraph:
    """WalkGraph of `drafts` cut at node steps and `attachments`; node 0 is START (exact coordinates)."""
    cuts = _cuts(drafts, attachments, params)
    occ = _occurrences(drafts, cuts, attachments, start_xy)
    node, rep = _merge(occ)
    node_xy = occ.xy[rep]
    edges = _edges(drafts, cuts, occ.offsets, node, node_xy)
    index = inner_index if inner_index is not None else CellIndex.build(inner)
    outside = outside_lengths(edges.geoms, inner, index)
    lengths = np.asarray([g.length for g in edges.geoms], dtype=np.float64)
    inside = np.where(lengths > 0.0, 1.0 - outside / np.where(lengths > 0.0, lengths, 1.0), 1.0)
    return make_walk_graph(
        node_xy, _final_kinds([occ.kind[i] for i in rep], edges.uv), [occ.ref[i] for i in rep], edges.uv,
        edges.geoms, edges.kinds, edges.refs, inside, np.asarray(connector_cost(lengths, outside, params.outside_penalty)),
        start_node=int(node[0]),
    )


def restrict_outside(g: WalkGraph, max_outside_m: float) -> WalkGraph:
    """Same graph without the connector edges that run more than `max_outside_m` outside `inner`.

    Nodes are kept (ids stay stable); components are recomputed.
    """
    keep = np.asarray([k != EdgeKind.CONNECTOR for k in g.edge_kind], dtype=bool) | (g.outside_len_m() <= max_outside_m)
    idx = np.flatnonzero(keep)
    return make_walk_graph(
        g.node_xy, g.node_kind, g.node_ref, g.edge_uv[idx], [g.edge_geom[i] for i in idx],
        [g.edge_kind[i] for i in idx], [g.edge_ref[i] for i in idx], g.edge_inside_frac[idx], g.edge_cost[idx],
        start_node=g.start_node,
    )


def component_report(g: WalkGraph) -> tuple[ComponentInfo, ...]:
    """One entry per connected component, START's first, then by edge length (longest first)."""
    edge_comp = g.component[g.edge_uv[:, 0]] if g.n_edges else np.zeros(0, dtype=np.int32)
    out: list[ComponentInfo] = []
    for label in np.unique(g.component):
        in_comp = edge_comp == label
        nodes = np.flatnonzero(g.component == label)
        centre = g.node_xy[nodes].mean(axis=0)
        ids = sorted({g.edge_ref[i] for i in np.flatnonzero(in_comp)
                      if g.edge_kind[i] == EdgeKind.INTERROW_CENTERLINE})
        out.append(ComponentInfo(int(label), bool(label == g.start_component), len(nodes), int(in_comp.sum()),
                                 float(g.edge_len_m[in_comp].sum()), float(centre[0]), float(centre[1]),
                                 tuple(ids)))
    return tuple(sorted(out, key=lambda c: (not c.has_start, -round(c.length_m, 6), c.label)))


__all__ = [
    "NODE_MERGE_M", "PASSAGE_NODE_STEP_M", "START_REF", "Attachment", "ComponentInfo", "GraphParams",
    "assemble_graph", "component_report", "cut_params", "outside_lengths", "restrict_outside",
]
