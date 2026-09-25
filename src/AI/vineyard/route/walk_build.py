"""Passable domain + walking graph from an AnnSet and the static route inputs (arch §4.11.1-2).

`prepare_lines` does the expensive, strategy-independent work once (domain, clipping indexes,
passage skeleton, interrow centerlines); `graph_for` adds the connectors of one headland strategy
and assembles the WalkGraph, so strategies can be compared on the same lines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import EdgeKind
from vineyard.route.cells import CellIndex
from vineyard.route.centerlines import (
    Centerline,
    CenterlineParams,
    fallback_centerlines,
    interrow_centerlines,
)
from vineyard.route.connectors import ConnectorParams, ConnectorPlan, HeadlandStrategy, plan_connectors
from vineyard.route.domain import (
    DomainParams,
    DomainSet,
    PassablePart,
    build_domain,
    passable_parts,
)
from vineyard.route.graph import (
    ComponentInfo,
    GraphParams,
    assemble_graph,
    component_report,
    restrict_outside,
)
from vineyard.route.graph_types import PolylineDraft, WalkGraph
from vineyard.route.skeleton import SkeletonParams, skeleton_lines

if TYPE_CHECKING:
    from vineyard.config.sections_post import RouteConfig

PASSAGE_REF_FMT: Final = "P{k:04d}"
PIECE_ID_COLUMN: Final = "piece_id"
INTERROW_ID_COLUMN: Final = "interrow_id"


@dataclass(frozen=True)
class PassableParams:
    domain: DomainParams
    skeleton: SkeletonParams
    centerline: CenterlineParams
    connector: ConnectorParams
    graph: GraphParams

    @classmethod
    def from_config(cls, route: RouteConfig, strategy: HeadlandStrategy | str | None = None) -> PassableParams:
        connector = ConnectorParams.from_config(route.graph)
        if strategy is not None:
            connector = replace(connector, strategy=HeadlandStrategy(strategy))
        return cls(domain=DomainParams.from_config(route.domain), skeleton=SkeletonParams.from_config(route.graph),
                   centerline=CenterlineParams.from_config(route.graph), connector=connector,
                   graph=GraphParams.from_config(route.graph))

    def with_strategy(self, strategy: HeadlandStrategy | str) -> PassableParams:
        return replace(self, connector=replace(self.connector, strategy=HeadlandStrategy(strategy)))


@dataclass(frozen=True, eq=False)
class PassableInputs:
    rows: gpd.GeoDataFrame        # row_id, vineyard_id, row_index, geometry (derive `rows`)
    pieces: gpd.GeoDataFrame      # piece_id, optional interrow_id, geometry (interrow pieces)
    canopies: tuple[BaseGeometry, ...]
    passages: BaseGeometry
    forbidden: BaseGeometry | None
    start_xy: tuple[float, float]


@dataclass(frozen=True, eq=False)
class LineSet:
    """Strategy-independent part of the walk graph."""

    domain: DomainSet
    inner_index: CellIndex
    parts: tuple[PassablePart, ...]
    skeleton: tuple[LineString, ...]
    centerlines: tuple[Centerline, ...]
    drafts: tuple[PolylineDraft, ...]
    joins: tuple[tuple[int, int], ...]
    start_xy: tuple[float, float]


@dataclass(frozen=True, eq=False)
class PassableResult:
    lines: LineSet
    plan: ConnectorPlan
    graph: WalkGraph
    components: tuple[ComponentInfo, ...]
    strategy: HeadlandStrategy


def _piece_refs(pieces: gpd.GeoDataFrame) -> list[str | None]:
    if INTERROW_ID_COLUMN not in pieces.columns:
        return [None] * len(pieces)
    return [None if v is None or pd.isna(v) or not str(v) else str(v) for v in pieces[INTERROW_ID_COLUMN]]


def _drafts(skeleton: Sequence[LineString], lines: Sequence[Centerline]) -> tuple[list[PolylineDraft], list]:
    drafts = [PolylineDraft(g, EdgeKind.PASSAGE_CENTERLINE, PASSAGE_REF_FMT.format(k=k))
              for k, g in enumerate(skeleton, start=1)]
    joins: list[tuple[int, int]] = []
    prev: Centerline | None = None
    for c in lines:
        drafts.append(PolylineDraft(c.geom, EdgeKind.INTERROW_CENTERLINE, c.interrow_id))
        if (prev is not None and c.source == "midline" and prev.source == "midline"
                and prev.interrow_id == c.interrow_id and c.piece == prev.piece + 1):
            joins.append((len(drafts) - 2, len(drafts) - 1))
        prev = c
    return drafts, joins


def _check_pieces(pieces: gpd.GeoDataFrame) -> None:
    if PIECE_ID_COLUMN not in pieces.columns:
        raise ValueError(f"interrow pieces lack the {PIECE_ID_COLUMN!r} column (got {list(pieces.columns)})")


def prepare_lines(inputs: PassableInputs, params: PassableParams) -> LineSet:
    """Domain, passage skeleton and interrow centerlines (midlines + skeleton fallback)."""
    _check_pieces(inputs.pieces)
    geoms = tuple(inputs.pieces.geometry)
    ids = tuple(str(v) for v in inputs.pieces[PIECE_ID_COLUMN])
    dom = build_domain(geoms, inputs.passages, inputs.forbidden, inputs.canopies, params.domain)
    eroded_index = CellIndex.build(dom.eroded)
    passage_region = shapely.intersection(inputs.passages, dom.raw, grid_size=params.domain.grid_size_m)
    skeleton = skeleton_lines(passage_region, params.skeleton, guard=dom.eroded)
    mids = interrow_centerlines(inputs.rows, eroded_index, params.centerline)
    extra = fallback_centerlines(geoms, ids, _piece_refs(inputs.pieces), mids, eroded_index, params.skeleton,
                                 params.centerline)
    lines = (*mids, *extra)
    drafts, joins = _drafts(skeleton, lines)
    return LineSet(domain=dom, inner_index=CellIndex.build(dom.inner),
                   parts=passable_parts(dom, geoms, ids, inputs.passages), skeleton=skeleton, centerlines=lines,
                   drafts=tuple(drafts), joins=tuple(joins), start_xy=inputs.start_xy)


def graph_for(lines: LineSet, connector: ConnectorParams, graph: GraphParams) -> PassableResult:
    """Connectors of `connector.strategy` + graph assembly; raises ValueError if START cannot be linked."""
    dom = lines.domain
    plan = plan_connectors(lines.drafts, lines.joins, lines.start_xy, dom.inner, lines.inner_index, connector)
    walk = assemble_graph((*lines.drafts, *plan.drafts()), plan.attachments(), lines.start_xy, dom.inner, graph,
                          lines.inner_index)
    return PassableResult(lines, plan, walk, component_report(walk), connector.strategy)


def restrict_result(base: PassableResult, connector: ConnectorParams) -> PassableResult:
    """`base` (a PENALTY build) reduced to `connector.strategy` without rebuilding the graph."""
    limit = connector.allowed_outside_m
    graph = restrict_outside(base.graph, limit)
    return PassableResult(base.lines, base.plan.restricted(limit), graph, component_report(graph), connector.strategy)


def build_passable(inputs: PassableInputs, params: PassableParams) -> PassableResult:
    """Graph of `params.connector.strategy` (built as PENALTY, then restricted: same connector set)."""
    base = graph_for(prepare_lines(inputs, params), replace(params.connector, strategy=HeadlandStrategy.PENALTY),
                     params.graph)
    return base if params.connector.strategy == HeadlandStrategy.PENALTY else restrict_result(base, params.connector)


__all__ = [
    "LineSet", "PassableInputs", "PassableParams", "PassableResult", "build_passable", "graph_for", "prepare_lines",
    "restrict_result",
]
