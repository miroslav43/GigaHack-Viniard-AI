"""WalkGraph and passable domain <-> contract layers (contract §2.5.12).

walk_nodes: node_id = graph node index; walk_edges: edge_id = graph edge index, u/v = node ids,
geometry oriented u -> v. passable_domain holds the raw domain (erosion_m = 0); inner and eroded
are rebuilt from it with `domain_from_raw`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from vineyard.contracts.schemas import coerce_layer, empty_layer, validate_layer
from vineyard.geo.tiling import CRS_EPSG
from vineyard.route.domain import DomainParams, DomainSet, PassablePart, domain_from_raw
from vineyard.route.graph_types import NodeKind, WalkGraph, make_walk_graph

NODES_LAYER: Final = "walk_nodes"
EDGES_LAYER: Final = "walk_edges"
DOMAIN_LAYER: Final = "passable_domain"
PARTS_LAYER: Final = "passable_parts"
DOMAIN_ID: Final = "D1"
FULL_CONFIDENCE: Final = 1.0


@dataclass(frozen=True)
class LayerProvenance:
    """Contract provenance columns shared by every row of a layer."""

    source: str
    run_id: str
    model_version: str
    confidence: float = FULL_CONFIDENCE
    qa_flags: str = ""

    def columns(self, n: int) -> dict[str, list]:
        return {"source": [self.source] * n, "run_id": [self.run_id] * n, "model_version": [self.model_version] * n,
                "confidence": [self.confidence] * n, "qa_flags": [self.qa_flags] * n}


def _layer(data: dict, geoms: Sequence, name: str) -> gpd.GeoDataFrame:
    if not geoms:
        return empty_layer(name)
    frame = coerce_layer(gpd.GeoDataFrame(data, geometry=list(geoms), crs=CRS_EPSG), name)
    validate_layer(frame, name)
    return frame


def graph_to_layers(g: WalkGraph, prov: LayerProvenance) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes = _layer({"node_id": np.arange(g.n_nodes, dtype=np.int64), "kind": list(g.node_kind),
                    "ref_id": list(g.node_ref), **prov.columns(g.n_nodes)},
                   [Point(xy) for xy in g.node_xy], NODES_LAYER)
    edges = _layer({"edge_id": np.arange(g.n_edges, dtype=np.int64), "u": g.edge_uv[:, 0], "v": g.edge_uv[:, 1],
                    "length_m": g.edge_len_m, "kind": list(g.edge_kind), "ref_id": list(g.edge_ref),
                    "inside_frac": g.edge_inside_frac.astype(np.float32), "cost": g.edge_cost,
                    **prov.columns(g.n_edges)},
                   list(g.edge_geom), EDGES_LAYER)
    return nodes, edges


def _check_ids(ids: np.ndarray, what: str) -> None:
    if not np.array_equal(np.sort(ids), np.arange(len(ids))):
        raise ValueError(f"{what} ids must be 0..{len(ids) - 1} without gaps or duplicates")


def graph_from_layers(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> WalkGraph:
    """Inverse of `graph_to_layers` (inside_frac comes back at float32 precision)."""
    nodes = nodes.sort_values("node_id", kind="mergesort")
    edges = edges.sort_values("edge_id", kind="mergesort")
    _check_ids(nodes["node_id"].to_numpy(np.int64), NODES_LAYER)
    _check_ids(edges["edge_id"].to_numpy(np.int64), EDGES_LAYER)
    kinds = [str(k) for k in nodes["kind"]]
    starts = [i for i, k in enumerate(kinds) if k == NodeKind.START]
    if len(starts) != 1:
        raise ValueError(f"{NODES_LAYER}: expected exactly one start node, found {len(starts)}")
    xy = np.column_stack([nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()])
    return make_walk_graph(
        xy, kinds, [str(r) for r in nodes["ref_id"]], edges[["u", "v"]].to_numpy(np.int64), list(edges.geometry),
        [str(k) for k in edges["kind"]], [str(r) for r in edges["ref_id"]],
        edges["inside_frac"].to_numpy(np.float64), edges["cost"].to_numpy(np.float64), start_node=starts[0],
    )


def domain_to_layer(dom: DomainSet, prov: LayerProvenance) -> gpd.GeoDataFrame:
    return _layer({"domain_id": [DOMAIN_ID], "area_m2": [float(dom.raw.area)], "erosion_m": [0.0],
                   "n_components": [dom.n_components], **prov.columns(1)}, [dom.raw], DOMAIN_LAYER)


def domain_from_layer(layer: gpd.GeoDataFrame, params: DomainParams) -> DomainSet:
    if len(layer) != 1:
        raise ValueError(f"{DOMAIN_LAYER}: expected exactly one row, got {len(layer)}")
    return domain_from_raw(layer.geometry.iloc[0], params)


def parts_to_layer(parts: Sequence[PassablePart], prov: LayerProvenance) -> gpd.GeoDataFrame:
    return _layer({"part_id": [p.part_id for p in parts], "kind": [p.kind for p in parts],
                   "ref_id": [p.ref_id for p in parts], "area_m2": [p.area_m2 for p in parts],
                   "erosion_m": [p.erosion_m for p in parts], **prov.columns(len(parts))},
                  [p.geom for p in parts], PARTS_LAYER)


__all__ = [
    "DOMAIN_ID", "DOMAIN_LAYER", "EDGES_LAYER", "NODES_LAYER", "PARTS_LAYER", "LayerProvenance", "domain_from_layer",
    "domain_to_layer", "graph_from_layers", "graph_to_layers", "parts_to_layer",
]
