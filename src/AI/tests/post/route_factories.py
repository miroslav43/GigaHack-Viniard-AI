"""Synthetic walking graphs, domains and targets for the route tests (no dependency on graph.py).

`graph_from_lines` snaps polyline endpoints into nodes, optionally splits every line into pieces of
`node_step_m`, and measures each edge's outside length against `inner` (cost = len + penalty * outside).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from vineyard.contracts.enums import EdgeKind
from vineyard.geo.tiling import CRS_EPSG
from vineyard.route.graph_types import WalkGraph, make_walk_graph
from vineyard.route.validate import outside_lengths

KEY_DECIMALS = 6
PENALTY = 10.0
IC, PC, CO = EdgeKind.INTERROW_CENTERLINE, EdgeKind.PASSAGE_CENTERLINE, EdgeKind.CONNECTOR


def _pieces(line: LineString, step: float | None) -> list[LineString]:
    if step is None or line.length <= step:
        return [line]
    cuts = np.linspace(0.0, line.length, int(np.ceil(line.length / step)) + 1)
    return [substring(line, a, b) for a, b in zip(cuts[:-1], cuts[1:], strict=True)]


def graph_from_lines(lines: Sequence[tuple[LineString, str, str]], start_xy: tuple[float, float], *,
                     node_step_m: float | None = None, inner: BaseGeometry | None = None,
                     penalty: float = PENALTY) -> WalkGraph:
    """WalkGraph whose nodes are the (rounded) endpoints of every piece; START must be one of them."""
    keys: dict[tuple[float, float], int] = {}
    xy: list[tuple[float, float]] = []
    uv, geoms, kinds, refs = [], [], [], []

    def node(pt: tuple[float, float]) -> int:
        key = (round(pt[0], KEY_DECIMALS), round(pt[1], KEY_DECIMALS))
        if key not in keys:
            keys[key] = len(xy)
            xy.append(key)
        return keys[key]

    for line, kind, ref in lines:
        for piece in _pieces(line, node_step_m):
            coords = list(piece.coords)
            u, v = node(coords[0]), node(coords[-1])
            coords[0], coords[-1] = xy[u], xy[v]
            uv.append((u, v))
            geoms.append(LineString(coords))
            kinds.append(str(kind))
            refs.append(ref)
    start = keys[(round(start_xy[0], KEY_DECIMALS), round(start_xy[1], KEY_DECIMALS))]
    lengths = np.array([g.length for g in geoms])
    outside = np.array([outside_lengths(g, inner, 1e-3).sum() for g in geoms]) if inner is not None \
        else np.zeros(len(geoms))
    inside_frac = 1.0 - outside / lengths
    kinds_out = ["start" if i == start else "passage" for i in range(len(xy))]
    return make_walk_graph(np.array(xy), kinds_out, [""] * len(xy), np.array(uv), geoms, kinds, refs,
                           inside_frac, lengths + penalty * outside, start_node=start)


@dataclass(frozen=True)
class MiniRoute:
    graph: WalkGraph
    raw: MultiPolygon
    inner: MultiPolygon
    eroded: MultiPolygon
    start_xy: tuple[float, float]
    row_ys: tuple[float, ...]
    centre_ys: tuple[float, ...]
    x0: float
    length_m: float


def _prepared(geom: BaseGeometry) -> MultiPolygon:
    parts = [p for p in shapely.get_parts(geom) if isinstance(p, Polygon) and p.area > 0]
    out = MultiPolygon(parts)
    shapely.prepare(out)
    return out


def mini_route(n_rows: int = 5, length_m: float = 60.0, spacing_m: float = 2.6, headland_m: float = 4.0,
               west_gap_m: float = 0.0, east_gap_m: float = 0.0, x0: float = 1000.0, y0: float = 2000.0,
               node_step_m: float | None = 2.0, row_half_m: float = 0.3,
               extra_lines: Sequence[tuple[LineString, str, str]] = ()) -> MiniRoute:
    """Horizontal rows, interrow strips between them, one vertical passage at each end.

    Graph: interrow centerlines, one passage centerline per side (split at every connector and START),
    connectors from centerline ends to the passage centerline. A gap > 0 leaves a headland strip that
    belongs to neither interrows nor passages, so every connector on that side crosses outside.
    """
    row_ys = tuple(y0 + k * spacing_m for k in range(n_rows))
    centre_ys = tuple((a + b) / 2.0 for a, b in zip(row_ys[:-1], row_ys[1:], strict=True))
    x1 = x0 + length_m
    strips = [box(x0, a + row_half_m, x1, b - row_half_m) for a, b in zip(row_ys[:-1], row_ys[1:], strict=True)]
    y_lo, y_hi = row_ys[0] - spacing_m, row_ys[-1] + spacing_m
    west = box(x0 - west_gap_m - headland_m, y_lo, x0 - west_gap_m, y_hi)
    east = box(x1 + east_gap_m, y_lo, x1 + east_gap_m + headland_m, y_hi)
    raw = _prepared(shapely.union_all([*strips, west, east]))
    inner = _prepared(raw.buffer(-0.05, join_style="mitre"))
    eroded = _prepared(raw.buffer(-0.3, join_style="mitre"))
    xw, xe = x0 - west_gap_m - headland_m / 2.0, x1 + east_gap_m + headland_m / 2.0
    start = (xw, y_lo + 1.0)
    lines: list[tuple[LineString, str, str]] = list(extra_lines)
    for k, yc in enumerate(centre_ys, start=1):
        lines.append((LineString([(x0 + 0.3, yc), (x1 - 0.3, yc)]), IC, f"V01-I{k:03d}"))
        lines.append((LineString([(xw, yc), (x0 + 0.3, yc)]), CO, f"W{k}"))
        lines.append((LineString([(x1 - 0.3, yc), (xe, yc)]), CO, f"E{k}"))
    ys_w = sorted({start[1], *centre_ys})
    for a, b in zip(ys_w[:-1], ys_w[1:], strict=True):
        lines.append((LineString([(xw, a), (xw, b)]), PC, "west"))
    for a, b in zip(centre_ys[:-1], centre_ys[1:], strict=True):
        lines.append((LineString([(xe, a), (xe, b)]), PC, "east"))
    graph = graph_from_lines(lines, start, node_step_m=node_step_m, inner=inner)
    return MiniRoute(graph, raw, inner, eroded, start, row_ys, centre_ys, x0, length_m)


def targets_frame(rows: Sequence[dict]) -> gpd.GeoDataFrame:
    """Targets layer subset used by the route: id, x, y, priority, route_role, reachable, reach_note."""
    base = {"priority": 2, "route_role": "must", "reachable": True, "reach_note": "", "gap_length_m": np.nan}
    records = [base | r for r in rows]
    return gpd.GeoDataFrame(records, geometry=gpd.points_from_xy([r["x"] for r in records],
                                                                  [r["y"] for r in records]), crs=CRS_EPSG)


__all__ = ["CO", "IC", "PC", "MiniRoute", "graph_from_lines", "mini_route", "targets_frame"]
