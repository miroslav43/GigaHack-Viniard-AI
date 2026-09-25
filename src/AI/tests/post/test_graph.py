"""Walking-graph assembly: cut drafts into node-spaced edges, merge nodes, label, cost (arch §4.11.2)."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, box

from vineyard.contracts.enums import EdgeKind
from vineyard.route.graph import (
    Attachment,
    GraphParams,
    assemble_graph,
    component_report,
    cut_params,
    restrict_outside,
)
from vineyard.route.graph_io import LayerProvenance
from vineyard.route.graph_types import NodeKind, PolylineDraft, WalkGraph, make_walk_graph

PARAMS = GraphParams(centerline_node_step_m=2.0, passage_node_step_m=5.0, outside_penalty=10.0)
BIG = box(-100, -100, 100, 100)
IR = EdgeKind.INTERROW_CENTERLINE
PS = EdgeKind.PASSAGE_CENTERLINE
CN = EdgeKind.CONNECTOR


def _draft(pts: list[tuple[float, float]], kind: EdgeKind, ref: str = "x") -> PolylineDraft:
    return PolylineDraft(LineString(pts), kind, ref)


def _t_graph() -> WalkGraph:
    drafts = [_draft([(0, 0), (10, 0)], IR, "V01-I001"), _draft([(10, -5), (10, 5)], PS, "P0001")]
    return assemble_graph(drafts, [Attachment(1, 5.0, "V01-I001:end")], (10.0, -5.0), BIG, PARAMS)


def test_params_from_config() -> None:
    from vineyard.config import load_config

    params = GraphParams.from_config(load_config().route.graph)
    assert params.centerline_node_step_m == 2.0 and params.outside_penalty == 10.0
    assert params.passage_node_step_m == 5.0


def test_params_validation() -> None:
    with pytest.raises(ValueError, match="node_step"):
        GraphParams(centerline_node_step_m=-1.0, passage_node_step_m=5.0, outside_penalty=10.0)


def test_cut_params_regular_and_extra() -> None:
    assert np.allclose(cut_params(5.0, 2.0, np.array([])), [0, 5 / 3, 10 / 3, 5])
    assert np.allclose(cut_params(4.0, 2.0, np.array([1.0, 2.0005, 9.0])), [0, 1, 2, 4])
    assert np.allclose(cut_params(4.0, 0.0, np.array([3.9995])), [0, 4])


def test_t_graph_nodes_edges_and_kinds() -> None:
    g = _t_graph()
    assert g.n_nodes == 8 and g.n_edges == 7
    assert g.node_kind[g.start_node] == NodeKind.START
    assert tuple(g.node_xy[g.start_node]) == (10.0, -5.0)
    kinds = dict(zip(map(tuple, np.round(g.node_xy, 6)), g.node_kind, strict=True))
    assert kinds[(0.0, 0.0)] == NodeKind.ROW_END and kinds[(10.0, 0.0)] == NodeKind.ROW_END
    assert kinds[(2.0, 0.0)] == NodeKind.INTERROW and kinds[(10.0, 5.0)] == NodeKind.PASSAGE
    assert set(g.component.tolist()) == {0}
    assert np.allclose(g.edge_inside_frac, 1.0) and np.allclose(g.edge_cost, g.edge_len_m)
    assert g.edge_len_m.sum() == pytest.approx(20.0)
    assert g.edge_kind.count(IR) == 5 and g.edge_kind.count(PS) == 2


def test_outside_fraction_and_penalised_cost() -> None:
    inner = shapely.union_all([box(0, 0, 4, 2), box(6, 0, 10, 2)])
    shapely.prepare(inner)
    g = assemble_graph([_draft([(1, 1), (9, 1)], CN, "c")], [], (1.0, 1.0), inner, PARAMS)
    assert g.n_edges == 1
    assert g.edge_inside_frac[0] == pytest.approx(0.75)
    assert g.edge_cost[0] == pytest.approx(8.0 + 10.0 * 2.0)
    assert g.outside_len_m()[0] == pytest.approx(2.0)


def test_nodes_within_1cm_merge_and_geometry_is_snapped() -> None:
    drafts = [_draft([(0, 0), (3, 0)], PS, "a"), _draft([(3.005, 0), (3.005, 3)], PS, "b")]
    g = assemble_graph(drafts, [], (0.0, 0.0), BIG, PARAMS)
    assert g.n_nodes == 3 and g.n_edges == 2
    assert len(set(g.component.tolist())) == 1


def test_three_passage_branches_make_a_junction() -> None:
    drafts = [_draft([(0, 0), (4, 0)], PS, "a"), _draft([(4, 0), (8, 0)], PS, "b"), _draft([(4, 0), (4, 4)], PS, "c")]
    g = assemble_graph(drafts, [], (0.0, 0.0), BIG, PARAMS)
    idx = int(np.flatnonzero(np.all(np.isclose(g.node_xy, (4.0, 0.0)), axis=1))[0])
    assert g.node_kind[idx] == NodeKind.JUNCTION


def test_attachment_labels_and_splits_the_target_line() -> None:
    drafts = [_draft([(0, 0), (8, 0)], PS, "p"), _draft([(3, 3), (3, 0)], CN, "V01-I001:start")]
    g = assemble_graph(drafts, [Attachment(0, 3.0, "V01-I001:start")], (0.0, 0.0), BIG, PARAMS)
    idx = int(np.flatnonzero(np.all(np.isclose(g.node_xy, (3.0, 0.0)), axis=1))[0])
    assert g.node_kind[idx] == NodeKind.ATTACH and g.node_ref[idx] == "V01-I001:start"
    assert g.n_edges == 4   # p cut at 0, 3, 4, 8 + the connector


def test_degenerate_draft_edge_is_dropped() -> None:
    drafts = [_draft([(0, 0), (0.004, 0)], PS, "tiny"), _draft([(0, 0), (5, 0)], PS, "p")]
    g = assemble_graph(drafts, [], (0.0, 0.0), BIG, PARAMS)
    assert g.n_edges == 1


def test_start_far_from_everything_is_an_isolated_node() -> None:
    g = assemble_graph([_draft([(0, 0), (5, 0)], PS, "p")], [], (50.0, 50.0), BIG, PARAMS)
    assert g.node_kind[g.start_node] == NodeKind.START
    assert g.component[g.start_node] != g.component[1]


def test_assembly_is_deterministic() -> None:
    assert _t_graph().equals(_t_graph())


def test_attachment_index_out_of_range() -> None:
    with pytest.raises(ValueError, match="attachment"):
        assemble_graph([_draft([(0, 0), (5, 0)], PS)], [Attachment(3, 1.0, "")], (0.0, 0.0), BIG, PARAMS)


def test_component_report_start_first() -> None:
    xy = np.array([[0, 0], [1, 0], [10, 10], [12, 10], [13, 10]], float)
    uv = np.array([[0, 1], [2, 3], [3, 4]])
    geoms = [LineString([xy[u], xy[v]]) for u, v in uv]
    g = make_walk_graph(xy, ["passage", "start", "row_end", "interrow", "row_end"], ["", "", *["V01-I001"] * 3],
                        uv, geoms, [PS, IR, IR], ["P1", "V01-I001", "V01-I001"], np.ones(3), start_node=1)
    rep = component_report(g)
    assert [c.has_start for c in rep] == [True, False]
    assert rep[0].length_m == pytest.approx(1.0) and rep[1].length_m == pytest.approx(3.0)
    assert rep[1].interrow_ids == ("V01-I001",)


def test_restrict_outside_drops_only_outside_connectors() -> None:
    inner = shapely.union_all([box(-1, -1, 4.5, 1), box(5.5, -1, 11, 1), box(-1, -1, 1, 9)])
    shapely.prepare(inner)
    drafts = [_draft([(0, 0), (4, 0)], PS, "a"), _draft([(6, 0), (10, 0)], PS, "b"),
              _draft([(4, 0), (6, 0)], CN, "gap"), _draft([(0, 0), (0, 8)], CN, "in")]
    g = assemble_graph(drafts, [], (0.0, 0.0), inner, PARAMS)
    assert len(set(g.component.tolist())) == 1
    strict = restrict_outside(g, 0.02)
    assert strict.n_nodes == g.n_nodes and strict.n_edges == g.n_edges - 1
    assert "gap" not in strict.edge_ref and "in" in strict.edge_ref
    assert len(set(strict.component.tolist())) == 2
    assert restrict_outside(g, 5.0).n_edges == g.n_edges


# ------------------------------------------------------------------ graph_io


def _prov() -> LayerProvenance:
    return LayerProvenance(source="reference", run_id="20260926T0100-post-abcdef", model_version="test")


def test_graph_layers_round_trip_through_parquet(tmp_path) -> None:
    from vineyard.geo.vector_io import read_layer, write_layer
    from vineyard.route.graph_io import graph_from_layers, graph_to_layers

    g = _t_graph()
    nodes, edges = graph_to_layers(g, _prov())
    assert list(nodes["node_id"]) == list(range(g.n_nodes)) and nodes["kind"].iloc[g.start_node] == "start"
    assert edges["inside_frac"].dtype == np.float32 and edges["cost"].dtype == np.float64
    back = graph_from_layers(read_layer(write_layer(nodes, "walk_nodes", tmp_path / "walk_nodes.parquet")),
                             read_layer(write_layer(edges, "walk_edges", tmp_path / "walk_edges.parquet")))
    assert back.equals(g, tol=1e-6)


def test_graph_from_layers_rejects_bad_ids_and_starts() -> None:
    from vineyard.route.graph_io import graph_from_layers, graph_to_layers

    nodes, edges = graph_to_layers(_t_graph(), _prov())
    with pytest.raises(ValueError, match="one start"):
        graph_from_layers(nodes.assign(kind="passage"), edges)
    with pytest.raises(ValueError, match="walk_edges ids"):
        graph_from_layers(nodes, edges.assign(edge_id=edges["edge_id"] + 1))


def test_domain_and_parts_layers() -> None:
    from vineyard.route.domain import DomainParams, build_domain, passable_parts
    from vineyard.route.graph_io import domain_from_layer, domain_to_layer, parts_to_layer

    params = DomainParams(0.001, 0.05, 0.3, 0.05, True)
    dom = build_domain((box(0, 0, 10, 4),), shapely.MultiPolygon([box(10, -5, 14, 9)]), None, (), params)
    layer = domain_to_layer(dom, _prov())
    assert layer["domain_id"].tolist() == ["D1"] and layer["n_components"].tolist() == [1]
    assert domain_from_layer(layer, params).eroded.area == pytest.approx(dom.eroded.area)
    parts = parts_to_layer(passable_parts(dom, (box(0, 0, 10, 4),), ("t:I001",), dom.raw), _prov())
    assert parts["kind"].tolist() == ["interrow", "passage"]
    assert len(parts_to_layer((), _prov())) == 0
    with pytest.raises(ValueError, match="exactly one row"):
        domain_from_layer(layer.iloc[0:0], params)
