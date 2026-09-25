"""GTSP candidate sets (arch §4.11.3, design 04 §3.8): projections per side, spurs, reachability."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, box

from vineyard.config import load_config
from vineyard.route.candidates import (
    NOTE_DISCONNECTED,
    NOTE_TOO_FAR,
    ROLE_MUST,
    ROLE_OPTIONAL,
    CandidateParams,
    TargetPoint,
    augment_with_targets,
    target_points,
)
from vineyard.route.graph_types import NodeKind

from .route_factories import CO, IC, PC, graph_from_lines, mini_route, targets_frame

PARAMS = CandidateParams(radius_m=1.9, max_per_side=3, max_snap_m=5.0, spur_standoff_m=1.5, node_snap_m=0.05,
                         merge_m=0.0)


def _tp(tid: str, x: float, y: float, **kw: object) -> TargetPoint:
    return TargetPoint(target_id=tid, x=x, y=y, role=str(kw.pop("role", ROLE_MUST)), **kw)


def _dist(g, node: int, t: TargetPoint) -> float:
    return float(np.hypot(*(g.node_xy[node] - (t.x, t.y))))


def test_params_from_config() -> None:
    p = CandidateParams.from_route_cfg(load_config().route)
    assert (p.radius_m, p.max_per_side, p.max_snap_m, p.spur_standoff_m) == (1.9, 3, 5.0, 1.5)


def test_target_on_middle_row_axis_gets_both_neighbouring_interrows() -> None:
    m = mini_route()
    t = _tp("T-GAP-0001", 1030.5, m.row_ys[2])
    aug = augment_with_targets(m.graph, [t], m.inner, PARAMS)
    (cset,) = aug.sets
    ys = {round(float(aug.graph.node_xy[n][1]), 3) for n in cset.nodes}
    assert ys == {round(m.centre_ys[1], 3), round(m.centre_ys[2], 3)}
    assert all(_dist(aug.graph, n, t) <= PARAMS.radius_m + 1e-9 for n in cset.nodes)
    assert min(_dist(aug.graph, n, t) for n in cset.nodes) == pytest.approx(1.3)
    assert len(cset.nodes) == 4  # the projection and the 2 m node before it, on each side


def test_one_candidate_per_side() -> None:
    m = mini_route()
    t = _tp("T-GAP-0001", 1030.5, m.row_ys[2])
    aug = augment_with_targets(m.graph, [t], m.inner, PARAMS.with_changes(max_per_side=1))
    (cset,) = aug.sets
    assert len(cset.nodes) == 2
    assert [_dist(aug.graph, n, t) for n in cset.nodes] == pytest.approx([1.3, 1.3])
    assert aug.reach[0].reachable and aug.reach[0].n_candidates == 2
    assert aug.reach[0].snap_dist_m == pytest.approx(1.3)


def test_split_keeps_total_length_and_cost_and_marks_target_nodes() -> None:
    m = mini_route()
    t = _tp("T-GAP-0001", 1030.5, m.row_ys[2])
    aug = augment_with_targets(m.graph, [t], m.inner, PARAMS)
    assert aug.graph.edge_len_m.sum() == pytest.approx(m.graph.edge_len_m.sum())
    assert aug.graph.edge_cost.sum() == pytest.approx(m.graph.edge_cost.sum())
    new = range(m.graph.n_nodes, aug.graph.n_nodes)
    assert len(new) == 2 and all(aug.graph.node_kind[i] == NodeKind.TARGET for i in new)
    assert all(aug.graph.node_ref[i] == "T-GAP-0001" for i in new)
    assert aug.graph.start_node == m.graph.start_node


def test_targets_sharing_all_nodes_are_one_set() -> None:
    m = mini_route()
    a = _tp("T-GAP-0001", 1030.0, m.row_ys[2])
    b = _tp("T-GAP-0002", 1030.01, m.row_ys[2], role=ROLE_OPTIONAL)
    aug = augment_with_targets(m.graph, [a, b], m.inner, PARAMS.with_changes(max_per_side=1))
    (cset,) = aug.sets
    assert cset.target_ids == ("T-GAP-0001", "T-GAP-0002")
    assert cset.role == ROLE_MUST
    assert aug.nodes_of("T-GAP-0002") == cset.nodes


def test_spur_for_target_beyond_radius_but_within_snap() -> None:
    wide = box(-5.0, -5.0, 50.0, 5.0)
    inner = wide.buffer(-0.05)
    shapely.prepare(inner)
    g = graph_from_lines([(LineString([(0.0, 0.0), (40.0, 0.0)]), PC, "p")], (0.0, 0.0), inner=inner)
    t = _tp("T-WST-0001", 20.0, 2.5)
    aug = augment_with_targets(g, [t], inner, PARAMS)
    (cset,) = aug.sets
    (node,) = cset.nodes
    assert _dist(aug.graph, node, t) == pytest.approx(1.5)
    assert "target_spur" in aug.graph.edge_kind
    spur = aug.graph.edge_geom[aug.graph.edge_kind.index("target_spur")]
    assert spur.length == pytest.approx(1.0)
    assert aug.reach[0].reachable and aug.reach[0].snap_dist_m == pytest.approx(2.5)


def test_spur_leaving_the_domain_is_unreachable() -> None:
    narrow = box(-5.0, -1.0, 50.0, 1.0)
    inner = narrow.buffer(-0.05)
    g = graph_from_lines([(LineString([(0.0, 0.0), (40.0, 0.0)]), PC, "p")], (0.0, 0.0), inner=inner)
    aug = augment_with_targets(g, [_tp("T-WST-0001", 20.0, 3.0)], inner, PARAMS)
    assert aug.sets == ()
    assert not aug.reach[0].reachable and aug.reach[0].note == NOTE_TOO_FAR


def test_too_far_and_disconnected() -> None:
    m = mini_route()
    lines = [(LineString([(0.0, 0.0), (20.0, 0.0)]), PC, "p"), (LineString([(0.0, 50.0), (20.0, 50.0)]), IC, "q")]
    g = graph_from_lines(lines, (0.0, 0.0))
    far = _tp("T-GAP-0001", 10.0, 6.0)
    island = _tp("T-GAP-0002", 10.0, 49.0)
    aug = augment_with_targets(g, [far, island], m.inner, PARAMS)
    assert aug.sets == ()
    assert [(r.reachable, r.note) for r in aug.reach] == [(False, NOTE_TOO_FAR), (False, NOTE_DISCONNECTED)]
    assert aug.reach[0].snap_dist_m == pytest.approx(6.0)


def test_preliminary_unreachable_is_dropped_with_its_note() -> None:
    m = mini_route()
    t = _tp("T-WST-0001", 1030.5, m.row_ys[2], reachable=False, note="in_forbidden")
    aug = augment_with_targets(m.graph, [t], m.inner, PARAMS)
    assert aug.sets == ()
    assert (aug.reach[0].reachable, aug.reach[0].note, aug.reach[0].n_candidates) == (False, "in_forbidden", 0)
    assert aug.graph.n_nodes == m.graph.n_nodes


def test_candidates_come_from_both_sides_of_a_single_line_target() -> None:
    lines = [(LineString([(0.0, 0.0), (20.0, 0.0)]), IC, "a"), (LineString([(0.0, 2.6), (20.0, 2.6)]), IC, "b"),
             (LineString([(0.0, 0.0), (0.0, 2.6)]), CO, "c")]
    g = graph_from_lines(lines, (0.0, 0.0), node_step_m=1.0)
    t = _tp("T-GAP-0001", 10.25, 1.3)
    aug = augment_with_targets(g, [t], None, PARAMS.with_changes(max_per_side=2))
    ys = sorted(float(aug.graph.node_xy[n][1]) for n in aug.sets[0].nodes)
    assert ys == [0.0, 0.0, 2.6, 2.6]


def test_augment_is_deterministic() -> None:
    m = mini_route()
    targets = [_tp(f"T-GAP-{k:04d}", 1010.0 + 7.3 * k, m.row_ys[1 + k % 3]) for k in range(1, 6)]
    a = augment_with_targets(m.graph, targets, m.inner, PARAMS)
    b = augment_with_targets(m.graph, targets, m.inner, PARAMS)
    assert a.graph.equals(b.graph) and a.sets == b.sets and a.reach == b.reach


def test_target_points_from_layer() -> None:
    frame = targets_frame([
        {"target_id": "T-GAP-0002", "x": 1.0, "y": 2.0, "priority": 3, "route_role": "optional"},
        {"target_id": "T-WST-0001", "x": 3.0, "y": 4.0, "priority": 1, "reachable": False,
         "reach_note": "in_forbidden"},
    ])
    pts = target_points(frame)
    assert [p.target_id for p in pts] == ["T-GAP-0002", "T-WST-0001"]
    assert (pts[0].role, pts[0].priority, pts[1].reachable, pts[1].note) == ("optional", 3, False, "in_forbidden")


def test_target_points_role_from_priority_when_column_missing() -> None:
    frame = targets_frame([{"target_id": "T-MSP-0001", "x": 1.0, "y": 2.0, "priority": 3},
                           {"target_id": "T-GAP-0001", "x": 1.0, "y": 2.0, "priority": 2}]).drop(columns="route_role")
    assert [p.role for p in target_points(frame)] == [ROLE_OPTIONAL, ROLE_MUST]


def test_target_points_rejects_missing_columns() -> None:
    frame = targets_frame([{"target_id": "T-GAP-0001", "x": 1.0, "y": 2.0}]).drop(columns="target_id")
    with pytest.raises(ValueError, match="target_id"):
        target_points(frame)
