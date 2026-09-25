"""End-to-end passable graph on the mini vineyard and the organizer examples; headland strategies (S5)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon

from tests.post.passable_factories import MiniVineyard, example_reference, mini_vineyard, route_inputs
from vineyard.contracts.enums import EdgeKind
from vineyard.route.connectors import HeadlandStrategy
from vineyard.route.graph_types import NodeKind
from vineyard.route.walk_build import (
    LineSet,
    PassableInputs,
    PassableParams,
    PassableResult,
    build_passable,
    graph_for,
    prepare_lines,
)
from vineyard.route.walk_report import (
    choose_strategy,
    compare_strategies,
    ends_near_passages,
    headland_metrics,
)

MAX_SHARE = 0.005


def _params() -> PassableParams:
    from vineyard.config import load_config

    return PassableParams.from_config(load_config().route)


def _base(lines: LineSet) -> PassableResult:
    return graph_for(lines, _params().connector, _params().graph)


def _inputs(mini: MiniVineyard, passages: MultiPolygon | None = None) -> PassableInputs:
    return PassableInputs(rows=mini.rows, pieces=mini.interrow_pieces, canopies=(),
                          passages=mini.passages if passages is None else passages, forbidden=None,
                          start_xy=mini.start_xy)


def test_params_from_config_strategy_override() -> None:
    from vineyard.config import load_config

    assert PassableParams.from_config(load_config().route).connector.strategy == HeadlandStrategy.PENALTY
    strict = PassableParams.from_config(load_config().route, "inside_only")
    assert strict.connector.strategy == HeadlandStrategy.INSIDE_ONLY
    assert _params().with_strategy("limit").connector.strategy == HeadlandStrategy.LIMIT


def test_touching_mini_vineyard_is_one_component_all_inside() -> None:
    mini = mini_vineyard(end_gap_m=0.0)
    res = build_passable(_inputs(mini), _params())
    g = res.graph
    assert len(res.components) == 1 and res.components[0].has_start
    assert res.components[0].interrow_ids == ("V01-I001", "V01-I002", "V01-I003")
    assert g.node_kind[g.start_node] == NodeKind.START
    assert tuple(g.node_xy[g.start_node]) == mini.start_xy
    assert np.allclose(g.outside_len_m(), 0.0, atol=0.03)
    ir = [i for i, k in enumerate(g.edge_kind) if k == EdgeKind.INTERROW_CENTERLINE]
    assert g.edge_len_m[ir].sum() == pytest.approx(180.0, abs=0.05)
    assert g.edge_len_m[ir].max() <= 2.0 + 1e-6          # a node every centerline_step_m
    m = headland_metrics(res, mini.passages)
    assert (m.n_row_ends, m.n_row_ends_near_passage) == (6, 6)
    assert m.n_interrows_reachable == 3 and m.outside_share_proxy < MAX_SHARE
    assert m.vineyards_reachable == {"V01": True}


def test_headland_gap_strategies() -> None:
    mini = mini_vineyard(end_gap_m=2.0)
    lines = prepare_lines(_inputs(mini), _params())
    assert lines.domain.n_components == 5
    assert ends_near_passages(lines, mini.passages) == (0, 6)
    results = {m.strategy: (m, r) for m, r in compare_strategies(_base(lines), _params(), mini.passages)}
    pen, _ = results["penalty"]
    # per end: the headland turn into the neighbour interrow (0.7 m outside) + the way to the passage (2.1 m)
    assert pen.n_interrows_reachable == 3 and pen.n_connectors_outside == 12
    assert pen.connector_outside_max_m == pytest.approx(2.1, abs=0.01)
    assert pen.outside_share_proxy > MAX_SHARE
    for name in ("limit", "inside_only"):
        m, r = results[name]
        assert m.n_interrows_reachable == 0
        assert m.start_outside_m == pytest.approx(0.0, abs=1e-6)
        assert len(r.components) > 1
    assert choose_strategy([m for m, _ in results.values()], MAX_SHARE) == HeadlandStrategy.LIMIT


def test_inside_only_keeps_dead_end_interrows_through_the_touching_end() -> None:
    mini = mini_vineyard(end_gap_m=0.0)
    west_only = MultiPolygon([mini.passages.geoms[0]])
    params = _params().with_strategy("inside_only")
    res = build_passable(_inputs(mini, west_only), params)
    m = headland_metrics(res, west_only)
    assert m.n_row_ends_near_passage == 3 and m.n_unconnected_ends == 0
    assert m.n_connectors_dropped == 3          # east ends: only the 0.7 m-outside headland turn exists
    assert m.n_interrows_reachable == 3 and m.start_outside_m == pytest.approx(0.0, abs=0.03)


def test_restricted_penalty_build_equals_direct_strategy_build() -> None:
    mini = mini_vineyard(end_gap_m=2.0)
    lines = prepare_lines(_inputs(mini), _params())
    for name in ("limit", "inside_only"):
        direct = graph_for(lines, _params().with_strategy(name).connector, _params().graph)
        restricted = build_passable(_inputs(mini), _params().with_strategy(name))
        assert restricted.strategy == name
        assert [c.kept for c in restricted.plan.connectors if c.kept] == [c.kept for c in direct.plan.kept]
        # node counts differ (the superset's attachment points stay), walkable length and reach do not
        assert [(round(c.length_m, 6), c.interrow_ids) for c in restricted.components if c.has_start] == \
            [(round(c.length_m, 6), c.interrow_ids) for c in direct.components if c.has_start]
    with pytest.raises(ValueError, match="PENALTY"):
        compare_strategies(direct, _params(), mini.passages)


def test_choose_strategy_prefers_reach_then_order() -> None:
    mini = mini_vineyard(end_gap_m=0.0)
    lines = prepare_lines(_inputs(mini), _params())
    metrics = [m for m, _ in compare_strategies(_base(lines), _params(), mini.passages)]
    assert choose_strategy(metrics, MAX_SHARE) == HeadlandStrategy.PENALTY
    assert choose_strategy([], MAX_SHARE) == HeadlandStrategy.INSIDE_ONLY


def test_missing_piece_id_column_is_an_error() -> None:
    mini = mini_vineyard()
    bad = PassableInputs(mini.rows, gpd.GeoDataFrame(geometry=list(mini.interrow_pieces.geometry)), (),
                         mini.passages, None, mini.start_xy)
    with pytest.raises(ValueError, match="piece_id"):
        prepare_lines(bad, _params())


def test_start_outside_domain_is_an_error() -> None:
    mini = mini_vineyard()
    inputs = PassableInputs(mini.rows, mini.interrow_pieces, (), mini.passages, None, (0.0, 0.0))
    lines = prepare_lines(inputs, _params())
    with pytest.raises(ValueError, match="START"):
        graph_for(lines, _params().connector, _params().graph)


@pytest.mark.examples
@pytest.mark.slow
def test_reference_examples_graph(examples_xml: bytes) -> None:
    rows, pieces, canopies = example_reference(examples_xml)
    real = route_inputs()
    inputs = PassableInputs(rows, pieces, tuple(canopies.geometry), real.passages, real.forbidden, real.start_xy)
    lines = prepare_lines(inputs, _params())
    assert len({c.interrow_id for c in lines.centerlines}) >= 20
    for m, res in compare_strategies(_base(lines), _params(), real.passages):
        g = res.graph
        assert g.node_kind[g.start_node] == NodeKind.START
        assert tuple(g.node_xy[g.start_node]) == real.start_xy
        assert m.n_row_ends > 0 and 0 <= m.n_row_ends_near_passage <= m.n_row_ends
        assert set(m.vineyards_reachable) == {"V01", "V02"}
