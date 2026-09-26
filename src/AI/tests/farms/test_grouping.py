from __future__ import annotations

import pytest
from shapely.geometry import LineString, MultiPolygon, box

from vineyard.errors import SchemaError
from vineyard.farms.grouping import FarmParams, components, farm_id, farm_of_block, group_blocks

X0, Y0 = 629000.0, 5220000.0
P = FarmParams(gap_max_m=10.0, touch_m=0.5, link_width_m=1.0, public_road_buffer_m=3.0, outline_simplify_m=0.2)


def square(dx: float, dy: float = 0.0, side: float = 40.0):
    return box(X0 + dx, Y0 + dy, X0 + dx + side, Y0 + dy + side)


def vertical_road(x: float) -> LineString:
    return LineString([(X0 + x, Y0 - 100), (X0 + x, Y0 + 200)])


def test_blocks_within_the_gap_are_one_farm() -> None:
    (farm,) = group_blocks(["V01", "V02"], [square(0), square(46)], None, P)
    assert farm.vineyard_ids == ("V01", "V02") and farm.farm_id == "F01"
    # the closing bridges the 6 m track: one outline covering both blocks and the track
    assert farm.outline.geom_type == "Polygon"
    assert farm.outline.area == pytest.approx(86 * 40, rel=0.01)


def test_blocks_farther_than_the_gap_are_two_farms() -> None:
    farms = group_blocks(["V01", "V02"], [square(0), square(55)], None, P)
    assert [f.vineyard_ids for f in farms] == [("V01",), ("V02",)]


def test_a_public_road_in_the_gap_splits_the_farm() -> None:
    farms = group_blocks(["V01", "V02"], [square(0), square(46)], vertical_road(43), P)
    assert len(farms) == 2
    assert all(not f.outline.intersects(vertical_road(43).buffer(2.9)) for f in farms)


def test_touching_blocks_stay_one_farm_whatever_the_roads() -> None:
    (farm,) = group_blocks(["V01", "V02"], [square(0), square(40.2)], vertical_road(40.1), P)
    assert farm.vineyard_ids == ("V01", "V02")


def test_a_chain_of_blocks_is_one_farm_and_ids_sort_naturally() -> None:
    geoms = [square(0), square(46), square(92)]
    (farm,) = group_blocks(["V10", "V2", "V1"], geoms, None, P)
    assert farm.vineyard_ids == ("V1", "V2", "V10")


def test_farms_are_numbered_north_first_then_west() -> None:
    geoms = [square(0, -200), square(300, 0), square(0, 0)]
    farms = group_blocks(["V03", "V02", "V01"], geoms, None, P)
    assert [(f.farm_id, f.vineyard_ids) for f in farms] == [("F01", ("V01",)), ("F02", ("V02",)), ("F03", ("V03",))]
    assert farm_of_block(farms) == {"V01": "F01", "V02": "F02", "V03": "F03"}


def test_a_road_through_a_farm_leaves_a_multipolygon_outline_largest_first() -> None:
    blocks = [square(0), square(40.2, side=10)]  # touching: one farm, the road cuts the outline in two
    (farm,) = group_blocks(["V01", "V02"], blocks, vertical_road(40.1), P)
    assert isinstance(farm.outline, MultiPolygon)
    assert farm.outline.geoms[0].area > farm.outline.geoms[1].area


def test_empty_and_mismatched_inputs() -> None:
    assert group_blocks([], [], None, P) == ()
    with pytest.raises(SchemaError):
        group_blocks(["V01"], [], None, P)


def test_farm_id_width_and_components() -> None:
    assert farm_id(0, 28) == "F01" and farm_id(99, 120) == "F100" and farm_id(4, 120) == "F005"
    assert components(5, [(0, 3), (3, 4)]) == ((0, 3, 4), (1,), (2,))
