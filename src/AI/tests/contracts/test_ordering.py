import math

import numpy as np
import pytest
from lxml import etree

from vineyard.contracts.ids import parse_tile_id, row_index_of
from vineyard.contracts.ordering import (
    NORMAL_NS_TOL,
    angle_deg_utm,
    angle_px_to_utm,
    block_sort_key,
    canonical_normal,
    mean_axial_angle_deg,
    order_blocks,
    order_by_normal,
)
from vineyard.errors import SchemaError

GRID_X0 = 628992.0
GRID_Y0 = 5221222.4
TILE_M = 51.2
GSD = 0.025


def test_angle_deg_utm_range_and_direction() -> None:
    assert angle_deg_utm((0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert angle_deg_utm((0.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)
    assert angle_deg_utm((0.0, 0.0), (-1.0, 1.0)) == pytest.approx(135.0)
    assert angle_deg_utm((1.0, 1.0), (0.0, 0.0)) == pytest.approx(45.0)
    assert angle_deg_utm((0.0, 0.0), (-1.0, 0.0)) == pytest.approx(0.0)
    for a in np.linspace(-720, 720, 97):
        v = angle_deg_utm((0.0, 0.0), (math.cos(math.radians(a)), math.sin(math.radians(a))))
        assert 0.0 <= v < 180.0


def test_angle_deg_utm_degenerate() -> None:
    with pytest.raises(SchemaError):
        angle_deg_utm((1.0, 2.0), (1.0, 2.0))


def test_angle_px_to_utm() -> None:
    assert angle_px_to_utm(0.0) == pytest.approx(0.0)
    assert angle_px_to_utm(30.0) == pytest.approx(150.0)
    assert angle_px_to_utm(-30.0) == pytest.approx(30.0)
    assert angle_px_to_utm(180.0) == pytest.approx(0.0)


def test_canonical_normal_rules() -> None:
    nx, ny = canonical_normal(90.0)
    assert (nx, ny) == pytest.approx((1.0, 0.0), abs=1e-12)
    nx, ny = canonical_normal(133.3)
    assert ny > 0
    assert math.hypot(nx, ny) == pytest.approx(1.0)
    assert canonical_normal(0.0) == pytest.approx((0.0, 1.0))
    assert canonical_normal(180.0) == pytest.approx((0.0, 1.0), abs=1e-12)
    # just above the N-S tolerance: n_y must still be positive
    tilt = math.degrees(math.asin(2 * NORMAL_NS_TOL))
    assert canonical_normal(90.0 + tilt)[1] > 0
    assert canonical_normal(90.0 - tilt)[1] > 0
    # inside the tolerance: n_x positive
    assert canonical_normal(90.3)[0] > 0
    assert canonical_normal(89.7)[0] > 0


def test_order_by_normal_simple() -> None:
    centroids = np.array([[0.0, 0.0], [0.0, 10.0], [0.0, 5.0]])
    assert order_by_normal(0.0, centroids).tolist() == [1, 2, 0]
    # N-S rows: easternmost first
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.0]])
    assert order_by_normal(90.0, centroids).tolist() == [1, 2, 0]
    assert order_by_normal(0.0, np.zeros((0, 2))).tolist() == []


def test_order_by_normal_stable_ties() -> None:
    centroids = np.array([[0.0, 1.0], [5.0, 1.0], [0.0, 2.0]])
    assert order_by_normal(0.0, centroids).tolist() == [2, 0, 1]


def test_order_by_normal_rejects_bad_shape() -> None:
    with pytest.raises(SchemaError):
        order_by_normal(0.0, np.zeros((3, 3)))


def test_block_sort_key_and_order() -> None:
    assert block_sort_key(10.4, 20.6) == (-21, 10)
    pts = np.array([[0.0, 0.0], [5.0, 10.0], [1.0, 10.0]])
    assert order_blocks(pts).tolist() == [2, 1, 0]


def test_mean_axial_angle() -> None:
    assert mean_axial_angle_deg([179.0, 1.0]) == pytest.approx(0.0, abs=1e-9)
    assert mean_axial_angle_deg([112.0, 114.0]) == pytest.approx(113.0)
    with pytest.raises(SchemaError):
        mean_axial_angle_deg([])


def _example_rows(xml_bytes: bytes) -> dict[str, list[tuple[str, np.ndarray]]]:
    tree = etree.fromstring(xml_bytes)
    out: dict[str, list[tuple[str, np.ndarray]]] = {}
    for image in tree.iter("image"):
        tile_id = image.get("name")[:-4]
        rows = []
        for pl in image.iter("polyline"):
            attrs = {a.get("name"): (a.text or "") for a in pl.iter("attribute")}
            uv = np.array([[float(c) for c in p.split(",")] for p in pl.get("points").split(";")])
            rows.append((attrs["row_id"], uv))
        out[tile_id] = rows
    return out


def _px_to_utm(tile_id: str, uv: np.ndarray) -> np.ndarray:
    row, col = parse_tile_id(tile_id)
    x0 = GRID_X0 + TILE_M * col
    y0 = GRID_Y0 - TILE_M * row
    return np.column_stack([x0 + GSD * uv[:, 0], y0 - GSD * uv[:, 1]])


def _centroid(xy: np.ndarray) -> np.ndarray:
    seg = np.diff(xy, axis=0)
    lengths = np.hypot(seg[:, 0], seg[:, 1])
    mids = (xy[:-1] + xy[1:]) / 2.0
    return (mids * lengths[:, None]).sum(axis=0) / lengths.sum()


@pytest.mark.examples
def test_order_by_normal_reproduces_example_row_ids(examples_xml: bytes) -> None:
    per_tile = _example_rows(examples_xml)
    assert set(per_tile) == {"siret3_r021_c012", "siret3_r006_c004"}
    for tile_id, rows in per_tile.items():
        axes = [_px_to_utm(tile_id, uv) for _, uv in rows]
        angles = [angle_deg_utm(tuple(a[0]), tuple(a[-1])) for a in axes]
        theta = mean_axial_angle_deg(angles)
        centroids = np.array([_centroid(a) for a in axes])
        order = order_by_normal(theta, centroids)
        indices = [row_index_of(rows[i][0]) for i in order]
        assert indices == list(range(1, len(rows) + 1)), tile_id
    assert len(per_tile["siret3_r021_c012"]) == 25
    assert len(per_tile["siret3_r006_c004"]) == 26
