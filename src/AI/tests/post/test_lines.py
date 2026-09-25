"""annset.lines: PCA direction, projections, offsets, axial median, midline and extension."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from vineyard.annset import lines as L
from vineyard.errors import SchemaError


def test_pca_direction_is_canonical_axial() -> None:
    pts = np.array([[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]])
    d = L.pca_direction(pts)
    assert d == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])
    # Same axis given in the opposite order: same canonical direction.
    assert L.pca_direction(pts[::-1]) == pytest.approx(d)
    # A 135° axis keeps angle 135 (not -45).
    assert L.axial_angle_of(L.pca_direction(np.array([[0.0, 0.0], [-3.0, 3.0]]))) == pytest.approx(135.0)


def test_pca_direction_rejects_degenerate_input() -> None:
    with pytest.raises(SchemaError):
        L.pca_direction(np.array([[1.0, 1.0]]))
    with pytest.raises(SchemaError):
        L.pca_direction(np.array([[1.0, 1.0], [1.0, 1.0]]))


def test_direction_angle_round_trip() -> None:
    for angle in (0.0, 30.0, 90.0, 127.5, 179.0):
        assert L.axial_angle_of(L.direction_of_angle(angle)) == pytest.approx(angle)
    assert L.axial_angle_of(np.array([-1.0, 0.0])) == pytest.approx(0.0)


def test_line_angle_uses_all_vertices() -> None:
    assert L.line_angle_deg(LineString([(0, 0), (0, 5)])) == pytest.approx(90.0)
    assert L.line_angle_deg(LineString([(0, 0), (4, 4), (8, 8)])) == pytest.approx(45.0)


def test_project_and_oriented() -> None:
    d = L.direction_of_angle(0.0)
    assert L.project(np.array([[2.0, 7.0], [-1.0, 0.0]]), d) == pytest.approx([2.0, -1.0])
    line = LineString([(5, 0), (0, 0)])
    assert list(L.oriented(line, d).coords) == [(0.0, 0.0), (5.0, 0.0)]
    same = LineString([(0, 0), (5, 0)])
    assert L.oriented(same, d).equals(same)


def test_offsets() -> None:
    d = L.direction_of_angle(0.0)
    assert L.lateral_offset(np.array([3.0, -0.4]), np.array([0.0, 0.0]), d) == pytest.approx(0.4)
    n = np.array([0.0, 1.0])
    assert L.signed_offset(np.array([3.0, -0.4]), np.array([0.0, 0.0]), n) == pytest.approx(-0.4)


def test_endpoint_direction() -> None:
    assert L.endpoint_direction(LineString([(0, 0), (1, 5), (0, 10)])) == pytest.approx([0.0, 1.0])
    with pytest.raises(SchemaError):
        L.endpoint_direction(LineString([(0, 0), (1, 1), (0, 0)]))


def test_axial_median_handles_wraparound() -> None:
    assert L.axial_median_deg([179.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert L.axial_median_deg([126.8, 127.5, 133.3]) == pytest.approx(127.5)
    with pytest.raises(SchemaError):
        L.axial_median_deg([])


def test_midline_of_parallel_rows_uses_overlap() -> None:
    a = LineString([(0, 0), (50, 0)])
    b = LineString([(5, 2.6), (45, 2.6)])
    mid = L.midline(a, b)
    assert mid is not None
    assert mid.length == pytest.approx(40.0)
    assert all(y == pytest.approx(1.3) for _, y in mid.coords)
    assert mid.distance(Point(5, 1.3)) < 1e-9 and mid.distance(Point(45, 1.3)) < 1e-9


def test_midline_of_fan_is_equidistant() -> None:
    a = LineString([(0, 0), (40, 0)])
    b = LineString([(0, 2.5), (40, 2.5 + 40 * math.tan(math.radians(2.0)))])
    mid = L.midline(a, b)
    assert mid is not None
    for s in np.linspace(0, 1, 7):
        p = mid.interpolate(s, normalized=True)
        assert abs(p.distance(a) - p.distance(b)) < 0.01


def test_midline_without_overlap_is_none() -> None:
    assert L.midline(LineString([(0, 0), (10, 0)]), LineString([(20, 2), (30, 2)])) is None


def test_extend_line() -> None:
    line = LineString([(0, 0), (10, 0)])
    ext = L.extend_line(line, before_m=2.0, after_m=3.0)
    assert list(ext.coords) == [(-2.0, 0.0), (0.0, 0.0), (10.0, 0.0), (13.0, 0.0)]
    assert L.extend_line(line).equals(line)
    with pytest.raises(SchemaError):
        L.extend_line(line, before_m=-1.0)
