"""Tests for the QA panel helpers (pure drawing, no tile I/O)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from fte.viz.panels import GREEN, _draw_polys, _new_geoms, _to_panel_px

X0, Y0 = 1000.0, 2000.0


def test_to_panel_px_scales_tile_to_1024() -> None:
    uv = _to_panel_px(np.array([[X0, Y0], [X0 + 51.2, Y0 - 51.2]]), X0, Y0)
    assert np.allclose(uv, [[0, 0], [1024, 1024]])


def test_new_geoms_and_draw_do_not_mutate() -> None:
    a, b, c = box(X0 + 1, Y0 - 2, X0 + 2, Y0 - 1), box(X0 + 3, Y0 - 2, X0 + 4, Y0 - 1), box(X0 + 5, Y0 - 2, X0 + 6, Y0 - 1)
    base = gpd.GeoDataFrame(geometry=[a, b])
    run = gpd.GeoDataFrame(geometry=[a, c])
    assert [g.equals(c) for g in _new_geoms(base, run)] == [True]
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    out = _draw_polys(img, [a], X0, Y0, GREEN, 1)
    assert img.sum() == 0 and out.sum() > 0
