"""geo.polygonize re-exports the P0 vectorizers from geo.raster (P1 may move them here)."""

from __future__ import annotations

from vineyard.geo import polygonize, raster


def test_reexports_are_the_raster_functions() -> None:
    assert polygonize.mask_to_polygons is raster.mask_to_polygons
    assert polygonize.valid_polygon is raster.valid_polygon
    assert set(polygonize.__all__) == {"mask_to_polygons", "valid_polygon"}
