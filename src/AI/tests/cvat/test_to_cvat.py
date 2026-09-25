"""cvat.to_cvat: AnnSet (UTM) -> per-tile CvatImages (px), with geometric cleaning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry

from vineyard.annset.model import AnnSet, empty_annset, make_meta
from vineyard.config import load_config
from vineyard.config.sections_io import CvatExportConfig
from vineyard.contracts.enums import Source
from vineyard.contracts.schemas import coerce_layer, get_schema, validate_layer
from vineyard.cvat.model import CvatDocument, CvatShape
from vineyard.cvat.to_cvat import TileWriteStats, annset_to_images, px_geom_to_utm
from vineyard.cvat.validator_doc import validate_document
from vineyard.geo.tiling import CRS_EPSG, TileRef, px_to_utm, tile_ref

T1 = "siret3_r021_c012"
T2 = "siret3_r006_c004"
PX = 2048
RUN = "20260926T0100-model-abcdef"

_DEFAULTS: Mapping[str, Any] = {"str": "", "bool": False}


def _default(kind: str) -> Any:
    return _DEFAULTS.get(kind, 0)


def make_layer(name: str, rows: Sequence[Mapping[str, Any]]) -> gpd.GeoDataFrame:
    """Schema-conformant layer; unspecified non-nullable columns get neutral defaults."""
    schema = get_schema(name)
    base = {c.name: _default(c.kind) for c in schema.all_columns if not c.nullable}
    base |= {"source": "model", "run_id": RUN, "model_version": "test", "confidence": 1.0}
    records = [base | {k: v for k, v in r.items() if k != "geometry"} for r in rows]
    gdf = gpd.GeoDataFrame(records, geometry=[r["geometry"] for r in rows], crs=CRS_EPSG)
    out = coerce_layer(gdf, name)
    validate_layer(out, name)
    return out


def utm(tile_id: str, geom_px: BaseGeometry) -> BaseGeometry:
    return px_geom_to_utm(geom_px, tile_ref(tile_id))


def px_square(u0: float, v0: float, size: float) -> Polygon:
    return box(u0, v0, u0 + size, v0 + size)


def canopy_row(tile_id: str, k: int, geom_px: BaseGeometry, vid: str = "V01") -> dict[str, Any]:
    return {"canopy_id": f"{tile_id}:C{k:04d}", "tile_id": tile_id, "vineyard_id": vid,
            "row_id": "V01-R001", "geometry": utm(tile_id, geom_px)}


def row_row(tile_id: str, r: int, geom_px: LineString, structure: str = "regular") -> dict[str, Any]:
    rid = f"V01-R{r:03d}"
    return {"piece_id": f"{rid}@{tile_id}", "row_id": rid, "vineyard_id": "V01", "tile_id": tile_id,
            "row_structure": structure, "geometry": utm(tile_id, geom_px)}


def interrow_row(tile_id: str, k: int, geom_px: Polygon, cover: str = "bare_soil") -> dict[str, Any]:
    iid = f"V01-I{k:03d}"
    return {"piece_id": f"{iid}@{tile_id}", "interrow_id": iid, "vineyard_id": "V01", "tile_id": tile_id,
            "interrow_cover": cover, "geometry": utm(tile_id, geom_px)}


def waste_row(tile_id: str, k: int, geom_px: Polygon, vid: str = "V01", dist: float = 0.0) -> dict[str, Any]:
    return {"waste_id": f"W{k:04d}", "tile_id": tile_id, "vineyard_id": vid, "dist_block_m": dist,
            "category": "rubbish", "detector": "rule", "geometry": utm(tile_id, geom_px)}


def make_annset(tile_ids: Sequence[str], **layers: Sequence[Mapping[str, Any]]) -> AnnSet:
    annset = empty_annset(make_meta(Source.MODEL, RUN, "test", tile_ids))
    for name, rows in layers.items():
        if rows:
            annset = annset.with_layer(name, make_layer(name, rows))
    return annset


@pytest.fixture(scope="module")
def cfg() -> CvatExportConfig:
    return load_config().export.cvat


def _convert(annset: AnnSet, cfg: CvatExportConfig, tiles: Sequence[str] = (T1,)):
    return annset_to_images(annset, [tile_ref(t) for t in tiles], cfg)


def _shoelace(points: Sequence[tuple[float, float]]) -> float:
    p = np.asarray(points)
    return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(np.roll(p[:, 0], -1), p[:, 1]))


def test_empty_annset_gives_empty_images_sorted(cfg: CvatExportConfig) -> None:
    images, stats = _convert(make_annset([T1, T2]), cfg, [T1, T2])
    assert [img.name for img in images] == [f"{T2}.tif", f"{T1}.tif"]
    assert [img.id for img in images] == [0, 1]
    assert all(img.shapes == () and img.width == PX and img.height == PX for img in images)
    assert all(isinstance(s, TileWriteStats) and sum(s.n_out.values()) == 0 for s in stats)


def test_shape_order_attributes_and_source(cfg: CvatExportConfig) -> None:
    annset = make_annset(
        [T1],
        canopies=[canopy_row(T1, 2, px_square(300, 300, 20)), canopy_row(T1, 1, px_square(10, 10, 20))],
        row_pieces=[row_row(T1, 1, LineString([(0, 100), (2048, 100)]))],
        interrow_pieces=[interrow_row(T1, 1, px_square(500, 500, 100), cover="mixed")],
        waste=[waste_row(T1, 1, box(50.04, 60.02, 70.0, 80.0))],
    )
    (image,), (stats,) = _convert(annset, cfg)
    assert [s.label for s in image.shapes] == ["row", "interrow_area", "vineyard", "vineyard", "waste"]
    assert [s.ref_id for s in image.shapes[2:4]] == [f"{T1}:C0001", f"{T1}:C0002"]
    row = image.shapes[0]
    assert row.tag == "polyline" and row.attributes == (
        ("vineyard_id", "V01"), ("row_id", "V01-R001"), ("row_structure", "regular"))
    assert row.points == ((0.0, 100.0), (2048.0, 100.0))
    assert image.shapes[1].attributes == (("vineyard_id", "V01"), ("interrow_cover", "mixed"))
    assert image.shapes[4].tag == "box" and image.shapes[4].box_xyxy == (50.0, 60.0, 70.0, 80.0)
    assert all(s.source == cfg.shape_source for s in image.shapes)
    assert stats.n_out == {"row": 1, "interrow_area": 1, "vineyard": 2, "waste": 1}
    assert validate_document_ok(image, cfg)


def validate_document_ok(image, cfg: CvatExportConfig) -> bool:
    rep = validate_document(CvatDocument((image,)), cfg=cfg, tile_px=PX)
    assert rep.ok, rep.summary_lines()
    return True


def test_polygons_are_ccw_in_utm_without_closing_vertex(cfg: CvatExportConfig) -> None:
    annset = make_annset([T1], canopies=[canopy_row(T1, 1, px_square(10.26, 10.0, 20))])
    (image,), _ = _convert(annset, cfg)
    pts = image.shapes[0].points
    assert pts[0] != pts[-1]
    assert _shoelace(pts) < 0
    assert all(round(c, 1) == c for p in pts for c in p)
    assert {p[0] for p in pts} == {10.3, 30.3}


def test_canopy_is_not_simplified_but_interrow_is(cfg: CvatExportConfig) -> None:
    wiggle = Polygon([(100, 100), (100, 150), (125, 150.3), (150, 150), (150, 100)])
    annset = make_annset([T1], canopies=[canopy_row(T1, 1, wiggle)],
                         interrow_pieces=[interrow_row(T1, 1, Polygon(
                             [(300, 100), (300, 150), (325, 150.3), (350, 150), (350, 100)]))])
    (image,), _ = _convert(annset, cfg)
    by_label = {s.label: s for s in image.shapes}
    assert len(by_label["vineyard"].points) == 5
    assert len(by_label["interrow_area"].points) == 4


def test_interrow_simplify_never_cuts_into_canopies(cfg: CvatExportConfig) -> None:
    # interrow hugging a canopy with a 0.4 px bevel: simplifying it overlaps the canopy by ~0.14 m2
    canopy = Polygon([(100, 100), (1200, 100), (1200, 1199.6), (1199.6, 1200), (100, 1200)])
    interrow = box(100, 100, 1300, 1300).difference(canopy)
    annset = make_annset([T1], canopies=[canopy_row(T1, 1, canopy)], interrow_pieces=[interrow_row(T1, 1, interrow)])
    (image,), (stats,) = _convert(annset, cfg)
    assert [i.code for i in stats.issues] == ["interrow_unsimplified"]
    ir = [s for s in image.shapes if s.label == "interrow_area"][0]
    assert len(ir.points) == len(interrow.exterior.coords) - 1
    assert validate_document(CvatDocument((image,)), cfg=cfg, tile_px=PX).ok


def test_clip_to_tile_and_drop_outside(cfg: CvatExportConfig) -> None:
    annset = make_annset(
        [T1],
        canopies=[canopy_row(T1, 1, px_square(2040, 100, 20)), canopy_row(T1, 2, px_square(2100, 100, 20))],
        row_pieces=[row_row(T1, 1, LineString([(-50, 10), (2100, 10)]))],
    )
    (image,), (stats,) = _convert(annset, cfg)
    canopy = [s for s in image.shapes if s.label == "vineyard"]
    assert len(canopy) == 1
    assert max(u for u, _ in canopy[0].points) == 2048.0
    row = [s for s in image.shapes if s.label == "row"][0]
    assert row.points[0][0] == 0.0 and row.points[-1][0] == 2048.0
    assert stats.n_dropped["vineyard"] == 1


def test_small_parts_dropped_and_multipart_split(cfg: CvatExportConfig) -> None:
    small = px_square(10, 10, 3)  # 9 px2 < 16
    bowtie = Polygon([(100, 100), (140, 140), (140, 100), (100, 140)])
    annset = make_annset([T1], canopies=[canopy_row(T1, 1, small)])
    (image,), (stats,) = _convert(annset, cfg)
    assert image.shapes == () and stats.n_dropped["vineyard"] == 1
    fixed = px_geom_to_utm(bowtie, tile_ref(T1))
    rows = [{**canopy_row(T1, 1, px_square(0, 0, 1)), "geometry": fixed}]
    gdf = make_layer_unchecked("canopies", rows)
    (image,), (stats,) = _convert(make_annset([T1]).with_layer("canopies", gdf), cfg)
    assert len(image.shapes) == 2 and stats.n_split == 1
    assert {s.ref_id for s in image.shapes} == {f"{T1}:C0001", f"{T1}:C0001#2"}


def make_layer_unchecked(name: str, rows: Sequence[Mapping[str, Any]]) -> gpd.GeoDataFrame:
    schema = get_schema(name)
    base = {c.name: _default(c.kind) for c in schema.all_columns if not c.nullable}
    base |= {"source": "model", "run_id": RUN, "model_version": "test", "confidence": 1.0}
    records = [base | {k: v for k, v in r.items() if k != "geometry"} for r in rows]
    return coerce_layer(gpd.GeoDataFrame(records, geometry=[r["geometry"] for r in rows], crs=CRS_EPSG), name)


def test_interrow_hole_is_notched(cfg: CvatExportConfig) -> None:
    holed = Polygon(px_square(100, 100, 200).exterior.coords, [px_square(180, 180, 40).exterior.coords])
    annset = make_annset([T1], interrow_pieces=[interrow_row(T1, 1, holed)])
    (image,), _ = _convert(annset, cfg)
    assert len(image.shapes) == 1
    poly = Polygon(image.shapes[0].points)
    assert poly.is_valid
    assert abs(poly.area - (200 * 200 - 40 * 40)) < 200 * cfg.notch_width_px + 1


def test_invalid_after_rounding_is_dropped_with_warning(cfg: CvatExportConfig) -> None:
    # a sliver thinner than the 0.1 px rounding grid collapses to a line after rounding
    sliver = Polygon([(100.0, 100.01), (400.0, 100.04), (400.0, 100.02)])
    rows = [{**canopy_row(T1, 1, px_square(0, 0, 1)), "geometry": px_geom_to_utm(sliver, tile_ref(T1))}]
    annset = make_annset([T1]).with_layer("canopies", make_layer_unchecked("canopies", rows))
    small_cfg = cfg.model_copy(update={"min_polygon_px2": 0.0})
    (image,), (stats,) = _convert(annset, small_cfg)
    assert image.shapes == ()
    assert [i.code for i in stats.issues] == ["invalid_after_rounding"]


def test_waste_box_clipped_and_degenerate_dropped(cfg: CvatExportConfig) -> None:
    annset = make_annset([T1], waste=[waste_row(T1, 1, box(2040, 10, 2060, 20), vid=""),
                                      waste_row(T1, 2, box(2050, 10, 2060, 20))])
    (image,), (stats,) = _convert(annset, cfg)
    assert [s.box_xyxy for s in image.shapes] == [(2040.0, 10.0, 2048.0, 20.0)]
    assert image.shapes[0].attributes == (("vineyard_id", ""),)
    assert stats.n_dropped["waste"] == 1


def test_objects_on_other_tiles_are_ignored(cfg: CvatExportConfig) -> None:
    annset = make_annset([T1, T2], canopies=[canopy_row(T2, 1, px_square(10, 10, 20))])
    (image,), (stats,) = _convert(annset, cfg, [T1])
    assert image.shapes == () and stats.tile_id == T1


def test_px_geom_to_utm_matches_tiling() -> None:
    t: TileRef = tile_ref(T1)
    geom = px_geom_to_utm(LineString([(0, 0), (2048, 2048)]), t)
    assert np.allclose(np.asarray(geom.coords), px_to_utm(t, np.array([[0, 0], [2048, 2048]])))


def test_conversion_is_deterministic(cfg: CvatExportConfig) -> None:
    annset = make_annset([T1], canopies=[canopy_row(T1, k, px_square(10 + 30 * k, 10, 20)) for k in range(1, 6)])
    a, _ = _convert(annset, cfg)
    b, _ = _convert(annset, cfg)
    assert a == b
    assert isinstance(a[0].shapes[0], CvatShape)
