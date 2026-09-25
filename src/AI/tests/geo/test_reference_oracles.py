"""Oracles measured on the organizers' examples: the test helpers, the pixel conventions and the
prototype canopy score must reproduce them (00_PLAN facts table, 01 §0, 05 F1/F2)."""

from __future__ import annotations

import numpy as np
import pytest
from shapely import union_all

from tests.helpers import proto_metrics as pm
from tests.helpers.examples import (
    ExampleImage,
    canopy_polygons_utm,
    interrow_polygons_utm,
    load_examples,
    parse_points,
    row_lines_utm,
)
from vineyard.geo.raster import mask_to_polygons, rasterize_utm, read_tile
from vineyard.geo.tiling import GSD_M, tile_ref
from vineyard.perception.vegmask import veg_mask

pytestmark = pytest.mark.examples

# tile -> (n canopies, Σ canopy m², n interrows, Σ interrow m², n rows, Σ row m, canopy pixel-set m²)
ORACLES = {
    "siret3_r021_c012": (399, 237.119, 24, 2068.031, 25, 910.104, 256.95),
    "siret3_r006_c004": (251, 299.056, 25, 1996.361, 26, 1031.451, 319.36),
}
MAX_LOST_PX = 100  # 1-px spurs have no area in the raw contour convention
PROTO_SCORE = {"siret3_r021_c012": 0.821, "siret3_r006_c004": 0.883}


@pytest.fixture(scope="module")
def examples(examples_xml: bytes) -> dict[str, ExampleImage]:
    return load_examples(examples_xml)


def test_parse_points() -> None:
    np.testing.assert_array_equal(parse_points("1.5,2;3,4.25"), [[1.5, 2.0], [3.0, 4.25]])


def test_examples_document_order_and_shapes(examples: dict[str, ExampleImage]) -> None:
    assert list(examples) == ["siret3_r021_c012", "siret3_r006_c004"]  # XML order is not sorted
    img = examples["siret3_r021_c012"]
    assert (img.image_id, img.width, img.height) == (0, 2048, 2048)
    row = img.by_label("row")[0]
    assert row.tag == "polyline" and set(row.attributes) == {"vineyard_id", "row_id", "row_structure"}
    assert not img.by_label("waste")


@pytest.mark.parametrize("tile_id", sorted(ORACLES))
def test_reference_counts_and_sums(examples: dict[str, ExampleImage], tile_id: str) -> None:
    n_can, a_can, n_ir, a_ir, n_rows, l_rows, _ = ORACLES[tile_id]
    img = examples[tile_id]
    canopies, interrows, rows = canopy_polygons_utm(img), interrow_polygons_utm(img), row_lines_utm(img)
    assert (len(canopies), len(interrows), len(rows)) == (n_can, n_ir, n_rows)
    assert sum(p.area for p in canopies) == pytest.approx(a_can, abs=0.01)
    assert sum(p.area for p in interrows) == pytest.approx(a_ir, abs=0.01)
    assert sum(r.length for r in rows) == pytest.approx(l_rows, abs=0.01)


@pytest.mark.parametrize("tile_id", sorted(ORACLES))
def test_reference_canopies_use_raw_contour_vertices(examples: dict[str, ExampleImage], tile_id: str) -> None:
    pts = np.concatenate(examples[tile_id].points("vineyard"))
    np.testing.assert_array_equal(pts, np.round(pts))
    assert pts.min() >= 0.0 and pts.max() <= 2047.0


@pytest.mark.parametrize("tile_id", sorted(ORACLES))
def test_index_convention_rebuilds_pixel_set_and_raw_vectorization_inverts_it(
    examples: dict[str, ExampleImage], tile_id: str
) -> None:
    t = tile_ref(tile_id)
    ref = canopy_polygons_utm(examples[tile_id])
    pixels = rasterize_utm(ref, t, convention="index")
    assert pixels.sum() * GSD_M**2 == pytest.approx(ORACLES[tile_id][6], abs=0.01)
    polys = mask_to_polygons(pixels, t, approx_eps_px=0.0, min_area_px=0.0)
    assert sum(p.area for p in polys) == pytest.approx(ORACLES[tile_id][1], rel=0.005)
    a, b = union_all(ref), union_all(polys)
    assert a.intersection(b).area / a.union(b).area >= 0.97
    back = rasterize_utm(polys, t, convention="index")
    assert int((back > pixels).sum()) == 0
    assert int((pixels > back).sum()) < MAX_LOST_PX


@pytest.mark.parametrize("tile_id", sorted(ORACLES))
def test_prototype_canopy_score_with_reference_axes(
    examples: dict[str, ExampleImage], example_tif, tile_id: str
) -> None:
    img = examples[tile_id]
    veg = veg_mask(read_tile(example_tif(tile_id)), np.ones((2048, 2048), bool), blur_sigma_px=2.5, threshold=4.0)
    canopies = img.points("vineyard")
    ref = pm.reference_labels(canopies)
    assert pm.score(ref, ref, len(canopies))[:2] == (1.0, 1.0)
    iou, f1, _ = pm.score(pm.canopy_labels(veg, img.points("row")), ref, len(canopies))
    assert pm.canopy_score(iou, f1) == pytest.approx(PROTO_SCORE[tile_id], abs=0.01)


@pytest.mark.parametrize("tile_id", sorted(ORACLES))
def test_prototype_row_f1_reference_vs_itself(examples: dict[str, ExampleImage], tile_id: str) -> None:
    rows = examples[tile_id].points("row")
    assert pm.row_f1(rows, rows) == (1.0, len(rows))
    assert pm.row_f1([_shift_perp(r, 0.3) for r in rows], rows) == (1.0, len(rows))
    assert pm.row_f1([_shift_perp(r, 0.5) for r in rows], rows)[1] == 0


def _shift_perp(row: np.ndarray, metres: float) -> np.ndarray:
    d = row[-1] - row[0]
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return row + n * metres / GSD_M
