"""Official metrics on the organizers' 2 example tiles: reference vs reference, shifted rows, and the
prototype-equivalent canopy prediction (validates the raw contour convention, S1)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from tests.eval.conftest import EXAMPLE_TILES
from tests.helpers import proto_metrics as pm
from tests.helpers.examples import ExampleImage, canopy_polygons_utm
from vineyard.annset.model import AnnSet
from vineyard.config import EvalConfig
from vineyard.eval.metrics import EvalParams, canopy_metrics
from vineyard.eval.report import evaluate_annsets
from vineyard.geo.raster import mask_to_polygons, read_tile
from vineyard.geo.tiling import tile_ref
from vineyard.perception.vegmask import veg_mask

pytestmark = pytest.mark.examples

SCORE_KEYS = ("canopy.score", "canopy.iou", "canopy.f1", "rows.f1", "interrow.iou", "interrow.f1",
              "attributes.row_structure.score", "attributes.interrow_cover.score", "attributes.score",
              "grouping.f1", "counts.score", "counts.blocks.score", "counts.rows.score",
              "counts.canopy_area_m2.score", "counts.interrow_area_m2.score", "counts.row_length_m.score", "waste.f1")
PROTO_BLUR_PX, PROTO_THRESHOLD, PROTO_EPS_PX = 2.5, 4.0, 2.5
PROTO_TARGET, PROTO_TOL = 0.85, 0.03


def test_reference_vs_reference_is_exactly_one(reference_annset: AnnSet, eval_cfg: EvalConfig) -> None:
    rep = evaluate_annsets(reference_annset, reference_annset, EXAMPLE_TILES, eval_cfg)
    for scope in (*rep.per_tile.values(), rep.mean, rep.pooled):
        for key in SCORE_KEYS:
            assert scope[key] == 1.0, key
        assert scope["counts.canopy_area_m2.rel_err"] == 0.0
    assert (rep.per_tile["siret3_r006_c004"]["rows.tp"], rep.per_tile["siret3_r021_c012"]["rows.tp"]) == (26, 25)
    assert rep.pooled["rows.n_ref"] == 51 and rep.pooled["interrow.n_ref"] == 49
    assert rep.pooled["canopy.n_ref"] == 650 and rep.pooled["counts.blocks.ref"] == 2
    assert rep.gates_passed


def test_rows_shifted_half_metre_score_zero(reference_annset: AnnSet, eval_cfg: EvalConfig) -> None:
    rows = reference_annset.row_pieces
    shifted = rows.assign(geometry=[g.offset_curve(0.5) for g in rows.geometry])
    rep = evaluate_annsets(reference_annset.with_layer("row_pieces", shifted), reference_annset, EXAMPLE_TILES,
                           eval_cfg)
    assert all(rep.per_tile[t]["rows.f1"] == 0.0 for t in EXAMPLE_TILES)
    assert rep.per_tile["siret3_r006_c004"]["attributes.row_structure.accuracy"] == 0.0
    assert not {g.name: g for g in rep.gates}["row_f1"].passed


def _proto_canopies(img: ExampleImage, tif: Path, *, offset: float, outset: float) -> list[Polygon]:
    """Reference axes -> Lab a* mask ∩ ±0.30 m corridor -> CC8 >= 0.19 m² -> contours + approxPolyDP 2.5."""
    rgb = read_tile(tif)
    veg = veg_mask(rgb, np.ones(rgb.shape[:2], bool), blur_sigma_px=PROTO_BLUR_PX, threshold=PROTO_THRESHOLD)
    labels = pm.canopy_labels(veg, img.points("row"))
    return mask_to_polygons(labels > 0, tile_ref(img.tile_id), approx_eps_px=PROTO_EPS_PX, min_area_px=0.0,
                            pixel_offset=offset, outset_px=outset)


@pytest.fixture(scope="module")
def proto_scores(example_images: dict[str, ExampleImage], example_tif: Callable[[str], Path],
                 eval_cfg: EvalConfig) -> dict[str, dict[str, float]]:
    p = EvalParams.from_config(eval_cfg)
    kw = {"match_iou": p.canopy_match_iou, "w_iou": p.canopy_w_iou, "w_f1": p.canopy_w_f1}
    out: dict[str, dict[str, float]] = {}
    for variant, (offset, outset) in {"raw": (0.0, 0.0), "outset": (0.5, 0.5)}.items():
        out[variant] = {t: canopy_metrics(_proto_canopies(img, example_tif(t), offset=offset, outset=outset),
                                          canopy_polygons_utm(img), **kw).score
                        for t, img in example_images.items()}
    return out


def test_prototype_equivalent_raw_convention_scores_about_085(proto_scores: dict[str, dict[str, float]]) -> None:
    raw = proto_scores["raw"]
    mean = sum(raw.values()) / len(raw)
    assert mean == pytest.approx(PROTO_TARGET, abs=PROTO_TOL), raw
    assert raw["siret3_r006_c004"] == pytest.approx(0.869, abs=0.02)
    assert raw["siret3_r021_c012"] == pytest.approx(0.800, abs=0.02)


def test_raw_convention_beats_contract_outset(proto_scores: dict[str, dict[str, float]]) -> None:
    raw, outset = proto_scores["raw"], proto_scores["outset"]
    assert sum(raw.values()) > sum(outset.values())


def test_prototype_canopies_in_full_report(reference_annset: AnnSet, example_images: dict[str, ExampleImage],
                                           example_tif: Callable[[str], Path], eval_cfg: EvalConfig) -> None:
    ref_can = reference_annset.canopies
    records = []
    for t, img in sorted(example_images.items()):
        vid = ref_can.loc[ref_can["tile_id"] == t, "vineyard_id"].iloc[0]
        polys = _proto_canopies(img, example_tif(t), offset=0.0, outset=0.0)
        records += [{"canopy_id": f"{t}:C{k:04d}", "tile_id": t, "vineyard_id": vid, "geometry": g}
                    for k, g in enumerate(polys, 1)]
    pred_can = gpd.GeoDataFrame(records, geometry="geometry", crs=ref_can.crs)
    rep = evaluate_annsets(reference_annset.with_layer("canopies", pred_can), reference_annset, EXAMPLE_TILES,
                           eval_cfg)
    assert rep.mean["canopy.score"] == pytest.approx(PROTO_TARGET, abs=PROTO_TOL)
    assert rep.mean["rows.f1"] == 1.0 and rep.mean["interrow.iou"] == 1.0 and rep.pooled["grouping.f1"] == 1.0
    assert {g.name: g.passed for g in rep.gates} == {"canopy_score": True, "row_f1": True, "interrow_iou": True,
                                                     "attributes": True}
