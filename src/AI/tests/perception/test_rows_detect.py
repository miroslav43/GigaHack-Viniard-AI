"""perception.rows_detect: per-tile row candidates (synthetic tiles + the 2 reference example tiles)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, box

from tests.helpers.examples import load_examples, row_lines_utm
from tests.helpers.synth import arc_mask, striped_mask
from vineyard.config import AppConfig, load_config
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.contracts.schemas import validate_layer
from vineyard.eval.metrics import row_axis_f1, row_matches
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_box, tile_ref, utm_to_px
from vineyard.perception.row_features import REASON_LOW_VINE_SCORE
from vineyard.perception.rows_detect import (
    EXTRA_COLUMNS,
    LAYER_NAME,
    METHOD_TEXTURE,
    METHOD_VEG,
    METHOD_VEG_NN,
    STATUS_LOW_SNR,
    STATUS_NO_PERIODICITY,
    STATUS_OK,
    DetectParams,
    TileDetection,
    detect_tile_rows,
    detection_to_candidates,
)
from vineyard.pipeline.runner import TileTask
from vineyard.pipeline.stages.tile_prep import prep_tile, tile_prep_outputs, tile_valid_wkb_path
from vineyard.pipeline.tile_cache import load_valid_mask, load_veg_mask

TILE_ID = "siret3_r021_c012"
TILE = tile_ref(TILE_ID)
SHAPE = (TILE_PX, TILE_PX)
FRAME = box(0.0, 0.0, float(TILE_PX), float(TILE_PX))
EXAMPLE_TILES = ("siret3_r021_c012", "siret3_r006_c004")
# 02 §5: UTM row angle and spacing of the two example tiles.
EXAMPLE_ANGLE_UTM = {"siret3_r021_c012": 127.5, "siret3_r006_c004": 112.8}
EXAMPLE_SPACING_M = {"siret3_r021_c012": (2.78, 0.25), "siret3_r006_c004": (2.53, 0.10)}
MAX_SECONDS_PER_TILE = 4.0
MIN_F1 = 0.95


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture(scope="module")
def params(cfg: AppConfig) -> DetectParams:
    return DetectParams.from_config(cfg)


@pytest.fixture(scope="module")
def stripes(params: DetectParams) -> TileDetection:
    veg = striped_mask(SHAPE, angle_deg=30.0, spacing_px=100.0, width_px=16.0)
    return detect_tile_rows(veg, FRAME, TILE_ID, params)


def _frame(det: TileDetection) -> gpd.GeoDataFrame:
    return detection_to_candidates(det, tile_ref(det.tile_id), run_id="test-run", model_version="pipe@test")


# ------------------------------------------------------------------ synthetic


def test_stripes_every_row_found_and_accepted(stripes: TileDetection) -> None:
    assert stripes.status_hint == STATUS_OK and stripes.method == METHOD_VEG
    accepted = [c for c in stripes.candidates if c.rejected_reason is None]
    assert len(stripes.orientations) == 1
    assert abs(stripes.orientations[0].angle_px_deg - 30.0) <= 0.5
    assert stripes.orientations[0].spacing.spacing_m == pytest.approx(2.5, abs=0.05)
    assert len(accepted) >= 25
    for c in accepted:
        assert c.local_spacing_m == pytest.approx(2.5, abs=0.1)
        assert c.near_edge_start and c.near_edge_end


def test_candidate_ids_and_contract_schema(stripes: TileDetection) -> None:
    frame = _frame(stripes)
    validate_layer(frame, LAYER_NAME)
    assert frame.crs.to_epsg() == CRS_EPSG
    assert list(frame["cand_id"]) == [f"{TILE_ID}:K{k:02d}" for k in range(1, len(frame) + 1)]
    assert all(is_valid_id(IdKind.ROW_CANDIDATE, cid) for cid in frame["cand_id"])
    assert set(dict(EXTRA_COLUMNS)) <= set(frame.columns)
    assert (frame["source"] == "model").all() and (frame["run_id"] == "test-run").all()
    assert frame["angle_deg"].sub(150.0).abs().max() <= 0.5  # UTM angle = (-30) mod 180
    minx, miny, maxx, maxy = tile_box(TILE).buffer(1e-6).bounds
    for geom in frame.geometry:
        gx0, gy0, gx1, gy1 = geom.bounds
        assert minx <= gx0 and gx1 <= maxx and miny <= gy0 and gy1 <= maxy


def test_offsets_are_signed_and_ordered(stripes: TileDetection) -> None:
    offsets = _frame(stripes)["offset_m"].to_numpy()
    assert len(np.unique(np.sign(offsets))) == 2
    steps = np.abs(np.diff(offsets))
    assert np.allclose(steps, 2.5, atol=0.1)


def test_two_orientations_are_both_found(params: DetectParams) -> None:
    left = striped_mask(SHAPE, angle_deg=30.0, spacing_px=100.0, width_px=16.0)
    right = striped_mask(SHAPE, angle_deg=80.0, spacing_px=110.0, width_px=16.0)
    u = np.broadcast_to(np.arange(TILE_PX)[None, :], SHAPE)
    det = detect_tile_rows(np.where(u < 1200, left, right), FRAME, TILE_ID, params)
    angles = sorted(round(o.angle_px_deg) for o in det.orientations)
    assert angles == [30, 80]
    kept = {o: sum(1 for c in det.candidates if c.orientation == o and c.rejected_reason is None)
            for o in range(len(det.orientations))}
    assert all(n >= 5 for n in kept.values())


def _tree_belt_rows() -> np.ndarray:
    """Rows at 53 deg (2.78 m) south of a solid 'tree belt': the histogram variance peaks at 0 deg."""
    v = np.broadcast_to(np.arange(TILE_PX)[:, None], SHAPE)
    return (striped_mask(SHAPE, angle_deg=53.0, spacing_px=111.0, width_px=16.0) & (v >= 1100)) | (v < 1040)


def test_angle_fallback_recovers_rows_next_to_a_tree_belt(params: DetectParams) -> None:
    detect = params.rows.detect
    off = replace(params, rows=params.rows.model_copy(
        update={"detect": detect.model_copy(update={"angle_fallback_enabled": False})}))
    on = replace(params, rows=params.rows.model_copy(
        update={"detect": detect.model_copy(update={"angle_fallback_enabled": True})}))
    mask = _tree_belt_rows()
    missed = detect_tile_rows(mask, FRAME, TILE_ID, off)
    assert missed.status_hint == STATUS_NO_PERIODICITY
    assert not [c for c in missed.candidates if c.rejected_reason is None]
    found = detect_tile_rows(mask, FRAME, TILE_ID, on)
    accepted = [c for c in found.candidates if c.rejected_reason is None]
    assert found.status_hint == STATUS_OK and len(accepted) >= 15
    assert abs(found.orientations[0].angle_px_deg - 53.0) <= 1.0
    assert all(c.local_spacing_m == pytest.approx(2.78, abs=0.1) for c in accepted)


def test_angle_fallback_keeps_tiles_that_already_work(params: DetectParams, stripes: TileDetection) -> None:
    off = replace(params, rows=params.rows.model_copy(
        update={"detect": params.rows.detect.model_copy(update={"angle_fallback_enabled": False})}))
    veg = striped_mask(SHAPE, angle_deg=30.0, spacing_px=100.0, width_px=16.0)
    plain = detect_tile_rows(veg, FRAME, TILE_ID, off)
    assert [c.line_px.wkt for c in plain.candidates] == [c.line_px.wkt for c in stripes.candidates]
    noise = np.random.default_rng(5).random(SHAPE) < 0.05  # nothing accepted, no periodic angle to retry with

    def lines(p: DetectParams) -> list[tuple[str, str | None]]:
        return [(c.line_px.wkt, c.rejected_reason) for c in detect_tile_rows(noise, FRAME, TILE_ID, p).candidates]

    assert lines(params) == lines(off)


def test_empty_mask_has_no_periodicity_and_no_candidates(params: DetectParams) -> None:
    det = detect_tile_rows(np.zeros(SHAPE, dtype=bool), FRAME, TILE_ID, params)
    assert det.status_hint == STATUS_NO_PERIODICITY and det.candidates == ()
    frame = _frame(det)
    assert frame.empty and set(dict(EXTRA_COLUMNS)) <= set(frame.columns)
    assert det.summary()["n_candidates"] == 0


def test_noise_is_soft_rejected_low_snr(params: DetectParams) -> None:
    noise = np.random.default_rng(0).random(SHAPE) < 0.2
    det = detect_tile_rows(noise, FRAME, TILE_ID, params)
    assert det.status_hint in (STATUS_LOW_SNR, STATUS_NO_PERIODICITY)
    assert all(c.rejected_reason is not None for c in det.candidates)


def test_prob_gives_vine_score_and_nn_profile_method(params: DetectParams) -> None:
    veg = striped_mask(SHAPE, angle_deg=0.0, spacing_px=100.0, width_px=16.0)
    low = np.full(SHAPE, 0.1, dtype=np.float32)
    det = detect_tile_rows(veg, FRAME, TILE_ID, params, prob=low)
    assert det.method == METHOD_VEG
    assert {c.rejected_reason for c in det.candidates} == {REASON_LOW_VINE_SCORE}
    nn_params = DetectParams(**{**params.__dict__, "use_prob_in_profile": True})
    high = np.full(SHAPE, 0.9, dtype=np.float32)
    det_nn = detect_tile_rows(veg, FRAME, TILE_ID, nn_params, prob=high)
    assert det_nn.method == METHOD_VEG_NN
    assert all(c.features.vine_score == pytest.approx(0.9, abs=1e-5) for c in det_nn.candidates)


def test_texture_fallback_points(params: DetectParams) -> None:
    veg = np.zeros(SHAPE, dtype=bool)
    points = striped_mask(SHAPE, angle_deg=0.0, spacing_px=100.0, width_px=16.0)
    det = detect_tile_rows(veg, FRAME, TILE_ID, params, fallback_points=points)
    assert det.method == METHOD_TEXTURE and len(det.candidates) > 0


def test_input_validation(params: DetectParams) -> None:
    with pytest.raises(ValueError, match="2-D bool mask"):
        detect_tile_rows(np.zeros(SHAPE, dtype=np.uint8), FRAME, TILE_ID, params)
    with pytest.raises(ValueError, match="prob shape"):
        detect_tile_rows(np.zeros(SHAPE, bool), FRAME, TILE_ID, params, prob=np.zeros((4, 4), np.float32))


def test_frame_conversion_rejects_a_foreign_tile(stripes: TileDetection) -> None:
    with pytest.raises(ValueError, match="TileRef"):
        detection_to_candidates(stripes, tile_ref("siret3_r006_c004"), run_id="r", model_version="m")


def test_summary_is_json_ready(stripes: TileDetection) -> None:
    summary = stripes.summary()
    assert summary["tile_id"] == TILE_ID and summary["status_hint"] == STATUS_OK
    assert summary["n_accepted"] <= summary["n_kept"] <= summary["n_candidates"]
    assert summary["orientations"][0]["spacing_ok"] is True


def test_tracking_flag_emits_polylines_for_curved_rows(params: DetectParams) -> None:
    r_px = 300.0 / GSD_M
    veg = arc_mask(SHAPE, centre=(1024.0, 1024.0 + r_px), r0_px=r_px - 400.0, spacing_px=100.0, width_px=16.0,
                   n_rows=9)
    detect = params.rows.detect.model_copy(update={"tracking_enabled": True})
    tracked = DetectParams(**{**params.__dict__, "rows": params.rows.model_copy(update={"detect": detect})})
    det = detect_tile_rows(veg, FRAME, TILE_ID, tracked)
    curved = [c for c in det.candidates if c.is_curved and c.rejected_reason is None]
    assert curved and all(len(c.line_px.coords) > 2 for c in curved)


# ------------------------------------------------------------------ reference examples


@dataclass(frozen=True, eq=False)
class ExampleRun:
    det: TileDetection
    frame: gpd.GeoDataFrame
    refs: list[LineString]
    ref_ids: list[str]
    seconds: float


def _prepare(tile_id: str, tif: Path, cache: Path, cfg: AppConfig) -> tuple[np.ndarray, np.ndarray, object]:
    prep_tile(TileTask(tile_id=tile_id, tif_path=tif, key="", cfg=cfg, inputs={"tif": tif},
                       outputs=tile_prep_outputs(cache, tile_id)))
    valid_utm = shapely.from_wkb(tile_valid_wkb_path(cache, tile_id).read_bytes())
    tile = tile_ref(tile_id)
    clip = shapely.transform(valid_utm.intersection(tile_box(tile)), lambda xy: utm_to_px(tile, xy))
    return load_veg_mask(cache, tile_id), load_valid_mask(cache, tile_id), clip


@pytest.fixture(scope="module")
def examples(examples_xml: bytes, example_tif: Callable[[str], Path], cfg: AppConfig,
             params: DetectParams, tmp_path_factory: pytest.TempPathFactory) -> dict[str, ExampleRun]:
    images = load_examples(examples_xml)
    cache = tmp_path_factory.mktemp("rows_detect_examples")
    runs: dict[str, ExampleRun] = {}
    for tile_id in EXAMPLE_TILES:
        veg, valid, clip = _prepare(tile_id, example_tif(tile_id), cache, cfg)
        start = time.perf_counter()
        det = detect_tile_rows(veg, clip, tile_id, params, valid=valid)
        seconds = time.perf_counter() - start
        img = images[tile_id]
        runs[tile_id] = ExampleRun(det=det, frame=_frame(det), refs=row_lines_utm(img), seconds=seconds,
                                   ref_ids=[s.attributes.get("row_id", "?") for s in img.by_label("row")])
    return runs


@pytest.mark.examples
@pytest.mark.parametrize("tile_id", EXAMPLE_TILES)
def test_example_row_f1(examples: dict[str, ExampleRun], cfg: AppConfig, tile_id: str) -> None:
    run = examples[tile_id]
    accepted = run.frame[run.frame["rejected_reason"].isna()]
    score = row_axis_f1(list(accepted.geometry), run.refs, tol_m=cfg.eval.row_tol_m,
                        min_cover=cfg.eval.row_min_cover)
    matched = {j for _, j in score.matches}
    missed = [run.ref_ids[j] for j in range(len(run.refs)) if j not in matched]
    assert score.f1 >= MIN_F1, f"{tile_id}: F1 {score.f1:.3f}, missed {missed}"


@pytest.mark.examples
@pytest.mark.parametrize("tile_id", EXAMPLE_TILES)
def test_example_no_true_row_is_hard_rejected(examples: dict[str, ExampleRun], cfg: AppConfig,
                                              tile_id: str) -> None:
    run = examples[tile_id]
    pairs = row_matches(list(run.frame.geometry), run.refs, tol_m=cfg.eval.row_tol_m,
                        min_cover=cfg.eval.row_min_cover)
    assert len(pairs) == len(run.refs)
    hard = [run.det.candidates[i] for i, _, _ in pairs if run.det.candidates[i].hard_rejected]
    assert hard == []


@pytest.mark.examples
@pytest.mark.parametrize("tile_id", EXAMPLE_TILES)
def test_example_angle_spacing_snr(examples: dict[str, ExampleRun], cfg: AppConfig, tile_id: str) -> None:
    det = examples[tile_id].det
    main = det.orientations[0]
    assert det.status_hint == STATUS_OK
    assert abs(main.angle_utm_deg - EXAMPLE_ANGLE_UTM[tile_id]) <= 1.0
    spacing, tol = EXAMPLE_SPACING_M[tile_id]
    assert main.spacing.spacing_m == pytest.approx(spacing, abs=tol)
    assert main.spacing.snr >= cfg.rows.detect.periodicity_min_snr


@pytest.mark.examples
def test_example_is_deterministic(examples: dict[str, ExampleRun], cfg: AppConfig, params: DetectParams,
                                  example_tif: Callable[[str], Path], tmp_path: Path) -> None:
    tile_id = EXAMPLE_TILES[0]
    veg, valid, clip = _prepare(tile_id, example_tif(tile_id), tmp_path, cfg)
    again = _frame(detect_tile_rows(veg, clip, tile_id, params, valid=valid))
    first = examples[tile_id].frame
    assert list(again["cand_id"]) == list(first["cand_id"])
    assert [g.wkb for g in again.geometry] == [g.wkb for g in first.geometry]
    assert again.drop(columns="geometry").equals(first.drop(columns="geometry"))


@pytest.mark.examples
@pytest.mark.slow
@pytest.mark.parametrize("tile_id", EXAMPLE_TILES)
def test_example_runtime_per_tile(examples: dict[str, ExampleRun], tile_id: str) -> None:
    assert examples[tile_id].seconds <= MAX_SECONDS_PER_TILE


def test_min_line_px_survives_the_utm_round_trip() -> None:
    # r027_c024: a 2.0 px vertical line is 0.049999999813 m in UTM, under the 0.05 m contract minimum
    from vineyard.contracts.schema_defs import MIN_LINE_LENGTH_M
    from vineyard.perception.rows_detect import MIN_LINE_PX

    tile = tile_ref("siret3_r027_c024")
    two_px = LineString(px_to_utm(tile, np.array([[757.0, 1535.5], [757.0, 1537.5]])))
    assert two_px.length < MIN_LINE_LENGTH_M and MIN_LINE_PX > 2.0
    shortest = LineString(px_to_utm(tile, np.array([[757.0, 1535.5], [757.0, 1535.5 + MIN_LINE_PX]])))
    assert shortest.length >= MIN_LINE_LENGTH_M


def test_ref_to_px_roundtrip_helper() -> None:
    uv = np.array([[0.0, 0.0], [2048.0, 2048.0]])
    np.testing.assert_allclose(utm_to_px(TILE, px_to_utm(TILE, uv)), uv, atol=1e-6)
