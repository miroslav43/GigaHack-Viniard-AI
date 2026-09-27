from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fte.waste.boxes import MatchScore
from fte.waste.evaluate import EvalCase, best_threshold, make_case, score_cases, split_gt
from fte.waste.export import HEADER, csv_rows, drop_duplicates_of, write_csv
from fte.waste.filters import VetoParams, veto_forbidden, veto_lookalike
from fte.waste.select import SelectParams, apply_caps, candidate_thresholds, choose_threshold
from fte.waste.train import lr_factor
from fte.waste.windows import ramp_weight, window_starts

# ---------------------------------------------------------------- select


def test_candidate_thresholds_descending_from_t0() -> None:
    assert candidate_thresholds(0.4, (0.1, 0.2, 0.3, 0.4, 0.5)) == [0.4, 0.3, 0.2, 0.1]


def test_choose_threshold_stops_at_recall() -> None:
    syn = {0.5: 0.6, 0.4: 0.7, 0.3: 0.82, 0.2: 0.9}
    ex = {0.5: 0, 0.4: 0, 0.3: 0, 0.2: 0}
    t, why = choose_threshold(0.5, syn, ex, SelectParams())
    assert t == 0.3 and "recall" in why


def test_choose_threshold_blocked_by_examples() -> None:
    syn = {0.5: 0.6, 0.4: 0.7, 0.3: 0.82}
    ex = {0.5: 0, 0.4: 2, 0.3: 5}
    t, why = choose_threshold(0.5, syn, ex, SelectParams())
    assert t == 0.5 and "stopped" in why


def test_choose_threshold_raised_when_t0_has_example_boxes() -> None:
    syn = {0.3: 0.9, 0.4: 0.8, 0.5: 0.6, 0.6: 0.5}
    ex = {0.3: 9, 0.4: 3, 0.5: 0, 0.6: 0}
    t, why = choose_threshold(0.4, syn, ex, SelectParams())
    assert t == 0.5 and why.startswith("raised")


def test_apply_caps() -> None:
    frame = pd.DataFrame({"tile_id": ["a"] * 7 + ["b"] * 2, "score": np.linspace(0.9, 0.1, 9),
                          "xtl": 0.0, "ytl": 0.0, "xbr": 1.0, "ybr": 1.0})
    out = apply_caps(frame, max_per_tile=5, max_total=6)
    assert len(out) == 6 and (out["tile_id"] == "a").sum() == 5
    assert out["score"].is_monotonic_decreasing
    assert len(frame) == 9


# ---------------------------------------------------------------- export


def test_csv_rows_format() -> None:
    frame = pd.DataFrame({"tile_id": ["siret3_r001_c002"], "xtl": [10.04], "ytl": [-3.0], "xbr": [20.0],
                          "ybr": [2050.0], "score": [0.834]})
    rows = csv_rows(frame)
    assert rows == [["siret3_r001_c002", "10.0", "0.0", "20.0", "2048.0", "add", "unknown",
                     "fte-waste@v1 (auto)", "score=0.83"]]
    with pytest.raises(ValueError):
        csv_rows(frame, category="plastic")


def test_write_csv_prepends_confirmed(tmp_path: Path) -> None:
    confirmed = ",".join(HEADER) + "\nsiret3_r0,1,2,3,4,accept,debris,x,\"a; b\"\n"
    path = write_csv(tmp_path / "o.csv", confirmed, [["t", "1.0", "2.0", "3.0", "4.0", "add", "unknown", "r", "score=0.50"]])
    text = path.read_text()
    assert text.startswith(confirmed)
    assert text.endswith("t,1.0,2.0,3.0,4.0,add,unknown,r,score=0.50\n")
    assert list(pd.read_csv(path).columns) == list(HEADER)


def test_drop_duplicates_of_confirmed() -> None:
    frame = pd.DataFrame({"tile_id": ["a", "a"], "xtl": [0.0, 50.0], "ytl": [0.0, 50.0], "xbr": [10.0, 60.0],
                          "ybr": [10.0, 60.0], "score": [0.9, 0.8]})
    conf = pd.DataFrame({"tile_id": ["a"], "xtl": [1.0], "ytl": [1.0], "xbr": [10.0], "ybr": [10.0]})
    out = drop_duplicates_of(frame, conf)
    assert out["xtl"].tolist() == [50.0]


# ---------------------------------------------------------------- filters


def test_veto_lookalike_iou_and_centre_rule() -> None:
    cands = np.array([[100, 100, 110, 104]], dtype=float)
    boxes = np.array([[100, 100, 110, 105, 0.9],  # same blob
                      [60, 60, 160, 160, 0.9],  # big heap containing the tube centre
                      [300, 300, 310, 310, 0.9]], dtype=float)
    assert veto_lookalike(boxes, cands, VetoParams()).tolist() == [True, False, False]


def test_veto_forbidden_intersects() -> None:
    from shapely.geometry import box

    zone = box(0, 0, 50, 50)
    boxes = np.array([[40, 40, 60, 60, 0.5], [100, 100, 110, 110, 0.5]], dtype=float)
    assert veto_forbidden(boxes, zone).tolist() == [True, False]
    assert veto_forbidden(boxes, None).tolist() == [False, False]


# ---------------------------------------------------------------- evaluate / windows / schedule


def test_split_gt_and_score_cases() -> None:
    kept, tiny = split_gt(np.array([[0, 0, 10, 10], [0, 0, 2, 2]], float), 24.0)
    assert len(kept) == 1 and len(tiny) == 1
    heat = np.zeros((64, 64), np.uint8)
    heat[10:20, 10:20] = 230
    heat[40:42, 40:60] = 230  # 40 px component in a don't-care region
    case = make_case("k", heat, np.array([[10, 10, 20, 20]], float), np.array([[38, 38, 62, 44]], float), 24.0)
    scores = score_cases([case], (0.5, 0.95))
    assert scores[0.5].tp == 1 and scores[0.5].n_pred == 1
    assert scores[0.95].n_pred == 0
    assert best_threshold(scores) == (0.5, 1.0)


def test_best_threshold_prefers_lower_on_tie() -> None:
    s = {0.3: MatchScore(1, 2, 2), 0.5: MatchScore(1, 2, 2)}
    assert best_threshold(s)[0] == 0.3


def test_window_starts_cover() -> None:
    assert window_starts(2048, 1024, 128) == [0, 896, 1024]
    assert window_starts(500, 1024, 128) == [0]
    starts = window_starts(3000, 768, 96)
    assert starts[0] == 0 and starts[-1] == 3000 - 768
    assert all(b - a <= 768 - 96 for a, b in zip(starts, starts[1:], strict=False))
    with pytest.raises(ValueError):
        window_starts(10, 8, 8)


def test_ramp_weight() -> None:
    w = ramp_weight(64, 16)
    assert w.shape == (64, 64) and w[32, 32] == 1.0 and 0 < w[0, 0] < 0.01


def test_lr_factor_warmup_then_cosine() -> None:
    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100) == pytest.approx(0.02)
    assert lr_factor(55, 10, 100) < 1.0


def test_eval_case_is_frozen() -> None:
    case = EvalCase("k", np.zeros((2, 2), np.uint8), np.zeros((0, 4)), np.zeros((0, 4)))
    with pytest.raises(AttributeError):
        case.key = "x"  # type: ignore[misc]


def test_crop_with_box_shape_and_red_box() -> None:
    from fte.waste.export import CROP_PX, crop_with_box

    rgb = np.zeros((2048, 2048, 3), np.uint8)
    crop = crop_with_box(rgb, (2040.0, 2040.0, 2048.0, 2048.0))
    assert crop.shape == (CROP_PX, CROP_PX, 3)
    assert (crop[..., 0] == 255).any() and rgb.max() == 0


def test_window_negatives_math() -> None:
    from fte.convert.waste_negatives import boxes_in_window, ignore_label, window_origin

    rng = np.random.default_rng(0)
    assert window_origin(10.0, 2040.0, rng, 0) == (0, 2048 - 384)
    boxes = np.array([[370.0, 10.0, 400.0, 20.0], [900.0, 900.0, 910.0, 910.0]])
    inside = boxes_in_window(boxes, 0, 0, pad=2.0)
    assert inside == [[368.0, 8.0, 384.0, 22.0]]
    lbl = ignore_label(inside, None)
    assert lbl[10, 370] == 255 and lbl[100, 100] == 0 and lbl.shape == (384, 384)


def test_holdout_tiles_deterministic_no_examples() -> None:
    from fte.convert.waste_negatives import holdout_tiles
    from fte.paths import EXAMPLE_TILES

    tiles = pd.DataFrame({"tile_id": [f"siret3_r{i:03d}_c000" for i in range(40)] + list(EXAMPLE_TILES),
                          "has_vineyard": [i % 2 == 0 for i in range(40)] + [True, True]})
    a = holdout_tiles(tiles, 10, 0.6)
    assert a == holdout_tiles(tiles, 10, 0.6) and len(a) == 10
    assert not set(a) & set(EXAMPLE_TILES)
