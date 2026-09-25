from __future__ import annotations

from dataclasses import replace

import pytest

from tests.waste.synth_waste import make_candidate
from vineyard.perception.waste.decide import (
    DecideParams,
    decide,
    degradation_level,
    rank_score,
    review_selection,
    rule_saliency,
)
from vineyard.perception.waste.types import CandidateScores, Category, Decision, Detector


@pytest.fixture
def params(cfg) -> DecideParams:
    return DecideParams.from_config(cfg.waste.decide)


@pytest.fixture
def auto_on(params: DecideParams) -> DecideParams:
    return replace(params, auto_accept_enabled=True)


def scores(probe, sam, margin, clip_p=None, rule=None, key="k") -> CandidateScores:
    return CandidateScores(
        cand_key=key,
        probe_p=probe,
        clip_pos_p=clip_p,
        clip_margin=margin,
        sam_score=sam,
        category=Category.UNKNOWN,
        rule_score=rule,
    )


def test_config_disables_auto(params: DecideParams) -> None:
    assert params.auto_accept_enabled is False
    d = decide(scores(0.99, 0.99, 0.9), None, params)
    assert not d.auto and d.review and d.detector == Detector.RULE


@pytest.mark.parametrize(
    ("probe", "sam", "margin", "auto"),
    [
        (0.95, 0.6, 0.0, True),
        (0.95, 0.4, 0.25, True),
        (0.95, 0.4, 0.1, False),
        (0.89, 0.9, 0.9, False),
        (0.95, None, 0.21, True),
        (0.95, None, None, False),
    ],
)
def test_two_model_rule(auto_on: DecideParams, probe, sam, margin, auto) -> None:
    d = decide(scores(probe, sam, margin), None, auto_on)
    assert d.auto is auto
    assert d.review is (not auto)


def test_tau_star_raises_the_threshold(auto_on: DecideParams) -> None:
    assert not decide(scores(0.92, 0.9, 0.9), 0.93, auto_on).auto
    assert decide(scores(0.94, 0.9, 0.9), 0.93, auto_on).auto


def test_detector_by_branch(auto_on: DecideParams) -> None:
    assert decide(scores(0.95, 0.6, 0.0), None, auto_on).detector == Detector.SAM3
    assert decide(scores(0.95, 0.1, 0.5), None, auto_on).detector == Detector.NN


def test_review_threshold(params: DecideParams) -> None:
    assert decide(scores(None, None, None, rule=0.30), None, params).review
    assert not decide(scores(None, None, None, rule=0.29), None, params).review


def test_rank_score_fallbacks() -> None:
    assert rank_score(scores(0.7, None, None, clip_p=0.5, rule=0.2)) == 0.7
    assert rank_score(scores(None, None, None, clip_p=0.5, rule=0.2)) == 0.5
    assert rank_score(scores(None, None, None, rule=0.2)) == 0.2
    assert rank_score(scores(None, None, None)) == 0.0


def test_rule_saliency_orders_by_colour_and_size() -> None:
    vivid_small = replace(make_candidate(area_px=24), mean_hsv=(110.0, 255.0, 200.0))
    vivid_big = replace(make_candidate(area_px=9600), mean_hsv=(110.0, 255.0, 200.0))
    white = replace(make_candidate(area_px=500, is_white=True), mean_hsv=(0.0, 10.0, 240.0))
    assert rule_saliency(vivid_small, 24, 9600) == pytest.approx(0.5)
    assert rule_saliency(vivid_big, 24, 9600) == pytest.approx(1.0)
    assert 0.3 < rule_saliency(white, 24, 9600) < 1.0
    with pytest.raises(ValueError):
        rule_saliency(white, 0, 9600)


def test_review_selection_cap_and_order() -> None:
    ds = [
        Decision("b", False, True, 0.5, Detector.RULE),
        Decision("a", False, True, 0.5, Detector.RULE),
        Decision("c", False, True, 0.9, Detector.RULE),
        Decision("d", False, False, 0.1, Detector.RULE),
    ]
    assert review_selection(ds, None) == ("c", "a", "b")
    assert review_selection(ds, 2) == ("c", "a")


def test_degradation_levels() -> None:
    assert degradation_level(True, True, True) == "L0"
    assert degradation_level(True, True, False) == "L1"
    assert degradation_level(False, True, False) == "L2"
    assert degradation_level(False, False, False) == "L3"
