"""decide: the two-model auto-accept rule with SAM 3 / CLIP scores and the SAM-boosted review rank."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from vineyard.config import load_config
from vineyard.perception.waste.decide import DecideParams, decide, rank_score
from vineyard.perception.waste.types import CandidateScores, Category, Detector


@pytest.fixture
def params(cfg) -> DecideParams:
    return DecideParams.from_config(cfg.waste.decide)


@pytest.fixture
def auto_on(params: DecideParams) -> DecideParams:
    return replace(params, auto_accept_enabled=True)


def scores(probe=None, sam=None, margin=None, clip_p=None, rule=None) -> CandidateScores:
    return CandidateScores("k", probe, clip_p, margin, sam, Category.UNKNOWN, rule)


def test_boost_comes_from_config(params: DecideParams) -> None:
    boost = load_config().waste.decide.sam_rank_boost
    assert params.sam_rank_boost == boost
    assert 0.0 < boost <= 1.0


@pytest.mark.parametrize("boost", [-0.1, 1.1, math.nan])
def test_invalid_boost_raises(params: DecideParams, boost: float) -> None:
    with pytest.raises(ValueError, match="sam_rank_boost"):
        replace(params, sam_rank_boost=boost)


def test_sam_boost_only_raises_the_rank() -> None:
    assert rank_score(scores(probe=0.35, sam=0.8), 0.5) == pytest.approx(0.35 + 0.5 * 0.8 * 0.65)
    assert rank_score(scores(probe=0.35, sam=0.0), 0.5) == pytest.approx(0.35)
    assert rank_score(scores(probe=0.35), 0.5) == pytest.approx(0.35)
    assert rank_score(scores(probe=0.35, sam=0.8), 0.0) == pytest.approx(0.35)
    assert rank_score(scores(probe=1.0, sam=1.0), 1.0) == pytest.approx(1.0)
    assert rank_score(scores(rule=0.2, sam=0.9), 0.5) == pytest.approx(0.2 + 0.5 * 0.9 * 0.8)


def test_sam_evidence_can_lift_a_candidate_into_review(params: DecideParams) -> None:
    low = scores(probe=0.25, sam=0.9)
    assert decide(low, None, params).review
    assert not decide(replace(low, sam_score=None), None, params).review


def test_auto_disabled_by_default_even_with_perfect_scores(params: DecideParams) -> None:
    d = decide(scores(probe=1.0, sam=1.0, margin=1.0), None, params)
    assert not d.auto and d.review and d.detector == Detector.RULE


def test_sam_branch_vs_clip_branch(auto_on: DecideParams) -> None:
    sam = decide(scores(probe=0.95, sam=0.6, margin=0.0), None, auto_on)
    clip = decide(scores(probe=0.95, sam=0.0, margin=0.25), None, auto_on)
    neither = decide(scores(probe=0.95, sam=0.49, margin=0.2), None, auto_on)
    assert (sam.auto, sam.detector) == (True, Detector.SAM3)
    assert (clip.auto, clip.detector) == (True, Detector.NN)
    assert (neither.auto, neither.review, neither.detector) == (False, True, Detector.RULE)


def test_sam_alone_never_auto_accepts(auto_on: DecideParams) -> None:
    assert not decide(scores(sam=0.99, margin=0.9, clip_p=0.99), None, auto_on).auto
    assert not decide(scores(probe=0.8, sam=0.99, margin=0.9), None, auto_on).auto


def test_probe_auto_disabled_by_infinite_threshold(auto_on: DecideParams) -> None:
    assert not decide(scores(probe=1.0, sam=1.0, margin=1.0), math.inf, auto_on).auto


def test_boundaries(auto_on: DecideParams) -> None:
    assert decide(scores(probe=0.9, sam=0.5), None, auto_on).auto  # both >=
    assert not decide(scores(probe=0.9, margin=0.2), None, auto_on).auto  # margin is strict >
