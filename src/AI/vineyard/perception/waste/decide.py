"""Waste decision rule (A§4.9 steps 4 and 6, design 03 W8) with auto-accept behind a config switch.

auto   = auto_accept_enabled and probe >= max(auto_probe_min, tau*) and (sam >= auto_sam3_min or
         clip margin > auto_clip_margin_min)   -- disabled before Publish (waste.decide.auto_accept_enabled);
rank   = probe_p, else clip_pos_p, else the rule saliency score;
review = not auto and rank >= candidate_min (capped in rule-only mode).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from vineyard.perception.waste.types import Candidate, CandidateScores, Decision, Detector

if TYPE_CHECKING:
    from vineyard.config import WasteDecideConfig

U8_MAX: Final = 255.0
SIZE_WEIGHT: Final = 0.5  # rule saliency = colour strength * (1 - w + w * log-size term)
LEVEL_FULL: Final = "L0"  # probe + SAM 3
LEVEL_PROBE_CLIP: Final = "L1"  # probe + CLIP margin
LEVEL_CLIP: Final = "L2"  # CLIP ranking only
LEVEL_RULE: Final = "L3"  # rule-only ranking


@dataclass(frozen=True)
class DecideParams:
    auto_accept_enabled: bool
    candidate_min: float
    auto_probe_min: float
    auto_sam3_min: float
    auto_clip_margin_min: float
    rule_only_max_review: int

    @classmethod
    def from_config(cls, cfg: WasteDecideConfig) -> DecideParams:
        return cls(
            auto_accept_enabled=cfg.auto_accept_enabled,
            candidate_min=cfg.candidate_min,
            auto_probe_min=cfg.auto_probe_min,
            auto_sam3_min=cfg.auto_sam3_min,
            auto_clip_margin_min=cfg.auto_clip_margin_min,
            rule_only_max_review=cfg.rule_only_max_review,
        )


def rank_score(s: CandidateScores) -> float:
    for value in (s.probe_p, s.clip_pos_p, s.rule_score):
        if value is not None:
            return float(value)
    return 0.0


def decide(s: CandidateScores, probe_auto_min: float | None, p: DecideParams) -> Decision:
    threshold = max(p.auto_probe_min, probe_auto_min if probe_auto_min is not None else 0.0)
    probe_ok = s.probe_p is not None and s.probe_p >= threshold
    sam_ok = s.sam_score is not None and s.sam_score >= p.auto_sam3_min
    clip_ok = s.clip_margin is not None and s.clip_margin > p.auto_clip_margin_min
    auto = p.auto_accept_enabled and probe_ok and (sam_ok or clip_ok)
    rank = rank_score(s)
    detector = (Detector.SAM3 if sam_ok else Detector.NN) if auto else Detector.RULE
    return Decision(
        cand_key=s.cand_key,
        auto=auto,
        review=not auto and rank >= p.candidate_min,
        rank_score=rank,
        detector=detector,
    )


def rule_saliency(c: Candidate, min_area_px: float, max_area_px: float) -> float:
    """[0, 1]: colour strength (S for vivid, V for bright) times a log-size term (0 at min, 1 at max area)."""
    if not 0 < min_area_px < max_area_px:
        raise ValueError(f"need 0 < min_area_px < max_area_px, got {min_area_px}, {max_area_px}")
    _, sat, val = c.mean_hsv
    colour = (val if c.is_white else sat) / U8_MAX
    size = math.log(max(c.area_px, min_area_px) / min_area_px) / math.log(max_area_px / min_area_px)
    return float(min(1.0, colour * (1.0 - SIZE_WEIGHT + SIZE_WEIGHT * min(size, 1.0))))


def review_selection(decisions: Sequence[Decision], cap: int | None) -> tuple[str, ...]:
    """cand_keys flagged for review, best rank first (ties by key), at most `cap` when given."""
    ranked = sorted((d for d in decisions if d.review), key=lambda d: (-d.rank_score, d.cand_key))
    keys = tuple(d.cand_key for d in ranked)
    return keys if cap is None else keys[:cap]


def degradation_level(has_probe: bool, has_clip: bool, has_sam: bool) -> str:
    if has_probe and has_sam:
        return LEVEL_FULL
    if has_probe and has_clip:
        return LEVEL_PROBE_CLIP
    return LEVEL_CLIP if has_clip else LEVEL_RULE
