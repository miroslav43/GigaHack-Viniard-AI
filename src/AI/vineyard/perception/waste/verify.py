"""Candidate verification interface (design 03 §2 verify.py). `load_verifier` is the rule-only verifier
(degradation level L3: no probe, no CLIP, no SAM 3, rank = rule saliency). The model verifier (probe + CLIP
margin + SAM 3, verify_ml.build_verifier) plugs in behind the same Protocol; `combine_scores` validates
and merges its per-candidate scores and `verify_status` names the resulting degradation level.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from vineyard.perception.waste.candidates import m2_to_px
from vineyard.perception.waste.decide import (
    LEVEL_RULE,
    DecideParams,
    decide,
    degradation_level,
    review_selection,
    rule_saliency,
)
from vineyard.perception.waste.nms import mark_suppressed
from vineyard.perception.waste.types import Candidate, CandidateScores, Category, Decision

if TYPE_CHECKING:
    from vineyard.config import WasteConfig
    from vineyard.perception.waste.sam3_adapter import Sam3Status, SamResult

RULE_DETECTOR_VERSION: Final = "1"
RULE_VERIFIER: Final = "rule"
REASON_RULE_ONLY: Final = "rule-only ranking (waste.probe and waste.sam3 disabled)"


class Verifier(Protocol):
    """`score` gets the live candidates plus the whole candidate pool (rejected ones included: the white
    tubes become SAM 3 negative boxes); `probe_auto_min` is the probe's calibrated tau* (None: no probe,
    inf: probe not trusted for auto-accept)."""

    @property
    def name(self) -> str: ...

    @property
    def probe_auto_min(self) -> float | None: ...

    def score(
        self, cands: Sequence[Candidate], pool: Sequence[Candidate] = ()
    ) -> tuple[CandidateScores, ...]: ...


@dataclass(frozen=True)
class VerifyStatus:
    level: str
    probe: bool
    clip: bool
    sam: bool
    reason: str


@dataclass(frozen=True)
class RuleOnlyVerifier:
    """Scores = rule saliency only; the pool is ignored."""

    min_area_px: float
    max_area_px: float
    name: str = RULE_VERIFIER
    probe_auto_min: float | None = None

    def score(
        self, cands: Sequence[Candidate], pool: Sequence[Candidate] = ()
    ) -> tuple[CandidateScores, ...]:
        return tuple(
            CandidateScores(
                cand_key=c.cand_key,
                probe_p=None,
                clip_pos_p=None,
                clip_margin=None,
                sam_score=None,
                category=Category.UNKNOWN,
                rule_score=rule_saliency(c, self.min_area_px, self.max_area_px),
            )
            for c in cands
        )


def rule_verifier(cfg: WasteConfig, gsd_m: float) -> RuleOnlyVerifier:
    return RuleOnlyVerifier(
        min_area_px=m2_to_px(cfg.min_area_m2, gsd_m), max_area_px=m2_to_px(cfg.max_area_m2, gsd_m)
    )


def load_verifier(cfg: WasteConfig, gsd_m: float) -> tuple[Verifier, VerifyStatus]:
    """The rule-only verifier and its degradation status (L3); the run's verifier is verify_setup.setup_verifier."""
    status = VerifyStatus(level=LEVEL_RULE, probe=False, clip=False, sam=False, reason=REASON_RULE_ONLY)
    return rule_verifier(cfg, gsd_m), status


def waste_version_string(status: VerifyStatus) -> str:
    """`<detector>@<version>` for model_version (contract §2.2), e.g. "rule@1"."""
    if status.probe or status.clip or status.sam:
        parts = [n for n, on in (("probe", status.probe), ("clip", status.clip), ("sam3", status.sam)) if on]
        return f"{'+'.join(parts)}@{RULE_DETECTOR_VERSION}"
    return f"{RULE_VERIFIER}@{RULE_DETECTOR_VERSION}"


@dataclass(frozen=True)
class RankedSet:
    candidates: tuple[Candidate, ...]  # input order; NMS losers carry reject_reason "nms"
    decisions: Mapping[str, Decision]  # per not-filtered candidate (cand_key)
    review_keys: tuple[str, ...]  # review list, best first


def rank_and_select(
    cands: Sequence[Candidate], verifier: Verifier, status: VerifyStatus, p: DecideParams, nms_iou: float
) -> RankedSet:
    """Score the filter survivors, decide, NMS by rank score, then pick the review list (capped at L3)."""
    live = [c for c in cands if not c.rejected]
    decisions = {s.cand_key: decide(s, verifier.probe_auto_min, p) for s in verifier.score(live, cands)}
    marked = mark_suppressed(cands, {k: d.rank_score for k, d in decisions.items()}, nms_iou)
    kept = [decisions[c.cand_key] for c in marked if not c.rejected]
    cap = p.rule_only_max_review if status.level == LEVEL_RULE else None
    return RankedSet(
        candidates=marked, decisions=MappingProxyType(decisions), review_keys=review_selection(kept, cap)
    )


# ------------------------------------------------------------------ model scores


def _checked(name: str, value: float | None, lo: float, hi: float, key: str) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not (math.isfinite(v) and lo <= v <= hi):
        raise ValueError(f"{key}: {name}={value} is not a finite value in [{lo}, {hi}]")
    return v


def combine_scores(
    cand_key: str,
    rule_score: float | None,
    probe_p: float | None,
    clip_pos_p: float | None,
    clip_margin: float | None,
    sam: SamResult | None,
) -> CandidateScores:
    """Validated CandidateScores; category from the SAM 3 prompt that fired (else unknown)."""
    sam_score = None if sam is None else _checked("sam_score", sam.score, 0.0, 1.0, cand_key)
    return CandidateScores(
        cand_key=cand_key,
        probe_p=_checked("probe_p", probe_p, 0.0, 1.0, cand_key),
        clip_pos_p=_checked("clip_pos_p", clip_pos_p, 0.0, 1.0, cand_key),
        clip_margin=_checked("clip_margin", clip_margin, -1.0, 1.0, cand_key),
        sam_score=sam_score,
        category=sam.category if sam is not None and sam.prompt is not None else Category.UNKNOWN,
        rule_score=_checked("rule_score", rule_score, 0.0, 1.0, cand_key),
    )


def verify_status(
    has_probe: bool, has_clip: bool, sam: Sam3Status | None, notes: Sequence[str] = ()
) -> VerifyStatus:
    """Degradation level L0..L3 of the available models plus a human-readable reason."""
    has_sam = sam is not None and sam.available
    level = degradation_level(has_probe, has_clip, has_sam)
    sam_part = "sam3: off" if sam is None else f"sam3: {sam.model_id if has_sam else sam.reason}"
    parts = [
        f"probe: {'on' if has_probe else 'off'}",
        f"clip: {'on' if has_clip else 'off'}",
        sam_part,
        *notes,
    ]
    return VerifyStatus(level=level, probe=has_probe, clip=has_clip, sam=has_sam, reason="; ".join(parts))
