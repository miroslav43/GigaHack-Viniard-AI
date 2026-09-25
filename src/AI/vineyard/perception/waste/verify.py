"""Candidate verification interface (design 03 §2 verify.py). Phase 1 ships the rule-only verifier
(degradation level L3): no probe, no CLIP, no SAM 3, so nothing can be auto-accepted and the rank is
the rule saliency. The CLIP probe / SAM 3 verifiers of W2/W3b plug in behind the same Protocol later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

from vineyard.perception.waste.candidates import m2_to_px
from vineyard.perception.waste.decide import LEVEL_RULE, DecideParams, decide, review_selection, rule_saliency
from vineyard.perception.waste.nms import mark_suppressed
from vineyard.perception.waste.types import Candidate, CandidateScores, Category, Decision

if TYPE_CHECKING:
    from vineyard.config import WasteConfig

RULE_DETECTOR_VERSION: Final = "1"
RULE_VERIFIER: Final = "rule"
REASON_PHASE1: Final = "probe / CLIP / SAM 3 not wired (phase 1: rule-only ranking)"


class Verifier(Protocol):
    name: str

    def score(
        self, cands: Sequence[Candidate], crops: Sequence[np.ndarray] | None = None
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
    """Scores = rule saliency only; crops are ignored."""

    min_area_px: float
    max_area_px: float
    name: str = RULE_VERIFIER

    def score(
        self, cands: Sequence[Candidate], crops: Sequence[np.ndarray] | None = None
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


def load_verifier(cfg: WasteConfig, gsd_m: float) -> tuple[Verifier, VerifyStatus]:
    """The verifier available in this build and its degradation status (always L3 in phase 1)."""
    verifier = RuleOnlyVerifier(
        min_area_px=m2_to_px(cfg.min_area_m2, gsd_m), max_area_px=m2_to_px(cfg.max_area_m2, gsd_m)
    )
    return verifier, VerifyStatus(level=LEVEL_RULE, probe=False, clip=False, sam=False, reason=REASON_PHASE1)


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
    decisions = {s.cand_key: decide(s, None, p) for s in verifier.score(live)}
    marked = mark_suppressed(cands, {k: d.rank_score for k, d in decisions.items()}, nms_iou)
    kept = [decisions[c.cand_key] for c in marked if not c.rejected]
    cap = p.rule_only_max_review if status.level == LEVEL_RULE else None
    return RankedSet(
        candidates=marked, decisions=MappingProxyType(decisions), review_keys=review_selection(kept, cap)
    )
