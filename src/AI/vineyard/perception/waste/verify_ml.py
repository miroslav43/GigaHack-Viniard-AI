"""Model verifier (design 03 §2 verify.py, W5/W6/W8): main-process scoring of the filter survivors.

1. CLIP crops (box + context, min side, resized) in batches -> embeddings -> probe P(waste), plus the CLIP
   zero-shot positive probability and margin (ranking / second-model signal only);
2. SAM 3 jobs for the survivors whose gate score (probe, else CLIP, else rule saliency) is at least
   `waste.sam3.min_probe_to_verify`, best first, one job per overlapping cluster, 512 px windows with the
   nearest rejected white tubes as negative boxes, within the global time budget (the rest keep None);
3. combine_scores -> CandidateScores.
Torch-free: the CLIP embedder, the probe and the SAM 3 backend come in as objects (Protocols).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.waste.crops import box_to_crop, crop_window, extract_crop, sam_window
from vineyard.perception.waste.sam3_adapter import (
    FileLister,
    Sam3Loader,
    Sam3Params,
    Sam3Status,
    Sam3Verifier,
    SamJob,
    SamResult,
    load_sam3,
    select_negative_boxes,
    verify_with_budget,
)
from vineyard.perception.waste.types import TILE_EXTENT_PX, BoxPx, Candidate, CandidateScores
from vineyard.perception.waste.verify import (
    RuleOnlyVerifier,
    VerifyStatus,
    combine_scores,
    rule_verifier,
    verify_status,
)

if TYPE_CHECKING:
    from vineyard.config import ProbeConfig, WasteConfig

TileReader = Callable[[str], np.ndarray]
MODEL_VERIFIER: Final = "model"
TILE_CACHE_SIZE: Final = 2  # SAM jobs are ordered by score, not tile: keep the last tiles decoded
EMBED_NDIM: Final = 2

_log = get_logger("perception.waste.verify")


class ClipEmbedder(Protocol):
    def embed(self, crops_u8: np.ndarray) -> np.ndarray: ...  # (N, S, S, 3) uint8 -> (N, D)

    def zero_shot(
        self, emb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...  # pos_p, margin, idx


@dataclass(frozen=True)
class ProbeHandle:
    """A trained probe: embeddings -> P(waste); auto_threshold = max(auto_probe_min, tau*) of the card."""

    score: Callable[[np.ndarray], np.ndarray]
    auto_threshold: float
    auto_allowed: bool  # False when recall_at_threshold < waste.probe.min_recall
    version: str


@dataclass(frozen=True)
class CropParams:
    context: float
    min_px: int
    resize_px: int
    batch: int
    extent_px: int

    @classmethod
    def from_config(cls, cfg: ProbeConfig, extent_px: int) -> CropParams:
        return cls(cfg.crop_context, cfg.crop_min_px, cfg.crop_resize_px, cfg.embed_batch, extent_px)


@dataclass(frozen=True)
class ClipScores:
    probe_p: float | None
    clip_pos_p: float | None
    clip_margin: float | None


@dataclass(frozen=True)
class SamPlanParams:
    min_gate: float
    dedupe_iou: float
    crop_px: int
    extent_px: int
    max_negatives: int
    budget_s: float

    @classmethod
    def from_config(cls, cfg: WasteConfig, extent_px: int) -> SamPlanParams:
        s = cfg.sam3
        return cls(
            s.min_probe_to_verify, cfg.nms_iou, s.crop_px, extent_px, s.max_negative_boxes, s.time_budget_s
        )


# ------------------------------------------------------------------ CLIP / probe


def _crop_batches(
    cands: Sequence[Candidate], read_tile: TileReader, p: CropParams
) -> Iterator[tuple[list[str], np.ndarray]]:
    """(cand_keys, uint8 crops) batches of at most p.batch, each tile read once."""
    keys: list[str] = []
    crops: list[np.ndarray] = []
    for tile_id in sorted({c.tile_id for c in cands}):
        rgb = read_tile(tile_id)
        for c in (c for c in cands if c.tile_id == tile_id):
            window = crop_window(c.box, p.context, p.min_px, p.extent_px)
            crops.append(extract_crop(rgb, window, p.resize_px))
            keys.append(c.cand_key)
            if len(crops) == p.batch:
                yield keys, np.stack(crops)
                keys, crops = [], []
    if crops:
        yield keys, np.stack(crops)


def _column(name: str, values: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} returned {arr.shape[0]} values for {n} crops")
    return arr


def clip_probe_scores(
    cands: Sequence[Candidate],
    read_tile: TileReader,
    embedder: ClipEmbedder,
    probe: ProbeHandle | None,
    p: CropParams,
) -> dict[str, ClipScores]:
    """Per cand_key: probe P(waste) (None without a probe), CLIP positive probability and margin."""
    out: dict[str, ClipScores] = {}
    for keys, crops in _crop_batches(cands, read_tile, p):
        n = len(keys)
        emb = np.asarray(embedder.embed(crops))
        if emb.ndim != EMBED_NDIM or emb.shape[0] != n:
            raise ValueError(f"CLIP embed returned shape {emb.shape} for {n} crops")
        pos_p, margin, _ = embedder.zero_shot(emb)
        pos = _column("CLIP zero_shot pos_p", pos_p, n)
        mar = _column("CLIP zero_shot margin", margin, n)
        prb = None if probe is None else _column(f"probe {probe.version}", probe.score(emb), n)
        for i, key in enumerate(keys):
            out[key] = ClipScores(None if prb is None else float(prb[i]), float(pos[i]), float(mar[i]))
    return out


# ------------------------------------------------------------------ SAM 3 jobs


def plan_sam_jobs(
    cands: Sequence[Candidate], gate: Mapping[str, float], pool: Sequence[Candidate], p: SamPlanParams
) -> tuple[SamJob, ...]:
    """Jobs best gate first (ties by key); a candidate overlapping (IoU > dedupe_iou) an already queued
    one of the same tile is skipped, since NMS keeps one box per cluster anyway."""
    eligible = [c for c in cands if gate.get(c.cand_key, 0.0) >= p.min_gate]
    ordered = sorted(eligible, key=lambda c: (-gate[c.cand_key], c.cand_key))
    queued: dict[str, tuple[BoxPx, ...]] = {}
    jobs: list[SamJob] = []
    for c in ordered:
        boxes = queued.get(c.tile_id, ())
        if any(c.box.iou(b) > p.dedupe_iou for b in boxes):
            continue
        queued = {**queued, c.tile_id: (*boxes, c.box)}
        window = sam_window(c.centroid_px, p.crop_px, p.extent_px)
        negatives = select_negative_boxes(c, pool, window, p.max_negatives)
        jobs.append(SamJob(c.cand_key, c.tile_id, window, box_to_crop(c.box.as_tuple(), window), negatives))
    return tuple(jobs)


def _gate(clip: ClipScores | None, rule: float | None) -> float:
    for value in (None if clip is None else clip.probe_p, None if clip is None else clip.clip_pos_p, rule):
        if value is not None:
            return float(value)
    return 0.0


def window_reader(read_tile: TileReader) -> Callable[[SamJob], np.ndarray]:
    """SamJob -> its uint8 window (no resize), keeping the last TILE_CACHE_SIZE tiles decoded."""
    cached = lru_cache(maxsize=TILE_CACHE_SIZE)(read_tile)

    def crop(job: SamJob) -> np.ndarray:
        w = job.window
        return np.ascontiguousarray(cached(job.tile_id)[w.y0 : w.y0 + w.side, w.x0 : w.x0 + w.side])

    return crop


# ------------------------------------------------------------------ the verifier


@dataclass(frozen=True)
class ModelVerifier:
    rule: RuleOnlyVerifier
    read_tile: TileReader
    crop: CropParams
    sam_plan: SamPlanParams
    embedder: ClipEmbedder | None = None
    probe: ProbeHandle | None = None
    sam: Sam3Verifier | None = None
    clock: Callable[[], float] = time.monotonic
    name: str = MODEL_VERIFIER

    def __post_init__(self) -> None:
        if self.probe is not None and self.embedder is None:
            raise ValueError(f"probe {self.probe.version} needs a CLIP embedder to produce its embeddings")

    @property
    def probe_auto_min(self) -> float | None:
        if self.probe is None:
            return None
        return self.probe.auto_threshold if self.probe.auto_allowed else math.inf

    def _sam_scores(
        self, cands: Sequence[Candidate], pool: Sequence[Candidate], gate: Mapping[str, float]
    ) -> dict[str, SamResult]:
        if self.sam is None:
            return {}
        jobs = plan_sam_jobs(cands, gate, pool, self.sam_plan)
        t0 = self.clock()
        results = verify_with_budget(
            self.sam, jobs, window_reader(self.read_tile), self.sam_plan.budget_s, self.clock
        )
        done = {j.cand_key: r for j, r in zip(jobs, results, strict=True) if r is not None}
        log_event(_log, "waste.sam3.verified", n_jobs=len(jobs), n_done=len(done), seconds=self.clock() - t0)
        return done

    def score(
        self, cands: Sequence[Candidate], pool: Sequence[Candidate] = ()
    ) -> tuple[CandidateScores, ...]:
        rule = {s.cand_key: s.rule_score for s in self.rule.score(cands)}
        clip = (
            {}
            if self.embedder is None
            else clip_probe_scores(cands, self.read_tile, self.embedder, self.probe, self.crop)
        )
        sam = self._sam_scores(cands, pool, {k: _gate(clip.get(k), r) for k, r in rule.items()})
        empty = ClipScores(None, None, None)
        return tuple(
            combine_scores(
                c.cand_key,
                rule[c.cand_key],
                clip.get(c.cand_key, empty).probe_p,
                clip.get(c.cand_key, empty).clip_pos_p,
                clip.get(c.cand_key, empty).clip_margin,
                sam.get(c.cand_key),
            )
            for c in cands
        )


def build_verifier(
    cfg: WasteConfig,
    gsd_m: float,
    models_dir: Path,
    read_tile: TileReader,
    *,
    embedder: ClipEmbedder | None = None,
    probe: ProbeHandle | None = None,
    sam_loader: Sam3Loader | None = None,
    sam_lister: FileLister | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[ModelVerifier, VerifyStatus, Sam3Status]:
    """The model verifier with whatever is available (SAM 3 loaded here, once) and its degradation status."""
    notes: list[str] = []
    if not cfg.probe.enabled and (embedder is not None or probe is not None):
        embedder, probe = None, None
        notes.append("probe disabled by config (waste.probe.enabled)")
    if probe is not None and not probe.auto_allowed:
        notes.append(f"probe {probe.version}: auto-accept not allowed (recall below waste.probe.min_recall)")
    sam, sam_status = load_sam3(Sam3Params.from_config(cfg.sam3, models_dir), sam_loader, sam_lister, clock)
    extent = int(TILE_EXTENT_PX)
    verifier = ModelVerifier(
        rule=rule_verifier(cfg, gsd_m),
        read_tile=read_tile,
        crop=CropParams.from_config(cfg.probe, extent),
        sam_plan=SamPlanParams.from_config(cfg, extent),
        embedder=embedder,
        probe=probe,
        sam=sam,
        clock=clock,
    )
    return verifier, verify_status(probe is not None, embedder is not None, sam_status, notes), sam_status
