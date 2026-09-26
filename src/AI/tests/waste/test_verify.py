"""verify / verify_ml: probe + CLIP + SAM 3 scores combined into CandidateScores (fakes, no torch)."""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest

from tests.waste.synth_waste import make_candidate, paint, soil
from vineyard.perception.waste.decide import DecideParams
from vineyard.perception.waste.sam3_adapter import XYXY, BackendOutput, Instance, Sam3Params, Sam3Status
from vineyard.perception.waste.types import Candidate, Category, Detector, RejectReason
from vineyard.perception.waste.verify import (
    VerifyStatus,
    combine_scores,
    load_verifier,
    rank_and_select,
    verify_status,
    waste_version_string,
)
from vineyard.perception.waste.verify_ml import (
    CropParams,
    ModelVerifier,
    ProbeHandle,
    SamPlanParams,
    build_verifier,
    clip_probe_scores,
    plan_sam_jobs,
)

TILE = 2048
BLUE = (20, 40, 230)
SAM_CROP = 512


def tile_with(cands: Sequence[Candidate], strengths: Sequence[float]) -> np.ndarray:
    """Soil tile with each candidate box painted blue scaled by its strength (0 = soil)."""
    img = soil(TILE)
    for c, s in zip(cands, strengths, strict=True):
        if s > 0:
            colour = tuple(
                int(round(v * s + sv * (1 - s))) for v, sv in zip(BLUE, (120, 100, 80), strict=True)
            )
            b = c.box
            img = paint(img, colour, slice(int(b.ytl), int(b.ybr)), slice(int(b.xtl), int(b.xbr)))
    return img


@dataclass
class FakeEmbedder:
    """Embedding = (blue fraction of the centre, 1 - it); zero-shot pos_p = blue fraction."""

    batches: list[tuple[int, ...]] = field(default_factory=list)

    def embed(self, crops_u8: np.ndarray) -> np.ndarray:
        self.batches.append(crops_u8.shape)
        n, h, w, _ = crops_u8.shape
        centre = crops_u8[:, h // 2 - 4 : h // 2 + 4, w // 2 - 4 : w // 2 + 4].astype(np.float64)
        blue = np.clip((centre[..., 2] - centre[..., 0]).mean(axis=(1, 2)) / 210.0, 0.0, 1.0)
        return np.stack([blue, 1.0 - blue], axis=1)

    def zero_shot(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return emb[:, 0], emb[:, 0] - 0.5, np.zeros(len(emb), np.int64)


def probe_handle(allowed: bool = True, threshold: float = 0.93) -> ProbeHandle:
    return ProbeHandle(
        score=lambda e: e[:, 0].copy(), auto_threshold=threshold, auto_allowed=allowed, version="v1"
    )


@dataclass
class FakeSam:
    """Counts every candidate: one instance equal to the blue pixels of the crop, score 0.7."""

    score: float = 0.7
    cost_s: float = 0.0
    now: list[float] = field(default_factory=lambda: [0.0])
    calls: list[tuple[XYXY, ...]] = field(default_factory=list)

    def detect(
        self, crop_rgb: np.ndarray, prompts: Sequence[str], negatives: Sequence[XYXY]
    ) -> BackendOutput:
        self.calls.append(tuple(negatives))
        self.now[0] += self.cost_s
        blue = (crop_rgb[..., 2].astype(int) - crop_rgb[..., 0]) > 50
        inst = (Instance(self.score, blue),) if blue.any() else ()
        return BackendOutput(
            per_prompt=tuple(inst for _ in prompts), vision_s=0.0, prompt_s=(0.0,) * len(prompts)
        )


@pytest.fixture
def cands() -> tuple[Candidate, ...]:
    return (
        make_candidate(centroid=(500.0, 500.0), key="strong"),
        make_candidate(centroid=(1500.0, 600.0), key="medium"),
        make_candidate(centroid=(900.0, 1500.0), key="soil"),
    )


@pytest.fixture
def reader(cands):
    img = tile_with(cands, (1.0, 0.5, 0.0))
    return lambda tile_id: img


def _loader(fake: FakeSam):
    return lambda source, params: fake


def _build(cfg, reader, *, fake: FakeSam | None = None, probe=None, embedder=None, tmp: Path, **kw):
    fake = fake or FakeSam()
    return build_verifier(
        cfg.waste,
        cfg.grid.gsd_m,
        tmp,
        reader,
        embedder=embedder if embedder is not None else FakeEmbedder(),
        probe=probe if probe is not None else probe_handle(),
        sam_loader=_loader(fake),
        sam_lister=lambda s: ["model.safetensors"],
        clock=lambda: fake.now[0],
        **kw,
    )


# ------------------------------------------------------------------ combine / status


def test_combine_scores_and_validation() -> None:
    s = combine_scores("k", 0.2, 0.9, 0.8, 0.3, None)
    assert (s.probe_p, s.clip_pos_p, s.clip_margin, s.sam_score, s.category) == (
        0.9,
        0.8,
        0.3,
        None,
        Category.UNKNOWN,
    )
    with pytest.raises(ValueError, match="probe_p"):
        combine_scores("k", 0.2, math.nan, None, None, None)
    with pytest.raises(ValueError, match="clip_pos_p"):
        combine_scores("k", 0.2, None, 1.5, None, None)
    with pytest.raises(ValueError, match="clip_margin"):
        combine_scores("k", 0.2, None, None, -1.5, None)


def test_verify_status_levels() -> None:
    ok = Sam3Status(True, "facebook/sam3", "ok", (), "cpu", 1.0)
    off = Sam3Status(False, None, "disabled", (), "cpu", 0.0)
    assert verify_status(True, True, ok).level == "L0"
    assert verify_status(True, True, off).level == "L1"
    assert verify_status(False, True, off).level == "L2"
    assert verify_status(False, False, None).level == "L3"
    s = verify_status(True, True, ok)
    assert (s.probe, s.clip, s.sam) == (True, True, True)
    assert waste_version_string(s) == "probe+clip+sam3@1"
    assert "sam3: disabled" in verify_status(True, True, off).reason


# ------------------------------------------------------------------ CLIP / probe scoring


def test_clip_probe_scores_batches_and_values(cfg, cands, reader) -> None:
    emb = FakeEmbedder()
    p = replace(CropParams.from_config(cfg.waste.probe, TILE), batch=2)
    got = clip_probe_scores(cands, reader, emb, probe_handle(), p)
    assert [b[0] for b in emb.batches] == [2, 1]
    assert all(b[1:] == (224, 224, 3) for b in emb.batches)
    assert got["strong"].probe_p == pytest.approx(1.0, abs=0.05)
    assert 0.3 < got["medium"].probe_p < 0.7
    assert got["soil"].probe_p == pytest.approx(0.0, abs=0.05)
    assert got["strong"].clip_margin == pytest.approx(got["strong"].clip_pos_p - 0.5)


def test_bad_embedder_output_raises(cfg, cands, reader) -> None:
    class Broken(FakeEmbedder):
        def embed(self, crops_u8: np.ndarray) -> np.ndarray:
            return np.zeros((len(crops_u8) + 1, 2))

    with pytest.raises(ValueError, match="embed"):
        clip_probe_scores(cands, reader, Broken(), None, CropParams.from_config(cfg.waste.probe, TILE))


# ------------------------------------------------------------------ SAM job planning


def _plan(cfg, **kw) -> SamPlanParams:
    base = SamPlanParams.from_config(cfg.waste, TILE)
    return replace(base, **kw)


def test_plan_orders_by_gate_and_skips_low(cfg, cands) -> None:
    gate = {"strong": 0.95, "medium": 0.5, "soil": 0.1}
    jobs = plan_sam_jobs(cands, gate, (), _plan(cfg))
    assert [j.cand_key for j in jobs] == ["strong", "medium"]
    j = jobs[0]
    assert j.window.side == SAM_CROP and (j.window.x0, j.window.y0) == (244, 244)
    assert j.cand_box == (492.0 - 244, 492.0 - 244, 508.0 - 244, 508.0 - 244)


def test_plan_dedupes_overlapping_candidates(cfg) -> None:
    a = make_candidate(centroid=(500.0, 500.0), key="a")
    b = make_candidate(centroid=(503.0, 500.0), key="b")
    c = replace(make_candidate(centroid=(503.0, 500.0), key="c"), tile_id="siret3_r000_c000")
    jobs = plan_sam_jobs((a, b, c), {"a": 0.6, "b": 0.9, "c": 0.5}, (), _plan(cfg))
    assert [j.cand_key for j in jobs] == ["b", "c"]


def test_plan_negatives_from_pool(cfg, cands) -> None:
    tube = replace(
        make_candidate(centroid=(530.0, 500.0), is_white=True, area_px=30, half=3.0),
        reject_reason=RejectReason.NEAR_AXIS,
    )
    jobs = plan_sam_jobs(cands[:1], {"strong": 0.9}, (tube,), _plan(cfg))
    assert jobs[0].negatives == ((527.0 - 244, 497.0 - 244, 533.0 - 244, 503.0 - 244),)


# ------------------------------------------------------------------ ModelVerifier end to end


def test_model_verifier_scores(cfg, cands, reader, tmp_path: Path) -> None:
    fake = FakeSam()
    verifier, status, sam_status = _build(cfg, reader, fake=fake, tmp=tmp_path)
    assert status.level == "L0" and sam_status.available
    scores = {s.cand_key: s for s in verifier.score(cands)}
    assert scores["strong"].sam_score == pytest.approx(0.7) and scores["strong"].category == Category.DEBRIS
    assert scores["medium"].sam_score == pytest.approx(0.7)
    assert scores["soil"].sam_score is None  # below min_probe_to_verify
    assert scores["soil"].rule_score is not None
    assert len(fake.calls) == 2


def test_budget_leaves_the_rest_unverified(cfg, cands, reader, tmp_path: Path) -> None:
    fake = FakeSam(cost_s=10.0)
    waste = cfg.waste.model_copy(update={"sam3": cfg.waste.sam3.model_copy(update={"time_budget_s": 10.0})})
    verifier, _, _ = build_verifier(
        waste,
        cfg.grid.gsd_m,
        tmp_path,
        reader,
        embedder=FakeEmbedder(),
        probe=probe_handle(),
        sam_loader=_loader(fake),
        sam_lister=lambda s: ["model.safetensors"],
        clock=lambda: fake.now[0],
    )
    scores = {s.cand_key: s for s in verifier.score(cands)}
    assert scores["strong"].sam_score is not None and scores["medium"].sam_score is None


def test_rank_and_select_auto_disabled_by_default(cfg, cands, reader, tmp_path: Path) -> None:
    verifier, status, _ = _build(cfg, reader, tmp=tmp_path)
    ranked = rank_and_select(
        cands, verifier, status, DecideParams.from_config(cfg.waste.decide), cfg.waste.nms_iou
    )
    assert not any(d.auto for d in ranked.decisions.values())
    assert ranked.review_keys[:2] == ("strong", "medium")
    assert "soil" not in ranked.review_keys


def test_rank_and_select_auto_when_enabled(cfg, cands, reader, tmp_path: Path) -> None:
    verifier, status, _ = _build(cfg, reader, tmp=tmp_path)
    p = replace(DecideParams.from_config(cfg.waste.decide), auto_accept_enabled=True)
    ranked = rank_and_select(cands, verifier, status, p, cfg.waste.nms_iou)
    strong = ranked.decisions["strong"]
    assert strong.auto and strong.detector == Detector.SAM3 and "strong" not in ranked.review_keys
    assert not ranked.decisions["medium"].auto


def test_probe_without_recall_never_auto(cfg, cands, reader, tmp_path: Path) -> None:
    verifier, status, _ = _build(cfg, reader, probe=probe_handle(allowed=False), tmp=tmp_path)
    assert verifier.probe_auto_min == math.inf
    p = replace(DecideParams.from_config(cfg.waste.decide), auto_accept_enabled=True)
    ranked = rank_and_select(cands, verifier, status, p, cfg.waste.nms_iou)
    assert not any(d.auto for d in ranked.decisions.values())


def test_rejected_candidates_feed_the_negatives(cfg, cands, reader, tmp_path: Path) -> None:
    tube = replace(
        make_candidate(centroid=(530.0, 500.0), is_white=True, area_px=30, half=3.0, key="tube"),
        reject_reason=RejectReason.NEAR_AXIS,
    )
    fake = FakeSam()
    verifier, status, _ = _build(cfg, reader, fake=fake, tmp=tmp_path)
    rank_and_select((*cands, tube), verifier, status, DecideParams.from_config(cfg.waste.decide), 0.3)
    assert fake.calls[0] == ((283.0, 253.0, 289.0, 259.0),)


# ------------------------------------------------------------------ degradation


def test_build_without_models_is_rule_only_plus_sam(cfg, cands, reader, tmp_path: Path) -> None:
    fake = FakeSam()
    verifier, status, _ = build_verifier(
        cfg.waste,
        cfg.grid.gsd_m,
        tmp_path,
        reader,
        sam_loader=_loader(fake),
        sam_lister=lambda s: ["model.safetensors"],
    )
    assert status.level == "L3" and verifier.probe_auto_min is None
    scores = {s.cand_key: s for s in verifier.score(cands)}
    assert all(s.probe_p is None and s.clip_pos_p is None for s in scores.values())
    assert scores["strong"].sam_score is not None  # rule saliency gates SAM when there is no probe


def test_sam_disabled_by_config(cfg, cands, reader, tmp_path: Path) -> None:
    waste = cfg.waste.model_copy(update={"sam3": cfg.waste.sam3.model_copy(update={"enabled": False})})
    verifier, status, sam_status = build_verifier(
        waste, cfg.grid.gsd_m, tmp_path, reader, embedder=FakeEmbedder(), probe=probe_handle()
    )
    assert status.level == "L1" and not sam_status.available and verifier.sam is None
    assert all(s.sam_score is None for s in verifier.score(cands))


def test_probe_disabled_by_config_drops_probe_and_clip(cfg, reader, tmp_path: Path) -> None:
    waste = cfg.waste.model_copy(update={"probe": cfg.waste.probe.model_copy(update={"enabled": False})})
    verifier, status, _ = build_verifier(
        waste,
        cfg.grid.gsd_m,
        tmp_path,
        reader,
        embedder=FakeEmbedder(),
        probe=probe_handle(),
        sam_loader=_loader(FakeSam()),
        sam_lister=lambda s: ["model.safetensors"],
    )
    assert verifier.probe is None and verifier.embedder is None
    assert "probe disabled" in status.reason


def test_probe_needs_an_embedder(cfg, reader) -> None:
    with pytest.raises(ValueError, match="embedder"):
        ModelVerifier(
            rule=load_verifier(cfg.waste, cfg.grid.gsd_m)[0],
            read_tile=reader,
            crop=CropParams.from_config(cfg.waste.probe, TILE),
            sam_plan=SamPlanParams.from_config(cfg.waste, TILE),
            probe=probe_handle(),
        )


def test_rule_only_path_unchanged(cfg) -> None:
    verifier, status = load_verifier(cfg.waste, cfg.grid.gsd_m)
    assert isinstance(status, VerifyStatus) and status.level == "L3" and verifier.probe_auto_min is None
    (s,) = verifier.score([make_candidate()], ())
    assert s.sam_score is None and s.rule_score is not None


def test_sam_params_follow_config(cfg, tmp_path: Path) -> None:
    p = Sam3Params.from_config(cfg.waste.sam3, tmp_path)
    plan = SamPlanParams.from_config(cfg.waste, TILE)
    assert plan.crop_px == p.crop_px and plan.min_gate == cfg.waste.sam3.min_probe_to_verify


def test_verify_modules_are_torch_free() -> None:
    code = (
        "import sys; import vineyard.perception.waste.verify, vineyard.perception.waste.verify_ml, "
        "vineyard.perception.waste.decide; print('torch' in sys.modules or 'transformers' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
