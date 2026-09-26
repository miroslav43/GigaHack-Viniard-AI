"""sam3_adapter: preflight, graceful loading, overlap rule, negative boxes, budget (fakes only; the real
model check is marked slow and skipped when the local weights are absent)."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx
import numpy as np
import pytest
from huggingface_hub.errors import GatedRepoError

from tests.waste.synth_waste import make_candidate
from vineyard.perception.waste.crops import CropWindow
from vineyard.perception.waste.sam3_adapter import (
    XYXY,
    BackendOutput,
    Instance,
    Sam3Params,
    Sam3Reason,
    Sam3Verifier,
    SamJob,
    box_prompt,
    has_transformers_weights,
    list_files,
    load_sam3,
    mask_overlap,
    model_sources,
    select_negative_boxes,
    verify_with_budget,
)
from vineyard.perception.waste.types import Category, RejectReason

CROP = 512
PROMPTS = ("trash", "plastic bag", "plastic bottle", "tire", "rubbish heap")
CAND_BOX: XYXY = (240.0, 240.0, 260.0, 260.0)  # 20 x 20 = 400 px


@pytest.fixture
def params(cfg, tmp_path: Path) -> Sam3Params:
    return Sam3Params.from_config(cfg.waste.sam3, tmp_path)


def crop() -> np.ndarray:
    return np.zeros((CROP, CROP, 3), np.uint8)


def mask_covering(frac: float, box: XYXY = CAND_BOX, extra_px: int = 0) -> np.ndarray:
    """Mask covering `frac` of the candidate box (full rows of the box) plus `extra_px` rows outside."""
    x0, y0, x1, y1 = (int(v) for v in box)
    m = np.zeros((CROP, CROP), bool)
    rows = round(frac * (y1 - y0))
    m[y0 : y0 + rows, x0:x1] = True
    m[:extra_px, :] = True
    return m


@dataclass
class FakeBackend:
    """Returns the same instances for every prompt unless `by_prompt` names the prompt."""

    instances: tuple[Instance, ...] = ()
    by_prompt: dict[str, tuple[Instance, ...]] = field(default_factory=dict)
    calls: list[tuple[tuple[str, ...], tuple[XYXY, ...]]] = field(default_factory=list)

    def detect(
        self, crop_rgb: np.ndarray, prompts: Sequence[str], negatives: Sequence[XYXY]
    ) -> BackendOutput:
        self.calls.append((tuple(prompts), tuple(negatives)))
        per = tuple(self.by_prompt.get(p, self.instances) for p in prompts)
        return BackendOutput(per_prompt=per, vision_s=0.5, prompt_s=tuple(0.1 for _ in prompts))


# ------------------------------------------------------------------ preflight / sources


def test_sam31_file_list_has_no_transformers_weights() -> None:
    assert not has_transformers_weights(["config.json", "processor_config.json", "sam3.1_multiplex.pt"])
    assert has_transformers_weights(["config.json", "model.safetensors"])
    assert has_transformers_weights(["model.safetensors.index.json"])
    assert has_transformers_weights(["pytorch_model.bin"])


def test_model_sources_prefers_local_mirror(tmp_path: Path) -> None:
    (tmp_path / "hf" / "sam3").mkdir(parents=True)
    ids = ("facebook/sam3.1", "facebook/sam3")
    assert model_sources(ids, tmp_path) == ("facebook/sam3.1", str(tmp_path / "hf" / "sam3"), "facebook/sam3")
    local = tmp_path / "mine"
    local.mkdir()
    assert model_sources((str(local),), tmp_path) == (str(local),)


def test_list_files_of_a_local_dir(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    (tmp_path / ".cache").mkdir()
    assert list_files(str(tmp_path)) == ("config.json", "model.safetensors")


def test_params_from_config(params: Sam3Params, cfg) -> None:
    assert params.text_prompts == PROMPTS
    assert params.prompt_categories["tire"] == Category.TYRE
    assert params.max_negative_boxes == 3 and params.threshold == 0.4
    assert params.device == "cpu" and params.sources[-1] == "facebook/sam3"
    with pytest.raises(ValueError, match="prompt_categories"):
        replace(params, text_prompts=(*PROMPTS, "car"))


# ------------------------------------------------------------------ load_sam3


def _gated(source: str) -> GatedRepoError:
    response = httpx.Response(403, request=httpx.Request("GET", f"https://huggingface.co/{source}"))
    return GatedRepoError(f"{source}: access to model is restricted", response=response)


def test_skip_without_weights_then_all_fail(params: Sam3Params) -> None:
    p = replace(params, sources=("facebook/sam3.1", "facebook/sam3"))
    files = {"facebook/sam3.1": ["config.json", "processor_config.json", "sam3.1_multiplex.pt"]}
    loaded: list[str] = []

    def lister(source: str) -> Sequence[str]:
        return files.get(source, ["config.json", "model.safetensors"])

    def loader(source: str, _: Sam3Params) -> FakeBackend:
        loaded.append(source)
        raise _gated(source)

    verifier, status = load_sam3(p, loader=loader, lister=lister)
    assert verifier is None and not status.available and status.model_id is None
    assert loaded == ["facebook/sam3"]
    assert [t[0] for t in status.tried] == ["facebook/sam3.1", "facebook/sam3"]
    assert status.tried[0][1].startswith(Sam3Reason.NO_WEIGHTS)
    assert status.tried[1][1].startswith(Sam3Reason.GATED)
    assert status.reason.startswith("no SAM 3 model loaded")


def test_gated_first_then_next_id_loads(params: Sam3Params) -> None:
    p = replace(params, sources=("a/one", "b/two"))
    ticks = iter([0.0, 12.5])

    def loader(source: str, _: Sam3Params) -> FakeBackend:
        if source == "a/one":
            raise _gated(source)
        return FakeBackend()

    verifier, status = load_sam3(
        p, loader=loader, lister=lambda s: ["model.safetensors"], clock=lambda: next(ticks)
    )
    assert isinstance(verifier, Sam3Verifier)
    assert status.available and status.model_id == "b/two" and status.reason == Sam3Reason.OK
    assert status.load_s == pytest.approx(12.5)
    assert len(status.tried) == 2


def test_listing_error_is_recorded_and_next_tried(params: Sam3Params) -> None:
    p = replace(params, sources=("offline/repo", "b/two"))

    def lister(source: str) -> Sequence[str]:
        if source == "offline/repo":
            raise OSError("connection refused")
        return ["model.safetensors"]

    verifier, status = load_sam3(p, loader=lambda s, _: FakeBackend(), lister=lister)
    assert verifier is not None
    assert status.tried[0] == ("offline/repo", f"{Sam3Reason.LIST_FAILED}: OSError: connection refused")


def test_offline_hub_listing_is_recorded(params: Sam3Params) -> None:
    def lister(source: str) -> Sequence[str]:
        raise httpx.ConnectError("[Errno 8] nodename nor servname provided")

    verifier, status = load_sam3(replace(params, sources=("facebook/sam3",)), lister=lister)
    assert verifier is None and status.tried[0][1].startswith(f"{Sam3Reason.LIST_FAILED}: ConnectError")


def test_import_error_disables(params: Sam3Params) -> None:
    def loader(source: str, _: Sam3Params) -> FakeBackend:
        raise ImportError("No module named 'transformers'")

    p = replace(params, sources=("x/y",))
    verifier, status = load_sam3(p, loader=loader, lister=lambda s: ["model.safetensors"])
    assert verifier is None and status.tried[0][1].startswith(Sam3Reason.IMPORT_FAILED)


def test_programming_errors_are_not_swallowed(params: Sam3Params) -> None:
    def loader(source: str, _: Sam3Params) -> FakeBackend:
        raise TypeError("bug")

    with pytest.raises(TypeError):
        load_sam3(replace(params, sources=("x/y",)), loader=loader, lister=lambda s: ["model.safetensors"])


def test_disabled_by_config_never_lists(params: Sam3Params) -> None:
    def boom(source: str) -> Sequence[str]:
        raise AssertionError("must not be called")

    verifier, status = load_sam3(replace(params, enabled=False), loader=None, lister=boom)
    assert verifier is None and status.reason == Sam3Reason.DISABLED and status.tried == ()


def test_no_sources(params: Sam3Params) -> None:
    verifier, status = load_sam3(replace(params, sources=()))
    assert verifier is None and status.reason == Sam3Reason.NO_IDS


# ------------------------------------------------------------------ overlap rule and verify


def test_mask_overlap_fraction() -> None:
    assert mask_overlap(mask_covering(0.25), CAND_BOX) == pytest.approx(0.25)
    assert mask_overlap(np.zeros((CROP, CROP), bool), CAND_BOX) == 0.0
    with pytest.raises(ValueError, match="outside"):
        mask_overlap(mask_covering(1.0), (600.0, 600.0, 610.0, 610.0))


@pytest.mark.parametrize(("frac", "counted"), [(0.15, False), (0.25, True)])
def test_overlap_threshold(params: Sam3Params, frac: float, counted: bool) -> None:
    v = Sam3Verifier(FakeBackend((Instance(0.8, mask_covering(frac)),)), params)
    r = v.verify(crop(), CAND_BOX, ())
    assert (r.score == pytest.approx(0.8)) is counted
    assert r.per_prompt[0].n_instances == 1 and r.per_prompt[0].n_counted == int(counted)
    assert r.category == (Category.DEBRIS if counted else Category.UNKNOWN)


def test_huge_masks_are_ignored(params: Sam3Params) -> None:
    big = Instance(0.9, mask_covering(1.0, extra_px=300))  # covers > 25 % of the crop
    r = Sam3Verifier(FakeBackend((big,)), params).verify(crop(), CAND_BOX, ())
    assert r.score == 0.0 and r.prompt is None


def test_best_prompt_and_category(params: Sam3Params) -> None:
    fake = FakeBackend(
        by_prompt={
            "plastic bag": (Instance(0.55, mask_covering(1.0)),),
            "tire": (Instance(0.7, mask_covering(0.5)), Instance(0.95, mask_covering(0.1))),
        }
    )
    r = Sam3Verifier(fake, params).verify(crop(), CAND_BOX, ())
    assert (r.score, r.prompt, r.category) == (pytest.approx(0.7), "tire", Category.TYRE)
    assert r.overlap == pytest.approx(0.5)
    assert [ps.prompt for ps in r.per_prompt] == list(PROMPTS)
    assert r.per_prompt[1].score == pytest.approx(0.55)
    assert r.seconds == pytest.approx(0.5 + 5 * 0.1)


def test_negatives_are_capped(params: Sam3Params) -> None:
    fake = FakeBackend()
    negs = tuple((float(10 * i), 0.0, float(10 * i + 5), 5.0) for i in range(5))
    r = Sam3Verifier(fake, params).verify(crop(), CAND_BOX, negs)
    assert fake.calls[0][1] == negs[:3] and r.n_negatives == 3
    assert fake.calls[0][0] == PROMPTS


def test_verify_validates_inputs(params: Sam3Params) -> None:
    v = Sam3Verifier(FakeBackend(), params)
    with pytest.raises(ValueError, match="uint8"):
        v.verify(np.zeros((CROP, CROP, 3), np.float32), CAND_BOX, ())
    with pytest.raises(ValueError, match="outside"):
        v.verify(crop(), (520.0, 500.0, 540.0, 520.0), ())
    assert v.verify(crop(), (500.0, 500.0, 520.0, 520.0), ()).score == 0.0  # partly inside: clipped


def test_backend_prompt_count_mismatch_raises(params: Sam3Params) -> None:
    class Short(FakeBackend):
        def detect(self, crop_rgb, prompts, negatives) -> BackendOutput:
            return BackendOutput(per_prompt=((),), vision_s=0.0, prompt_s=(0.0,))

    with pytest.raises(RuntimeError, match="prompts"):
        Sam3Verifier(Short(), params).verify(crop(), CAND_BOX, ())


def test_box_prompt_labels_are_negative() -> None:
    boxes, labels = box_prompt([(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)] * 3, 3) or ([], [])
    assert boxes == [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]]]
    assert labels == [[0, 0, 0]]
    assert box_prompt([], 3) is None
    assert box_prompt([(1.0, 2.0, 3.0, 4.0)], 0) is None


# ------------------------------------------------------------------ negative boxes


def _tube(cx: float, cy: float, reason: RejectReason | None = RejectReason.NEAR_AXIS, white: bool = True):
    c = make_candidate(centroid=(cx, cy), is_white=white, area_px=30, half=3.0)
    return replace(c, reject_reason=reason)


def test_select_negative_boxes_nearest_tubes_in_window() -> None:
    cand = make_candidate(centroid=(1000.0, 1000.0), half=8.0)
    window = CropWindow(744, 744, 512)
    pool = [
        _tube(1100, 1000),  # d = 100
        _tube(1020, 1000),  # d = 20
        _tube(1050, 1000, RejectReason.TUBE_SHAPE),  # d = 50
        _tube(1010, 1000),  # touches the candidate box -> excluded
        _tube(1030, 1000, RejectReason.BRIGHT_SOIL),  # not a tube reason
        _tube(1040, 1000, white=False),  # not white
        _tube(1300, 1000),  # outside the window
        _tube(1200, 1000, None),  # not rejected
        replace(_tube(1025, 1000), tile_id="siret3_r000_c000"),  # another tile
    ]
    got = select_negative_boxes(cand, pool, window, 3)
    assert got == (
        (1017.0 - 744, 997.0 - 744, 1023.0 - 744, 1003.0 - 744),
        (1047.0 - 744, 997.0 - 744, 1053.0 - 744, 1003.0 - 744),
        (1097.0 - 744, 997.0 - 744, 1103.0 - 744, 1003.0 - 744),
    )
    assert select_negative_boxes(cand, pool, window, 1) == got[:1]
    assert select_negative_boxes(cand, pool, window, 0) == ()


# ------------------------------------------------------------------ budget


def _job(i: int) -> SamJob:
    return SamJob(f"k{i}", "t", CropWindow(0, 0, CROP), CAND_BOX, ())


def test_budget_with_fake_clock(params: Sam3Params) -> None:
    now = [0.0]

    class Slow(FakeBackend):
        def detect(self, crop_rgb, prompts, negatives) -> BackendOutput:
            now[0] += 10.0
            return super().detect(crop_rgb, prompts, negatives)

    fake = Slow()
    out = verify_with_budget(
        Sam3Verifier(fake, params), [_job(i) for i in range(5)], lambda j: crop(), 30.0, clock=lambda: now[0]
    )
    assert [r is not None for r in out] == [True, True, True, False, False]
    assert len(fake.calls) == 3


def test_zero_budget_runs_nothing(params: Sam3Params) -> None:
    fake = FakeBackend()
    out = verify_with_budget(Sam3Verifier(fake, params), [_job(0)], lambda j: crop(), 0.0)
    assert out == (None,) and fake.calls == []


# ------------------------------------------------------------------ hygiene and the real model


def test_adapter_import_is_torch_free() -> None:
    code = (
        "import sys; import vineyard.perception.waste.sam3_adapter; "
        "print('torch' in sys.modules or 'transformers' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.slow
def test_real_sam3_on_a_crop(cfg) -> None:
    """Loads the local transformers weights (models/hf/sam3) on CPU and runs one crop."""
    p = Sam3Params.from_config(cfg.waste.sam3, cfg.paths.models_dir)
    local = [s for s in p.sources if Path(s).is_dir() and has_transformers_weights(list_files(s))]
    if not local:
        pytest.skip("local SAM 3 transformers weights not present")
    verifier, status = load_sam3(replace(p, sources=(local[0],)))
    assert verifier is not None, status
    img = np.full((CROP, CROP, 3), (120, 100, 80), np.uint8)
    img[240:260, 240:260] = (30, 60, 220)
    r = verifier.verify(img, CAND_BOX, ((100.0, 100.0, 106.0, 106.0),))
    assert len(r.per_prompt) == len(p.text_prompts) and 0.0 <= r.score <= 1.0
