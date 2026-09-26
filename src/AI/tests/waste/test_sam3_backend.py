"""sam3_backend plumbing with duck-typed fakes of Sam3Model / Sam3Processor (imports torch, no weights)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
backend_mod = pytest.importorskip("vineyard.perception.waste.sam3_backend")

from vineyard.perception.waste.sam3_adapter import Sam3Params  # noqa: E402

CROP = 64
PIX = 8


@dataclass
class FakeProcessor:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self, images=None, text=None, return_tensors=None, input_boxes=None, input_boxes_labels=None
    ):
        self.calls.append(
            {"images": images is not None, "text": text, "boxes": input_boxes, "labels": input_boxes_labels}
        )
        out: dict[str, Any] = {}
        if images is not None:
            out["pixel_values"] = torch.zeros(1, 3, PIX, PIX)
        if text is not None:
            out["input_ids"] = torch.ones(1, 4, dtype=torch.long)
            out["attention_mask"] = torch.ones(1, 4, dtype=torch.long)
        if input_boxes is not None:
            out["input_boxes"] = torch.tensor(input_boxes, dtype=torch.float32)
            out["input_boxes_labels"] = torch.tensor(input_boxes_labels, dtype=torch.long)
        return out

    def post_process_instance_segmentation(self, outputs, threshold, mask_threshold, target_sizes):
        h, w = target_sizes[0]
        keep = outputs.score > threshold
        scores = torch.tensor([outputs.score] if keep else [])
        masks = torch.ones(1 if keep else 0, h, w, dtype=torch.long)
        return [{"scores": scores, "masks": masks, "boxes": torch.zeros(len(scores), 4)}]


@dataclass
class FakeModel:
    score: float = 0.8
    vision_calls: int = 0
    forwards: list[dict[str, Any]] = field(default_factory=list)

    def get_vision_features(self, pixel_values):
        self.vision_calls += 1
        return SimpleNamespace(tag="vision", shape=tuple(pixel_values.shape))

    def get_text_features(self, input_ids, attention_mask):
        return SimpleNamespace(pooler_output=torch.zeros(1, 4, 2))

    def __call__(self, vision_embeds, text_embeds, attention_mask, **geometry):
        self.forwards.append({"vision": vision_embeds.tag, "geometry": geometry, "mask": attention_mask})
        return SimpleNamespace(score=self.score)

    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def params(cfg, tmp_path: Path) -> Sam3Params:
    return Sam3Params.from_config(cfg.waste.sam3, tmp_path)


def make_backend(params: Sam3Params, score: float = 0.8):
    model, proc = FakeModel(score=score), FakeProcessor()
    texts = backend_mod._text_prompts(model, proc, params)
    return backend_mod.TransformersSam3Backend(model, proc, texts, params), model, proc


def test_vision_once_prompts_separately_negatives_label_zero(params: Sam3Params) -> None:
    backend, model, proc = make_backend(params)
    negs = [(float(i), 0.0, float(i) + 4.0, 4.0) for i in range(5)]
    out = backend.detect(np.zeros((CROP, CROP, 3), np.uint8), params.text_prompts, negs)
    assert model.vision_calls == 1 and len(model.forwards) == len(params.text_prompts)
    geo = model.forwards[0]["geometry"]
    assert geo["input_boxes"].shape == (1, 3, 4)
    assert geo["input_boxes_labels"].tolist() == [[0, 0, 0]]
    assert len(out.per_prompt) == len(params.text_prompts) == len(out.prompt_s)
    (inst,) = out.per_prompt[0]
    assert inst.score == pytest.approx(0.8) and inst.mask.dtype == bool and inst.mask.shape == (CROP, CROP)
    image_calls = [c for c in proc.calls if c["images"]]
    assert len(image_calls) == 1 and image_calls[0]["labels"] == [[0, 0, 0]]


def test_no_negatives_means_text_only(params: Sam3Params) -> None:
    backend, model, _ = make_backend(params, score=0.1)
    out = backend.detect(np.zeros((CROP, CROP, 3), np.uint8), params.text_prompts[:2], [])
    assert model.forwards[0]["geometry"] == {}
    assert out.per_prompt == ((), ())


def test_unknown_prompt_raises(params: Sam3Params) -> None:
    backend, _, _ = make_backend(params)
    with pytest.raises(KeyError, match="car"):
        backend.detect(np.zeros((CROP, CROP, 3), np.uint8), ["car"], [])


def test_threads() -> None:
    assert backend_mod._threads(4) == 4
    assert backend_mod._threads(0) == max(1, os.cpu_count() or 1)


def test_loader_is_local_only_fp32(params: Sam3Params, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class M:
        @classmethod
        def from_pretrained(cls, source, **kw):
            seen["model"] = (source, kw)
            return FakeModel()

    class P:
        @classmethod
        def from_pretrained(cls, source, **kw):
            seen["proc"] = (source, kw)
            return FakeProcessor()

    monkeypatch.setattr(backend_mod, "Sam3Model", M)
    monkeypatch.setattr(backend_mod, "Sam3Processor", P)
    monkeypatch.setattr(backend_mod.torch, "set_num_threads", lambda n: seen.setdefault("threads", n))
    b = backend_mod.load_transformers_backend("some/dir", params)
    assert seen["model"] == ("some/dir", {"dtype": torch.float32, "local_files_only": True})
    assert seen["proc"] == ("some/dir", {"local_files_only": True})
    assert seen["threads"] >= 1 and set(b.texts) == set(params.text_prompts)
