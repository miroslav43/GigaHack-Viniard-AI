"""transformers SAM 3 backend (A§4.9, design 03 W6): Sam3Model + Sam3Processor, fp32, eval, on CPU by config.

Imported only by sam3_adapter's default loader (main process, never the pipeline workers). Per crop the
vision encoder runs once and its features are reused for every text prompt; the text embeddings of the
prompts are computed once at load time. Negative boxes enter as `input_boxes` with label 0.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor

from vineyard.perception.waste.sam3_adapter import XYXY, BackendOutput, Instance, Sam3Params, box_prompt

DTYPE = torch.float32


def _threads(requested: int) -> int:
    return requested if requested > 0 else max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class TextPrompt:
    embeds: Any  # BaseModelOutputWithPooling of Sam3Model.get_text_features
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class TransformersSam3Backend:
    model: Sam3Model
    processor: Sam3Processor
    texts: Mapping[str, TextPrompt]
    params: Sam3Params

    def _encode(self, crop_rgb: np.ndarray, negatives: Sequence[XYXY]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Pixel values plus the normalised negative boxes (one processor call)."""
        prompt = box_prompt(negatives, self.params.max_negative_boxes)
        boxes = {} if prompt is None else {"input_boxes": prompt[0], "input_boxes_labels": prompt[1]}
        enc = self.processor(images=crop_rgb, return_tensors="pt", **boxes)
        geometry = {k: enc[k].to(self.params.device) for k in boxes}
        return enc["pixel_values"].to(self.params.device, DTYPE), geometry

    def _instances(self, outputs: Any, size: tuple[int, int]) -> tuple[Instance, ...]:
        res = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.params.threshold,
            mask_threshold=self.params.mask_threshold,
            target_sizes=[size],
        )[0]
        scores = res["scores"].detach().cpu().numpy()
        masks = res["masks"].detach().cpu().numpy().astype(bool)
        return tuple(Instance(float(s), m) for s, m in zip(scores, masks, strict=True))

    def detect(
        self, crop_rgb: np.ndarray, prompts: Sequence[str], negatives: Sequence[XYXY]
    ) -> BackendOutput:
        unknown = [t for t in prompts if t not in self.texts]
        if unknown:
            raise KeyError(f"SAM 3 prompts without precomputed text embeddings: {unknown}")
        size = (int(crop_rgb.shape[0]), int(crop_rgb.shape[1]))
        with torch.inference_mode():
            t0 = time.perf_counter()
            pixels, geometry = self._encode(crop_rgb, negatives)
            vision = self.model.get_vision_features(pixel_values=pixels)
            vision_s = time.perf_counter() - t0
            per, secs = [], []
            for prompt in prompts:
                t1 = time.perf_counter()
                text = self.texts[prompt]
                out = self.model(
                    vision_embeds=vision,
                    text_embeds=text.embeds,
                    attention_mask=text.attention_mask,
                    **geometry,
                )
                per.append(self._instances(out, size))
                secs.append(time.perf_counter() - t1)
        return BackendOutput(per_prompt=tuple(per), vision_s=vision_s, prompt_s=tuple(secs))


def _text_prompts(model: Sam3Model, processor: Sam3Processor, p: Sam3Params) -> Mapping[str, TextPrompt]:
    out: dict[str, TextPrompt] = {}
    with torch.inference_mode():
        for prompt in p.text_prompts:
            enc = processor(text=prompt, return_tensors="pt")
            ids, mask = enc["input_ids"].to(p.device), enc["attention_mask"].to(p.device)
            out[prompt] = TextPrompt(model.get_text_features(input_ids=ids, attention_mask=mask), mask)
    return MappingProxyType(out)


def load_transformers_backend(source: str, p: Sam3Params) -> TransformersSam3Backend:
    """Sam3Model/Sam3Processor from a local dir or hub id (no download unless allow_download)."""
    torch.set_num_threads(_threads(p.torch_threads))
    local_only = not p.allow_download
    processor = Sam3Processor.from_pretrained(source, local_files_only=local_only)
    model = Sam3Model.from_pretrained(source, dtype=DTYPE, local_files_only=local_only)
    model = model.to(p.device).eval()
    return TransformersSam3Backend(model, processor, _text_prompts(model, processor, p), p)
