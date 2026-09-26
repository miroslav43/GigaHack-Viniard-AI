"""CLIP crop embeddings for the waste probe (A§4.9 step 3, design 03 W3).

- `OpenClipEmbedder`: OpenCLIP ViT-B-32 (laion2b_s34b_b79k) on MPS when available, else CPU; batched,
  fp32, L2-normalised image features. Crops arrive as uint8 (N, 224, 224, 3) RGB made by
  `crops.extract_crop`, so preprocessing is only the model's mean/std normalisation.
- Zero-shot (ranking only, never a threshold): softmax over positive + negative prompts;
  margin = max positive prob - max negative prob; pos_p = total positive prob.
- `embed_crop_store`: embeddings of a crop store cached next to it, keyed by crops sha256 + model id.
- `probe_data_from_stores`: labelled stores + cached embeddings -> ProbeData for probe.train_probe.
torch / open_clip are imported lazily, inside the loader and the embed call only.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol

import numpy as np

from vineyard.errors import VineyardError
from vineyard.perception.waste.crop_store import CropStore, read_crop_store
from vineyard.perception.waste.probe import ProbeData
from vineyard.pipeline.atomic import atomic_path, atomic_write_json

if TYPE_CHECKING:
    from vineyard.config.sections_waste import ProbeConfig

U8_MAX: Final = 255.0
RGB: Final = 3
_NDIM_CROPS: Final = 4
_TAG_UNSAFE: Final = re.compile(r"[^A-Za-z0-9_.-]+")
EMB_PREFIX: Final = "emb_"
_EPS: Final = 1e-12
DEVICES: Final = ("mps", "cuda", "cpu")


class ClipEmbedError(VineyardError):
    """CLIP loading or embedding failure."""


@dataclass(frozen=True)
class ClipParams:
    model_name: str
    pretrained: str
    device: str
    batch: int
    crop_px: int
    pos_prompts: tuple[str, ...]
    neg_prompts: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg: ProbeConfig) -> ClipParams:
        return cls(
            model_name=cfg.clip_model,
            pretrained=cfg.clip_pretrained,
            device=cfg.device,
            batch=cfg.embed_batch,
            crop_px=cfg.crop_resize_px,
            pos_prompts=tuple(cfg.clip_pos_prompts),
            neg_prompts=tuple(cfg.clip_neg_prompts),
        )


class ZeroShot(NamedTuple):
    """Unpacks as (pos_p, margin, best_idx), the tuple verify_ml.ClipEmbedder.zero_shot returns."""

    pos_p: np.ndarray  # (N,) total softmax probability of the positive prompts
    margin: np.ndarray  # (N,) max positive prob - max negative prob
    best_idx: np.ndarray  # (N,) index into pos_prompts + neg_prompts


class ClipEmbedder(Protocol):
    model_id: str

    def embed(self, crops_u8: np.ndarray) -> np.ndarray: ...


# ------------------------------------------------------------------ pure helpers


def model_tag(model_name: str, pretrained: str) -> str:
    return f"open_clip:{model_name}/{pretrained}"


def clip_input(crops_u8: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """uint8 (N, H, W, 3) -> float32 (N, 3, H, W) normalised with the model's mean / std."""
    if crops_u8.dtype != np.uint8:
        raise ClipEmbedError("crops must be uint8", dtype=str(crops_u8.dtype))
    if crops_u8.ndim != _NDIM_CROPS or crops_u8.shape[-1] != RGB:
        raise ClipEmbedError("crops must have shape (N, H, W, 3)", shape=crops_u8.shape)
    x = crops_u8.astype(np.float32) / U8_MAX
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < _EPS):
        raise ClipEmbedError("cannot L2-normalise a zero embedding", n_zero=int((norms < _EPS).sum()))
    return (x / norms).astype(np.float32)


def batch_slices(n: int, size: int) -> Iterator[slice]:
    if size <= 0:
        raise ClipEmbedError("batch size must be positive", batch=size)
    return (slice(i, min(i + size, n)) for i in range(0, n, size))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def zero_shot_scores(emb: np.ndarray, text_emb: np.ndarray, n_pos: int, logit_scale: float) -> ZeroShot:
    """Softmax over all prompts (the first n_pos are the positive ones)."""
    if not 0 < n_pos < len(text_emb):
        raise ClipEmbedError("need 0 < n_pos < number of prompts", n_pos=n_pos, n_prompts=len(text_emb))
    probs = _softmax(logit_scale * emb.astype(np.float64) @ text_emb.astype(np.float64).T)
    pos, neg = probs[:, :n_pos], probs[:, n_pos:]
    return ZeroShot(
        pos_p=pos.sum(axis=1), margin=pos.max(axis=1) - neg.max(axis=1), best_idx=probs.argmax(axis=1)
    )


def select_device(pref: str, *, mps_available: bool, cuda_available: bool) -> str:
    """Preferred device if present, else CPU."""
    if pref not in DEVICES:
        raise ClipEmbedError("unknown device", device=pref, allowed=DEVICES)
    available = {"mps": mps_available, "cuda": cuda_available, "cpu": True}
    return pref if available[pref] else "cpu"


def embed_batched(embedder: ClipEmbedder, crops: np.ndarray, batch: int) -> np.ndarray:
    """(N, D) float32 embeddings, computed `batch` crops at a time (works on memmaps)."""
    parts = [
        np.asarray(embedder.embed(np.asarray(crops[s])), np.float32) for s in batch_slices(len(crops), batch)
    ]
    if not parts:
        raise ClipEmbedError("no crops to embed")
    return np.concatenate(parts)


# ------------------------------------------------------------------ OpenCLIP


@dataclass(frozen=True)
class OpenClipEmbedder:
    model: Any
    device: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    image_px: int
    text_emb: np.ndarray
    n_pos: int
    logit_scale: float
    model_id: str

    def embed(self, crops_u8: np.ndarray) -> np.ndarray:
        import torch

        if crops_u8.shape[1:3] != (self.image_px, self.image_px):
            raise ClipEmbedError("crop size does not match the model", shape=crops_u8.shape, px=self.image_px)
        x = torch.from_numpy(clip_input(crops_u8, self.mean, self.std)).to(self.device)
        with torch.inference_mode():
            feats = self.model.encode_image(x).float().cpu().numpy()
        return l2_normalise(feats)

    def zero_shot(self, emb: np.ndarray) -> ZeroShot:
        return zero_shot_scores(emb, self.text_emb, self.n_pos, self.logit_scale)


def _image_px(model: Any) -> int:
    size = model.visual.image_size
    return int(size[0] if isinstance(size, tuple | list) else size)


def _norm_stats(model: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    cfg = getattr(model.visual, "preprocess_cfg", None) or {}
    return tuple(cfg.get("mean", OPENAI_DATASET_MEAN)), tuple(cfg.get("std", OPENAI_DATASET_STD))


def _text_features(model: Any, name: str, prompts: Sequence[str], device: str) -> np.ndarray:
    import open_clip
    import torch

    tokens = open_clip.get_tokenizer(name)(list(prompts)).to(device)
    with torch.inference_mode():
        return l2_normalise(model.encode_text(tokens).float().cpu().numpy())


def load_openclip(p: ClipParams, device_pref: str | None = None) -> OpenClipEmbedder:
    """Load the OpenCLIP model (weights from the HF cache; the first call downloads them)."""
    import open_clip
    import torch

    device = select_device(
        device_pref or p.device,
        mps_available=torch.backends.mps.is_available(),
        cuda_available=torch.cuda.is_available(),
    )
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            p.model_name, pretrained=p.pretrained, device=device
        )
    except (RuntimeError, OSError, ValueError) as exc:
        raise ClipEmbedError(
            "OpenCLIP load failed", model=p.model_name, pretrained=p.pretrained, error=str(exc)
        ) from exc
    model.eval()
    mean, std = _norm_stats(model)
    text = _text_features(model, p.model_name, (*p.pos_prompts, *p.neg_prompts), device)
    return OpenClipEmbedder(
        model=model,
        device=device,
        mean=mean,
        std=std,
        image_px=_image_px(model),
        text_emb=text,
        n_pos=len(p.pos_prompts),
        logit_scale=float(model.logit_scale.exp().item()),
        model_id=model_tag(p.model_name, p.pretrained),
    )


# ------------------------------------------------------------------ embedding cache next to a crop store


def embeddings_path(store_dir: Path, model_id: str) -> Path:
    return Path(store_dir) / f"{EMB_PREFIX}{_TAG_UNSAFE.sub('_', model_id)}.npy"


def read_embeddings(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    meta_path = Path(path).with_suffix(".json")
    if not meta_path.is_file():
        raise ClipEmbedError("embedding meta file missing", path=str(meta_path))
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        emb = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ClipEmbedError("unreadable embeddings", path=str(path), error=str(exc)) from exc
    return emb, meta


def _cache_hit(path: Path, model_id: str, crops_sha: str) -> bool:
    meta_path = path.with_suffix(".json")
    if not path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClipEmbedError("unreadable embedding meta", path=str(meta_path), error=str(exc)) from exc
    return meta.get("model_id") == model_id and meta.get("crops_sha256") == crops_sha


def embed_crop_store(store_dir: Path, embedder: ClipEmbedder, batch: int) -> Path:
    """Embed every crop of a store (cached: skipped when crops and model are unchanged)."""
    store = read_crop_store(store_dir)
    path = embeddings_path(store_dir, embedder.model_id)
    if _cache_hit(path, embedder.model_id, store.crops_sha256):
        return path
    t0 = time.perf_counter()
    emb = embed_batched(embedder, store.crops, batch)
    elapsed = time.perf_counter() - t0
    with atomic_path(path) as tmp, tmp.open("wb") as fh:
        np.save(fh, emb, allow_pickle=False)
    meta = {
        "model_id": embedder.model_id,
        "n": int(len(emb)),
        "dim": int(emb.shape[1]),
        "crops_sha256": store.crops_sha256,
        "s_per_crop": elapsed / max(len(emb), 1),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    atomic_write_json(path.with_suffix(".json"), meta)
    return path


# ------------------------------------------------------------------ stores -> probe training data


def store_embeddings(store_dir: Path, model_id: str) -> tuple[CropStore, np.ndarray]:
    """A store and its cached embeddings (which must match the current crops and the model)."""
    store = read_crop_store(store_dir)
    path = embeddings_path(store_dir, model_id)
    if not _cache_hit(path, model_id, store.crops_sha256):
        raise ClipEmbedError(
            "embeddings missing or stale: run embed_crop_store", store=str(store_dir), model=model_id
        )
    emb, _ = read_embeddings(path)
    if len(emb) != len(store.records):
        raise ClipEmbedError("embedding count differs from the store", store=str(store_dir), n_emb=len(emb))
    return store, emb


def _train_source(store: CropStore, store_dir: Path) -> dict[str, Any]:
    keep = ("kind", "dataset", "doi", "licence", "attribution")
    return {
        "store": Path(store_dir).name,
        "n": len(store.records),
        **{k: store.meta[k] for k in keep if k in store.meta},
    }


def probe_data_from_stores(
    store_dirs: Sequence[Path], model_id: str
) -> tuple[ProbeData, tuple[dict[str, Any], ...]]:
    """Concatenate labelled stores (positives and negatives) into ProbeData + model-card train_data."""
    if not store_dirs:
        raise ClipEmbedError("no crop stores given")
    loaded = [store_embeddings(d, model_id) for d in store_dirs]
    records = [r for store, _ in loaded for r in store.records]
    data = ProbeData(
        emb=np.concatenate([emb for _, emb in loaded]),
        label=np.array([r.label for r in records], np.int8),
        group=np.array([r.group for r in records]),
        is_example=np.array([bool(r.meta.get("is_example", False)) and r.label == 0 for r in records]),
        survives=np.array([bool(r.meta.get("survives", False)) and r.label == 0 for r in records]),
        source=np.array([r.source for r in records]),
    )
    return data, tuple(_train_source(store, d) for (store, _), d in zip(loaded, store_dirs, strict=True))
