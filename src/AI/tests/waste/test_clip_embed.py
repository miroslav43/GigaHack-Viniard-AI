"""CLIP crop embedder (design 03 W3): pure-numpy helpers, batching, zero-shot margins, embedding cache."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from vineyard.perception.waste.clip_embed import (
    ClipEmbedError,
    ClipParams,
    batch_slices,
    clip_input,
    embed_batched,
    embed_crop_store,
    l2_normalise,
    model_tag,
    probe_data_from_stores,
    read_embeddings,
    select_device,
    zero_shot_scores,
)
from vineyard.perception.waste.crop_store import CropRecord, write_crop_store

MEAN = (0.5, 0.5, 0.5)
STD = (0.25, 0.25, 0.25)


def test_clip_input_normalises_and_transposes() -> None:
    crops = np.zeros((2, 4, 4, 3), np.uint8)
    crops[1] = 255
    x = clip_input(crops, MEAN, STD)
    assert x.shape == (2, 3, 4, 4) and x.dtype == np.float32
    np.testing.assert_allclose(x[0], -2.0)
    np.testing.assert_allclose(x[1], 2.0)


def test_clip_input_rejects_bad_shapes() -> None:
    with pytest.raises(ClipEmbedError, match="uint8"):
        clip_input(np.zeros((1, 4, 4, 3), np.float32), MEAN, STD)
    with pytest.raises(ClipEmbedError, match="shape"):
        clip_input(np.zeros((4, 4, 3), np.uint8), MEAN, STD)


def test_l2_normalise_rows_and_zero_rows_error() -> None:
    x = np.array([[3.0, 4.0], [1.0, 0.0]])
    np.testing.assert_allclose(l2_normalise(x), [[0.6, 0.8], [1.0, 0.0]])
    with pytest.raises(ClipEmbedError, match="zero"):
        l2_normalise(np.zeros((1, 2)))


def test_batch_slices_cover_everything() -> None:
    assert list(batch_slices(5, 2)) == [slice(0, 2), slice(2, 4), slice(4, 5)]
    assert list(batch_slices(0, 2)) == []
    with pytest.raises(ClipEmbedError):
        list(batch_slices(3, 0))


def test_zero_shot_margin_and_pos_probability() -> None:
    text = np.eye(3, dtype=np.float32)  # prompt 0 = positive, prompts 1-2 = negative
    emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    zs = zero_shot_scores(emb, text, n_pos=1, logit_scale=10.0)
    e = np.exp(10.0)
    np.testing.assert_allclose(zs.pos_p, [e / (e + 2), 1 / (e + 2)], rtol=1e-6)
    np.testing.assert_allclose(zs.margin, [(e - 1) / (e + 2), (1 - e) / (e + 2)], rtol=1e-6)
    assert zs.best_idx.tolist() == [0, 1]
    pos_p, margin, idx = zs  # verify_ml.ClipEmbedder.zero_shot contract: a 3-tuple
    assert pos_p is zs.pos_p and margin is zs.margin and idx is zs.best_idx


def test_zero_shot_needs_both_prompt_groups() -> None:
    with pytest.raises(ClipEmbedError, match="n_pos"):
        zero_shot_scores(np.ones((1, 2), np.float32), np.ones((2, 2), np.float32), n_pos=2, logit_scale=1.0)


def test_select_device_order() -> None:
    assert select_device("mps", mps_available=True, cuda_available=False) == "mps"
    assert select_device("mps", mps_available=False, cuda_available=False) == "cpu"
    assert select_device("cuda", mps_available=True, cuda_available=False) == "cpu"
    assert select_device("cpu", mps_available=True, cuda_available=True) == "cpu"


def test_model_tag_is_filename_safe() -> None:
    assert model_tag("ViT-B-32", "laion2b_s34b_b79k") == "open_clip:ViT-B-32/laion2b_s34b_b79k"


@dataclass
class FakeEmbedder:
    """Mean colour as a 3-d embedding; counts calls to check batching."""

    model_id: str = "fake:mean-rgb"
    calls: list[int] = field(default_factory=list)

    def embed(self, crops_u8: np.ndarray) -> np.ndarray:
        self.calls.append(len(crops_u8))
        return l2_normalise(crops_u8.reshape(len(crops_u8), -1, 3).mean(axis=1) + 1.0)


def _crops(n: int, px: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(n, px, px, 3), dtype=np.uint8)


def test_embed_batched_respects_batch_size() -> None:
    fake = FakeEmbedder()
    out = embed_batched(fake, _crops(5), batch=2)
    assert fake.calls == [2, 2, 1]
    assert out.shape == (5, 3) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)


def _store(tmp_path: Path, n: int) -> Path:
    recs = tuple(
        CropRecord(key=f"k{i}", source="random", group="t", label=0, meta={"i": i}) for i in range(n)
    )
    return write_crop_store(tmp_path / "store", iter(_crops(n)), recs, crop_px=8, meta={"kind": "test"})


def test_embed_crop_store_writes_and_reuses_cache(tmp_path: Path) -> None:
    store = _store(tmp_path, 4)
    fake = FakeEmbedder()
    path = embed_crop_store(store, fake, batch=3)
    assert fake.calls == [3, 1]
    emb, meta = read_embeddings(path)
    assert emb.shape == (4, 3) and meta["model_id"] == "fake:mean-rgb" and meta["n"] == 4
    assert "s_per_crop" in meta
    again = embed_crop_store(store, fake, batch=3)
    assert again == path and fake.calls == [3, 1]  # cache hit: no new calls


def test_embed_crop_store_recomputes_when_crops_change(tmp_path: Path) -> None:
    store = _store(tmp_path, 2)
    fake = FakeEmbedder()
    embed_crop_store(store, fake, batch=8)
    _store(tmp_path, 3)  # rewrite the store with different content
    path = embed_crop_store(store, fake, batch=8)
    assert fake.calls == [2, 3]
    assert json.loads(path.with_suffix(".json").read_text())["n"] == 3


def test_read_embeddings_missing_meta(tmp_path: Path) -> None:
    np.save(tmp_path / "e.npy", np.zeros((1, 2), np.float32))
    with pytest.raises(ClipEmbedError, match="meta"):
        read_embeddings(tmp_path / "e.npy")


def _labelled_store(tmp_path: Path, name: str, label: int, n: int, meta: dict) -> Path:
    recs = tuple(
        CropRecord(key=f"{name}{i}", source=name, group=f"g{i % 2}", label=label, meta=dict(meta))
        for i in range(n)
    )
    return write_crop_store(
        tmp_path / name, iter(_crops(n)), recs, crop_px=8, meta={"kind": name, "licence": "L"}
    )


def test_probe_data_from_stores(tmp_path: Path) -> None:
    pos = _labelled_store(tmp_path, "pos", 1, 3, {"is_example": True})  # flag ignored on positives
    neg = _labelled_store(tmp_path, "neg", 0, 2, {"is_example": True, "survives": True})
    fake = FakeEmbedder()
    for store in (pos, neg):
        embed_crop_store(store, fake, batch=8)
    data, sources = probe_data_from_stores((pos, neg), fake.model_id)
    assert data.emb.shape == (5, 3)
    assert data.label.tolist() == [1, 1, 1, 0, 0]
    assert data.is_example.tolist() == [False, False, False, True, True]
    assert data.survives.tolist() == [False, False, False, True, True]
    assert data.group.tolist() == ["g0", "g1", "g0", "g0", "g1"]
    assert sources[0] == {"store": "pos", "n": 3, "kind": "pos", "licence": "L"}


def test_probe_data_needs_fresh_embeddings(tmp_path: Path) -> None:
    pos = _labelled_store(tmp_path, "pos", 1, 2, {})
    with pytest.raises(ClipEmbedError, match="stale"):
        probe_data_from_stores((pos,), "fake:mean-rgb")
    with pytest.raises(ClipEmbedError, match="no crop stores"):
        probe_data_from_stores((), "fake:mean-rgb")


def test_clip_params_from_config() -> None:
    from vineyard.config import load_config

    cfg = load_config()
    p = ClipParams.from_config(cfg.waste.probe)
    assert p.model_name == "ViT-B-32" and p.pretrained == "laion2b_s34b_b79k"
    assert p.pos_prompts[0] == "trash" and "a white plastic vine tube" in p.neg_prompts


class _FakeVisual:
    image_size = (8, 8)
    preprocess_cfg = {"mean": MEAN, "std": STD}


class _FakeClipModel:
    """Torch stand-in for an open_clip model: image feature = per-channel mean, text feature = one-hot."""

    def __init__(self) -> None:
        import torch

        self.visual = _FakeVisual()
        self.logit_scale = torch.tensor(np.log(10.0))

    def eval(self) -> _FakeClipModel:
        return self

    def encode_image(self, x):  # noqa: ANN001, ANN201 - torch tensors
        return x.mean(dim=(2, 3)) + 3.0

    def encode_text(self, tokens):  # noqa: ANN001, ANN201
        import torch

        return torch.nn.functional.one_hot(tokens % 3, 3).float()


def _patch_open_clip(monkeypatch: pytest.MonkeyPatch, fail: bool = False) -> None:
    import open_clip
    import torch

    def create(name: str, pretrained: str, device: str):  # noqa: ANN202
        if fail:
            raise RuntimeError("no weights")
        return _FakeClipModel(), None, None

    monkeypatch.setattr(open_clip, "create_model_and_transforms", create)
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda name: lambda texts: torch.arange(len(texts)))


def _params() -> ClipParams:
    return ClipParams("ViT-B-32", "laion2b_s34b_b79k", "cpu", 4, 8, ("trash",), ("soil", "grass"))


def test_load_openclip_with_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from vineyard.perception.waste.clip_embed import load_openclip

    _patch_open_clip(monkeypatch)
    emb_model = load_openclip(_params(), device_pref="cpu")
    assert emb_model.device == "cpu" and emb_model.image_px == 8 and emb_model.n_pos == 1
    assert emb_model.logit_scale == pytest.approx(10.0)
    assert emb_model.model_id == "open_clip:ViT-B-32/laion2b_s34b_b79k"
    emb = emb_model.embed(_crops(3))
    assert emb.shape == (3, 3) and emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-6)
    zs = emb_model.zero_shot(emb)
    assert zs.pos_p.shape == (3,)
    with pytest.raises(ClipEmbedError, match="crop size"):
        emb_model.embed(_crops(1, px=16))


def test_load_openclip_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    from vineyard.perception.waste.clip_embed import load_openclip

    _patch_open_clip(monkeypatch, fail=True)
    with pytest.raises(ClipEmbedError, match="OpenCLIP load failed"):
        load_openclip(_params(), device_pref="cpu")


def test_select_device_rejects_unknown() -> None:
    with pytest.raises(ClipEmbedError, match="device"):
        select_device("tpu", mps_available=False, cuda_available=False)


@pytest.mark.network
@pytest.mark.slow
def test_real_openclip_embeds_224_crops() -> None:  # pragma: no cover - needs weights
    from vineyard.config import load_config
    from vineyard.perception.waste.clip_embed import load_openclip

    cfg = load_config()
    emb_model = load_openclip(ClipParams.from_config(cfg.waste.probe), device_pref="cpu")
    crops = _crops(3, px=224)
    emb = emb_model.embed(crops)
    assert emb.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-5)
    zs = emb_model.zero_shot(emb)
    assert zs.pos_p.shape == (3,) and np.all((zs.pos_p >= 0) & (zs.pos_p <= 1))
