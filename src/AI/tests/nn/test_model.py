"""U-Net factory and state_dict helpers (design 03 §5, CPU only, no network)."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from vineyard.config import NnConfig, load_config
from vineyard.nn import model as model_mod
from vineyard.nn.model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ModelLoadError,
    VineUnet,
    build_unet,
    count_parameters,
    images_to_tensor,
    load_model,
    load_state_dict_file,
    predict_proba,
    save_state_dict,
)


@pytest.fixture(scope="module")
def nn_cfg() -> NnConfig:
    return load_config().nn


@pytest.fixture(scope="module")
def unet(nn_cfg: NnConfig) -> VineUnet:
    torch.manual_seed(0)
    return build_unet(nn_cfg, pretrained=False).eval()


def test_output_shape_batch(unet: VineUnet) -> None:
    with torch.no_grad():
        assert unet(torch.rand(2, 3, 256, 256)).shape == (2, 1, 256, 256)


@pytest.mark.slow
def test_output_shape_full_tile(unet: VineUnet) -> None:
    with torch.no_grad():
        assert unet(torch.rand(1, 3, 1024, 1024)).shape == (1, 1, 1024, 1024)


def test_parameter_count_is_resnet18_unet(unet: VineUnet) -> None:
    assert 13_000_000 < count_parameters(unet) < 16_000_000


def test_no_pretrained_weights_means_no_network(nn_cfg: NnConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: Any, **_k: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert isinstance(build_unet(nn_cfg, pretrained=False), VineUnet)


class _FakeUnet(nn.Module):
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        _FakeUnet.calls.append(kwargs)
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_pretrained_asks_smp_for_the_configured_encoder_weights(nn_cfg: NnConfig,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_mod.smp, "Unet", _FakeUnet)
    _FakeUnet.calls.clear()
    build_unet(nn_cfg, pretrained=True)
    build_unet(nn_cfg, pretrained=False)
    none_cfg = nn_cfg.model_copy(update={"train": nn_cfg.train.model_copy(update={"encoder_weights": None})})
    build_unet(none_cfg, pretrained=True)
    assert [c["encoder_weights"] for c in _FakeUnet.calls] == ["imagenet", None, None]
    assert all(c["encoder_name"] == "resnet18" and c["classes"] == 1 and c["activation"] is None
               for c in _FakeUnet.calls)


def test_forward_normalises_with_imagenet_statistics() -> None:
    model = VineUnet(nn.Identity())
    x = torch.rand(1, 3, 4, 4)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    assert torch.allclose(model(x), (x - mean) / std)
    assert "mean" not in model.state_dict() and "std" not in model.state_dict()


def test_save_load_round_trip(unet: VineUnet, nn_cfg: NnConfig, tmp_path: Path) -> None:
    path = save_state_dict(unet, tmp_path / "vine-unet" / "v1" / "weights.pt")
    state = load_state_dict_file(path)
    assert not any(k.startswith("net.") for k in state)  # a plain smp.Unet state_dict
    loaded = load_model(nn_cfg, path, device="cpu")
    assert not loaded.training
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(loaded(x), unet(x))


def test_load_model_errors(nn_cfg: NnConfig, tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="missing"):
        load_model(nn_cfg, tmp_path / "absent.pt")
    bad = tmp_path / "bad.pt"
    torch.save({"conv.weight": torch.zeros(1)}, bad)
    with pytest.raises(ModelLoadError, match="state_dict"):
        load_model(nn_cfg, bad)
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(ModelLoadError, match="unreadable"):
        load_state_dict_file(junk)


def test_images_to_tensor() -> None:
    one = np.full((8, 6, 3), 255, dtype=np.uint8)
    t = images_to_tensor(one)
    assert t.shape == (1, 3, 8, 6) and t.dtype == torch.float32 and float(t.max()) == 1.0
    batch = np.zeros((2, 8, 6, 3), dtype=np.uint8)
    batch[1, 0, 0] = (51, 102, 255)
    tb = images_to_tensor(batch)
    assert tb.shape == (2, 3, 8, 6)
    assert torch.allclose(tb[1, :, 0, 0], torch.tensor([0.2, 0.4, 1.0]))
    with pytest.raises(ValueError, match="uint8"):
        images_to_tensor(np.zeros((8, 6, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="RGB"):
        images_to_tensor(np.zeros((8, 6), dtype=np.uint8))


def test_predict_proba_is_a_probability_without_grad(unet: VineUnet) -> None:
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    p = predict_proba(unet, x)
    assert p.shape == (1, 1, 64, 64)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    assert not p.requires_grad
