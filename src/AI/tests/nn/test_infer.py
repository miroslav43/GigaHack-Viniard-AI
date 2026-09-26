"""nn.infer: batched main-process inference -> contract §6 PNG (+ .key), nodata = 0, CPU retry (CPU only)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch

from vineyard.geo.raster import write_mask_png
from vineyard.nn import infer
from vineyard.nn.infer import (
    InferJob,
    infer_tiles,
    load_model,
    predict_batch,
    predict_tiles,
    select_device,
    state_dict_bytes,
    to_input,
    valid_to_net,
)
from vineyard.nn.probs import PROB_PX, read_prob_png
from vineyard.nn.weights import ModelCard, load_weights, save_weights
from vineyard.pipeline.cache import read_key

TILE_PX = 2 * PROB_PX


class ConstLogit(torch.nn.Module):
    """Returns a constant logit map of the input's spatial size."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(float(np.log(p / (1 - p)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logit.expand(x.shape[0], 1, x.shape[2], x.shape[3])


def _job(tmp_path: Path, tile_id: str, valid: np.ndarray) -> InferJob:
    valid_path = write_mask_png(tmp_path / "cache" / "valid" / f"{tile_id}.png", valid)
    return InferJob(tile_id=tile_id, tif_path=tmp_path / "tiles" / f"{tile_id}.tif", valid_path=valid_path,
                    out_path=tmp_path / "nn" / f"{tile_id}.png", key=f"key-{tile_id}")


def _rgb(_path: Path) -> np.ndarray:
    return np.full((TILE_PX, TILE_PX, 3), 100, np.uint8)


def test_to_input_scales_to_unit_range() -> None:
    batch = np.zeros((2, 4, 4, 3), np.uint8)
    batch[0, 0, 0] = (255, 0, 51)
    x = to_input(batch, torch.device("cpu"))
    assert x.shape == (2, 3, 4, 4) and x.dtype == torch.float32
    np.testing.assert_allclose(x[0, :, 0, 0].numpy(), [1.0, 0.0, 0.2], rtol=1e-6)


def test_vine_unet_state_dict_is_the_plain_network(tmp_path: Path) -> None:
    from vineyard.nn.model import build_unet_from

    model = build_unet_from("resnet18", 1, None)
    raw = torch.load(io.BytesIO(state_dict_bytes(model)), weights_only=True)
    assert not any(k.startswith("net.") for k in raw) and "encoder.conv1.weight" in raw
    card = ModelCard(name="vine-unet", version="v8", arch="unet", encoder="resnet18", in_gsd_m=0.05,
                     classes=("canopy_prob",), train_data=(), metrics={}, sha256="", created_at="now", train_cmd="t")
    save_weights(state_dict_bytes(model), card, tmp_path)
    back = load_model(load_weights(tmp_path, "vine-unet", "v8", None), torch.device("cpu"))
    x = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        assert torch.allclose(back(x), model.eval()(x), atol=1e-6)


def test_valid_to_net_majority() -> None:
    valid = np.zeros((4, 4), bool)
    valid[0, :2] = True  # half of block (0, 0)
    valid[2:, 2:] = True
    assert valid_to_net(valid, 2).tolist() == [[True, False], [False, True]]


def test_constant_prob_and_nodata(tmp_path: Path) -> None:
    valid = np.ones((TILE_PX, TILE_PX), bool)
    valid[:, :100] = False
    outcomes = infer_tiles([_job(tmp_path, "siret3_r021_c012", valid)], ConstLogit(0.7), torch.device("cpu"),
                           batch_size=2, factor=2, read_rgb=_rgb)
    assert [o.ok for o in outcomes] == [True]
    png = read_prob_png(tmp_path / "nn" / "siret3_r021_c012.png")
    assert png.shape == (PROB_PX, PROB_PX)
    assert np.all(png[:, 50:] == 178)  # rint(0.7 * 255 = 178.5) = 178 (half to even)
    assert np.all(png[:, :50] == 0)
    assert read_key(tmp_path / "nn" / "siret3_r021_c012.png") == "key-siret3_r021_c012"
    assert outcomes[0].mean_prob == pytest.approx(0.7 * (TILE_PX - 100) / TILE_PX, abs=1e-3)


def test_batch_of_three_and_a_failing_tile(tmp_path: Path) -> None:
    valid = np.ones((TILE_PX, TILE_PX), bool)
    ids = ("siret3_r021_c012", "siret3_r006_c004", "siret3_r005_c004", "siret3_r008_c002")
    jobs = [_job(tmp_path, t, valid) for t in ids]

    def read(path: Path) -> np.ndarray:
        if "r008" in path.name:
            raise OSError("corrupt tile")
        return _rgb(path)

    outcomes = infer_tiles(jobs, ConstLogit(0.3), torch.device("cpu"), batch_size=2, factor=2, read_rgb=read)
    assert [o.tile_id for o in outcomes] == list(ids)
    assert [o.ok for o in outcomes] == [True, True, True, False]
    assert "corrupt tile" in (outcomes[3].error or "")
    assert sorted(p.name for p in (tmp_path / "nn").iterdir()) == sorted(
        [f"{t}.png" for t in ids[:3]] + [f"{t}.png.key" for t in ids[:3]])


def test_predict_tiles_order_and_batches() -> None:
    inputs = [np.full((32, 32, 3), v, np.uint8) for v in (0, 50, 100)]
    out = predict_tiles(ConstLogit(0.2), inputs, torch.device("cpu"), batch_size=2)
    assert len(out) == 3 and all(o.shape == (32, 32) for o in out)
    assert np.allclose(out[1], 0.2, atol=1e-6)


def test_cpu_retry_on_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = infer._forward

    def fake(model: torch.nn.Module, batch: np.ndarray, device: torch.device) -> np.ndarray:
        calls.append(device.type)
        if device.type != "cpu":
            raise RuntimeError("The operator 'aten::foo' is not currently implemented for the MPS device")
        return real(model, batch, device)

    monkeypatch.setattr(infer, "_forward", fake)
    probs, device = predict_batch(ConstLogit(0.6), np.zeros((1, 8, 8, 3), np.uint8), torch.device("mps"))
    assert calls == ["mps", "cpu"] and device.type == "cpu"
    assert np.allclose(probs, 0.6, atol=1e-6)


def test_other_runtime_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(model: torch.nn.Module, batch: np.ndarray, device: torch.device) -> np.ndarray:
        raise RuntimeError("out of memory")

    monkeypatch.setattr(infer, "_forward", boom)
    with pytest.raises(RuntimeError, match="out of memory"):
        predict_batch(ConstLogit(0.6), np.zeros((1, 8, 8, 3), np.uint8), torch.device("mps"))


def test_select_device(monkeypatch: pytest.MonkeyPatch) -> None:
    assert select_device("cpu", "cpu").type == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("mps", "cpu").type == "cpu"
    with pytest.raises(ValueError, match="device"):
        select_device("tpu", "cpu")


def test_load_model_round_trip(tmp_path: Path) -> None:
    model = ConstLogit(0.42)
    card = ModelCard(name="vine-unet", version="v9", arch="unet", encoder="resnet18", in_gsd_m=0.05,
                     classes=("canopy_prob",), train_data=(), metrics={}, sha256="", created_at="now", train_cmd="t")
    save_weights(state_dict_bytes(model), card, tmp_path)
    loaded = load_weights(tmp_path, "vine-unet", "v9", None)
    back = load_model(loaded, torch.device("cpu"), factory=lambda enc, n, w: ConstLogit(0.5))
    assert not back.training
    assert torch.allclose(back.logit, model.logit)
    buf = io.BytesIO(state_dict_bytes(model))
    assert set(torch.load(buf, weights_only=True)) == {"logit"}
