"""nn.train: epoch loop, early stopping on the official score, weights + model card, smoke / F variants,
drop-noisiest, NaN guard and the iteration benchmark. CPU only; tiny injected parts or a tiny real store."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.nn.n2_synth import TILE_ID, synth_tile, write_tiny_store
from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import FusionVariant
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.nn.train import (
    F_SUFFIX,
    SMOKE_VERSION,
    EpochBatches,
    NnTrainingError,
    TrainParts,
    TrainSettings,
    benchmark_iterations,
    default_loader,
    default_parts,
    default_patch_losses,
    train_model,
)
from vineyard.nn.validate import TileScore, ValResult, ValTile, input_factor, make_val_tile, predict_canopies
from vineyard.nn.weights import load_weights, sha256_file

PATCH = 32


def _cfg(**train: object) -> AppConfig:
    cfg = load_config(overrides=["nn.device=cpu", "nn.train.num_workers=0", "nn.batch_size=2"])
    return cfg.model_copy(update={"nn": cfg.nn.model_copy(update={"train": cfg.nn.train.model_copy(update=train)})})


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _batches(n: int = 4) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        x = torch.rand((2, 3, PATCH, PATCH), generator=g)
        y = (x[:, 1] > 0.5).long()
        y[:, :2] = 255
        out.append((x, y))
    return out


def _masked_bce(settings: TrainSettings) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        keep = labels != 255
        target = labels.float() * (1 - settings.label_smoothing) + settings.label_smoothing / 2
        return F.binary_cross_entropy_with_logits(logits[:, 0][keep], target[keep])

    return loss


def _parts(scores: Iterable[float] | None = None, *, loss=None, seen: list[TrainSettings] | None = None) -> TrainParts:  # noqa: ANN001
    it: Iterator[float] | None = iter(scores) if scores is not None else None

    def make_loader(store_dir: Path, settings: TrainSettings, indices: Sequence[int] | None) -> EpochBatches:
        if seen is not None:
            seen.append(settings)
        return lambda epoch: _batches()

    def evaluate(model: torch.nn.Module, device: torch.device) -> ValResult:
        assert it is not None
        s = next(it)
        return ValResult("B", (TileScore(TILE_ID, "B", s, s, s, 1, 1),))

    return TrainParts(make_model=lambda enc, n, w: Tiny(), make_loader=make_loader,
                      make_loss=loss or _masked_bce, evaluate=evaluate if scores is not None else None)


@pytest.fixture(scope="module")
def val_tile() -> ValTile:
    cfg = load_config()
    s = synth_tile()
    base = make_val_tile(TILE_ID, s.rgb, s.valid, s.veg, s.pieces, tile_box(tile_ref(TILE_ID)), (), input_factor(cfg))
    return replace(base, ref=tuple(predict_canopies(base, None, FusionVariant.A, cfg)))


def test_settings_from_config() -> None:
    cfg = _cfg()
    main = TrainSettings.from_config(cfg, seed=3)
    assert (main.version, main.epochs, main.smoke, main.variant, main.max_batches) == ("v1", 20, False, "main", None)
    assert main.label_smoothing == pytest.approx(0.05) and main.seed == 3 and main.drop_noisy_after is None
    assert main.holdout == tuple(cfg.nn.holdout_tiles)
    smoke = TrainSettings.from_config(cfg, seed=0, smoke=True)
    assert (smoke.version, smoke.epochs, smoke.max_batches) == (SMOKE_VERSION, 1, cfg.nn.train.smoke_max_batches)
    f = TrainSettings.from_config(cfg, seed=0, variant="F")
    assert f.version == "v1" + F_SUFFIX and f.label_smoothing == 0.0
    with pytest.raises(ValueError, match="variant"):
        TrainSettings.from_config(cfg, seed=0, variant="G")


def test_smoke_run_writes_weights_card_and_history(tmp_path: Path, val_tile: ValTile) -> None:
    cfg = _cfg(smoke_max_batches=2)
    history = tmp_path / "metrics" / "nn_train.json"
    res = train_model(cfg, 0, tmp_path / "store", (val_tile,), tmp_path / "models", smoke=True, parts=_parts(),
                      history_path=history, train_cmd="vineyard nn train --smoke")
    assert res.card.version == SMOKE_VERSION and len(res.history) == 1
    assert np.isfinite(res.history[0].train_loss) and res.history[0].n_batches == 2
    assert 0.0 <= res.history[0].val_score <= 1.0 and res.device == "cpu"
    loaded = load_weights(tmp_path / "models", "vine-unet", SMOKE_VERSION, None)
    assert loaded.card.sha256 == sha256_file(loaded.weights_path)
    assert loaded.card.extra["smoke"] is True and loaded.card.train_cmd == "vineyard nn train --smoke"
    assert {"val_score", "best_epoch", "img_per_s", f"val_score:{TILE_ID}"} <= set(loaded.card.metrics)
    assert [d.licence for d in loaded.card.train_data]
    doc = json.loads(history.read_text(encoding="utf-8"))
    assert len(doc["history"]) == 1 and doc["best_epoch"] == 1 and doc["settings"]["smoke"] is True


def test_early_stopping_keeps_the_best_epoch(tmp_path: Path) -> None:
    cfg = _cfg(epochs=10, patience=3)
    res = train_model(cfg, 0, tmp_path / "store", (), tmp_path / "models",
                      parts=_parts([0.70, 0.74, 0.73, 0.735, 0.72, 0.9]))
    assert res.stopped_early and len(res.history) == 5 and res.best_epoch == 2
    assert res.card.metrics["val_score"] == pytest.approx(0.74)
    assert res.card.metrics["val_score_last"] == pytest.approx(0.72)
    lrs = [h.lr for h in res.history]
    assert lrs[0] == pytest.approx(cfg.nn.train.lr) and lrs == sorted(lrs, reverse=True)


def test_best_weights_are_saved_not_the_last(tmp_path: Path) -> None:
    snapshots: list[torch.Tensor] = []

    def evaluate(model: torch.nn.Module, device: torch.device) -> ValResult:
        snapshots.append(model.conv.weight.detach().clone())
        s = [0.9, 0.1][len(snapshots) - 1]
        return ValResult("B", (TileScore(TILE_ID, "B", s, s, s, 1, 1),))

    parts = replace(_parts(), evaluate=evaluate)
    res = train_model(_cfg(epochs=2, patience=5), 0, tmp_path / "s", (), tmp_path / "m", parts=parts)
    raw = torch.load(load_weights(tmp_path / "m", "vine-unet", "v1", None).weights_path, weights_only=True)
    assert res.best_epoch == 1 and torch.equal(raw["conv.weight"], snapshots[0])


def test_f_variant_trains_without_smoothing(tmp_path: Path) -> None:
    seen: list[TrainSettings] = []
    res = train_model(_cfg(epochs=1), 0, tmp_path / "store", (), tmp_path / "models", variant="F",
                      parts=_parts([0.5], seen=seen))
    assert res.card.version == "v1" + F_SUFFIX and seen[0].label_smoothing == 0.0
    assert res.card.extra["variant"] == "F"


def test_nan_loss_raises_with_context(tmp_path: Path) -> None:
    def nan_loss(settings: TrainSettings) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return lambda logits, labels: logits.mean() * float("nan")

    with pytest.raises(NnTrainingError, match="batch=0") as err:
        train_model(_cfg(epochs=1), 0, tmp_path / "store", (), tmp_path / "models", parts=_parts([0.5], loss=nan_loss))
    assert "epoch=1" in str(err.value) and "nn.device=cpu" in str(err.value)


def test_needs_validation_tiles_or_an_evaluator(tmp_path: Path) -> None:
    with pytest.raises(NnTrainingError, match="validation"):
        train_model(_cfg(epochs=1), 0, tmp_path / "store", (), tmp_path / "models", parts=_parts())


def test_empty_loader_is_an_error(tmp_path: Path) -> None:
    parts = replace(_parts([0.5]), make_loader=lambda store, settings, idx: (lambda epoch: []))
    with pytest.raises(NnTrainingError, match="no batch"):
        train_model(_cfg(epochs=1), 0, tmp_path / "store", (), tmp_path / "models", parts=parts)


# ------------------------------------------------------------------ real WP-N1 parts on a tiny store


def test_default_parts_on_a_tiny_store(tmp_path: Path, val_tile: ValTile) -> None:
    store = write_tiny_store(tmp_path / "store")
    small = lambda enc, n, w: default_parts().make_model("resnet18", n, None)  # noqa: E731  (offline)
    parts = replace(default_parts(), make_model=small)
    cfg = _cfg(epochs=1)
    res = train_model(cfg, 0, store, (val_tile,), tmp_path / "models", parts=parts, version="vtest")
    assert np.isfinite(res.history[0].train_loss) and res.history[0].n_batches == 3  # 6 patches / batch 2
    raw = torch.load(load_weights(tmp_path / "models", "vine-unet", "vtest", None).weights_path, weights_only=True)
    assert "encoder.conv1.weight" in raw and not any(k.startswith("net.") for k in raw)


def test_default_loader_is_deterministic_per_epoch(tmp_path: Path) -> None:
    store = write_tiny_store(tmp_path / "store")
    settings = TrainSettings.from_config(_cfg(), seed=1)
    a = [x.clone() for x, _ in default_loader(store, settings, None)(3)]
    b = [x.clone() for x, _ in default_loader(store, settings, None)(3)]
    assert all(torch.equal(p, q) for p, q in zip(a, b, strict=True))
    kept = [x for x, _ in default_loader(store, settings, [0, 1])(0)]
    assert sum(int(x.shape[0]) for x in kept) == 2


def test_drop_noisy_rebuilds_the_loader(tmp_path: Path) -> None:
    store = write_tiny_store(tmp_path / "store")
    cfg = _cfg(epochs=3, patience=5)
    cfg = cfg.model_copy(update={"nn": cfg.nn.model_copy(update={"train": cfg.nn.train.model_copy(update={
        "drop_noisy": cfg.nn.train.drop_noisy.model_copy(update={"enabled": True, "after_epoch": 1, "frac": 0.5})})})})
    calls: list[Sequence[int] | None] = []
    base = default_parts()

    def loader(store_dir: Path, settings: TrainSettings, indices: Sequence[int] | None) -> EpochBatches:
        calls.append(indices)
        return base.make_loader(store_dir, settings, indices)

    parts = replace(_parts([0.1, 0.2, 0.3]), make_model=lambda e, n, w: Tiny(), make_loader=loader,
                    make_loss=base.make_loss, patch_losses=default_patch_losses)
    res = train_model(cfg, 0, store, (), tmp_path / "models", parts=parts)
    assert len(res.history) == 3 and calls[0] is None and len(calls) == 2 and len(calls[1] or ()) == 3
    assert res.history[1].n_batches == 2  # 3 kept patches / batch 2


def test_benchmark_iterations_on_cpu() -> None:
    res = benchmark_iterations(_cfg(), n_iter=3, batch_size=2, size_px=PATCH, parts=_parts(), warmup=1)
    assert res.n_iter == 3 and res.s_per_iter > 0 and res.img_per_s > 0 and res.device == "cpu"
