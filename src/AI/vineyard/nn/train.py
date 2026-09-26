"""U-Net training on pseudo-label patches (A§4.6, design 03 N3-N5). Imports torch: CLI / main process only.

Epoch loop: AdamW + per-epoch cosine LR (nn.schedule), Dice + BCE with label smoothing ignoring 255
(nn.losses), fp32 on MPS without autocast, deterministic seeds, persistent spawn DataLoader workers
(nn.dataset). After every epoch the model predicts the full 1024² held-out reference tiles and
nn.validate scores fusion variant B with the official vector metric; early stopping (patience) keeps
the best epoch, whose state_dict is saved with its model card. Optional drop-noisiest after
nn.train.drop_noisy.after_epoch.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal
from zoneinfo import ZoneInfo

import numpy as np
import torch

from vineyard.config import AppConfig
from vineyard.errors import VineyardError
from vineyard.logging_setup import get_logger, log_event
from vineyard.nn.dataset import NO_AUGMENT, AugmentParams, EpochSampler, PatchDataset, build_loader
from vineyard.nn.infer import default_model_factory, predict_tiles, select_device, state_dict_bytes, sync
from vineyard.nn.losses import DiceBceLoss, LossParams, per_sample_loss
from vineyard.nn.schedule import EarlyStopState, build_scheduler, drop_noisiest, early_stop_update
from vineyard.nn.validate import ValResult, ValTile, validate_probs
from vineyard.nn.weights import F_SUFFIX, SMOKE_VERSION, DataSource, ModelCard, save_weights, training_version
from vineyard.pipeline.atomic import atomic_write_json

VARIANTS: Final = ("main", "F")
CLASSES: Final = ("canopy_prob",)
RGB_BANDS: Final = 3
BENCH_WARMUP: Final = 3
EVENT_EPOCH: Final = "nn.train.epoch"
EVENT_DROP: Final = "nn.train.drop_noisy"
EVENT_DONE: Final = "nn.train.done"
NAN_HINT: Final = "--set nn.device=cpu"
TRAIN_DATA: Final = (
    DataSource("Sireț3 GigaHack 2026 orthophoto tiles (pseudo-labels from the classic a*∩corridor pipeline)",
               "https://github.com/miroslav43/GigaHack-Viniard-AI", "GigaHack 2026 challenge data, organiser terms"),
    DataSource("ResNet-18 ImageNet encoder weights (segmentation-models-pytorch / torchvision)",
               "https://github.com/qubvel-org/segmentation_models.pytorch", "BSD-3-Clause (torchvision weights)"),
)

Batch = tuple[torch.Tensor, torch.Tensor]
EpochBatches = Callable[[int], Iterable[Batch]]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_log = get_logger("nn.train")

__all__ = ["F_SUFFIX", "SMOKE_VERSION", "BenchResult", "EpochStats", "NnTrainingError", "TrainParts", "TrainResult",
           "TrainSettings", "benchmark_iterations", "default_parts", "train_model"]


class NnTrainingError(VineyardError):
    """Training cannot continue (NaN loss, no data); the context names epoch and batch."""


@dataclass(frozen=True)
class TrainSettings:
    version: str
    variant: str
    smoke: bool
    seed: int
    epochs: int
    max_batches: int | None
    lr: float
    weight_decay: float
    batch_size: int
    num_workers: int
    patience: int
    label_smoothing: float
    bce_weight: float
    dice_weight: float
    colour_jitter: float
    ignore_band_px: int
    encoder_weights: str | None
    holdout: tuple[str, ...]
    drop_noisy_after: int | None  # epoch after which the noisiest patches are dropped (None = off)
    drop_noisy_frac: float
    val_variant: str  # fusion variant scored for early stopping (nn.train.val_variant)

    @classmethod
    def from_config(cls, cfg: AppConfig, *, seed: int, smoke: bool = False, variant: str = "main",
                    version: str | None = None) -> TrainSettings:
        if variant not in VARIANTS:
            raise ValueError(f"unknown training variant {variant!r}; expected one of {VARIANTS}")
        nn, tr = cfg.nn, cfg.nn.train
        is_f = variant == "F"
        drop = tr.drop_noisy
        return cls(version=version or training_version(nn.version, smoke=smoke, variant=variant), variant=variant,
                   smoke=smoke, seed=seed, epochs=1 if smoke else tr.epochs,
                   max_batches=tr.smoke_max_batches if smoke else None, lr=tr.lr, weight_decay=tr.weight_decay,
                   batch_size=nn.batch_size, num_workers=tr.num_workers, patience=tr.patience,
                   label_smoothing=0.0 if is_f else tr.label_smoothing, bce_weight=tr.bce_weight,
                   dice_weight=tr.dice_weight, colour_jitter=tr.colour_jitter,
                   ignore_band_px=nn.pseudolabels.ignore_band_px, encoder_weights=tr.encoder_weights,
                   holdout=tuple(nn.holdout_tiles), drop_noisy_after=drop.after_epoch if drop.enabled else None,
                   drop_noisy_frac=drop.frac, val_variant=tr.val_variant)

    def loss_params(self) -> LossParams:
        return LossParams(bce_weight=self.bce_weight, dice_weight=self.dice_weight,
                          label_smoothing=self.label_smoothing)


@dataclass(frozen=True)
class TrainParts:
    """Pluggable pieces (tests inject tiny ones); `default_parts` wires nn.model / dataset / losses."""

    make_model: Callable[[str, int, str | None], torch.nn.Module]
    make_loader: Callable[[Path, TrainSettings, Sequence[int] | None], EpochBatches]
    make_loss: Callable[[TrainSettings], LossFn]
    patch_losses: Callable[[torch.nn.Module, torch.device, Path, TrainSettings], Sequence[float]] | None = None
    evaluate: Callable[[torch.nn.Module, torch.device], ValResult] | None = None


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    train_loss: float
    val_score: float
    val_iou: float
    val_f1: float
    val_tiles: Mapping[str, float]
    lr: float
    img_per_s: float
    wall_s: float
    n_batches: int


@dataclass(frozen=True)
class TrainResult:
    card: ModelCard
    best_epoch: int
    history: tuple[EpochStats, ...]
    stopped_early: bool
    device: str


@dataclass(frozen=True)
class BenchResult:
    n_iter: int
    batch_size: int
    size_px: int
    s_per_iter: float
    img_per_s: float
    device: str


@dataclass(frozen=True)
class _Loop:
    """Training objects for the epoch helpers (torch updates model / optimizer / scheduler in place)."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_fn: LossFn
    device: torch.device


# ------------------------------------------------------------------ default parts (WP-N1 modules)


def default_loader(store_dir: Path, settings: TrainSettings, indices: Sequence[int] | None) -> EpochBatches:
    """PatchDataset + EpochSampler + persistent spawn workers; the returned callable sets the epoch."""
    augment = AugmentParams(flips=True, rot90=True, colour_jitter=settings.colour_jitter)
    dataset = PatchDataset(store_dir, seed=settings.seed, augment=augment, holdout=settings.holdout, indices=indices)
    if len(dataset) == 0:
        raise NnTrainingError("the patch store holds no patches", store=str(store_dir))
    sampler = EpochSampler(len(dataset), settings.seed)
    loader = build_loader(dataset, sampler, batch_size=settings.batch_size, num_workers=settings.num_workers)

    def batches(epoch: int) -> Iterable[Batch]:
        sampler.set_epoch(epoch)
        return loader

    return batches


def default_loss(settings: TrainSettings) -> LossFn:
    return DiceBceLoss(settings.loss_params())


def default_patch_losses(model: torch.nn.Module, device: torch.device, store_dir: Path,
                         settings: TrainSettings) -> list[float]:
    """Per-patch loss (eval mode, no augmentation) in store order, for drop-noisiest."""
    dataset = PatchDataset(store_dir, seed=settings.seed, augment=NO_AUGMENT, holdout=settings.holdout)
    loader = build_loader(dataset, EpochSampler(len(dataset), settings.seed, shuffle=False),
                          batch_size=settings.batch_size, num_workers=0)
    model.eval()
    out: list[float] = []
    with torch.inference_mode():
        for images, labels in loader:
            out.extend(per_sample_loss(model(images.to(device)), labels.to(device), settings.loss_params()).tolist())
    return out


def default_parts() -> TrainParts:
    return TrainParts(make_model=default_model_factory, make_loader=default_loader, make_loss=default_loss,
                      patch_losses=default_patch_losses)


# ------------------------------------------------------------------ epoch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _step(loop: _Loop, batch: Batch, epoch: int, k: int) -> float:
    images, labels = batch
    logits = loop.model(images.to(loop.device, dtype=torch.float32))
    loss = loop.loss_fn(logits, labels.to(loop.device).long())
    value = float(loss.detach().cpu())
    if not np.isfinite(value):
        raise NnTrainingError("non-finite training loss", epoch=epoch, batch=k, loss=value, hint=NAN_HINT)
    loop.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    loop.optimizer.step()
    return value


def _train_epoch(loop: _Loop, batches: Iterable[Batch], epoch: int, max_batches: int | None
                 ) -> tuple[float, int, int, float]:
    """(mean loss, n batches, n images, synced seconds)."""
    loop.model.train()
    sync(loop.device)
    t0 = time.perf_counter()
    losses: list[float] = []
    n_img = 0
    for k, batch in enumerate(batches):
        if max_batches is not None and k >= max_batches:
            break
        losses.append(_step(loop, batch, epoch, k))
        n_img += int(batch[0].shape[0])
    sync(loop.device)
    if not losses:
        raise NnTrainingError("the training loader yielded no batch", epoch=epoch)
    return float(np.mean(losses)), len(losses), n_img, time.perf_counter() - t0


def _evaluate(loop: _Loop, parts: TrainParts, val: Sequence[ValTile], cfg: AppConfig) -> ValResult:
    if parts.evaluate is not None:
        return parts.evaluate(loop.model, loop.device)
    loop.model.eval()
    probs = predict_tiles(loop.model, [t.net_input for t in val], loop.device, cfg.nn.infer_batch_size)
    return validate_probs(val, {t.tile_id: p for t, p in zip(val, probs, strict=True)}, cfg)


def _epoch(loop: _Loop, parts: TrainParts, batches: EpochBatches, settings: TrainSettings, val: Sequence[ValTile],
           cfg: AppConfig, epoch: int) -> EpochStats:
    t0 = time.perf_counter()
    lr = float(loop.optimizer.param_groups[0]["lr"])
    loss, n_batches, n_img, train_s = _train_epoch(loop, batches(epoch - 1), epoch, settings.max_batches)
    loop.scheduler.step()
    res = _evaluate(loop, parts, val, cfg)
    return EpochStats(epoch=epoch, train_loss=loss, val_score=res.mean_score, val_iou=res.mean_iou,
                      val_f1=res.mean_f1, val_tiles={s.tile_id: s.score for s in res.scores}, lr=lr,
                      img_per_s=n_img / train_s if train_s > 0 else 0.0, wall_s=time.perf_counter() - t0,
                      n_batches=n_batches)


def _maybe_drop(loop: _Loop, parts: TrainParts, settings: TrainSettings, store_dir: Path, epoch: int,
                batches: EpochBatches) -> EpochBatches:
    if settings.drop_noisy_after is None or epoch != settings.drop_noisy_after:
        return batches
    if parts.patch_losses is None:
        raise NnTrainingError("drop_noisy is enabled but no per-patch loss is available", epoch=epoch)
    losses = parts.patch_losses(loop.model, loop.device, store_dir, settings)
    kept = drop_noisiest(losses, settings.drop_noisy_frac)
    log_event(_log, EVENT_DROP, epoch=epoch, n_patches=len(losses), n_kept=len(kept))
    return parts.make_loader(store_dir, settings, kept)


# ------------------------------------------------------------------ outputs


def _history_doc(settings: TrainSettings, history: Sequence[EpochStats], best: EarlyStopState, device: str
                 ) -> dict[str, Any]:
    return {"settings": asdict(settings), "device": device, "stopped_early": best.stop, "best_epoch": best.best_epoch,
            "best_val_score": best.best_score, "val_variant": settings.val_variant,
            "history": [{**asdict(h), "val_tiles": dict(h.val_tiles)} for h in history]}


def _card(cfg: AppConfig, settings: TrainSettings, history: Sequence[EpochStats], best: EarlyStopState, device: str,
          store_dir: Path, train_cmd: str) -> ModelCard:
    top = history[best.best_epoch - 1]
    metrics = {"val_score": top.val_score, "val_iou": top.val_iou, "val_f1": top.val_f1,
               "val_score_last": history[-1].val_score, "best_epoch": float(best.best_epoch),
               "epochs_run": float(len(history)), "img_per_s": float(np.mean([h.img_per_s for h in history])),
               **{f"val_score:{t}": s for t, s in top.val_tiles.items()}}
    extra = {"seed": settings.seed, "device": device, "smoke": settings.smoke, "variant": settings.variant,
             "store": str(store_dir), "holdout_tiles": list(settings.holdout), "val_variant": settings.val_variant,
             "label_smoothing": settings.label_smoothing, "ignore_band_px": settings.ignore_band_px,
             "encoder_weights": settings.encoder_weights, "drop_noisy_after": settings.drop_noisy_after}
    nn = cfg.nn
    return ModelCard(name=nn.name, version=settings.version, arch=nn.arch, encoder=nn.encoder, in_gsd_m=nn.in_gsd_m,
                     classes=CLASSES[: nn.out_channels], train_data=TRAIN_DATA, metrics=metrics, sha256="",
                     created_at=datetime.now(ZoneInfo(cfg.logging.tz)).isoformat(timespec="seconds"),
                     train_cmd=train_cmd, extra=extra)


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _setup(cfg: AppConfig, settings: TrainSettings, parts: TrainParts) -> _Loop:
    seed_everything(settings.seed)
    device = select_device(cfg.nn.device, cfg.nn.fallback_device)
    model = parts.make_model(cfg.nn.encoder, cfg.nn.out_channels, settings.encoder_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    return _Loop(model, optimizer, build_scheduler(optimizer, settings.epochs), parts.make_loss(settings), device)


# ------------------------------------------------------------------ entry points


def train_model(cfg: AppConfig, seed: int, store_dir: Path, val: Sequence[ValTile], models_dir: Path, *,
                smoke: bool = False, variant: Literal["main", "F"] = "main", parts: TrainParts | None = None,
                history_path: Path | None = None, train_cmd: str = "", version: str | None = None) -> TrainResult:
    """Train, early-stop on the official score of the held-out tiles, save the best epoch's weights + card."""
    settings = TrainSettings.from_config(cfg, seed=seed, smoke=smoke, variant=variant, version=version)
    use = parts or default_parts()
    if use.evaluate is None and not val:
        raise NnTrainingError("no validation tiles for early stopping", holdout=", ".join(settings.holdout))
    loop = _setup(cfg, settings, use)
    batches = use.make_loader(Path(store_dir), settings, None)
    history: list[EpochStats] = []
    best = EarlyStopState()
    best_weights: dict[str, torch.Tensor] = {}
    for epoch in range(1, settings.epochs + 1):
        stats = _epoch(loop, use, batches, settings, val, cfg, epoch)
        history.append(stats)
        best = early_stop_update(best, epoch, stats.val_score, settings.patience)
        best_weights = _snapshot(loop.model) if best.improved else best_weights
        log_event(_log, EVENT_EPOCH, **{k: v for k, v in asdict(stats).items() if k != "val_tiles"},
                  best_epoch=best.best_epoch)
        if history_path is not None:
            atomic_write_json(history_path, _history_doc(settings, history, best, loop.device.type))
        if best.stop:
            break
        batches = _maybe_drop(loop, use, settings, Path(store_dir), epoch, batches)
    loop.model.load_state_dict(best_weights)
    card = save_weights(state_dict_bytes(loop.model), _card(cfg, settings, history, best, loop.device.type,
                                                             Path(store_dir), train_cmd), models_dir)
    log_event(_log, EVENT_DONE, version=card.version, best_epoch=best.best_epoch, val_score=best.best_score,
              epochs_run=len(history), stopped_early=best.stop, sha256=card.sha256)
    return TrainResult(card, best.best_epoch, tuple(history), best.stop, loop.device.type)


def benchmark_iterations(cfg: AppConfig, *, n_iter: int, batch_size: int, size_px: int,
                         parts: TrainParts | None = None, warmup: int = BENCH_WARMUP) -> BenchResult:
    """Seconds per optimisation step on random patches (forward + backward + AdamW), GPU-synced."""
    use = parts or default_parts()
    settings = replace(TrainSettings.from_config(cfg, seed=cfg.runtime.seed), encoder_weights=None)
    loop = _setup(cfg, settings, use)
    g = torch.Generator().manual_seed(settings.seed)
    images = torch.rand((batch_size, RGB_BANDS, size_px, size_px), generator=g)
    labels = torch.randint(0, 2, (batch_size, size_px, size_px), generator=g)
    loop.model.train()
    for k in range(warmup):
        _step(loop, (images, labels), 0, k)
    sync(loop.device)
    t0 = time.perf_counter()
    for k in range(n_iter):
        _step(loop, (images, labels), 0, k)
    sync(loop.device)
    per = (time.perf_counter() - t0) / n_iter
    return BenchResult(n_iter, batch_size, size_px, per, batch_size / per, loop.device.type)
