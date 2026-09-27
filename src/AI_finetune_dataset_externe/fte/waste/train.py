"""Train the waste detector (smp U-Net resnet34, 1 head) on the patch mixture of ``fte.waste.dataset``.

AdamW, 1 warm-up epoch then cosine, fp32 (MPS). Every epoch: DroneWaste val-site box F1 over a threshold
grid (early stopping on the best F1, patience), synthetic Sireț3 recall at that threshold, and the box
count on the 2 example tiles. Writes best.pt / last.pt / model_card.json into ``--out`` and a JSON-lines
log to ``work/logs/waste_train.jsonl``.

    python -m fte.waste.train --epochs 30 --patches-per-epoch 4000 --batch 12 --lr 3e-4 \\
        --time-box-min 90 --patience 5 --workers 3 --out models/fte-waste/v1 [--smoke] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fte.device import pick_device
from fte.nn.losses import HeadLoss, multi_head_loss
from fte.nn.model import MultiHeadUnet, save_weights
from fte.paths import FTE_ROOT, LOGS
from fte.waste.boxes import BoxParams
from fte.waste.card import write_card
from fte.waste.dataset import SOURCES, EpochSampler, WasteDataset
from fte.waste.evaluate import (
    THRESHOLDS,
    best_threshold,
    count_boxes,
    dronewaste_cases,
    example_heats,
    score_cases,
    summarise,
    synval_cases,
)
from fte.waste.sources import WasteSources, load_sources
from fte.waste.synval import build_synval, load_synval

log = logging.getLogger("fte.waste.train")

ENCODER = "resnet34"
LOSS = HeadLoss(focal_gamma=2.0, dice_weight=1.0)
LR_FLOOR = 0.02


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    patches_per_epoch: int = 4000
    batch: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    time_box_min: float = 90.0
    patience: int = 5
    workers: int = 3
    out: str = "models/fte-waste/v1"
    smoke: bool = False
    device: str = "auto"
    seed: int = 0
    down_weight: float = 0.3
    log_path: str = str(LOGS / "waste_train.jsonl")


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    ap = argparse.ArgumentParser(description="train the fte waste detector")
    for f, default in asdict(TrainConfig()).items():
        flag = "--" + f.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(flag, action="store_true")
        else:
            ap.add_argument(flag, type=type(default), default=default)
    cfg = TrainConfig(**vars(ap.parse_args(argv)))
    if cfg.smoke:
        return TrainConfig(**{**asdict(cfg), "epochs": min(cfg.epochs, 2), "patches_per_epoch": 24,
                              "batch": 4, "workers": min(cfg.workers, 2), "time_box_min": 5.0})
    return cfg


def lr_factor(step: int, warmup: int, total: int, floor: float = LR_FLOOR) -> float:
    """Linear warm-up over ``warmup`` steps, then cosine from 1 to ``floor`` at ``total``."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class ValData:
    dw_images: tuple
    synval_items: tuple
    synval_images: dict
    example_crop: int | None
    min_area_px: float


def val_data(src: WasteSources, smoke: bool) -> ValData:
    items, images = load_synval(build_synval(src))
    dw = src.dw_val[:12] if smoke else src.dw_val
    syn = items[:16] if smoke else items
    return ValData(tuple(dw), tuple(syn), images, 1024 if smoke else None, BoxParams().area_px()[0])


def validate(model: MultiHeadUnet, device: torch.device, src: WasteSources, vd: ValData) -> dict:
    model.eval()
    t0 = time.time()
    dw = score_cases(dronewaste_cases(model, device, src.dronewaste, vd.dw_images, vd.min_area_px), THRESHOLDS)
    syn = score_cases(synval_cases(model, device, vd.synval_items, vd.synval_images, vd.min_area_px), THRESHOLDS)
    ex = count_boxes(example_heats(model, device, vd.example_crop), THRESHOLDS)
    t_best, f1 = best_threshold(dw)
    model.train()
    return {"val_f1": round(f1, 4), "val_threshold": t_best, "syn_recall_at_t": round(syn[t_best].recall, 4),
            "syn_f1_at_t": round(syn[t_best].f1, 4), "example_boxes_at_t": ex[t_best],
            "dronewaste": summarise(dw), "synthetic": summarise(syn),
            "example_boxes": {f"{t:.2f}": n for t, n in ex.items()}, "val_s": round(time.time() - t0, 1)}


def _loader(ds: WasteDataset, sampler: EpochSampler, cfg: TrainConfig) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch, sampler=sampler, num_workers=cfg.workers, drop_last=True,
                      persistent_workers=cfg.workers > 0, prefetch_factor=4 if cfg.workers > 0 else None)


def train_epoch(model: MultiHeadUnet, loader: DataLoader, opt: torch.optim.Optimizer,
                sched: torch.optim.lr_scheduler.LambdaLR, device: torch.device, cfg: TrainConfig) -> dict:
    model.train()
    t0, n, total, per_source = time.time(), 0, 0.0, torch.zeros(len(SOURCES))
    for x, y, w, s in loader:
        x, y = x.to(device), y.to(device)
        pw = (1.0 - (1.0 - cfg.down_weight) * w.to(device).float())
        loss, _ = multi_head_loss(model(x), y, [LOSS], pixel_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        total += float(loss.detach()) * len(x)
        n += len(x)
        per_source += torch.bincount(s, minlength=len(SOURCES)).float()
    dt = time.time() - t0
    return {"loss": round(total / max(1, n), 5), "patches": n, "train_s": round(dt, 1),
            "patches_per_s": round(n / max(dt, 1e-6), 2), "lr": sched.get_last_lr()[0],
            "mix": dict(zip(SOURCES, per_source.int().tolist(), strict=True))}


def _append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def run(cfg: TrainConfig) -> dict:
    device = pick_device(cfg.device)
    out = Path(cfg.out) if Path(cfg.out).is_absolute() else FTE_ROOT / cfg.out
    torch.manual_seed(cfg.seed)
    src = load_sources()
    vd = val_data(src, cfg.smoke)
    ds = WasteDataset(src, cfg.patches_per_epoch, cfg.epochs, seed=cfg.seed)
    sampler = EpochSampler(cfg.patches_per_epoch, seed=cfg.seed)
    loader = _loader(ds, sampler, cfg)
    model = MultiHeadUnet(ENCODER, 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: lr_factor(k, steps, steps * cfg.epochs))
    return _loop(cfg, model, loader, sampler, opt, sched, device, src, vd, out)


def _loop(cfg: TrainConfig, model: MultiHeadUnet, loader: DataLoader, sampler: EpochSampler,
          opt: torch.optim.Optimizer, sched: torch.optim.lr_scheduler.LambdaLR, device: torch.device,
          src: WasteSources, vd: ValData, out: Path) -> dict:
    t_start, best, best_epoch, bad, history = time.time(), -1.0, -1, 0, []
    log_path = Path(cfg.log_path)
    _append_log(log_path, {"event": "start", "config": asdict(cfg), "device": str(device), "counts": src.counts(),
                           "val_sites": list(src.val_sites)})
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        rec = {"event": "epoch", "epoch": epoch, **train_epoch(model, loader, opt, sched, device, cfg)}
        rec.update(validate(model, device, src, vd))
        save_weights(model, out / "last.pt")
        if rec["val_f1"] > best:
            best, best_epoch, bad = rec["val_f1"], epoch, 0
            save_weights(model, out / "best.pt")
        else:
            bad += 1
        rec["elapsed_min"] = round((time.time() - t_start) / 60.0, 2)
        history.append({k: v for k, v in rec.items() if k not in ("dronewaste", "synthetic")})
        _append_log(log_path, rec)
        log.info("epoch %d loss %.4f f1 %.3f@%.2f syn_recall %.3f ex_boxes %d (%.1f patches/s, %.1f min)",
                 epoch, rec["loss"], rec["val_f1"], rec["val_threshold"], rec["syn_recall_at_t"],
                 rec["example_boxes_at_t"], rec["patches_per_s"], rec["elapsed_min"])
        write_card(out, cfg, src, history, best_epoch, rec if best_epoch == epoch else None)
        per_epoch_min = rec["elapsed_min"] / (epoch + 1)
        if bad >= cfg.patience or rec["elapsed_min"] + per_epoch_min > cfg.time_box_min:
            log.info("stopping after epoch %d (patience %d/%d, %.1f min)", epoch, bad, cfg.patience,
                     rec["elapsed_min"])
            break
    return {"best_epoch": best_epoch, "best_f1": best, "out": str(out)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(json.dumps(run(parse_args()), indent=2))
