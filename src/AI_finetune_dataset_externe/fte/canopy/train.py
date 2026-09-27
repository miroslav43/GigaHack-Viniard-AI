"""Train the 3-head canopy U-Net (c0 canopy, c1 plant contact, c2 ground vegetation) at 2.5 cm/px.

AdamW (encoder / decoder learning rates), 1 warm-up epoch then cosine, fp32 (MPS-safe). After every
epoch: official canopy metric on the 2 example tiles (fusion B and C, base-run axes) and the ICAERUS
val instance F1. Keeps ``best_mask.pt`` (mean of B/C), ``best_contact.pt`` (instance F1),
``last.pt`` and ``model_card.json``; one JSON line per epoch in ``work/logs/canopy_train.jsonl``.

CLI: ``python -m fte.canopy.train --epochs 40 --patches-per-epoch 1000 --batch 6 --out models/fte-canopy/v1``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fte.canopy.dataset import SOURCE_NAMES, CanopyPatches, MixSpec
from fte.canopy.icaerus_samples import IcaerusSample, load_sample
from fte.canopy.infer import HEAD_NAMES, WIN
from fte.canopy.validate import ExampleValidator, icaerus_instance_f1
from fte.device import pick_device
from fte.nn.losses import HeadLoss, multi_head_loss
from fte.nn.model import MultiHeadUnet, save_weights
from fte.paths import FTE_ROOT, LOGS, MODELS, STORES

log = logging.getLogger(__name__)

HEAD_SPECS = (HeadLoss(weight=1.0), HeadLoss(weight=0.5, pos_weight=5.0), HeadLoss(weight=0.3))
LICENCES = {
    "icaerus": {"name": "ICAERUS AI4Leafhopper vine segmentation (YOLO-seg subset)", "licence": "CC-BY-NC-4.0",
                "doi": "10.5281/zenodo.14605849", "use": "c0 fg, c1 contact band, c2 ground vegetation"},
    "siret3": {"name": "Sireț3 orthomosaic tiles + base-run pseudo-labels (complete-v4)", "licence": "CC-BY-4.0",
               "use": "c0 pseudo-labels (canopy polygons + row corridors), c2 ExG labels, negative tiles"},
}


@dataclass(frozen=True)
class TrainArgs:
    store: Path
    out: Path
    epochs: int = 40
    patches_per_epoch: int = 1000
    batch: int = 6
    lr: float = 3e-4
    encoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    encoder: str = "resnet34"
    time_box_min: float = 75.0
    patience: int = 5
    workers: int = 3
    val_tta: int = 1
    seed: int = 0
    smoke: bool = False
    device: str = "auto"
    amp: bool = False  # fp16 autocast + GradScaler; CUDA only (fp32 on MPS)
    init_weights: Path | None = None  # warm start from a saved smp state_dict (e.g. an interrupted run)


def lr_factor(step: int, warmup_steps: int, total_steps: int, floor: float = 0.02) -> float:
    """Linear warm-up to 1, then cosine down to ``floor``."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    t = min(1.0, (step - warmup_steps) / span)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))


def build_optimizer(model: MultiHeadUnet, args: TrainArgs) -> torch.optim.Optimizer:
    enc = [p for n, p in model.net.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.net.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.AdamW([{"params": enc, "lr": args.encoder_lr}, {"params": rest, "lr": args.lr}],
                             weight_decay=args.weight_decay)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_icaerus_val(store: Path) -> list[IcaerusSample]:
    index = json.loads((store / "index.json").read_text())
    return [load_sample(store / "icaerus" / p) for p in index.get("icaerus", {}).get("val", [])]


def _step(model: MultiHeadUnet, x: torch.Tensor, y: torch.Tensor, opt: torch.optim.Optimizer,
          scaler: torch.amp.GradScaler | None) -> tuple[torch.Tensor, list[float]]:
    """One optimisation step; with a scaler the forward runs under fp16 autocast (CUDA)."""
    opt.zero_grad(set_to_none=True)
    if scaler is None:
        loss, parts = multi_head_loss(model(x), y, HEAD_SPECS)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {float(loss)} (heads {parts})")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        return loss, parts
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
    loss, parts = multi_head_loss(logits.float(), y, HEAD_SPECS)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss {float(loss)} (heads {parts})")
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(opt)
    scaler.update()
    return loss, parts


def train_epoch(model: MultiHeadUnet, loader: DataLoader, opt: torch.optim.Optimizer,
                sched: torch.optim.lr_scheduler.LambdaLR, device: torch.device,
                scaler: torch.amp.GradScaler | None = None) -> dict:
    model.train()
    losses, heads, src_counts, n_seen, t0 = [], [], np.zeros(len(SOURCE_NAMES), int), 0, time.time()
    for x, y, src in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss, parts = _step(model, x, y, opt, scaler)
        sched.step()
        losses.append(float(loss.detach()))
        heads.append(parts)
        src_counts += np.bincount(src.numpy(), minlength=len(SOURCE_NAMES))
        n_seen += x.shape[0]
    dt = time.time() - t0
    return {"loss": float(np.mean(losses)), "head_loss": np.mean(heads, axis=0).round(4).tolist(),
            "patches": n_seen, "train_s": round(dt, 1), "patches_per_s": round(n_seen / max(dt, 1e-6), 2),
            "sources": dict(zip(SOURCE_NAMES, src_counts.tolist(), strict=True))}


def validate(model: MultiHeadUnet, device: torch.device, examples: ExampleValidator,
             ica_val: Sequence[IcaerusSample], tta: int) -> dict:
    model.eval()
    t0 = time.time()
    official = examples.evaluate(model, device, tta=tta)
    out = {"official": official, "mask_score": float(np.mean([official[v]["mean"] for v in official]))}
    if ica_val:
        out["icaerus_val"] = icaerus_instance_f1(model, ica_val, device)
    out["val_s"] = round(time.time() - t0, 1)
    return out


@dataclass(frozen=True)
class Best:
    mask: float = -1.0
    contact: float = -1.0
    mask_epoch: int = -1
    contact_epoch: int = -1
    stale: int = 0


def update_best(best: Best, epoch: int, val: dict, model: MultiHeadUnet, out: Path) -> Best:
    mask = val["mask_score"]
    contact = val.get("icaerus_val", {}).get("f1", -1.0)
    improved_mask, improved_contact = mask > best.mask, contact > best.contact
    if improved_mask:
        save_weights(model, out / "best_mask.pt")
    if improved_contact:
        save_weights(model, out / "best_contact.pt")
    return Best(max(mask, best.mask), max(contact, best.contact),
                epoch if improved_mask else best.mask_epoch, epoch if improved_contact else best.contact_epoch,
                0 if (improved_mask or improved_contact) else best.stale + 1)


def write_card(args: TrainArgs, history: list[dict], best: Best, baseline: dict, n_train: dict) -> None:
    weights = {n: sha256(args.out / n) for n in ("best_mask.pt", "best_contact.pt", "last.pt")
               if (args.out / n).is_file()}
    card = {"model": "fte-canopy", "arch": "smp.Unet + scse decoder attention", "encoder": args.encoder,
            "heads": list(HEAD_NAMES), "head_meaning": {"c0": "vine canopy", "c1": "plant-plant contact band",
                                                        "c2": "ground vegetation (non-vine green)"},
            "in_gsd_m": 0.025, "input": "RGB uint8 / 255, ImageNet normalisation inside the model",
            "patch": WIN, "infer": {"window": WIN, "overlap": 96, "tta": "dihedral x4 optional"},
            "loss": [asdict(h) for h in HEAD_SPECS], "train_args": {k: str(v) for k, v in asdict(args).items()},
            "datasets": LICENCES, "store_counts": n_train, "baseline_variant_A": baseline,
            "best": asdict(best), "sha256": weights, "metrics": history}
    tmp = args.out / "model_card.json.tmp"
    tmp.write_text(json.dumps(card, indent=1))
    tmp.replace(args.out / "model_card.json")


def make_loader(args: TrainArgs) -> tuple[CanopyPatches, DataLoader]:
    ds = CanopyPatches(args.store, args.patches_per_epoch, MixSpec(), seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None,
                        pin_memory=torch.cuda.is_available())
    return ds, loader


def train(args: TrainArgs) -> Best:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    ds, loader = make_loader(args)
    counts = {"vineyard_tiles": len(ds.vine), "negative_tiles": len(ds.neg), "icaerus_train": len(ds.icaerus_paths)}
    ica_val = load_icaerus_val(args.store)
    examples = ExampleValidator()
    baseline = examples.baseline()
    log.info("device %s, store %s, val icaerus %d, baseline A %.4f", device, counts, len(ica_val), baseline["mean"])
    model = MultiHeadUnet(args.encoder, len(HEAD_NAMES), encoder_weights="imagenet")
    if args.init_weights is not None:
        model.net.load_state_dict(torch.load(args.init_weights, map_location="cpu", weights_only=True))
        log.info("warm start from %s", args.init_weights)
    model = model.to(device)
    opt = build_optimizer(model, args)
    steps = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, steps, steps * args.epochs))
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    log.info("mixed precision: %s", "fp16 autocast" if scaler else "off (fp32)")
    return fit(args, model, loader, opt, sched, device, examples, ica_val, baseline, counts, scaler)


def fit(args: TrainArgs, model: MultiHeadUnet, loader: DataLoader, opt: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LambdaLR, device: torch.device, examples: ExampleValidator,
        ica_val: Sequence[IcaerusSample], baseline: dict, counts: dict,
        scaler: torch.amp.GradScaler | None = None) -> Best:
    t_start, best, history = time.time(), Best(), []
    log_path = LOGS / ("canopy_train_smoke.jsonl" if args.smoke else "canopy_train.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        rec = {"epoch": epoch, "lr": [g["lr"] for g in opt.param_groups], **train_epoch(model, loader, opt, sched, device, scaler)}
        rec.update(validate(model, device, examples, ica_val, args.val_tta))
        save_weights(model, args.out / "last.pt")
        best = update_best(best, epoch, rec, model, args.out)
        rec.update({"epoch_s": round(time.time() - t_ep, 1), "elapsed_min": round((time.time() - t_start) / 60, 2),
                    "best_mask": best.mask, "best_contact": best.contact, "smoke": args.smoke})
        history.append(rec)
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        write_card(args, history, best, baseline, counts)
        log.info("epoch %d: loss %.4f | B %.4f C %.4f | ica F1 %s | %.0f s", epoch, rec["loss"],
                 rec["official"]["B"]["mean"], rec["official"]["C"]["mean"],
                 rec.get("icaerus_val", {}).get("f1"), rec["epoch_s"])
        if best.stale >= args.patience:
            log.info("early stop: no improvement for %d epochs", best.stale)
            break
        elapsed_min, epoch_min = (time.time() - t_start) / 60, rec["epoch_s"] / 60
        if elapsed_min + epoch_min > args.time_box_min:
            log.info("time box: %.1f + %.1f min > %.1f min, stopping", elapsed_min, epoch_min, args.time_box_min)
            break
    return best


def parse_args(argv: Sequence[str] | None) -> TrainArgs:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=STORES / "canopy_v1")
    ap.add_argument("--out", type=Path, default=MODELS / "fte-canopy" / "v1")
    for name, typ, default in (("epochs", int, 40), ("patches-per-epoch", int, 1000), ("batch", int, 6),
                               ("lr", float, 3e-4), ("encoder-lr", float, 1e-4), ("time-box-min", float, 75.0),
                               ("patience", int, 5), ("workers", int, 3), ("val-tta", int, 1), ("seed", int, 0)):
        ap.add_argument(f"--{name}", type=typ, default=default)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--smoke", action="store_true", help="2 tiny epochs (48 patches) for a pipeline check")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast + GradScaler (CUDA only)")
    ap.add_argument("--init-weights", type=Path, default=None, help="warm start from a state_dict (.pt)")
    a = ap.parse_args(argv)
    out = a.out if a.out.is_absolute() else FTE_ROOT / a.out
    store = a.store if a.store.is_absolute() else FTE_ROOT / a.store
    epochs, ppe, tbox = (2, 48, 3.0) if a.smoke else (a.epochs, a.patches_per_epoch, a.time_box_min)
    return TrainArgs(store=store, out=out, epochs=epochs, patches_per_epoch=ppe, batch=a.batch, lr=a.lr,
                     encoder_lr=a.encoder_lr, encoder=a.encoder, time_box_min=tbox, patience=a.patience,
                     workers=a.workers, val_tta=a.val_tta, seed=a.seed, smoke=a.smoke, device=a.device,
                     amp=a.amp, init_weights=a.init_weights)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if not (args.store / "index.json").is_file():
        raise SystemExit(f"store not found: {args.store} (build it with python -m fte.canopy.store)")
    best = train(args)
    log.info("done: best mask %.4f (epoch %d), best contact %.4f (epoch %d) -> %s", best.mask, best.mask_epoch,
             best.contact, best.contact_epoch, args.out)


if __name__ == "__main__":
    main()
