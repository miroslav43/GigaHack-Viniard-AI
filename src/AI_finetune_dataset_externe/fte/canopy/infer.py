"""Sliding-window inference of the canopy model on 2048² Sireț3 tiles (2.5 cm/px).

Windows of 512 px with 96 px overlap are blended with a raised-cosine weight (no seams); optional
dihedral TTA (identity, h-flip, v-flip, hv-flip). Output per tile and head: ``<out>/c<k>/<tile>.png``
uint8 = round(p * 255), 2048², nodata (all-zero RGB) = 0.

CLI: ``python -m fte.canopy.infer --weights models/fte-canopy/v1/best_mask.pt --tiles examples --tta 4``
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch

from fte.device import pick_device
from fte.nn.model import MultiHeadUnet, load_model
from fte.paths import AI_RUNS, AI_TILES, BASE_RUN, EXAMPLE_TILES, PREDS

log = logging.getLogger(__name__)

WIN = 512
OVERLAP = 96
HEAD_NAMES = ("c0", "c1", "c2")
_FLIPS: tuple[tuple[int, ...], ...] = ((), (3,), (2,), (2, 3))  # dims of a (B, C, H, W) tensor


def window_starts(size: int, win: int = WIN, overlap: int = OVERLAP) -> list[int]:
    """Start offsets covering [0, size) with windows of ``win`` and at least ``overlap`` overlap."""
    if win > size:
        raise ValueError(f"window {win} larger than the image side {size}")
    if not 0 <= overlap < win:
        raise ValueError(f"overlap must be in [0, {win}), got {overlap}")
    stride = win - overlap
    starts = list(range(0, size - win + 1, stride))
    if starts[-1] != size - win:
        starts.append(size - win)
    return starts


def blend_weight(win: int = WIN, overlap: int = OVERLAP, floor: float = 1e-3) -> np.ndarray:
    """(win, win) float32 separable weight: raised-cosine ramps over ``overlap`` px at each border."""
    ramp = np.ones(win, np.float32)
    if overlap > 0:
        t = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        edge = 0.5 - 0.5 * np.cos(np.pi * t)
        ramp[:overlap], ramp[-overlap:] = edge, edge[::-1]
    w = np.outer(ramp, ramp)
    return np.maximum(w, floor).astype(np.float32)


@torch.no_grad()
def forward_tta(model: torch.nn.Module, x: torch.Tensor, tta: int) -> torch.Tensor:
    """Mean sigmoid over the first ``tta`` flips (1..4)."""
    flips = _FLIPS[:max(1, min(tta, len(_FLIPS)))]
    acc = None
    for dims in flips:
        xi = torch.flip(x, dims) if dims else x
        p = torch.sigmoid(model(xi))
        p = torch.flip(p, dims) if dims else p
        acc = p if acc is None else acc + p
    return acc / len(flips)


@torch.no_grad()
def predict_image(model: torch.nn.Module, rgb: np.ndarray, device: torch.device, *, tta: int = 1,
                  batch: int = 6, win: int = WIN, overlap: int = OVERLAP) -> np.ndarray:
    """(K, H, W) float32 probabilities of an HxWx3 uint8 image (H, W >= win)."""
    h, w = rgb.shape[:2]
    boxes = [(y, x) for y in window_starts(h, win, overlap) for x in window_starts(w, win, overlap)]
    weight = blend_weight(win, overlap)
    acc, norm = None, np.zeros((h, w), np.float32)
    for i in range(0, len(boxes), batch):
        chunk = boxes[i:i + batch]
        arr = np.stack([rgb[y:y + win, x:x + win] for y, x in chunk]).transpose(0, 3, 1, 2)
        x_t = torch.from_numpy(np.ascontiguousarray(arr)).to(device).float().div_(255.0)
        probs = forward_tta(model, x_t, tta).cpu().numpy()
        if acc is None:
            acc = np.zeros((probs.shape[1], h, w), np.float32)
        for (y, x), p in zip(chunk, probs, strict=True):
            acc[:, y:y + win, x:x + win] += p * weight
            norm[y:y + win, x:x + win] += weight
    out = acc / norm[None]
    out[:, rgb.max(axis=2) == 0] = 0.0
    return out


def to_u8(prob: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(prob, 0.0, 1.0) * 255.0).astype(np.uint8)


def read_rgb(tile_id: str) -> np.ndarray:
    from vineyard.contracts.ids import file_name_from_tile_id
    from vineyard.geo.raster import read_tile

    return read_tile(AI_TILES / file_name_from_tile_id(tile_id))


def resolve_tiles(spec: str) -> list[str]:
    """examples | vineyard (>= 1 base canopy + examples) | all (every tile) | comma-separated ids."""
    import pandas as pd

    if spec == "examples":
        return list(EXAMPLE_TILES)
    if spec == "vineyard":
        can = pd.read_parquet(AI_RUNS / BASE_RUN / "annset" / "canopies.parquet", columns=["tile_id"])
        return sorted(set(can["tile_id"].astype(str)) | set(EXAMPLE_TILES))
    if spec == "all":
        return sorted(p.stem for p in AI_TILES.glob("*.tif"))
    return [t.strip() for t in spec.split(",") if t.strip()]


def model_from_weights(weights: Path, device: torch.device, encoder: str | None = None) -> MultiHeadUnet:
    card = weights.parent / "model_card.json"
    meta = json.loads(card.read_text()) if card.is_file() else {}
    enc = encoder or meta.get("encoder", "resnet34")
    n_heads = len(meta.get("heads", HEAD_NAMES))
    return load_model(weights, enc, n_heads, device)


def run(weights: Path, tiles: Sequence[str], out_dir: Path, *, tta: int, batch: int,
        encoder: str | None = None) -> dict[str, float]:
    device = pick_device()
    model = model_from_weights(weights, device, encoder)
    timings: dict[str, float] = {}
    for tile_id in tiles:
        t0 = time.time()
        probs = predict_image(model, read_rgb(tile_id), device, tta=tta, batch=batch)
        for name, p in zip(HEAD_NAMES, probs, strict=False):
            path = out_dir / name / f"{tile_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), to_u8(p)):
                raise OSError(f"cannot write {path}")
        timings[tile_id] = time.time() - t0
        log.info("infer %s: %.1f s", tile_id, timings[tile_id])
    if timings:
        log.info("infer: %d tiles, %.2f s/tile (tta=%d, device=%s)", len(timings),
                 float(np.mean(list(timings.values()))), tta, device)
    return timings


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--tiles", default="examples")
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--encoder", default=None, help="default: from model_card.json next to the weights")
    ap.add_argument("--out", type=Path, default=PREDS / "canopy" / "v1")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.weights.is_file():
        raise SystemExit(f"weights not found: {args.weights}")
    run(args.weights, resolve_tiles(args.tiles), args.out, tta=args.tta, batch=args.batch, encoder=args.encoder)


if __name__ == "__main__":
    main()
