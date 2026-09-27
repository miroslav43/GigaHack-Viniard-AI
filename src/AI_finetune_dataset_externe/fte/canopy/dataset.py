"""Patch dataset for the canopy model: Sireț3 vineyard / ICAERUS mosaic / Sireț3 negatives mixture.

Every item is an independent random 512x512 patch (``len`` = patches per epoch):
  * Sireț3 vineyard (default 50 %): a window of a vineyard tile, centred on a row corridor with
    probability ``corridor_p`` (coarse weights from the store), else anywhere on the tile;
  * ICAERUS (25 %): 2x2 mosaic of four labelled crops (each ~281 px at 2.5 cm, random dihedral);
  * Sireț3 negatives (25 %): a window of a no-vineyard tile (c0 = 0), weighted to green areas.
RGB windows come from the original GeoTIFF tiles, labels from the store's tiled label rasters. Items
are (image float32 3xHxW in [0, 1], labels uint8 3xHxW, source id).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from fte.canopy.icaerus_samples import IcaerusSample, load_sample
from fte.convert.targets import IGNORE
from fte.nn.augment import AugmentSpec, augment, dihedral

log = logging.getLogger(__name__)

SRC_VINE, SRC_ICAERUS, SRC_NEG = 0, 1, 2
SOURCE_NAMES = ("siret3_vineyard", "icaerus", "siret3_negative")
TILE_PX = 2048
MAX_TRIES = 4
MAX_IGNORE_FRAC = 0.9


@dataclass(frozen=True)
class MixSpec:
    p_vine: float = 0.5
    p_icaerus: float = 0.25
    p_neg: float = 0.25
    patch: int = 512
    read_px: int = 608  # >= patch / (1 - scale_jitter): the scale-jitter crop never upsamples much
    corridor_p: float = 0.85
    cell_px: int = 64
    augment: AugmentSpec = field(default_factory=AugmentSpec)

    def probs(self, has_icaerus: bool, has_neg: bool) -> np.ndarray:
        p = np.array([self.p_vine, self.p_icaerus if has_icaerus else 0.0, self.p_neg if has_neg else 0.0])
        if p.sum() <= 0:
            raise ValueError("mixture has no available source")
        return p / p.sum()


# ------------------------------------------------------------------ pure sampling helpers


def draw_centre(weights: np.ndarray, cell_px: int, weighted_p: float, rng: np.random.Generator,
                tile_px: int = TILE_PX) -> tuple[int, int]:
    """(cx, cy) patch centre: a cell drawn by weight (prob ``weighted_p``) + uniform jitter, else uniform."""
    w = np.clip(weights.astype(np.float64).ravel(), 0.0, None)
    if rng.random() < weighted_p and w.sum() > 0:
        k = int(rng.choice(w.size, p=w / w.sum()))
        cy, cx = divmod(k, weights.shape[1])
        return (int(cx * cell_px + rng.integers(0, cell_px)), int(cy * cell_px + rng.integers(0, cell_px)))
    return int(rng.integers(0, tile_px)), int(rng.integers(0, tile_px))


def window_at(cx: int, cy: int, size: int, tile_px: int = TILE_PX) -> tuple[int, int]:
    """Top-left (x0, y0) of a ``size`` window centred at (cx, cy), clamped inside the tile."""
    if size > tile_px:
        raise ValueError(f"window {size} larger than tile {tile_px}")
    x0 = min(max(cx - size // 2, 0), tile_px - size)
    y0 = min(max(cy - size // 2, 0), tile_px - size)
    return int(x0), int(y0)


def mosaic_2x2(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Four (rgb HxWx3, lbl HxWxK) pairs -> one 2S x 2S canvas (S = largest side); padding = 0 / 255."""
    if len(parts) != 4:
        raise ValueError(f"mosaic needs 4 parts, got {len(parts)}")
    s = max(max(r.shape[0], r.shape[1]) for r, _ in parts)
    k = parts[0][1].shape[2]
    img = np.zeros((2 * s, 2 * s, 3), np.uint8)
    lbl = np.full((2 * s, 2 * s, k), IGNORE, np.uint8)
    for n, (r, lab) in enumerate(parts):
        y, x = (n // 2) * s, (n % 2) * s
        img[y:y + r.shape[0], x:x + r.shape[1]] = r
        lbl[y:y + lab.shape[0], x:x + lab.shape[1]] = lab
    return img, lbl


# ------------------------------------------------------------------ dataset


class CanopyPatches(Dataset):
    def __init__(self, store_dir: Path, n_patches: int, spec: MixSpec | None = None, seed: int = 0) -> None:
        self.store_dir = Path(store_dir)
        index = json.loads((self.store_dir / "index.json").read_text())
        self.vine = [t for t in index["tiles"] if t["kind"] != "negative"]
        self.neg = [t for t in index["tiles"] if t["kind"] == "negative"]
        if not self.vine:
            raise ValueError(f"store {store_dir} has no vineyard tiles")
        self.icaerus_paths = [self.store_dir / "icaerus" / p for p in index.get("icaerus", {}).get("train", [])]
        self.weights = {t["tile_id"]: np.load(self.store_dir / t["weights"]) for t in index["tiles"]}
        spec = spec or MixSpec()
        self.n_patches, self.spec, self.seed = int(n_patches), spec, seed
        self.p = spec.probs(bool(self.icaerus_paths), bool(self.neg))
        self._icaerus: list[IcaerusSample] | None = None
        self._handles: dict[str, rasterio.DatasetReader] = {}
        self._pid = os.getpid()

    def __len__(self) -> int:
        return self.n_patches

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_handles"], state["_icaerus"] = {}, None
        return state

    def _open(self, path: str) -> rasterio.DatasetReader:
        if self._pid != os.getpid():
            self._handles, self._pid = {}, os.getpid()
        if path not in self._handles:
            self._handles[path] = rasterio.open(path)
        return self._handles[path]

    def _read(self, path: str, x0: int, y0: int, size: int) -> np.ndarray:
        arr = self._open(path).read(window=Window(x0, y0, size, size))
        return np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))

    def _siret(self, tiles: list[dict], rng: np.random.Generator, weighted_p: float) -> tuple[np.ndarray, np.ndarray]:
        size = self.spec.read_px
        for _ in range(MAX_TRIES):
            t = tiles[int(rng.integers(0, len(tiles)))]
            cx, cy = draw_centre(self.weights[t["tile_id"]], self.spec.cell_px, weighted_p, rng)
            x0, y0 = window_at(cx, cy, size)
            lbl = self._read(str(self.store_dir / t["label"]), x0, y0, size)
            if (lbl[..., 0] == IGNORE).mean() < MAX_IGNORE_FRAC:
                break
        return self._read(t["rgb"], x0, y0, size), lbl

    def _icaerus_mosaic(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self._icaerus is None:
            self._icaerus = [load_sample(p) for p in self.icaerus_paths]
        picks = rng.choice(len(self._icaerus), size=4, replace=len(self._icaerus) < 4)
        parts = [dihedral(self._icaerus[int(i)].rgb, self._icaerus[int(i)].lbl, int(rng.integers(0, 4)),
                          bool(rng.integers(0, 2))) for i in picks]
        return mosaic_2x2(parts)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
        src = int(rng.choice(3, p=self.p))
        if src == SRC_VINE:
            img, lbl = self._siret(self.vine, rng, self.spec.corridor_p)
        elif src == SRC_ICAERUS:
            img, lbl = self._icaerus_mosaic(rng)
        else:
            img, lbl = self._siret(self.neg, rng, 0.7)
        img2, lbl2 = augment(img, lbl, self.spec.patch, self.spec.augment, rng)
        return img2, lbl2, src

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rng = np.random.default_rng([self.seed, i, int.from_bytes(os.urandom(4), "little")])
        img, lbl, src = self.sample(rng)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div_(255.0)
        y = torch.from_numpy(np.ascontiguousarray(lbl.transpose(2, 0, 1)))
        return x, y, src
