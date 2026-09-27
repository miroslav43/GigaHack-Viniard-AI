"""Per-epoch patch mixture for the waste detector (torch Dataset + epoch-slice sampler).

Mixture (fractions of the patches of an epoch): paste-on-Sireț3 0.30, Sireț3 hard negatives 0.25,
DroneWaste 0.30 (of which 30 % empty / pile images as negatives), UAVVaste 0.15. Positive images are
pre-cropped around a random waste object so the random crop of the augmentation keeps it: >= 50 % of the
patches carry a positive. Label-0 pixels of DroneWaste positive images far (> 20 cm) from any
annotation are flagged for a reduced loss weight (unlabelled small items). Index i of epoch e is ``e * n + i``; the sample is a pure function of
(seed, index), so workers need no shared state.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from fte.convert.coco import IGNORE_LABEL, StoreImage, WasteStore, object_mask, rasterize, read_rgb
from fte.convert.paste import PasteSpec, paste_many
from fte.nn.augment import AugmentSpec, augment
from fte.waste.sources import WasteSources, read_window

SOURCES: Final = ("paste", "hardneg", "dronewaste", "uavvaste")
PATCH: Final = 384
WASTE_AUGMENT: Final = AugmentSpec(hue=0.05)
DOWN_MARGIN_PX: Final = 8  # 20 cm
Sample = tuple[np.ndarray, np.ndarray, np.ndarray]  # rgb, label, down-weight flag


@dataclass(frozen=True)
class MixSpec:
    paste: float = 0.30
    hardneg: float = 0.25
    dronewaste: float = 0.30
    uavvaste: float = 0.15
    dw_negative_frac: float = 0.30

    def probs(self) -> np.ndarray:
        p = np.array([self.paste, self.hardneg, self.dronewaste, self.uavvaste], dtype=np.float64)
        if (p < 0).any() or p.sum() <= 0:
            raise ValueError(f"invalid mixture {p}")
        return p / p.sum()


def pad_with_ignore(img: np.ndarray, lbl: np.ndarray, side: int, down: np.ndarray | None = None) -> Sample:
    """Pad to at least side x side: image reflected/edge, label 255, down flag 0 (new arrays)."""
    h, w = lbl.shape
    flag = np.zeros_like(lbl) if down is None else down
    ph, pw = max(0, side - h), max(0, side - w)
    if ph == 0 and pw == 0:
        return img.copy(), lbl.copy(), flag.copy()
    top, left = ph // 2, pw // 2
    pads = ((top, ph - top), (left, pw - left))
    mode = "reflect" if h > ph and w > pw else "edge"
    return (np.pad(img, pads + ((0, 0),), mode=mode),
            np.pad(lbl, pads, mode="constant", constant_values=IGNORE_LABEL),
            np.pad(flag, pads, mode="constant", constant_values=0))


def crop_around(img: np.ndarray, lbl: np.ndarray, down: np.ndarray, centre: tuple[float, float], side: int,
                rng: np.random.Generator) -> Sample:
    """``side`` crop (arrays already >= side) containing ``centre`` at a random position."""
    h, w = lbl.shape
    cx, cy = centre
    lo_x, hi_x = max(0, int(cx) - side + 1), min(w - side, int(cx))
    lo_y, hi_y = max(0, int(cy) - side + 1), min(h - side, int(cy))
    x0 = int(rng.integers(lo_x, hi_x + 1)) if hi_x >= lo_x else int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(rng.integers(lo_y, hi_y + 1)) if hi_y >= lo_y else int(np.clip(cy - side // 2, 0, h - side))
    sl = np.s_[y0:y0 + side, x0:x0 + side]
    return img[sl].copy(), lbl[sl].copy(), down[sl].copy()


def unannotated_far(img_rec: StoreImage, lbl: np.ndarray, margin_px: int) -> np.ndarray:
    """uint8 flag: label-0 pixels farther than ``margin_px`` from any annotated object (any role).

    DroneWaste annotators skipped small / dubious items, so these pixels are only weak negatives.
    """
    annotated = np.zeros(lbl.shape, dtype=np.uint8)
    for obj in img_rec.objects:
        annotated[object_mask(obj, lbl.shape)] = 1
    k = 2 * margin_px + 1
    near = cv2.dilate(annotated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return ((lbl == 0) & (near == 0)).astype(np.uint8)


def _pre_side(patch: int, spec: AugmentSpec) -> int:
    return int(np.ceil(patch / (1.0 - spec.scale_jitter))) + 2


class WasteDataset(Dataset):
    """Samples are generated on the fly from ``WasteSources`` (length = n_per_epoch * max_epochs)."""

    def __init__(self, src: WasteSources, n_per_epoch: int, max_epochs: int, seed: int = 0,
                 mix: MixSpec = MixSpec(), patch: int = PATCH, aug: AugmentSpec = WASTE_AUGMENT,
                 paste: PasteSpec = PasteSpec(), down_margin_px: int = DOWN_MARGIN_PX) -> None:
        self.src, self.n, self.max_epochs, self.seed = src, n_per_epoch, max_epochs, seed
        self.down_margin_px = down_margin_px
        self.mix, self.patch, self.aug, self.paste_spec = mix, patch, aug, paste
        self.pre = _pre_side(patch, aug)
        self.dw_pos = tuple(i for i in src.dw_train if i.n_waste)
        self.dw_neg = tuple(i for i in src.dw_train if not i.n_waste)
        self.uav_pos = tuple(i for i in src.uav_train if i.n_waste)

    def __len__(self) -> int:
        return self.n * self.max_epochs

    def source_of(self, rng: np.random.Generator) -> str:
        return SOURCES[int(rng.choice(len(SOURCES), p=self.mix.probs()))]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """(x float 3xPxP in [0, 1], label uint8 1xPxP, down-weight flag uint8 1xPxP, source index)."""
        rng = np.random.default_rng([self.seed, int(index)])
        name = self.source_of(rng)
        img, lbl, down = getattr(self, f"_{name}")(rng)
        img, lbl2 = augment(img, np.stack([lbl, down], axis=2), self.patch, self.aug, rng)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(np.ascontiguousarray(lbl2[:, :, 0]))[None]
        w = torch.from_numpy(np.ascontiguousarray(lbl2[:, :, 1]))[None]
        return x, y, w, SOURCES.index(name)

    def _window(self, rng: np.random.Generator) -> Sample:
        win = self.src.windows[int(rng.integers(len(self.src.windows)))]
        img, lbl = read_window(self.src.siret3_dir, win)
        return img, lbl, np.zeros_like(lbl)

    def _paste(self, rng: np.random.Generator) -> Sample:
        img, lbl, down = self._window(rng)
        img, lbl, _ = paste_many(img, lbl, self.src.bank, self.paste_spec, rng)
        return img, lbl, down

    def _hardneg(self, rng: np.random.Generator) -> Sample:
        return self._window(rng)

    def _store_sample(self, store: WasteStore, img_rec: StoreImage, rng: np.random.Generator,
                      down_weight: bool) -> Sample:
        rgb = read_rgb(store.image_path(img_rec))
        lbl = rasterize(img_rec.objects, rgb.shape[:2])
        down = unannotated_far(img_rec, lbl, self.down_margin_px) if down_weight and img_rec.n_waste else (
            np.zeros_like(lbl))
        rgb, lbl, down = pad_with_ignore(rgb, lbl, self.pre, down)
        ys, xs = np.nonzero(lbl == 1)
        if len(ys) == 0:
            return rgb, lbl, down
        k = int(rng.integers(len(ys)))
        return crop_around(rgb, lbl, down, (float(xs[k]), float(ys[k])), self.pre, rng)

    def _dronewaste(self, rng: np.random.Generator) -> Sample:
        use_neg = bool(self.dw_neg) and rng.random() < self.mix.dw_negative_frac
        pool = self.dw_neg if use_neg else self.dw_pos
        return self._store_sample(self.src.dronewaste, pool[int(rng.integers(len(pool)))], rng, True)

    def _uavvaste(self, rng: np.random.Generator) -> Sample:
        pool = self.uav_pos
        return self._store_sample(self.src.uavvaste, pool[int(rng.integers(len(pool)))], rng, False)


class EpochSampler(Sampler[int]):
    """Yields ``epoch * n .. epoch * n + n - 1`` in a seeded random order; call ``set_epoch`` first."""

    def __init__(self, n_per_epoch: int, seed: int = 0) -> None:
        self.n, self.seed, self.epoch = n_per_epoch, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[int]:
        order = np.random.default_rng([self.seed, 10_000 + self.epoch]).permutation(self.n)
        return iter((self.epoch * self.n + order).tolist())
