"""Torch dataset over a patch store (design 03 §3 N3) with deterministic augmentation.

A sample is addressed by (epoch, i): its RNG is np.random.default_rng((seed, epoch, manifest index)), so
a run is reproducible whatever the worker count. EpochSampler yields those keys; like torch's
DistributedSampler it carries the epoch (set_epoch) because persistent DataLoader workers keep the
dataset but the sampler runs in the main process. Geometric augmentation (flips, rot90) moves image
and label together; colour jitter (brightness, contrast, saturation) touches the image only, never
the hue (hue is the vine signal). Images are float32 (3, P, P) in [0, 1]; labels int64 {0, 1, 255}.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from vineyard.nn.pseudolabels import PseudoLabelError
from vineyard.nn.store_format import PatchRecord, read_patch, verify_store

if TYPE_CHECKING:
    from vineyard.config import TrainConfig

U8_SCALE: Final = 255.0
LUMA: Final = np.array([0.299, 0.587, 0.114], dtype=np.float32)
N_ROT: Final = 4
FLIP_P: Final = 0.5
N_JITTER: Final = 3  # brightness, contrast, saturation
SPAWN: Final = "spawn"

type SampleKey = int | tuple[int, int]


@dataclass(frozen=True)
class AugmentParams:
    flips: bool
    rot90: bool
    colour_jitter: float  # factors drawn from U[1 - j, 1 + j]

    def __post_init__(self) -> None:
        if not 0.0 <= self.colour_jitter < 1.0:
            raise ValueError(f"colour_jitter must be in [0, 1), got {self.colour_jitter}")

    @classmethod
    def from_config(cls, train: TrainConfig) -> AugmentParams:
        return cls(flips=True, rot90=True, colour_jitter=train.colour_jitter)


NO_AUGMENT: Final = AugmentParams(flips=False, rot90=False, colour_jitter=0.0)


def colour_jitter(x: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    """New float32 HWC image in [0, 1]: brightness scale, contrast around the mean grey, saturation around grey."""
    out = x * np.float32(brightness)
    mean_grey = float((out @ LUMA).mean())
    out = (out - mean_grey) * np.float32(contrast) + np.float32(mean_grey)
    grey = (out @ LUMA)[..., None]
    out = grey + (out - grey) * np.float32(saturation)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def augment_sample(img: np.ndarray, lbl: np.ndarray, rng: np.random.Generator,
                   p: AugmentParams) -> tuple[np.ndarray, np.ndarray]:
    """uint8 (P, P, 3) + (P, P) -> float32 image in [0, 1] and uint8 label, both freshly allocated."""
    if img.shape[:2] != lbl.shape:
        raise ValueError(f"image shape {img.shape} and label shape {lbl.shape} differ")
    k = int(rng.integers(N_ROT)) if p.rot90 else 0
    flip = bool(rng.random() < FLIP_P) if p.flips else False
    img_t, lbl_t = np.rot90(img, k), np.rot90(lbl, k)
    if flip:
        img_t, lbl_t = img_t[:, ::-1], lbl_t[:, ::-1]
    x = img_t.astype(np.float32) / np.float32(U8_SCALE)
    if p.colour_jitter > 0:
        b, c, s = rng.uniform(1.0 - p.colour_jitter, 1.0 + p.colour_jitter, size=N_JITTER)
        x = colour_jitter(x, float(b), float(c), float(s))
    return np.ascontiguousarray(x), np.ascontiguousarray(lbl_t)


class PatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Map-style dataset; key = patch position i (epoch 0) or (epoch, i)."""

    def __init__(self, store_dir: Path, *, seed: int, augment: AugmentParams, holdout: Sequence[str] = (),
                 indices: Sequence[int] | None = None) -> None:
        manifest = verify_store(store_dir, holdout)
        n = len(manifest.patches)
        chosen = tuple(range(n)) if indices is None else tuple(int(i) for i in indices)
        if any(not 0 <= i < n for i in chosen):
            raise PseudoLabelError("dataset indices outside the patch store", n_patches=n, store=str(store_dir))
        self.store_dir = Path(store_dir)
        self.seed = int(seed)
        self.augment = augment
        self.indices = chosen
        self.records: tuple[PatchRecord, ...] = tuple(manifest.patches[i] for i in chosen)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: SampleKey) -> tuple[torch.Tensor, torch.Tensor]:
        epoch, i = key if isinstance(key, tuple) else (0, key)
        if not 0 <= i < len(self.records):
            raise IndexError(f"patch position {i} outside [0, {len(self.records)})")
        img, lbl = read_patch(self.store_dir, self.records[i])
        rng = np.random.default_rng((self.seed, int(epoch), self.indices[i]))
        x, y = augment_sample(img, lbl, rng, self.augment)
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(y.astype(np.int64))


class EpochSampler(Sampler[tuple[int, int]]):
    """Yields (epoch, i) for a per-epoch permutation of range(n) seeded by (seed, epoch)."""

    def __init__(self, n: int, seed: int, shuffle: bool = True) -> None:
        self.n = int(n)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        order = np.random.default_rng((self.seed, self.epoch)).permutation(self.n) if self.shuffle else range(self.n)
        return iter([(self.epoch, int(i)) for i in order])

    def __len__(self) -> int:
        return self.n


def build_loader(dataset: PatchDataset, sampler: EpochSampler, *, batch_size: int, num_workers: int,
                 drop_last: bool = False) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """DataLoader with spawn workers kept alive across epochs; call it under `if __name__ == "__main__"`."""
    workers = max(int(num_workers), 0)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      persistent_workers=workers > 0, multiprocessing_context=SPAWN if workers > 0 else None,
                      drop_last=drop_last, pin_memory=False)
