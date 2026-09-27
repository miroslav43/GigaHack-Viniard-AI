from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fte.canopy.icaerus_samples import build_all, build_sample, load_sample
from fte.convert.targets import IGNORE


def _write_crop(root: Path, split: str, name: str) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    bgra = np.zeros((100, 100, 4), np.uint8)
    bgra[..., :3] = (90, 110, 120)  # BGR soil
    bgra[:, :50, 3] = 255  # left half labelled
    bgra[20:80, 10:22, :3] = (40, 150, 50)  # green vine (BGR)
    bgra[20:80, 23:35, :3] = (40, 150, 50)
    cv2.imwrite(str(root / "images" / split / f"{name}.png"), bgra)
    (root / "labels" / split / f"{name}.txt").write_text(
        "0 0.10 0.20 0.22 0.20 0.22 0.80 0.10 0.80\n0 0.23 0.20 0.35 0.20 0.35 0.80 0.23 0.80\n")


def test_build_sample_degrades_and_labels(tmp_path: Path) -> None:
    _write_crop(tmp_path, "train", "a")
    s = build_sample(tmp_path / "images/train/a.png", tmp_path / "labels/train/a.txt", "train")
    assert s.rgb.shape == (56, 56, 3) and s.lbl.shape == (56, 56, 3) and s.inst.shape == (56, 56)
    assert set(np.unique(s.inst)) == {0, 1, 2}
    assert (s.lbl[~s.valid] == IGNORE).all() and s.valid[:, :20].all() and not s.valid[:, 40:].any()
    assert s.lbl[28, 9, 0] == 1 and s.lbl[28, 2, 0] == 0
    assert (s.lbl[..., 1] == 1).any()  # the two plants are 1 px apart at native res -> contact band
    assert s.rgb[28, 9, 1] > s.rgb[28, 9, 0]  # BGR -> RGB: green stays green


def test_build_all_roundtrip(tmp_path: Path) -> None:
    for split, name in (("train", "a"), ("val", "b")):
        _write_crop(tmp_path / "raw", split, name)
    out = build_all(tmp_path / "raw", tmp_path / "store")
    assert out == {"train": ["train/a.npz"], "val": ["val/b.npz"]}
    s = load_sample(tmp_path / "store" / "val" / "b.npz")
    assert s.split == "val" and s.inst.dtype == np.uint16 and s.valid.dtype == bool
