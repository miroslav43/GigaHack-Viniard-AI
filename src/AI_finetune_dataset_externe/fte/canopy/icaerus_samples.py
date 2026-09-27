"""ICAERUS YOLO-seg crops -> canopy training samples at the Sireț3 GSD (2.5 cm/px).

Each RGBA crop (~500x502 at 1.4 cm/px) is rasterised to an instance map at native resolution, then
RGB is degraded through the 3.52 cm capture GSD to 2.5 cm and the instance map / alpha are resized
with nearest neighbour to the same shape. Labels: c0 = vine fg, c1 = contact band, c2 = ground
vegetation; alpha == 0 (outside the labelled 5x5 m cell) is 255 in every channel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fte.convert.gsd import degrade_to_target, resize_mask
from fte.convert.targets import (
    GroundVegSpec,
    contact_band,
    dilate,
    fg_from_instances,
    ground_veg_labels,
    stack_labels,
)
from fte.convert.yolo_seg import rasterize_instances, read_yolo_seg

log = logging.getLogger(__name__)

SRC_GSD_M = 0.014
SPLITS = ("train", "val")
CONTACT_RADIUS_PX = 2
VINE_REGION_DILATE_PX = 4


@dataclass(frozen=True, eq=False)
class IcaerusSample:
    name: str
    split: str
    rgb: np.ndarray  # uint8 HxWx3 at 2.5 cm/px
    lbl: np.ndarray  # uint8 HxWx3 {0, 1, 255}
    inst: np.ndarray  # uint16 HxW instance ids at 2.5 cm/px (0 = background)
    valid: np.ndarray  # bool HxW (alpha > 0)


def read_rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(RGB uint8, alpha-valid bool) of a PNG; a PNG without alpha is fully valid."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot decode image {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        raise ValueError(f"{path}: expected 8-bit image, got {img.dtype}")
    if img.shape[2] == 4:
        return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB), img[..., 3] > 0
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB), np.ones(img.shape[:2], bool)


def build_sample(image_path: Path, label_path: Path, split: str, src_gsd: float = SRC_GSD_M) -> IcaerusSample:
    rgb, valid = read_rgba(image_path)
    h, w = rgb.shape[:2]
    inst_native = rasterize_instances(read_yolo_seg(label_path, w, h), (h, w))
    rgb_t = degrade_to_target(rgb, src_gsd)
    shape = rgb_t.shape[:2]
    inst = resize_mask(inst_native, shape)
    valid_t = resize_mask(valid.astype(np.uint8), shape) > 0
    return IcaerusSample(image_path.stem, split, rgb_t, labels_for(rgb_t, inst, valid_t), inst, valid_t)


def labels_for(rgb: np.ndarray, inst: np.ndarray, valid: np.ndarray) -> np.ndarray:
    fg = fg_from_instances(inst)
    contact = contact_band(inst, CONTACT_RADIUS_PX).astype(np.uint8)
    ground = ground_veg_labels(rgb, dilate(fg, VINE_REGION_DILATE_PX), GroundVegSpec())
    return stack_labels(fg, contact, ground, ignore=~valid)


def list_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"ICAERUS split {split!r} missing: {img_dir}")
    pairs = []
    for img in sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.is_file():
            log.warning("icaerus: no label for %s, skipped", img.name)
            continue
        pairs.append((img, lbl))
    return pairs


def save_sample(sample: IcaerusSample, out_dir: Path) -> Path:
    path = out_dir / sample.split / f"{sample.name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rgb=sample.rgb, lbl=sample.lbl, inst=sample.inst, valid=sample.valid)
    return path


def load_sample(path: Path) -> IcaerusSample:
    with np.load(path) as z:
        return IcaerusSample(path.stem, path.parent.name, z["rgb"], z["lbl"], z["inst"], z["valid"])


def build_all(root: Path, out_dir: Path) -> dict[str, list[str]]:
    """Convert every split under ``root`` into ``out_dir/<split>/<name>.npz``; returns relative paths."""
    out: dict[str, list[str]] = {}
    for split in SPLITS:
        paths = [save_sample(build_sample(img, lbl, split), out_dir) for img, lbl in list_pairs(root, split)]
        out[split] = [str(p.relative_to(out_dir)) for p in paths]
        log.info("icaerus %s: %d samples", split, len(paths))
    return out
