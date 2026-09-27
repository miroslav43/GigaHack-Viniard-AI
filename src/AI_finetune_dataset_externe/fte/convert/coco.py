"""Waste stores in the common "scaled COCO" layout (DroneWaste, UAVVaste) and their label rasterisation.

A store is ``<root>/images/*.jpg`` (already at 2.5 cm/px) plus ``<root>/annotations.json``::

    {images: [{id, file_name, site, role: pos|neg_pile|neg_empty, src_gsd_m, width, height, [split]}],
     annotations: [{image_id, category_id, role: waste|ignore|background, bbox (xywh), segmentation}]}

Labels: ``waste`` -> 1, ``ignore`` -> 255, ``background`` (rubble, soil, wood, vehicles) stays 0; waste is
drawn after ignore so an item annotated inside an ambiguous pile stays positive.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

WASTE: Final = "waste"
IGNORE_ROLE: Final = "ignore"
BACKGROUND: Final = "background"
ROLES: Final = frozenset({WASTE, IGNORE_ROLE, BACKGROUND})
IGNORE_LABEL: Final = 255
POS_IMAGE: Final = "pos"


class StoreError(ValueError):
    """Malformed waste store."""


@dataclass(frozen=True)
class StoreObject:
    role: str
    bbox: tuple[float, float, float, float]  # x, y, w, h (px at 2.5 cm/px)
    polygons: tuple[tuple[float, ...], ...]  # flat x0, y0, x1, y1, ...


@dataclass(frozen=True)
class StoreImage:
    image_id: int
    file_name: str
    group: str  # DroneWaste site / UAVVaste flight batch (split unit)
    role: str
    width: int
    height: int
    objects: tuple[StoreObject, ...]
    split: str | None = None  # pre-assigned split (UAVVaste); None = decide by group

    @property
    def n_waste(self) -> int:
        return sum(o.role == WASTE for o in self.objects)


@dataclass(frozen=True)
class WasteStore:
    name: str
    root: Path
    images: tuple[StoreImage, ...]

    def image_path(self, img: StoreImage) -> Path:
        return self.root / "images" / img.file_name


def _object(d: Mapping) -> StoreObject:
    role = str(d["role"])
    if role not in ROLES:
        raise StoreError(f"unknown annotation role {role!r}")
    bbox = tuple(float(v) for v in d["bbox"])
    if len(bbox) != 4:
        raise StoreError(f"bbox must be xywh, got {d['bbox']!r}")
    polys = tuple(tuple(float(v) for v in p) for p in d.get("segmentation") or [] if isinstance(p, list))
    return StoreObject(role, bbox, tuple(p for p in polys if len(p) >= 6))


def load_store(root: Path, name: str) -> WasteStore:
    """Parse ``<root>/annotations.json``; images whose JPEG is missing are dropped."""
    path = Path(root) / "annotations.json"
    if not path.is_file():
        raise StoreError(f"{name}: annotations missing at {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    by_image: dict[int, list[StoreObject]] = {}
    for ann in doc["annotations"]:
        by_image.setdefault(int(ann["image_id"]), []).append(_object(ann))
    images = []
    for d in doc["images"]:
        if not (Path(root) / "images" / d["file_name"]).is_file():
            continue
        images.append(StoreImage(
            image_id=int(d["id"]), file_name=str(d["file_name"]), group=str(d.get("site", "misc")),
            role=str(d["role"]), width=int(d["width"]), height=int(d["height"]),
            objects=tuple(by_image.get(int(d["id"]), ())), split=d.get("split"),
        ))
    return WasteStore(name=name, root=Path(root), images=tuple(sorted(images, key=lambda i: i.image_id)))


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise StoreError(f"cannot read image {path}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def write_jpeg(path: Path, rgb: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise StoreError(f"cannot write {path}")


def object_mask(obj: StoreObject, shape_hw: tuple[int, int]) -> np.ndarray:
    """Bool mask of one object (polygons, or its bbox when it has none)."""
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if obj.polygons:
        pts = [np.round(np.asarray(p, dtype=np.float64).reshape(-1, 2)).astype(np.int32) for p in obj.polygons]
        cv2.fillPoly(mask, pts, 1)
    else:
        x, y, w, h = obj.bbox
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = int(np.ceil(x + w)), int(np.ceil(y + h))
        mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1
    return mask.astype(bool)


def rasterize(objects: Sequence[StoreObject], shape_hw: tuple[int, int]) -> np.ndarray:
    """uint8 label: 0 background, 255 ignore, 1 waste (waste drawn last)."""
    label = np.zeros(shape_hw, dtype=np.uint8)
    for obj in objects:
        if obj.role == IGNORE_ROLE:
            label[object_mask(obj, shape_hw)] = IGNORE_LABEL
    for obj in objects:
        if obj.role == WASTE:
            label[object_mask(obj, shape_hw)] = 1
    return label


def xywh_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in bbox)
    return x, y, x + w, y + h


def choose_val_groups(pos_per_group: Mapping[str, int], frac: float = 0.15, n_groups: int = 3) -> tuple[str, ...]:
    """The ``n_groups`` groups whose positive count is closest to ``frac`` of the total (ties: names)."""
    groups = sorted(g for g, n in pos_per_group.items() if n > 0)
    if len(groups) <= n_groups:
        raise StoreError(f"need more than {n_groups} groups with positives, got {len(groups)}")
    total = sum(pos_per_group[g] for g in groups)
    target = frac * total
    best = min(itertools.combinations(groups, n_groups),
               key=lambda c: (abs(sum(pos_per_group[g] for g in c) - target), c))
    return tuple(best)


def split_store(store: WasteStore, val_groups: Sequence[str]) -> tuple[tuple[StoreImage, ...], tuple[StoreImage, ...]]:
    """(train, val): a pre-assigned ``split`` wins; otherwise val = images of ``val_groups``."""
    val_set = frozenset(val_groups)

    def is_val(img: StoreImage) -> bool:
        return img.split in ("val", "test") if img.split else img.group in val_set

    return (tuple(i for i in store.images if not is_val(i)), tuple(i for i in store.images if is_val(i)))


def positives_per_group(images: Sequence[StoreImage]) -> dict[str, int]:
    out: dict[str, int] = {}
    for img in images:
        out[img.group] = out.get(img.group, 0) + img.n_waste
    return out
