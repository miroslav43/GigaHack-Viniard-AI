"""Training/validation sources of the waste detector, loaded once in the main process.

DroneWaste is split by site (3 sites holding ~15 % of the positives -> val), UAVVaste by its official
split; Sireț3 windows come from ``fte.convert.waste_negatives`` (train tiles only; held-out tiles feed
the synthetic validation set). The instance bank for copy-paste holds TRAIN instances only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from fte.convert.coco import (
    StoreImage,
    WasteStore,
    choose_val_groups,
    load_store,
    positives_per_group,
    read_rgb,
    split_store,
)
from fte.convert.paste import Instance, instance_bank
from fte.convert.waste_negatives import OUT_DIR as SIRET3_DIR
from fte.convert.waste_negatives import ignore_label
from fte.paths import DATA

log = logging.getLogger(__name__)

DRONEWASTE_DIR: Final = DATA / "dronewaste"
UAVVASTE_DIR: Final = DATA / "uavvaste"
NODATA_MAX: Final = 2  # a window pixel with max(R, G, B) <= 2 is nodata


@dataclass(frozen=True)
class Siret3Window:
    key: str
    tile_id: str
    kind: str
    has_vineyard: bool
    ignore_boxes: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class WasteSources:
    dronewaste: WasteStore
    uavvaste: WasteStore
    dw_train: tuple[StoreImage, ...]
    dw_val: tuple[StoreImage, ...]
    uav_train: tuple[StoreImage, ...]
    uav_val: tuple[StoreImage, ...]
    windows: tuple[Siret3Window, ...]  # train split
    holdout: tuple[Siret3Window, ...]
    bank: tuple[Instance, ...]
    val_sites: tuple[str, ...]
    siret3_dir: Path = SIRET3_DIR

    def counts(self) -> dict[str, int]:
        def pos(imgs: tuple[StoreImage, ...]) -> int:
            return sum(i.n_waste > 0 for i in imgs)

        return {"dw_train_images": len(self.dw_train), "dw_train_pos": pos(self.dw_train),
                "dw_val_images": len(self.dw_val), "dw_val_pos": pos(self.dw_val),
                "dw_train_objects": sum(i.n_waste for i in self.dw_train),
                "uav_train_images": len(self.uav_train), "uav_val_images": len(self.uav_val),
                "uav_train_objects": sum(i.n_waste for i in self.uav_train),
                "siret3_train_windows": len(self.windows), "siret3_holdout_windows": len(self.holdout),
                "paste_bank_instances": len(self.bank)}


def _windows(index: pd.DataFrame, split: str) -> tuple[Siret3Window, ...]:
    sub = index[index["split"] == split]
    return tuple(Siret3Window(r.key, r.tile_id, r.kind, bool(r.has_vineyard),
                              tuple(tuple(b) for b in json.loads(r.ignore_boxes)))
                 for r in sub.itertuples(index=False))


def load_sources(dw_dir: Path = DRONEWASTE_DIR, uav_dir: Path = UAVVASTE_DIR,
                 siret3_dir: Path = SIRET3_DIR, val_frac: float = 0.15) -> WasteSources:
    dw = load_store(dw_dir, "dronewaste")
    uav = load_store(uav_dir, "uavvaste")
    val_sites = choose_val_groups(positives_per_group(dw.images), frac=val_frac, n_groups=3)
    dw_train, dw_val = split_store(dw, val_sites)
    uav_train, uav_val = split_store(uav, ())
    index = pd.read_parquet(siret3_dir / "index.parquet")
    bank = instance_bank(dw, dw_train) + instance_bank(uav, uav_train)
    src = WasteSources(dw, uav, dw_train, dw_val, uav_train, uav_val, _windows(index, "train"),
                       _windows(index, "holdout"), bank, val_sites, siret3_dir)
    log.info("waste sources: val sites %s, %s", val_sites, src.counts())
    return src


def read_window(siret3_dir: Path, win: Siret3Window) -> tuple[np.ndarray, np.ndarray]:
    """(rgb, label) of a Sireț3 window: 0, 255 in unknown boxes and on nodata."""
    rgb = read_rgb(siret3_dir / "windows" / f"{win.key}.jpg")
    valid = rgb.max(axis=2) > NODATA_MAX
    return rgb, ignore_label(win.ignore_boxes, valid, size=rgb.shape[0])
