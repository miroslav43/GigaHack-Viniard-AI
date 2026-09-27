"""Synthetic Sireț3 validation set: VAL instances (DroneWaste val sites + UAVVaste val) pasted with a fixed
seed onto windows of the held-out tiles; a quarter of the windows stay empty (false-positive check).

Saved once to ``work/data/siret3_waste/synval/{images/*.jpg, synval.json}``; ``synval.json`` holds per
window the GT boxes (window px) and the don't-care boxes (unknown `kept` candidates, nodata excluded).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from fte.convert.coco import read_rgb, write_jpeg
from fte.convert.paste import PasteSpec, instance_bank, paste_many
from fte.waste.sources import WasteSources, read_window

log = logging.getLogger(__name__)

SYNVAL_SUBDIR: Final = "synval"
EMPTY_FRAC: Final = 0.25
SEED: Final = 20260926


@dataclass(frozen=True)
class SynvalItem:
    key: str
    gt: np.ndarray  # (N, 4) xyxy window px
    ignore: np.ndarray  # (M, 4)


def synval_dir(src: WasteSources) -> Path:
    return src.siret3_dir / SYNVAL_SUBDIR


def build_synval(src: WasteSources, seed: int = SEED, force: bool = False) -> Path:
    """Write the set (idempotent unless ``force``); returns the directory."""
    out = synval_dir(src)
    if (out / "synval.json").is_file() and not force:
        return out
    bank = instance_bank(src.dronewaste, src.dw_val) + instance_bank(src.uavvaste, src.uav_val)
    if not bank:
        raise ValueError("no validation instances to paste")
    rng = np.random.default_rng(seed)
    items = []
    for win in src.holdout:
        img, lbl = read_window(src.siret3_dir, win)
        boxes: list[tuple] = []
        if rng.random() >= EMPTY_FRAC:
            img, _, boxes = paste_many(img, lbl, bank, PasteSpec(), rng, n=int(rng.integers(1, 4)))
        write_jpeg(out / "images" / f"{win.key}.jpg", img)
        items.append({"key": win.key, "tile_id": win.tile_id, "gt": [list(b) for b in boxes],
                      "ignore": [list(b) for b in win.ignore_boxes]})
    doc = {"seed": seed, "n_instances_bank": len(bank), "items": items}
    (out / "synval.json").write_text(json.dumps(doc), encoding="utf-8")
    log.info("synval: %d windows, %d GT boxes, bank %d", len(items), sum(len(i["gt"]) for i in items), len(bank))
    return out


def load_synval(directory: Path) -> tuple[list[SynvalItem], dict[str, np.ndarray]]:
    """Items + key -> rgb image."""
    doc = json.loads((directory / "synval.json").read_text(encoding="utf-8"))
    items, images = [], {}
    for it in doc["items"]:
        items.append(SynvalItem(it["key"], np.asarray(it["gt"], float).reshape(-1, 4),
                                np.asarray(it["ignore"], float).reshape(-1, 4)))
        images[it["key"]] = read_rgb(directory / "images" / f"{it['key']}.jpg")
    return items, images
