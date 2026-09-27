from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fte.convert.coco import (
    StoreError,
    StoreImage,
    StoreObject,
    WasteStore,
    choose_val_groups,
    load_store,
    rasterize,
    split_store,
    write_jpeg,
)


def test_rasterize_roles() -> None:
    waste = StoreObject("waste", (2, 2, 4, 4), ((2, 2, 6, 2, 6, 6, 2, 6),))
    ignore = StoreObject("ignore", (0, 0, 10, 10), ())
    bg = StoreObject("background", (12, 12, 4, 4), ())
    lbl = rasterize([waste, ignore, bg], (20, 20))
    assert lbl[4, 4] == 1  # waste drawn after ignore
    assert lbl[8, 8] == 255
    assert lbl[14, 14] == 0
    assert set(np.unique(lbl)) <= {0, 1, 255}


def test_choose_val_groups_closest_to_fraction() -> None:
    counts = {"a": 50, "b": 5, "c": 5, "d": 5, "e": 35}
    assert choose_val_groups(counts, frac=0.15, n_groups=3) == ("b", "c", "d")
    with pytest.raises(StoreError):
        choose_val_groups({"a": 1, "b": 1}, n_groups=3)


def test_split_prefers_preassigned(tmp_path: Path) -> None:
    imgs = (StoreImage(1, "a.jpg", "s1", "pos", 4, 4, ()), StoreImage(2, "b.jpg", "s2", "pos", 4, 4, ()),
            StoreImage(3, "c.jpg", "s1", "pos", 4, 4, (), split="val"))
    train, val = split_store(WasteStore("x", tmp_path, imgs), ["s2"])
    assert [i.image_id for i in train] == [1]
    assert sorted(i.image_id for i in val) == [2, 3]


def test_load_store_drops_missing_images(tmp_path: Path) -> None:
    write_jpeg(tmp_path / "images" / "a.jpg", np.zeros((8, 8, 3), np.uint8))
    doc = {"images": [{"id": 1, "file_name": "a.jpg", "site": "s", "role": "pos", "width": 8, "height": 8},
                      {"id": 2, "file_name": "missing.jpg", "site": "s", "role": "pos", "width": 8, "height": 8}],
           "annotations": [{"image_id": 1, "category_id": 5, "role": "waste", "bbox": [1, 1, 2, 2],
                            "segmentation": [[1, 1, 3, 1, 3, 3]]}], "categories": []}
    (tmp_path / "annotations.json").write_text(json.dumps(doc))
    store = load_store(tmp_path, "t")
    assert len(store.images) == 1 and store.images[0].n_waste == 1
