"""Crop store: streamed memmap + manifest written last; validation errors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vineyard.perception.waste.crop_store import (
    CROPS_FILE,
    MANIFEST_FILE,
    CropRecord,
    CropStoreError,
    read_crop_store,
    write_crop_store,
)


def test_round_trip(tmp_path: Path) -> None:
    crops = [np.full((8, 8, 3), i, np.uint8) for i in range(3)]
    recs = tuple(CropRecord(f"k{i}", "src", f"g{i}", i % 2, {"x": i}) for i in range(3))
    out = write_crop_store(tmp_path / "s", iter(crops), recs, 8, {"kind": "t"})
    store = read_crop_store(out)
    assert store.records == recs and store.meta == {"kind": "t"}
    assert store.crops.shape == (3, 8, 8, 3) and int(store.crops[2, 0, 0, 0]) == 2
    assert len(store.crops_sha256) == 64
    assert (out / CROPS_FILE).is_file() and (out / MANIFEST_FILE).is_file()


def test_crop_store_validation(tmp_path: Path) -> None:
    rec = (CropRecord("a", "x", "g", 0), CropRecord("a", "x", "g", 0))
    with pytest.raises(CropStoreError, match="duplicate"):
        write_crop_store(tmp_path / "s", iter([]), rec, 8, {})
    one = (CropRecord("a", "x", "g", 0),)
    with pytest.raises(CropStoreError, match="fewer"):
        write_crop_store(tmp_path / "s", iter([]), one, 8, {})
    with pytest.raises(CropStoreError, match="more"):
        write_crop_store(tmp_path / "s", iter([np.zeros((8, 8, 3), np.uint8)] * 2), one, 8, {})
    with pytest.raises(CropStoreError, match="shape"):
        write_crop_store(tmp_path / "s", iter([np.zeros((4, 4, 3), np.uint8)]), one, 8, {})
    with pytest.raises(CropStoreError, match="manifest"):
        read_crop_store(tmp_path / "s")
    assert not (tmp_path / "s" / "crops.npy").exists()
