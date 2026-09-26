"""Crop store shared by the probe positives and negatives (design 03 W4/W5).

<dir>/crops.npy (N, px, px, 3) uint8 written through a memmap + <dir>/manifest.json (records, meta, crops
sha256). The manifest is removed first and written last, so a store without one is incomplete.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from vineyard.errors import VineyardError
from vineyard.pipeline.atomic import atomic_path, atomic_write_json

CROPS_FILE: Final = "crops.npy"
MANIFEST_FILE: Final = "manifest.json"
STORE_VERSION: Final = 1
RGB: Final = 3
_HASH_CHUNK: Final = 1 << 20


class CropStoreError(VineyardError):
    """Crop store read / write failure."""


@dataclass(frozen=True)
class CropRecord:
    key: str
    source: str
    group: str
    label: int
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "source": self.source, "group": self.group, "label": self.label, **self.meta}


@dataclass(frozen=True)
class CropStore:
    crops: np.ndarray  # read-only memmap (N, px, px, 3) uint8
    records: tuple[CropRecord, ...]
    meta: Mapping[str, Any]
    crops_sha256: str


_RECORD_KEYS: Final = ("key", "source", "group", "label")


def _record_from_json(d: Mapping[str, Any]) -> CropRecord:
    extra = {k: v for k, v in d.items() if k not in _RECORD_KEYS}
    return CropRecord(
        key=str(d["key"]), source=str(d["source"]), group=str(d["group"]), label=int(d["label"]), meta=extra
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _fill(mm: np.ndarray, crops: Iterable[np.ndarray], crop_px: int) -> int:
    n = 0
    for crop in crops:
        if n >= len(mm):
            raise CropStoreError("more crops than records", n_records=len(mm))
        if crop.shape != (crop_px, crop_px, RGB) or crop.dtype != np.uint8:
            raise CropStoreError(
                "crop has the wrong shape or dtype", index=n, shape=crop.shape, dtype=str(crop.dtype)
            )
        mm[n] = crop
        n += 1
    return n


def write_crop_store(
    out_dir: Path,
    crops: Iterable[np.ndarray],
    records: Sequence[CropRecord],
    crop_px: int,
    meta: Mapping[str, Any],
) -> Path:
    """Stream crops into crops.npy (memmap), then write the manifest; returns out_dir."""
    out = Path(out_dir)
    keys = [r.key for r in records]
    if len(set(keys)) != len(keys):
        raise CropStoreError("duplicate crop keys", out_dir=str(out), n=len(keys), n_unique=len(set(keys)))
    (out / MANIFEST_FILE).unlink(missing_ok=True)  # the store is invalid until the new manifest lands
    with atomic_path(out / CROPS_FILE) as tmp:
        mm = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.uint8, shape=(len(records), crop_px, crop_px, RGB)
        )
        n = _fill(mm, crops, crop_px)
        mm.flush()
        del mm
        if n != len(records):
            raise CropStoreError(
                "fewer crops than records", out_dir=str(out), n_crops=n, n_records=len(records)
            )
    doc = {
        "version": STORE_VERSION,
        "n": len(records),
        "crop_px": crop_px,
        "crops_sha256": _sha256(out / CROPS_FILE),
        "meta": dict(meta),
        "records": [r.to_json() for r in records],
    }
    atomic_write_json(out / MANIFEST_FILE, doc)
    return out


def read_crop_store(store_dir: Path) -> CropStore:
    folder = Path(store_dir)
    manifest = folder / MANIFEST_FILE
    if not manifest.is_file():
        raise CropStoreError("crop store manifest missing (store incomplete?)", path=str(manifest))
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        crops = np.load(folder / CROPS_FILE, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CropStoreError("unreadable crop store", path=str(folder), error=str(exc)) from exc
    records = tuple(_record_from_json(r) for r in doc["records"])
    if len(crops) != len(records):
        raise CropStoreError(
            "crops / records count mismatch", path=str(folder), n_crops=len(crops), n_records=len(records)
        )
    return CropStore(crops=crops, records=records, meta=doc["meta"], crops_sha256=doc["crops_sha256"])
