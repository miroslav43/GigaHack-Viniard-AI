"""On-disk format of the NN patch store (design 03 §3 N3), torch-free.

    <store>/img_<shard>.npy   uint8 (n, P, P, 3), one shard per tile
    <store>/lbl_<shard>.npy   uint8 (n, P, P) in {0, 1, 255}
    <store>/manifest.json     StoreManifest: tiles, patches, selection summary, label params, timing

Readers (PatchDataset, training, QA) go through load_manifest / verify_store / read_patch; a holdout
tile anywhere in a manifest raises HoldoutLeakError, which is also an AssertionError.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from vineyard.nn.pseudolabels import IGNORE, PseudoLabelError
from vineyard.pipeline.atomic import atomic_write_json

STORE_FORMAT: Final = 1
MANIFEST_NAME: Final = "manifest.json"
IMG_NAME: Final = "img_{:04d}.npy"
LBL_NAME: Final = "lbl_{:04d}.npy"
KIND_POSITIVE: Final = "positive"
KIND_EMPTY: Final = "empty"
NATIVE_STAT_FIELDS: Final = ("n_small_px", "n_clump_px", "n_tree_px", "n_ineligible_px", "n_overhang_px")


class HoldoutLeakError(PseudoLabelError, AssertionError):
    """A holdout (reference) tile reached the patch store."""


@dataclass(frozen=True)
class PatchRecord:
    tile_id: str
    shard: int
    index: int
    x0: int  # label px (column) of the patch origin in the tile
    y0: int
    n_pos: int
    n_neg: int
    n_ignore: int


@dataclass(frozen=True)
class TileEntry:
    tile_id: str
    kind: str
    shard: int
    n_patches: int
    n_pos: int  # whole-tile label counts (before patch filtering)
    n_neg: int
    n_ignore: int
    n_small_px: int  # native px of in-corridor vegetation per ignore rule
    n_clump_px: int
    n_tree_px: int
    n_ineligible_px: int
    seconds: float
    n_overhang_px: int = 0  # vegetation ignored just outside the corridors (last, so older manifests load)


@dataclass(frozen=True)
class StoreManifest:
    key: str
    format: int
    run_id: str
    created_at: str
    patch_px: int
    gsd_m: float
    holdout: tuple[str, ...]
    tiles: tuple[TileEntry, ...]
    patches: tuple[PatchRecord, ...]
    selection: Mapping[str, Any]
    params: Mapping[str, Any]
    failed: tuple[str, ...]
    build_s: float

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "selection": dict(self.selection), "params": dict(self.params)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> StoreManifest:
        return cls(**{**raw, "holdout": tuple(raw["holdout"]), "failed": tuple(raw["failed"]),
                      "tiles": tuple(TileEntry(**t) for t in raw["tiles"]),
                      "patches": tuple(PatchRecord(**p) for p in raw["patches"])})


def write_manifest(store_dir: Path, manifest: StoreManifest) -> Path:
    return atomic_write_json(Path(store_dir) / MANIFEST_NAME, manifest.to_dict())


def load_manifest(store_dir: Path) -> StoreManifest:
    path = Path(store_dir) / MANIFEST_NAME
    try:
        return StoreManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PseudoLabelError(f"unreadable patch-store manifest: {type(exc).__name__}: {exc}", path=str(path)) from exc


def check_holdout(tile_ids: Sequence[str], holdout: Sequence[str], where: str) -> None:
    """HoldoutLeakError naming every holdout tile found in `tile_ids`."""
    leaked = sorted(set(tile_ids) & set(holdout))
    if leaked:
        raise HoldoutLeakError(f"holdout tiles in the patch store: {', '.join(leaked)}", where=where)


def verify_store(store_dir: Path, holdout: Sequence[str]) -> StoreManifest:
    """Load the manifest and assert that no holdout tile is in it (tiles or patches)."""
    manifest = load_manifest(store_dir)
    ids = [t.tile_id for t in manifest.tiles] + [p.tile_id for p in manifest.patches]
    check_holdout(ids, holdout, str(store_dir))
    return manifest


def _read_row(path: Path, index: int, record: PatchRecord) -> np.ndarray:
    if not path.is_file():
        raise PseudoLabelError("patch shard missing", path=str(path), tile_id=record.tile_id, shard=record.shard)
    arr = np.load(path, mmap_mode="r")
    if not 0 <= index < arr.shape[0]:
        raise PseudoLabelError("patch index outside the shard", path=str(path), index=index, n=arr.shape[0])
    # copying drops the last reference to the memmap, so no file handle stays open (spawned workers x shards)
    return np.array(arr[index])


def read_patch(store_dir: Path, record: PatchRecord) -> tuple[np.ndarray, np.ndarray]:
    """(uint8 (P, P, 3) image, uint8 (P, P) label) of one stored patch."""
    root = Path(store_dir)
    return (_read_row(root / IMG_NAME.format(record.shard), record.index, record),
            _read_row(root / LBL_NAME.format(record.shard), record.index, record))


def store_stats(manifest: StoreManifest, store_dir: Path) -> dict[str, Any]:
    """Patch / tile counts, label class balance over the stored patches, and bytes on disk."""
    pos = sum(p.n_pos for p in manifest.patches)
    neg = sum(p.n_neg for p in manifest.patches)
    ign = sum(p.n_ignore for p in manifest.patches)
    total = max(pos + neg + ign, 1)
    size = sum(f.stat().st_size for f in Path(store_dir).iterdir() if f.is_file())
    kinds = {k: sum(t.kind == k for t in manifest.tiles) for k in (KIND_POSITIVE, KIND_EMPTY)}
    return {"n_tiles": len(manifest.tiles), "n_tiles_by_kind": kinds, "n_patches": len(manifest.patches),
            "pos_frac": pos / total, "neg_frac": neg / total, "ignore_frac": ign / total,
            "pos_frac_of_labeled": pos / max(pos + neg, 1), "bytes": size, "build_s": manifest.build_s,
            "ignore_value": IGNORE}
