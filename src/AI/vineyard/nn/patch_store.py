"""NN patch store builder (design 03 §3 N3), torch-free: work/nn/stores/<key>/ (format: nn.store_format).

<key> is a sha1 of the config subtrees, the label parameters and the inputs (tile sha256, tile_prep
key, rows file sha256), so a changed input builds a new store. The holdout tiles are checked before
extraction and again on the finished manifest; a leak raises HoldoutLeakError (an AssertionError).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from vineyard.config.hashing import canonical_json, cfg_hash
from vineyard.geo.raster import read_tile
from vineyard.geo.tiling import tile_ref
from vineyard.geo.vector_io import read_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.nn.pseudolabels import (
    LabelCounts,
    LabelParams,
    PseudoLabelError,
    SelectionPolicy,
    TileSelection,
    build_pseudolabel,
    classify_corridor_veg,
    corridor_inputs,
    downsample_rgb,
    empty_tile_label,
    label_counts,
    read_train_tiles_file,
    select_training_tiles,
)
from vineyard.nn.store_format import (
    IMG_NAME,
    KIND_EMPTY,
    KIND_POSITIVE,
    LBL_NAME,
    MANIFEST_NAME,
    NATIVE_STAT_FIELDS,
    STORE_FORMAT,
    HoldoutLeakError,
    PatchRecord,
    StoreManifest,
    TileEntry,
    check_holdout,
    load_manifest,
    read_patch,
    store_stats,
    verify_store,
    write_manifest,
)
from vineyard.pipeline.context import make_run_paths
from vineyard.pipeline.parallel import parallel_map
from vineyard.pipeline.stages.canopy import local_pieces
from vineyard.pipeline.tile_cache import load_valid_mask, load_veg_mask, tile_prep_key
from vineyard.pipeline.tile_index import tile_path

if TYPE_CHECKING:
    from vineyard.config import AppConfig

__all__ = [
    "IMG_NAME", "KIND_EMPTY", "KIND_POSITIVE", "LBL_NAME", "MANIFEST_NAME", "STORE_FORMAT",
    "HoldoutLeakError", "Patch", "PatchRecord", "StoreManifest", "StorePlan", "TileEntry", "TileJob", "TileResult",
    "build_store", "cut_patches", "default_stores_root", "extract_tile_patches", "load_manifest", "patch_offsets",
    "plan_store", "read_patch", "store_stats", "tile_arrays", "verify_store", "write_manifest",
]

STORES_SUBDIR: Final = ("nn", "stores")
KEY_LEN: Final = 16
ROWS_FILE: Final = "rows.parquet"
TILE_STATUS_FILE: Final = "tile_status.parquet"
QA_ISSUES_FILE: Final = "qa_issues.parquet"
ERROR_SEVERITY: Final = "error"
HASH_CHUNK: Final = 1 << 20
CFG_KEYS: Final = ("nn.pseudolabels", "nn.in_gsd_m", "nn.holdout_tiles", "canopy", "orchard")

_log = get_logger("nn.patch_store")


# ------------------------------------------------------------------ jobs


@dataclass(frozen=True, eq=False)
class Patch:
    x0: int
    y0: int
    img: np.ndarray
    lbl: np.ndarray
    counts: LabelCounts


@dataclass(frozen=True)
class TileJob:
    tile_id: str
    kind: str
    shard: int
    tif_path: Path
    cache_dir: Path
    rows_path: Path
    cfg: AppConfig
    params: LabelParams
    patch_px: int
    stride_px: int
    min_labeled_frac: float


@dataclass(frozen=True)
class TileResult:
    entry: TileEntry
    patches: tuple[PatchRecord, ...]


@dataclass(frozen=True)
class StorePlan:
    key: str
    store_dir: Path
    run_id: str
    selection: TileSelection
    jobs: tuple[TileJob, ...]
    holdout: tuple[str, ...]
    patch_px: int
    gsd_m: float
    params: Mapping[str, Any]
    allow_failures: bool


# ------------------------------------------------------------------ patches


def patch_offsets(size: int, patch: int, stride: int) -> tuple[int, ...]:
    """Patch origins along one axis; a last patch is aligned to the end when the stride leaves a gap."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if patch > size:
        raise ValueError(f"patch {patch} px is larger than the raster ({size} px)")
    starts = list(range(0, size - patch + 1, stride))
    return tuple(starts if starts[-1] == size - patch else [*starts, size - patch])


def cut_patches(img: np.ndarray, lbl: np.ndarray, patch: int, stride: int, min_labeled_frac: float) -> tuple[Patch, ...]:
    """Row-major patches (y0 outer, x0 inner) with at least min_labeled_frac non-ignored pixels."""
    if img.shape[:2] != lbl.shape or img.shape[0] != img.shape[1]:
        raise ValueError(f"image shape {img.shape} and label shape {lbl.shape} must be square and equal")
    offsets = patch_offsets(lbl.shape[0], patch, stride)
    out: list[Patch] = []
    for y0 in offsets:
        for x0 in offsets:
            window = lbl[y0 : y0 + patch, x0 : x0 + patch]
            counts = label_counts(window)
            if (counts.n_pos + counts.n_neg) / window.size < min_labeled_frac:
                continue
            out.append(Patch(x0=x0, y0=y0, img=img[y0 : y0 + patch, x0 : x0 + patch].copy(), lbl=window.copy(),
                             counts=counts))
    return tuple(out)


def _positive_label(job: TileJob, veg: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    tile = tile_ref(job.tile_id)
    pieces = local_pieces(read_layer(job.rows_path, "rows"), tile, job.cfg.canopy.rows_margin_m)
    if pieces.empty:
        raise PseudoLabelError("positive training tile has no row pieces", tile_id=job.tile_id,
                               rows=str(job.rows_path))
    inputs = corridor_inputs(pieces, tile, veg, valid, job.cfg, job.params)
    masks = classify_corridor_veg(veg, inputs.labels, inputs.eligible, inputs.tree, job.params)
    stats = {"n_small_px": masks.n_small_px, "n_clump_px": masks.n_clump_px, "n_tree_px": masks.n_tree_px,
             "n_ineligible_px": masks.n_ineligible_px, "n_overhang_px": masks.n_overhang_px}
    return build_pseudolabel(masks.vine, masks.ignore, valid, job.params), stats


def tile_arrays(job: TileJob) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """(NN input RGB, label, native stats) of one tile at the NN resolution."""
    valid = load_valid_mask(job.cache_dir, job.tile_id)
    if job.kind == KIND_POSITIVE:
        label, stats = _positive_label(job, load_veg_mask(job.cache_dir, job.tile_id), valid)
    elif job.kind == KIND_EMPTY:
        label = empty_tile_label(valid, job.params)
        stats = dict.fromkeys(NATIVE_STAT_FIELDS, 0)
    else:
        raise PseudoLabelError("unknown training-tile kind", tile_id=job.tile_id, kind=job.kind)
    return downsample_rgb(read_tile(job.tif_path), valid, job.params.factor), label, stats


def extract_tile_patches(job: TileJob, out_dir: Path) -> TileResult:
    """Worker: label one tile, cut its patches and write its shard pair into out_dir."""
    start = time.perf_counter()
    img, label, stats = tile_arrays(job)
    patches = cut_patches(img, label, job.patch_px, job.stride_px, job.min_labeled_frac)
    if patches:
        np.save(out_dir / IMG_NAME.format(job.shard), np.stack([p.img for p in patches]))
        np.save(out_dir / LBL_NAME.format(job.shard), np.stack([p.lbl for p in patches]))
    records = tuple(PatchRecord(tile_id=job.tile_id, shard=job.shard, index=i, x0=p.x0, y0=p.y0, n_pos=p.counts.n_pos,
                                n_neg=p.counts.n_neg, n_ignore=p.counts.n_ignore) for i, p in enumerate(patches))
    totals = label_counts(label)
    entry = TileEntry(tile_id=job.tile_id, kind=job.kind, shard=job.shard, n_patches=len(records), n_pos=totals.n_pos,
                      n_neg=totals.n_neg, n_ignore=totals.n_ignore, seconds=round(time.perf_counter() - start, 4),
                      **stats)
    return TileResult(entry=entry, patches=records)


@dataclass(frozen=True)
class _Work:
    job: TileJob
    out_dir: Path


def _run_work(work: _Work) -> TileResult:
    return extract_tile_patches(work.job, work.out_dir)


# ------------------------------------------------------------------ planning


def default_stores_root(cfg: AppConfig) -> Path:
    return Path(cfg.paths.work_dir).joinpath(*STORES_SUBDIR)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_status(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "layers" / TILE_STATUS_FILE
    if not path.is_file():
        raise PseudoLabelError("tile_status layer of the model run missing; run assemble first", path=str(path))
    return pd.read_parquet(path).drop(columns=["geometry"], errors="ignore")


def _error_tiles(run_dir: Path) -> frozenset[str]:
    path = run_dir / "qa" / QA_ISSUES_FILE
    if not path.is_file():
        log_event(_log, "nn.qa_issues_missing", path=str(path))
        return frozenset()
    issues = pd.read_parquet(path, columns=["severity", "tile_id"])
    return frozenset(str(t) for t in issues.loc[issues["severity"] == ERROR_SEVERITY, "tile_id"] if t)


def _tile_shas(work_dir: Path, tile_ids: Sequence[str]) -> dict[str, str]:
    path = work_dir / "tile_index.parquet"
    if not path.is_file():
        raise PseudoLabelError("tile_index missing; run ingest first", path=str(path))
    index = pd.read_parquet(path, columns=["tile_id", "sha256"])
    shas = dict(zip(index["tile_id"].astype(str), index["sha256"].astype(str), strict=True))
    missing = sorted(set(tile_ids) - set(shas))
    if missing:
        raise PseudoLabelError("training tiles missing from tile_index", tiles=missing)
    return {t: shas[t] for t in tile_ids}


def _store_key(cfg: AppConfig, params: LabelParams, rows_sha: str, inputs: Sequence[tuple[str, ...]]) -> str:
    payload = {"format": STORE_FORMAT, "cfg": cfg_hash(cfg, CFG_KEYS), "params": asdict(params),
               "min_labeled_frac": cfg.nn.pseudolabels.min_labeled_frac, "rows": rows_sha, "tiles": [list(t) for t in inputs]}
    return hashlib.sha1(canonical_json(payload).encode("utf-8")).hexdigest()[:KEY_LEN]


def _selection(cfg: AppConfig, run_dir: Path) -> TileSelection:
    listed = cfg.nn.pseudolabels.train_tiles_file
    approved = read_train_tiles_file(listed) if listed is not None else None
    return select_training_tiles(_read_status(run_dir), approved, cfg.nn.holdout_tiles,
                                 SelectionPolicy.from_config(cfg), _error_tiles(run_dir))


def plan_store(cfg: AppConfig, run_id: str, *, stores_root: Path | None = None) -> StorePlan:
    """Select the training tiles of model run `run_id` and derive the store key and per-tile jobs."""
    paths = make_run_paths(cfg, run_id)
    selection = _selection(cfg, paths.run_dir)
    rows_path = paths.layers_dir / ROWS_FILE
    if not rows_path.is_file():
        raise PseudoLabelError("rows layer of the model run missing", path=str(rows_path))
    params = LabelParams.from_config(cfg)
    kinds = [(t, KIND_POSITIVE) for t in selection.positives] + [(t, KIND_EMPTY) for t in selection.empties]
    shas = _tile_shas(paths.work_dir, [t for t, _ in kinds])
    inputs = [(t, k, shas[t], tile_prep_key(paths.cache_dir, t)) for t, k in kinds]
    key = _store_key(cfg, params, _sha256_file(rows_path), inputs)
    pl = cfg.nn.pseudolabels
    jobs = tuple(TileJob(tile_id=t, kind=k, shard=i, tif_path=tile_path(paths, t), cache_dir=paths.cache_dir,
                         rows_path=rows_path, cfg=cfg, params=params, patch_px=pl.patch_px, stride_px=pl.patch_stride_px,
                         min_labeled_frac=pl.min_labeled_frac) for i, (t, k) in enumerate(kinds))
    return StorePlan(key=key, store_dir=(stores_root or default_stores_root(cfg)) / key, run_id=run_id,
                     selection=selection, jobs=jobs, holdout=tuple(cfg.nn.holdout_tiles), patch_px=pl.patch_px,
                     gsd_m=cfg.nn.in_gsd_m, params={**asdict(params), "min_labeled_frac": pl.min_labeled_frac},
                     allow_failures=cfg.runtime.allow_failures)


# ------------------------------------------------------------------ build


def _collect(plan: StorePlan, tmp: Path, workers: int) -> tuple[list[TileResult], list[str]]:
    work = [_Work(job=j, out_dir=tmp) for j in plan.jobs]
    results: list[TileResult] = []
    failed: list[str] = []
    for outcome in parallel_map(_run_work, work, key=lambda w: w.job.tile_id, workers=workers):
        if outcome.ok and outcome.value is not None:
            results.append(outcome.value)
            continue
        failed.append(outcome.key)
        log_event(_log, "nn.patch_store.tile_failed", level=logging.ERROR, tile_id=outcome.key, error=outcome.error,
                  traceback=outcome.traceback)
    return results, failed


def _manifest(plan: StorePlan, results: Sequence[TileResult], failed: Sequence[str], build_s: float) -> StoreManifest:
    ordered = sorted(results, key=lambda r: r.entry.shard)
    return StoreManifest(
        key=plan.key, format=STORE_FORMAT, run_id=plan.run_id, created_at=datetime.now().astimezone().isoformat(),
        patch_px=plan.patch_px, gsd_m=plan.gsd_m, holdout=plan.holdout, tiles=tuple(r.entry for r in ordered),
        patches=tuple(p for r in ordered for p in r.patches), selection=plan.selection.summary(), params=plan.params,
        failed=tuple(sorted(failed)), build_s=round(build_s, 3),
    )


def _cached(plan: StorePlan) -> StoreManifest | None:
    if not (plan.store_dir / MANIFEST_NAME).is_file():
        return None
    manifest = verify_store(plan.store_dir, plan.holdout)
    return manifest if manifest.key == plan.key else None


def build_store(plan: StorePlan, *, workers: int = 1) -> StoreManifest:
    """Extract every job's patches (process pool when workers > 1) into plan.store_dir; cached by key."""
    check_holdout([j.tile_id for j in plan.jobs], plan.holdout, "plan")
    cached = _cached(plan)
    if cached is not None:
        log_event(_log, "nn.patch_store.cached", key=plan.key, n_patches=len(cached.patches))
        return cached
    tmp = plan.store_dir.with_name(f".{plan.store_dir.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    start = time.perf_counter()
    results, failed = _collect(plan, tmp, workers)
    if failed and not plan.allow_failures:
        shutil.rmtree(tmp, ignore_errors=True)
        raise PseudoLabelError("patch extraction failed (runtime.allow_failures=false)", failed=sorted(failed))
    manifest = _manifest(plan, results, failed, time.perf_counter() - start)
    write_manifest(tmp, manifest)
    verify_store(tmp, plan.holdout)
    shutil.rmtree(plan.store_dir, ignore_errors=True)
    tmp.rename(plan.store_dir)
    log_event(_log, "nn.patch_store.built", key=plan.key, n_tiles=len(manifest.tiles),
              n_patches=len(manifest.patches), n_failed=len(failed), build_s=manifest.build_s)
    return manifest
