"""Sireț3 384 px windows for the waste detector: hard negatives, random backgrounds, held-out tiles.

Per non-example tile: windows centred (jittered) on candidates the rule stage rejected for a look-alike
reason (tube_shape, near_axis, bright_soil, too_small_white, hose, periodic_tube, vehicle), stratified by
reason, plus random windows (more on non-vineyard tiles). Label = 0, except 255 inside the boxes of
`kept` candidates (status unknown: we may not hand-label Sireț3), inside team-confirmed / run waste boxes
and on nodata. ~30 held-out tiles (fixed hash order) only get random windows, used as backgrounds of the
synthetic validation set; the 2 example tiles get none (they are checked whole).

Output: ``work/data/siret3_waste/{windows/*.jpg, index.parquet}``; ``ignore_boxes`` is a JSON list of
window-relative xyxy boxes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from fte.convert.coco import write_jpeg
from fte.paths import AI_CACHE, AI_RUNS, AI_TILES, AI_WORK, BASE_RUN, DATA, EXAMPLE_TILES

log = logging.getLogger(__name__)

OUT_DIR: Final = DATA / "siret3_waste"
WINDOW: Final = 384
TILE_PX: Final = 2048
HARD_REASONS: Final = ("tube_shape", "near_axis", "bright_soil", "too_small_white", "hose",
                       "periodic_tube", "vehicle")
KEPT_REASON: Final = None
UNKNOWN_PAD_PX: Final = 4
MIN_VALID_FRAC: Final = 0.7
CONFIRMED_CSV: Final = AI_WORK.parent / "configs" / "waste_confirmed.csv"


@dataclass(frozen=True)
class WindowPlan:
    per_reason: int = 3
    random_vineyard: int = 2
    random_other: int = 5
    random_holdout: int = 10
    n_holdout: int = 30
    holdout_vineyard_frac: float = 0.6
    jitter_px: int = 96
    seed: int = 0


def tile_table() -> pd.DataFrame:
    """tile_id, has_vineyard of every Sireț3 tile (base run status)."""
    status = pd.read_parquet(AI_RUNS / BASE_RUN / "layers" / "tile_status.parquet",
                             columns=["tile_id", "has_vineyard"])
    return status.sort_values("tile_id").reset_index(drop=True)


def _hash_key(tile_id: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}:{tile_id}".encode()).hexdigest()


def holdout_tiles(tiles: pd.DataFrame, n: int, vineyard_frac: float) -> tuple[str, ...]:
    """Deterministic held-out set: hash order, ``vineyard_frac`` vineyard tiles, never an example."""
    pool = tiles[~tiles["tile_id"].isin(EXAMPLE_TILES)]
    n_vin = int(round(n * vineyard_frac))
    picks: list[str] = []
    for flag, k in ((True, n_vin), (False, n - n_vin)):
        ids = sorted(pool.loc[pool["has_vineyard"] == flag, "tile_id"], key=lambda t: _hash_key(t, "waste"))
        picks.extend(ids[:k])
    return tuple(sorted(picks))


def window_origin(cx: float, cy: float, rng: np.random.Generator, jitter: int,
                  size: int = WINDOW, tile_px: int = TILE_PX) -> tuple[int, int]:
    """Top-left of a ``size`` window around (cx, cy) +- jitter, clipped inside the tile."""
    jx, jy = rng.integers(-jitter, jitter + 1, size=2) if jitter > 0 else (0, 0)
    x0 = int(np.clip(round(cx - size / 2 + jx), 0, tile_px - size))
    y0 = int(np.clip(round(cy - size / 2 + jy), 0, tile_px - size))
    return x0, y0


def boxes_in_window(boxes: np.ndarray, x0: int, y0: int, size: int = WINDOW, pad: float = 0.0) -> list[list[float]]:
    """Boxes (xyxy tile px) intersecting the window, shifted to window px, padded, clipped."""
    out = []
    for xtl, ytl, xbr, ybr in boxes[:, :4]:
        b = [max(0.0, xtl - pad - x0), max(0.0, ytl - pad - y0),
             min(float(size), xbr + pad - x0), min(float(size), ybr + pad - y0)]
        if b[2] > b[0] and b[3] > b[1]:
            out.append([round(v, 1) for v in b])
    return out


def ignore_label(ignore_boxes: Sequence[Sequence[float]], valid: np.ndarray | None,
                 size: int = WINDOW) -> np.ndarray:
    """uint8 (size, size): 0, with 255 inside ``ignore_boxes`` (window px) and on invalid pixels."""
    lbl = np.zeros((size, size), dtype=np.uint8)
    for x0, y0, x1, y1 in ignore_boxes:
        lbl[int(np.floor(y0)):int(np.ceil(y1)), int(np.floor(x0)):int(np.ceil(x1))] = 255
    if valid is not None:
        lbl[~valid] = 255
    return lbl


def _confirmed_boxes(tile_id: str) -> np.ndarray:
    parts = []
    if CONFIRMED_CSV.is_file():
        df = pd.read_csv(CONFIRMED_CSV)
        parts.append(df.loc[df["tile_id"] == tile_id, ["xtl", "ytl", "xbr", "ybr"]].to_numpy(float))
    run = AI_RUNS / BASE_RUN / "layers" / "waste.parquet"
    if run.is_file():
        w = pd.read_parquet(run, columns=["tile_id", "px_xtl", "px_ytl", "px_xbr", "px_ybr"])
        parts.append(w.loc[w["tile_id"] == tile_id].iloc[:, 1:].to_numpy(float))
    return np.concatenate(parts) if parts else np.zeros((0, 4))


def load_candidates(tile_id: str) -> pd.DataFrame:
    path = AI_CACHE / "waste" / f"{tile_id}.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=["xtl", "ytl", "xbr", "ybr", "cx", "cy", "reject_reason"])
    return pd.read_parquet(path, columns=["xtl", "ytl", "xbr", "ybr", "cx", "cy", "reject_reason"])


def _centres(tile_id: str, cands: pd.DataFrame, plan: WindowPlan, n_random: int, hard: bool,
             rng: np.random.Generator) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    if hard:
        for reason in HARD_REASONS:
            sub = cands[cands["reject_reason"] == reason]
            if len(sub):
                idx = rng.choice(len(sub), size=min(plan.per_reason, len(sub)), replace=False)
                out.extend((f"hard:{reason}", float(sub.iloc[i]["cx"]), float(sub.iloc[i]["cy"])) for i in idx)
    out.extend(("random", float(rng.uniform(0, TILE_PX)), float(rng.uniform(0, TILE_PX))) for _ in range(n_random))
    return out


@dataclass(frozen=True)
class TileJob:
    tile_id: str
    has_vineyard: bool
    split: str  # train | holdout
    plan: WindowPlan
    out_dir: Path


def process_tile(job: TileJob) -> list[dict]:
    """Cut and save the windows of one tile; returns their index rows."""
    from vineyard.geo.raster import read_tile
    from vineyard.pipeline.tile_cache import load_valid_mask

    rng = np.random.default_rng(int(_hash_key(job.tile_id, f"win{job.plan.seed}")[:8], 16))
    rgb = read_tile(AI_TILES / f"{job.tile_id}.tif")
    try:
        valid = load_valid_mask(AI_CACHE, job.tile_id)
    except Exception as exc:  # noqa: BLE001 - cache miss: derive from the pixels
        log.warning("%s: no valid mask (%s), using non-black pixels", job.tile_id, exc)
        valid = rgb.max(axis=2) > 0
    cands = load_candidates(job.tile_id)
    kept = cands[cands["reject_reason"].isna()][["xtl", "ytl", "xbr", "ybr"]].to_numpy(float)
    unknown = np.concatenate([kept, _confirmed_boxes(job.tile_id)])
    holdout = job.split == "holdout"
    n_random = job.plan.random_holdout if holdout else (
        job.plan.random_vineyard if job.has_vineyard else job.plan.random_other)
    rows = []
    for kind, cx, cy in _centres(job.tile_id, cands, job.plan, n_random, not holdout, rng):
        x0, y0 = window_origin(cx, cy, rng, job.plan.jitter_px)
        win_valid = valid[y0:y0 + WINDOW, x0:x0 + WINDOW]
        if win_valid.mean() < MIN_VALID_FRAC:
            continue
        key = f"{job.tile_id}_{x0:04d}_{y0:04d}"
        write_jpeg(job.out_dir / "windows" / f"{key}.jpg", rgb[y0:y0 + WINDOW, x0:x0 + WINDOW])
        rows.append({"key": key, "tile_id": job.tile_id, "x0": x0, "y0": y0, "kind": kind,
                     "has_vineyard": job.has_vineyard, "split": job.split,
                     "valid_frac": float(win_valid.mean()),
                     "ignore_boxes": json.dumps(boxes_in_window(unknown, x0, y0, pad=UNKNOWN_PAD_PX))})
    return rows


def build(out_dir: Path = OUT_DIR, plan: WindowPlan = WindowPlan(), workers: int = 6,
          limit_tiles: int | None = None) -> pd.DataFrame:
    tiles = tile_table()
    hold = frozenset(holdout_tiles(tiles, plan.n_holdout, plan.holdout_vineyard_frac))
    jobs = [TileJob(t, bool(v), "holdout" if t in hold else "train", plan, out_dir)
            for t, v in zip(tiles["tile_id"], tiles["has_vineyard"], strict=True) if t not in EXAMPLE_TILES]
    jobs = jobs[:limit_tiles] if limit_tiles else jobs
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(process_tile, jobs, chunksize=4)):
            rows.extend(part)
            if (i + 1) % 50 == 0:
                log.info("siret3 windows: %d / %d tiles, %d windows", i + 1, len(jobs), len(rows))
    index = pd.DataFrame(rows)
    index.to_parquet(out_dir / "index.parquet", index=False)
    log.info("siret3 windows: %s", index.groupby(["split", "kind"]).size().to_dict())
    return index


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-tiles", type=int, default=None)
    args = ap.parse_args()
    idx = build(workers=args.workers, limit_tiles=args.limit_tiles)
    print(idx.groupby(["split", "kind"]).size().to_string())
