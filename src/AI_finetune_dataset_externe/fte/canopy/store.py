"""Canopy training store: Sireț3 pseudo-label rasters + ICAERUS npz samples + index.json.

Layout (``work/stores/canopy_v1``)::

    index.json                    tiles (id, kind, paths, stats) + ICAERUS sample lists
    labels/<tile>.tif             uint8 3-band (c0, c1, c2), tiled 256 + deflate (windowed reads)
    weights/<tile>.npy            coarse patch-centre weights (32x32 for 64 px cells)
    icaerus/{train,val}/<n>.npz   rgb / lbl / inst / valid at 2.5 cm/px

RGB is not copied: training reads windows of the original GeoTIFF tiles under src/AI/work/tiles.

CLI: ``python -m fte.canopy.store [--out DIR] [--max-vineyard N] [--n-neg 40] [--workers 6]``
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry.base import BaseGeometry

from fte.canopy import icaerus_samples
from fte.convert.siret3_pseudo import (
    PseudoSpec,
    TileKind,
    axis_distance_px,
    lines_to_px,
    pseudo_labels,
    sampling_weights,
)
from fte.paths import AI_CACHE, AI_ROOT, AI_RUNS, AI_TILES, BASE_RUN, EXAMPLE_TILES, RAW, STORES

log = logging.getLogger(__name__)

STORE_VERSION = 1
DEFAULT_STORE = STORES / "canopy_v1"
REVIEW_CSV = AI_ROOT / "configs" / "tile_review.csv"
MIN_CANOPIES = 30
N_NEGATIVES = 40
N_NEG_TOP_VEG = 30  # most vegetated negatives; the rest drawn at random for diversity
UNTRUSTED_REVIEW = ("missed", "partial")
MIN_VALID_FRAC = 0.2


@dataclass(frozen=True)
class TileEntry:
    tile_id: str
    kind: str
    n_canopies: int


def _review_status(path: Path) -> dict[str, str]:
    if not path.is_file():
        log.warning("tile review file missing: %s", path)
        return {}
    with path.open(newline="") as fh:
        return {r["tile_id"]: r["status"] for r in csv.DictReader(fh)}


def select_tiles(canopy_counts: pd.Series, status: pd.DataFrame, review: dict[str, str], *,
                 min_canopies: int = MIN_CANOPIES, n_neg: int = N_NEGATIVES, seed: int = 0,
                 exclude: Sequence[str] = EXAMPLE_TILES) -> list[TileEntry]:
    """Vineyard tiles (>= min_canopies, not examples) + vegetated negative tiles (status no_vineyard)."""
    vine = [TileEntry(t, TileKind.PARTIAL.value if review.get(t) in UNTRUSTED_REVIEW else TileKind.VINEYARD.value,
                      int(n)) for t, n in canopy_counts.items() if n >= min_canopies and t not in exclude]
    neg_ok = status[(status["status"] == "no_vineyard") & (status["n_row_pieces"] == 0)
                    & ~status["tile_id"].isin(list(review)) & ~status["tile_id"].isin(list(exclude))
                    & ~status["issues"].fillna("").str.contains("block_too_few_rows")]
    neg_ok = neg_ok.sort_values("veg_frac", ascending=False)
    top = list(neg_ok["tile_id"].iloc[:N_NEG_TOP_VEG])
    veg = neg_ok.set_index("tile_id")["veg_frac"]
    rest = [t for t in veg.index if t not in top and veg[t] > 0.05]
    rng = np.random.default_rng(seed)
    extra = list(rng.choice(rest, size=min(len(rest), max(0, n_neg - len(top))), replace=False)) if rest else []
    negs = [TileEntry(str(t), TileKind.NEGATIVE.value, 0) for t in (top + extra)[:n_neg]]
    return sorted(vine, key=lambda e: e.tile_id) + negs


def _write_label_tif(path: Path, lbl: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    meta = {"driver": "GTiff", "width": lbl.shape[1], "height": lbl.shape[0], "count": 3, "dtype": "uint8",
            "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate",
            "crs": profile.get("crs"), "transform": profile.get("transform")}
    with rasterio.open(tmp, "w", **meta) as ds:
        ds.write(np.transpose(lbl, (2, 0, 1)))
    tmp.replace(path)


def build_tile(entry: TileEntry, canopies: list[BaseGeometry], axes: list[BaseGeometry], out_dir: Path) -> dict:
    """Pseudo-labels of one tile -> labels/<tile>.tif + weights/<tile>.npy; returns its index record."""
    from vineyard.contracts.ids import file_name_from_tile_id
    from vineyard.geo.raster import rasterize_utm
    from vineyard.geo.tiling import tile_ref, utm_to_px
    from vineyard.pipeline.tile_cache import load_valid_mask

    tref = tile_ref(entry.tile_id)
    tif = AI_TILES / file_name_from_tile_id(entry.tile_id)
    with rasterio.open(tif) as ds:
        rgb = np.transpose(ds.read(), (1, 2, 0))
        profile = dict(ds.profile)
    valid = load_valid_mask(AI_CACHE, entry.tile_id).astype(bool)
    kind = TileKind(entry.kind)
    shape = rgb.shape[:2]
    canopy = rasterize_utm(canopies, tref, shape) if canopies else np.zeros(shape, np.uint8)
    dist = axis_distance_px(lines_to_px(axes, lambda xy: utm_to_px(tref, xy)), shape)
    lbl = pseudo_labels(rgb, valid, canopy, dist, kind, PseudoSpec())
    label_rel, weights_rel = f"labels/{entry.tile_id}.tif", f"weights/{entry.tile_id}.npy"
    _write_label_tif(out_dir / label_rel, lbl, profile)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / weights_rel, sampling_weights(lbl, dist, kind))
    c0 = lbl[..., 0]
    return {**asdict(entry), "label": label_rel, "weights": weights_rel, "rgb": str(tif),
            "valid_frac": float(valid.mean()), "c0_pos": float((c0 == 1).mean()),
            "c0_neg": float((c0 == 0).mean()), "c2_pos": float((lbl[..., 2] == 1).mean())}


def _geoms_by_tile(frame: gpd.GeoDataFrame) -> dict[str, list[BaseGeometry]]:
    return {str(t): list(g.geometry) for t, g in frame.groupby("tile_id")}


def build_siret3(entries: Sequence[TileEntry], out_dir: Path, workers: int) -> list[dict]:
    from vineyard.annset.io import read_annset

    annset = read_annset(AI_RUNS / BASE_RUN / "annset")
    can, axes = _geoms_by_tile(annset.layer("canopies")), _geoms_by_tile(annset.layer("row_pieces"))
    jobs = [(e, can.get(e.tile_id, []) if e.kind != TileKind.NEGATIVE.value else [],
             axes.get(e.tile_id, []) if e.kind != TileKind.NEGATIVE.value else []) for e in entries]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(build_tile, e, c, a, out_dir) for e, c, a in jobs]
        records = []
        for e, f in zip(entries, futures, strict=True):
            try:
                records.append(f.result())
            except Exception as exc:  # keep going: one bad tile must not kill the store
                log.error("store: tile %s failed: %s", e.tile_id, exc)
    return [r for r in records if r["valid_frac"] >= MIN_VALID_FRAC]


def build_store(out_dir: Path, *, max_vineyard: int | None, n_neg: int, workers: int,
                icaerus_root: Path = RAW / "icaerus") -> dict:
    t0 = time.time()
    run_dir = AI_RUNS / BASE_RUN
    counts = pd.read_parquet(run_dir / "annset" / "canopies.parquet", columns=["tile_id"])["tile_id"].value_counts()
    status = pd.read_parquet(run_dir / "layers" / "tile_status.parquet",
                             columns=["tile_id", "status", "n_row_pieces", "veg_frac", "issues"])
    entries = select_tiles(counts, status, _review_status(REVIEW_CSV), n_neg=n_neg)
    if max_vineyard is not None:
        vine = [e for e in entries if e.kind != TileKind.NEGATIVE.value][:max_vineyard]
        entries = vine + [e for e in entries if e.kind == TileKind.NEGATIVE.value][:max(1, max_vineyard // 2)]
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = build_siret3(entries, out_dir, workers)
    ica = icaerus_samples.build_all(icaerus_root, out_dir / "icaerus") if (icaerus_root / "images").is_dir() else {}
    if not ica:
        log.warning("store: no ICAERUS data under %s (samples skipped)", icaerus_root)
    index = {"version": STORE_VERSION, "base_run": BASE_RUN, "gsd_m": 0.025, "created": time.strftime("%FT%T"),
             "pseudo_spec": asdict(PseudoSpec()), "tiles": tiles, "icaerus": ica}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, default=str))
    kinds = pd.Series([t["kind"] for t in tiles]).value_counts().to_dict()
    log.info("store %s: %s tiles, icaerus %s, %.0f s", out_dir, kinds, {k: len(v) for k, v in ica.items()},
             time.time() - t0)
    return index


def add_icaerus(out_dir: Path, icaerus_root: Path = RAW / "icaerus") -> dict:
    """Add (or rebuild) the ICAERUS samples of an existing store and update its index.json."""
    index_path = out_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no store index at {index_path}")
    if not (icaerus_root / "images").is_dir():
        raise FileNotFoundError(f"ICAERUS data not downloaded yet: {icaerus_root / 'images'}")
    index = json.loads(index_path.read_text())
    new_index = {**index, "icaerus": icaerus_samples.build_all(icaerus_root, out_dir / "icaerus")}
    tmp = index_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new_index, indent=1, default=str))
    tmp.replace(index_path)
    log.info("store %s: icaerus %s", out_dir, {k: len(v) for k, v in new_index["icaerus"].items()})
    return new_index


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--max-vineyard", type=int, default=None, help="smoke: cap the vineyard tiles")
    ap.add_argument("--n-neg", type=int, default=N_NEGATIVES)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--icaerus-only", action="store_true", help="(re)build only the ICAERUS part of an existing store")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.icaerus_only:
        add_icaerus(args.out)
    else:
        build_store(args.out, max_vineyard=args.max_vineyard, n_neg=args.n_neg, workers=args.workers)


if __name__ == "__main__":
    main()
