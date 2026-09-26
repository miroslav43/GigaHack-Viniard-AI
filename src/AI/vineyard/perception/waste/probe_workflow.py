"""Waste ML workflows behind `vineyard waste probe-data | probe-train | sam3-check` (design 03 W2/W3b).

probe-data : UAVVaste positives (plain + pasted on Sireț3 backgrounds) and Sireț3 negatives of a model run
             -> work/probe/{pos,neg} crop stores (needs the run's cache/waste/<tile>.parquet).
probe-train: OpenCLIP embeddings of the stores (cached next to them) -> grouped-CV logistic probe ->
             models/<waste.probe.name>/<version>/{probe.npz, model_card.json}.
sam3-check : SAM 3 on a few 512 px candidate crops -> load time, s/crop, crops per time budget.
Torch-free at import; OpenCLIP / SAM 3 load inside the functions.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from vineyard.errors import VineyardError
from vineyard.geo.raster import read_tile
from vineyard.perception.waste.clip_embed import (
    ClipParams,
    embed_crop_store,
    load_openclip,
    probe_data_from_stores,
)
from vineyard.perception.waste.crop_store import read_crop_store
from vineyard.perception.waste.negatives import (
    NegativeParams,
    background_sampler,
    build_negative_store,
    run_tile_ids,
    work_sources,
)
from vineyard.perception.waste.positives import PositiveParams, build_positive_store
from vineyard.perception.waste.probe import CvReport, ProbeModel, ProbeParams, save_probe, train_probe
from vineyard.perception.waste.sam3_adapter import (
    NEGATIVE_REASONS,
    FileLister,
    Sam3Loader,
    Sam3Params,
    Sam3Status,
    SamResult,
    load_sam3,
    verify_with_budget,
)
from vineyard.perception.waste.types import TILE_EXTENT_PX, Candidate, candidate_from_record
from vineyard.perception.waste.uavvaste import DATASET_DIR, extract_dataset, load_dataset
from vineyard.perception.waste.verify_ml import SamPlanParams, TileReader, plan_sam_jobs, window_reader

if TYPE_CHECKING:
    from vineyard.config import AppConfig
    from vineyard.perception.waste.verify_ml import ClipEmbedder

PROBE_SUBDIR: Final = "probe"
POS_STORE: Final = "pos"
NEG_STORE: Final = "neg"
UAVVASTE_SUBDIR: Final = Path("external") / "uavvaste"
RUNS_SUBDIR: Final = "runs"
WASTE_CACHE: Final = Path("cache") / "waste"
TILES_SUBDIR: Final = "tiles"
TILE_SUFFIX: Final = ".tif"
FULL_GATE: Final = 1.0  # sam3-check: every chosen candidate is eligible


class WasteWorkflowError(VineyardError):
    """A waste ML workflow cannot run (missing inputs)."""


@dataclass(frozen=True)
class ProbeStores:
    pos: Path
    neg: Path
    n_pos: int
    n_neg: int
    n_tiles: int


def probe_root(cfg: AppConfig) -> Path:
    return Path(cfg.paths.work_dir) / PROBE_SUBDIR


def uavvaste_root(cfg: AppConfig) -> Path:
    return Path(cfg.paths.work_dir) / UAVVASTE_SUBDIR / DATASET_DIR


def extract_uavvaste(cfg: AppConfig, zip_path: Path) -> Path:
    """md5-checked extraction into work/external/uavvaste/dataset (idempotent)."""
    return extract_dataset(zip_path, uavvaste_root(cfg).parent)


def build_probe_stores(cfg: AppConfig, run_id: str, out_root: Path | None = None) -> ProbeStores:
    """Positive and negative crop stores for the tiles of model run `run_id`."""
    work = Path(cfg.paths.work_dir)
    run_dir = work / RUNS_SUBDIR / run_id
    if not run_dir.is_dir():
        raise WasteWorkflowError("model run not found", run_dir=str(run_dir))
    tiles = run_tile_ids(run_dir)
    missing = [t for t in tiles if not (work / WASTE_CACHE / f"{t}.parquet").is_file()]
    if missing:
        raise WasteWorkflowError("waste tile cache missing; run the waste stage on this run first",
                                 n_missing=len(missing), first=missing[:3])
    root = out_root or probe_root(cfg)
    sources = work_sources(work)
    npar = NegativeParams.from_config(cfg)
    ds = load_dataset(uavvaste_root(cfg))
    pos = build_positive_store(ds, PositiveParams.from_config(cfg), root / POS_STORE,
                               background=background_sampler(tiles, sources, npar))
    neg = build_negative_store(tiles, sources, npar, root / NEG_STORE)
    return ProbeStores(pos, neg, len(read_crop_store(pos).records), len(read_crop_store(neg).records), len(tiles))


def train_probe_from_stores(
    cfg: AppConfig,
    stores: Sequence[Path],
    version: str,
    *,
    train_cmd: str,
    models_dir: Path | None = None,
    embedder: ClipEmbedder | None = None,
) -> tuple[ProbeModel, CvReport]:
    """Embed the stores (cached), train + calibrate the probe, save it under the models dir."""
    probe_cfg = cfg.waste.probe
    clip = embedder if embedder is not None else load_openclip(ClipParams.from_config(probe_cfg))
    model_id = getattr(clip, "model_id", None)
    if model_id is None:
        raise WasteWorkflowError("the CLIP embedder has no model_id", embedder=type(clip).__name__)
    for store in stores:
        embed_crop_store(store, clip, probe_cfg.embed_batch)
    data, sources = probe_data_from_stores(stores, model_id)
    result = train_probe(data, ProbeParams.from_config(cfg.waste, cfg.runtime.seed), model_id, version)
    saved = save_probe(result, models_dir or cfg.paths.models_dir, probe_cfg.name, train_data=sources,
                       train_cmd=train_cmd)
    return saved, result.cv


# ------------------------------------------------------------------ SAM 3 speed check


@dataclass(frozen=True)
class Sam3Check:
    status: Sam3Status
    results: tuple[tuple[str, SamResult | None], ...]  # (cand_key, result) in job order

    @property
    def seconds(self) -> tuple[float, ...]:
        return tuple(r.seconds for _, r in self.results if r is not None)

    @property
    def steady_s_per_crop(self) -> float:
        """Mean s/crop without the first (warm-up) crop; the only one when a single crop ran."""
        s = self.seconds
        if not s:
            return float("nan")
        return statistics.fmean(s[1:]) if len(s) > 1 else s[0]


def _tile_candidates(work: Path, tile_id: str) -> tuple[Candidate, ...]:
    path = work / WASTE_CACHE / f"{tile_id}.parquet"
    if not path.is_file():
        raise WasteWorkflowError("waste tile cache missing; run the waste stage first", path=str(path))
    return tuple(candidate_from_record(r) for r in pd.read_parquet(path).to_dict("records"))


def check_candidates(pool: Sequence[Candidate], n_crops: int) -> tuple[Candidate, ...]:
    """Filter survivors first, then white tube rejects (the hard negatives), at most n_crops."""
    live = sorted((c for c in pool if not c.rejected), key=lambda c: c.cand_key)
    tubes = sorted((c for c in pool if c.is_white and c.reject_reason in NEGATIVE_REASONS),
                   key=lambda c: c.cand_key)
    step = max(1, len(tubes) // max(1, n_crops))
    return tuple([*live, *tubes[::step]][:n_crops])


def sam3_check(
    cfg: AppConfig,
    tile_ids: Sequence[str],
    n_crops: int,
    *,
    loader: Sam3Loader | None = None,
    lister: FileLister | None = None,
    read: TileReader | None = None,
) -> Sam3Check:
    """Load SAM 3 once and verify up to n_crops candidate crops of the tiles (budget from config)."""
    work = Path(cfg.paths.work_dir)
    params = Sam3Params.from_config(cfg.waste.sam3, cfg.paths.models_dir)
    verifier, status = load_sam3(params, loader, lister)
    if verifier is None:
        return Sam3Check(status, ())
    pool = tuple(c for t in tile_ids for c in _tile_candidates(work, t))
    chosen = check_candidates(pool, n_crops)
    plan = SamPlanParams.from_config(cfg.waste, int(TILE_EXTENT_PX))
    jobs = plan_sam_jobs(chosen, dict.fromkeys((c.cand_key for c in chosen), FULL_GATE), pool, plan)
    reader = read or (lambda t: read_tile(work / TILES_SUBDIR / f"{t}{TILE_SUFFIX}"))
    results = verify_with_budget(verifier, jobs, window_reader(reader), plan.budget_s, time.monotonic)
    return Sam3Check(status, tuple((j.cand_key, r) for j, r in zip(jobs, results, strict=True)))
