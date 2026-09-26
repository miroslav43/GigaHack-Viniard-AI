"""The work behind `vineyard nn ...` (train from a run, ablation, panels, fetch, bench), torch-free at import:
functions that need torch import nn.infer / nn.train inside their body.

Outputs: <run>/metrics/nn_train_<version>.json, <run>/metrics/nn_ablation.json, reports/nn_ablation.md and
<run>/qa/nn_panels/<tile>.jpg, where <run> is the model run whose pseudo-labels / rows are used.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from vineyard.errors import StageError
from vineyard.logging_setup import get_logger, log_event
from vineyard.nn.ablation import (
    AXES_MODEL,
    AXES_REFERENCE,
    F_WEIGHTS,
    MAIN_WEIGHTS,
    AblationReport,
    run_ablation,
    write_report,
)
from vineyard.nn.panels import disagreement, pick_hard_tiles, render_panel, write_jpeg
from vineyard.nn.probs import load_canopy_prob, prob_png_path
from vineyard.nn.validate import (
    ValTile,
    load_model_rows,
    model_axes,
    prob_to_tile,
    reference_val_tiles,
    tile_reader,
)
from vineyard.nn.weights import (
    F_SUFFIX,
    LoadedWeights,
    WeightsError,
    fetch_weights,
    load_weights,
    training_version,
    weights_exist,
)
from vineyard.perception.vegmask import neg_a
from vineyard.pipeline.context import make_run_paths
from vineyard.pipeline.tile_cache import load_valid_mask, load_veg_mask
from vineyard.pipeline.tile_index import indexed_tile_ids

if TYPE_CHECKING:
    from vineyard.config import AppConfig
    from vineyard.nn.train import BenchResult, TrainParts, TrainResult

REPORTS_DIR: Final = "reports"
ABLATION_MD: Final = "nn_ablation.md"
ABLATION_JSON: Final = "nn_ablation.json"
TRAIN_JSON: Final = "nn_train_{version}.json"
PANELS_DIR: Final = "nn_panels"
PROB_CHANNEL: Final = "canopy_prob"
EVENT_NO_F: Final = "nn.ablation.no_f_weights"
EVENT_NO_CACHE: Final = "nn.panels.no_cache"

_log = get_logger("nn.workflows")

__all__ = ["AblationOutputs", "ablate", "bench", "fetch", "panels_from_cache", "train_from_run", "write_panels"]


@dataclass(frozen=True)
class AblationOutputs:
    report: AblationReport
    json_path: Path
    md_path: Path
    panels: tuple[Path, ...]


# ------------------------------------------------------------------ train


def train_from_run(cfg: AppConfig, run_id: str, *, store: Path | None, smoke: bool, variant: str,
                   reference: str | None, seed: int | None, train_cmd: str,
                   parts: TrainParts | None = None) -> TrainResult:
    """Train on the patch store of model run `run_id` (its default store unless `store` is given),
    validating on the reference AnnSet's holdout tiles; the epoch history goes to <run>/metrics."""
    from vineyard.nn.patch_store import MANIFEST_NAME, plan_store, verify_store
    from vineyard.nn.train import train_model

    store_dir = Path(store) if store is not None else plan_store(cfg, run_id).store_dir
    if not (store_dir / MANIFEST_NAME).is_file():
        flag = " --variant F" if variant == "F" else ""
        raise StageError("patch store not built", store=str(store_dir),
                         hint=f"vineyard nn pseudolabels --run {run_id}{flag}")
    verify_store(store_dir, cfg.nn.holdout_tiles)
    val = reference_val_tiles(cfg, cfg.nn.holdout_tiles, reference or cfg.eval.reference_annset)
    version = training_version(cfg.nn.version, smoke=smoke, variant=variant)
    history = make_run_paths(cfg, run_id).metrics_dir / TRAIN_JSON.format(version=version)
    return train_model(cfg, cfg.runtime.seed if seed is None else seed, store_dir, val, cfg.paths.models_dir,
                       smoke=smoke, variant="F" if variant == "F" else "main", parts=parts, history_path=history,
                       train_cmd=train_cmd)


# ------------------------------------------------------------------ probabilities of the validation tiles


def _predict(cfg: AppConfig, loaded: LoadedWeights, tiles: Sequence[ValTile]) -> tuple[dict[str, np.ndarray], float]:
    from vineyard.nn.infer import load_model, predict_tiles, select_device, sync

    device = select_device(cfg.nn.device, cfg.nn.fallback_device)
    model = load_model(loaded, device)
    sync(device)
    t0 = time.perf_counter()
    probs = predict_tiles(model, [t.net_input for t in tiles], device, cfg.nn.infer_batch_size)
    sync(device)
    per_tile = (time.perf_counter() - t0) / max(len(tiles), 1)
    return {t.tile_id: p for t, p in zip(tiles, probs, strict=True)}, per_tile


def weights_probs(cfg: AppConfig, version: str, tiles: Sequence[ValTile], expected_sha256: str | None
                  ) -> tuple[dict[str, np.ndarray], float] | None:
    """(probabilities per tile, inference s/tile) of weights `version`, or None when they are absent."""
    if not weights_exist(cfg.paths.models_dir, cfg.nn.name, version):
        return None
    return _predict(cfg, load_weights(cfg.paths.models_dir, cfg.nn.name, version, expected_sha256), tiles)


# ------------------------------------------------------------------ panels


def write_panels(cfg: AppConfig, out_dir: Path, probs: Mapping[str, np.ndarray], cache_dir: Path,
                 read_rgb: Callable[[str], np.ndarray]) -> tuple[Path, ...]:
    """One RGB | a* | NN | diff JPEG per tile of `probs` (1024² NN probabilities)."""
    written = []
    for tile_id, prob in sorted(probs.items()):
        rgb = read_rgb(tile_id)
        img = render_panel(rgb, neg_a(rgb, cfg.veg.blur_sigma_px), prob_to_tile(prob), load_veg_mask(cache_dir, tile_id),
                           cfg.nn.prob_threshold)
        written.append(write_jpeg(Path(out_dir) / f"{tile_id}.jpg", img))
    return tuple(written)


def _cached_probs(cfg: AppConfig, cache_dir: Path, tile_ids: Sequence[str]) -> dict[str, np.ndarray]:
    out = {}
    for tile_id in tile_ids:
        prob = load_canopy_prob(prob_png_path(cache_dir, cfg.nn.version, PROB_CHANNEL, tile_id))
        if prob is not None:
            out[tile_id] = prob
    return out


def _hard_tiles(cfg: AppConfig, cache_dir: Path, cached: Mapping[str, np.ndarray], n: int,
                exclude: Sequence[str]) -> tuple[str, ...]:
    scores = {t: disagreement(prob_to_tile(p), load_veg_mask(cache_dir, t), load_valid_mask(cache_dir, t),
                              cfg.nn.prob_threshold) for t, p in cached.items() if t not in exclude}
    return pick_hard_tiles(scores, n)


def panels_from_cache(cfg: AppConfig, run_id: str, tiles: Sequence[str], n_auto: int | None) -> tuple[Path, ...]:
    """Panels from the nn_infer cache: the given tiles (else nn.ablation.panel_tiles) plus the n_auto tiles
    (else nn.ablation.n_auto_panels) where the NN and the a* mask disagree most."""
    paths = make_run_paths(cfg, run_id)
    explicit = tuple(tiles) or tuple(cfg.nn.ablation.panel_tiles)
    cached = _cached_probs(cfg, paths.cache_dir, indexed_tile_ids(paths))
    missing = [t for t in explicit if t not in cached]
    if missing:
        raise StageError("no NN probabilities cached for panel tiles", tiles=", ".join(missing),
                         hint="vineyard nn infer --tiles ...")
    if not cached:
        log_event(_log, EVENT_NO_CACHE, level=logging.WARNING, version=cfg.nn.version)
    n = cfg.nn.ablation.n_auto_panels if n_auto is None else n_auto
    auto = _hard_tiles(cfg, paths.cache_dir, cached, n, explicit)
    chosen = {t: cached[t] for t in (*explicit, *auto)}
    return write_panels(cfg, paths.qa_dir / PANELS_DIR, chosen, paths.cache_dir, tile_reader(paths.tiles_dir))


# ------------------------------------------------------------------ ablation


def _probs_by_weights(cfg: AppConfig, tiles: Sequence[ValTile], f_version: str | None
                      ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    main = weights_probs(cfg, cfg.nn.version, tiles, cfg.nn.weights_sha256)
    if main is None:
        raise StageError("no NN weights for the ablation", name=cfg.nn.name, version=cfg.nn.version,
                         hint="vineyard nn train | vineyard nn fetch")
    f_ver = f_version or cfg.nn.version + F_SUFFIX
    f = weights_probs(cfg, f_ver, tiles, None)
    if f is None:
        log_event(_log, EVENT_NO_F, level=logging.WARNING, version=f_ver)
    probs = {MAIN_WEIGHTS: main[0], **({F_WEIGHTS: f[0]} if f else {})}
    timing = {MAIN_WEIGHTS: main[1], **({F_WEIGHTS: f[1]} if f else {})}
    return probs, timing


def ablate(cfg: AppConfig, run_id: str, *, reference: str | None = None, f_version: str | None = None) -> AblationOutputs:
    """Ablation A-F on the holdout tiles with the model rows of `run_id` and the reference axes, the JSON +
    Markdown report and the panels of the holdout tiles."""
    paths = make_run_paths(cfg, run_id)
    ref_tiles = reference_val_tiles(cfg, cfg.nn.holdout_tiles, reference or cfg.eval.reference_annset)
    rows = load_model_rows(cfg.paths.work_dir, run_id)
    by_axes = {AXES_MODEL: model_axes(ref_tiles, rows, cfg.canopy.rows_margin_m), AXES_REFERENCE: ref_tiles}
    probs, timing = _probs_by_weights(cfg, ref_tiles, f_version)
    report = run_ablation(by_axes, probs, cfg, infer_s_per_tile=timing)
    json_path, md_path = write_report(report, paths.metrics_dir / ABLATION_JSON,
                                      cfg.paths.project_root / REPORTS_DIR / ABLATION_MD)
    panels = write_panels(cfg, paths.qa_dir / PANELS_DIR, probs[MAIN_WEIGHTS], paths.cache_dir,
                          tile_reader(paths.tiles_dir))
    return AblationOutputs(report, json_path, md_path, panels)


# ------------------------------------------------------------------ fetch / bench


def fetch(cfg: AppConfig, *, url: str | None, sha256: str | None, version: str | None, card_url: str | None,
          opener: Callable[[str], bytes] | None = None) -> LoadedWeights:
    """Download + verify the configured (or given) weights into models/<name>/<version>."""
    src, sha = url or cfg.nn.weights_url, sha256 or cfg.nn.weights_sha256
    if not src or not sha:
        raise WeightsError("fetch needs a URL and a sha256", hint="--url/--sha256 or nn.weights_url/weights_sha256")
    return fetch_weights(src, cfg.paths.models_dir, cfg.nn.name, version or cfg.nn.version, sha,
                         card_url=card_url, opener=opener)


def bench(cfg: AppConfig, *, iters: int, batch: int | None, size: int | None,
          parts: TrainParts | None = None) -> BenchResult:
    from vineyard.nn.train import benchmark_iterations

    return benchmark_iterations(cfg, n_iter=iters, batch_size=batch or cfg.nn.batch_size,
                                size_px=size or cfg.nn.pseudolabels.patch_px, parts=parts)
