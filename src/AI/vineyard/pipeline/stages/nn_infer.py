"""Stage nn_infer (global, main process only; contract §6, design 03 N1/N6).

nn.enabled and verified weights -> batched inference on nn.device over the selected tiles, writing
cache/nn/<nn.version>/canopy_prob/<tile>.png (1024², uint8) + .key (the path the canopy and rows_detect
consumers read). Without nn.enabled or without weights the stage is skipped and every consumer falls back
to the classic recipe (canopy_prob=None). Tampered weights (sha256 mismatch) fail the stage.
torch is imported only when inference really runs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from vineyard.logging_setup import get_logger, log_event
from vineyard.nn.probs import prob_png_path
from vineyard.nn.weights import LoadedWeights, load_weights, nn_version_string, weights_exist
from vineyard.pipeline.cache import all_fresh, cache_key, read_key, remove_key
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult, TileFailure, record_tile_failures
from vineyard.pipeline.tile_cache import tile_prep_key, valid_mask_path
from vineyard.pipeline.tile_index import indexed_tile_ids, tile_path

if TYPE_CHECKING:
    from vineyard.nn.infer import InferJob, InferOutcome
    from vineyard.pipeline.context import RunContext

STAGE_NAME: Final = "nn_infer"
STAGE_VERSION: Final = "1"
PROB_CHANNEL: Final = "canopy_prob"
CFG_KEYS: Final = ("nn.name", "nn.version", "nn.arch", "nn.encoder", "nn.in_gsd_m", "nn.out_channels", "grid.gsd_m",
                   "grid.tile_px")
EVENT_SKIPPED: Final = "nn_infer.skipped"
EVENT_WEIGHTS_MISSING: Final = "nn.weights_missing"
EVENT_DONE: Final = "nn_infer.done"
NO_DEVICE: Final = "none"
GPU_DEVICES: Final = ("mps", "cuda")

_log = get_logger("pipeline.stages.nn_infer")


def _skipped(reason: str, **metrics: float) -> StageResult:
    log_event(_log, EVENT_SKIPPED, stage=STAGE_NAME, reason=reason)
    return StageResult(stage=STAGE_NAME, n_items=0, n_cached=0, n_failed=0, metrics={"skipped": 1.0, **metrics})


def _job(ctx: RunContext, tile_id: str, digest: str, weights: LoadedWeights) -> InferJob:
    from vineyard.nn.infer import InferJob

    inputs = [f"tile_prep:{tile_prep_key(ctx.paths.cache_dir, tile_id)}", f"weights:{weights.card.sha256}"]
    return InferJob(tile_id=tile_id, tif_path=tile_path(ctx, tile_id),
                    valid_path=valid_mask_path(ctx.paths.cache_dir, tile_id),
                    out_path=prob_png_path(ctx.paths.cache_dir, ctx.cfg.nn.version, PROB_CHANNEL, tile_id),
                    key=cache_key(STAGE_NAME, STAGE_VERSION, digest, inputs))


def _todo(ctx: RunContext, jobs: Sequence[InferJob]) -> list[InferJob]:
    force = ctx.should_force(STAGE_NAME)
    todo = [j for j in jobs if force or not all_fresh([j.out_path], j.key)]
    for job in todo:  # an artifact being recomputed must not look fresh to a reader meanwhile
        if read_key(job.out_path) != job.key:
            remove_key(job.out_path)
    return todo


def _infer(ctx: RunContext, weights: LoadedWeights, todo: Sequence[InferJob]
           ) -> tuple[tuple[InferOutcome, ...], float, str]:
    from vineyard.nn.infer import load_model, select_device, timed_infer
    from vineyard.nn.validate import input_factor

    nn = ctx.cfg.nn
    device = select_device(nn.device, nn.fallback_device)
    model = load_model(weights, device)
    outcomes, seconds = timed_infer(todo, model, device, batch_size=nn.infer_batch_size,
                                    factor=input_factor(ctx.cfg))
    return outcomes, seconds, device.type


def _result(tiles: Sequence[str], n_todo: int, outcomes: Sequence[InferOutcome], failures: Sequence[TileFailure],
            seconds: float, device: str) -> StageResult:
    n_done = len(outcomes) - len(failures)
    metrics = {"skipped": 0.0, "n_inferred": float(n_done), "infer_s": seconds,
               "s_per_tile": seconds / len(outcomes) if outcomes else 0.0, "device_gpu": float(device in GPU_DEVICES)}
    return StageResult(stage=STAGE_NAME, n_items=len(tiles), n_cached=len(tiles) - n_todo, n_failed=len(failures),
                       failed=tuple(sorted(f.tile_id for f in failures)),
                       outputs=tuple(o.path for o in outcomes if o.path is not None), metrics=metrics)


def run(ctx: RunContext) -> StageResult:
    nn = ctx.cfg.nn
    if not nn.enabled:
        return _skipped("nn.enabled=false")
    models_dir = ctx.cfg.paths.models_dir
    if not weights_exist(models_dir, nn.name, nn.version):
        log_event(_log, EVENT_WEIGHTS_MISSING, level=logging.WARNING, stage=STAGE_NAME, name=nn.name,
                  version=nn.version, models_dir=str(models_dir), hint="vineyard nn train | vineyard nn fetch")
        return _skipped("weights missing", weights_missing=1.0)
    weights = load_weights(models_dir, nn.name, nn.version, nn.weights_sha256)
    tiles = ctx.selected_tiles(indexed_tile_ids(ctx))
    digest = ctx.stage_cfg_digest(STAGE)
    todo = _todo(ctx, [_job(ctx, t, digest, weights) for t in tiles])
    outcomes, seconds, device = _infer(ctx, weights, todo) if todo else ((), 0.0, NO_DEVICE)
    failures = [TileFailure(STAGE_NAME, o.tile_id, o.error or "unknown error") for o in outcomes if not o.ok]
    record_tile_failures(ctx.paths, STAGE_NAME, tiles, failures)
    log_event(_log, EVENT_DONE, stage=STAGE_NAME, model_version=nn_version_string(weights.card), n_tiles=len(tiles),
              n_inferred=len(outcomes) - len(failures), n_failed=len(failures), seconds=round(seconds, 3),
              device=device)
    return _result(tiles, len(todo), outcomes, failures, seconds, device)


STAGE: Final = StageSpec(
    name=STAGE_NAME,
    version=STAGE_VERSION,
    scope="global",
    cfg_keys=CFG_KEYS,
    requires=("tile_prep",),
    run=run,
    description="NN canopy_prob per tile (main-process batched inference; skipped without nn.enabled/weights).",
)
