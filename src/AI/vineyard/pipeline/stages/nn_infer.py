"""Stage nn_infer (Phase-1 placeholder): a no-op while `nn.enabled` is false, so `vineyard run` can pass it.

The real inference stage (design 03) replaces this module; it must never import torch while nn is disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from vineyard.errors import StageNotImplemented
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext

STAGE_NAME: Final = "nn_infer"
EVENT_SKIPPED: Final = "nn_infer.skipped"

_log = get_logger("pipeline.stages.nn_infer")


def run(ctx: RunContext) -> StageResult:
    """Skip when nn is disabled; fail loudly when it is enabled but no inference exists yet."""
    if ctx.cfg.nn.enabled:
        raise StageNotImplemented("nn inference is not implemented yet; set nn.enabled=false",
                                  stage=STAGE_NAME)
    log_event(_log, EVENT_SKIPPED, stage=STAGE_NAME, reason="nn.enabled=false")
    return StageResult(stage=STAGE_NAME, n_items=0, n_cached=0, n_failed=0, metrics={"skipped": 1.0})


STAGE = StageSpec(
    name=STAGE_NAME,
    version="0",
    scope="global",
    cfg_keys=("nn",),
    requires=("tile_prep",),
    run=run,
    description="NN probability maps (Phase-1 placeholder: skipped while nn.enabled is false).",
)
