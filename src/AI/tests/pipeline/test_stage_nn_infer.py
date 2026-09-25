"""Stage nn_infer placeholder: skipped while nn is disabled, explicit error when enabled; never loads torch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.errors import StageNotImplemented
from vineyard.pipeline.context import RunContext, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages import nn_infer


def _ctx(nn_enabled: bool) -> RunContext:
    cfg = load_config()
    cfg = cfg.model_copy(update={"nn": cfg.nn.model_copy(update={"enabled": nn_enabled})})
    return new_run_context(cfg, source=Source.MODEL, run_id="nn-test", workers=1)


def test_registered_and_skipped_when_disabled(tmp_work: Path) -> None:
    assert load_stage("nn_infer") is nn_infer.STAGE
    res = nn_infer.run(_ctx(False))
    assert (res.stage, res.n_items, res.n_failed) == ("nn_infer", 0, 0)
    assert res.metrics["skipped"] == 1.0


def test_enabled_nn_fails_explicitly(tmp_work: Path) -> None:
    with pytest.raises(StageNotImplemented, match="nn inference"):
        nn_infer.run(_ctx(True))


def test_import_does_not_load_torch() -> None:
    code = "import sys, vineyard.pipeline.stages.nn_infer; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
