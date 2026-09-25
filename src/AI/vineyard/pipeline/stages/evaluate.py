"""Stage evaluate (`vineyard eval-examples`): AnnSet(--annset) vs AnnSet(reference) on eval.example_tiles.

Writes <evaluated run>/metrics/eval_examples.{json,md}. eval.baseline_file adds a regression check,
eval.write_baseline stores this report as the new baseline, and eval.enforce_gates (--gates) fails the
stage on a failed gate or a regression, after the reports are written.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from vineyard.annset.io import ANNSET_DIRNAME, read_annset, resolve_run_dir
from vineyard.annset.model import AnnSet
from vineyard.config import EvalConfig
from vineyard.errors import StageError
from vineyard.eval.report import (
    HEADLINE_KEYS,
    EvalReport,
    compare_to_baseline,
    evaluate_annsets,
    load_baseline,
    report_to_dict,
    write_report_json,
    write_report_markdown,
)
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.atomic import atomic_write_json
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext

STAGE_NAME: Final = "evaluate"
STAGE_VERSION: Final = "1"
CFG_KEYS: Final = ("eval",)
REPORT_NAME: Final = "eval_examples"
METRICS_DIRNAME: Final = "metrics"
DEFAULT_ANNSET_REF: Final = "LATEST_MODEL"
EVENT_DONE: Final = "eval.done"
EVENT_NO_BASELINE: Final = "eval.baseline_new"
HINTS: Final = {"reference": "run `vineyard import-reference` first", "evaluated": "check --annset"}

_log = get_logger("pipeline.stages.evaluate")


def _load(work_dir: Path, ref: str, role: str) -> tuple[Path, AnnSet]:
    try:
        run_dir = resolve_run_dir(work_dir, ref)
    except StageError as exc:
        raise StageError(f"{role} AnnSet not found", ref=ref, hint=HINTS[role]) from exc
    annset_dir = run_dir / ANNSET_DIRNAME
    if not annset_dir.is_dir():
        raise StageError(f"{role} AnnSet not found", ref=ref, run_dir=str(run_dir), hint=HINTS[role])
    return run_dir, read_annset(annset_dir)


def baseline_path(ctx: RunContext) -> Path:
    configured = ctx.cfg.eval.baseline_file
    return Path(configured) if configured is not None else ctx.paths.work_dir / ctx.cfg.eval.default_baseline_name


def _with_regressions(report: EvalReport, cfg: EvalConfig) -> EvalReport:
    if cfg.baseline_file is None:
        return report
    source = Path(cfg.baseline_file)
    if cfg.write_baseline and not source.is_file():
        # First --write-baseline to a new file: nothing to compare against yet.
        log_event(_log, EVENT_NO_BASELINE, stage=STAGE_NAME, path=str(source))
        return report
    baseline = load_baseline(source)
    return report.with_regressions(compare_to_baseline(report, baseline, max_drop=cfg.regression_max_drop))


def _write_outputs(ctx: RunContext, run_dir: Path, report: EvalReport) -> tuple[Path, ...]:
    metrics_dir = run_dir / METRICS_DIRNAME
    outputs = (write_report_json(report, metrics_dir / f"{REPORT_NAME}.json"),
               write_report_markdown(report, metrics_dir / f"{REPORT_NAME}.md"))
    if ctx.cfg.eval.write_baseline:
        outputs += (atomic_write_json(baseline_path(ctx), report_to_dict(report)),)
    return outputs


def _stage_metrics(report: EvalReport) -> dict[str, float]:
    out = {f"mean.{k}": float(v) for k in HEADLINE_KEYS if (v := report.mean.get(k)) is not None}
    return out | {"gates_passed": float(report.gates_passed), "regressions": float(len(report.regressions))}


def _enforce(report: EvalReport, cfg: EvalConfig, report_path: Path) -> None:
    if not cfg.enforce_gates or (report.gates_passed and not report.regressions):
        return
    failed = [g.name for g in report.gates if not g.passed]
    raise StageError("eval gates failed", failed_gates=failed, regressions=list(report.regressions),
                     report=str(report_path))


def run(ctx: RunContext) -> StageResult:
    cfg = ctx.cfg.eval
    pred_dir, pred = _load(ctx.paths.work_dir, ctx.annset_ref or DEFAULT_ANNSET_REF, "evaluated")
    _, ref = _load(ctx.paths.work_dir, cfg.reference_annset, "reference")
    tiles = ctx.selected_tiles(cfg.example_tiles)
    report = _with_regressions(evaluate_annsets(pred, ref, tiles, cfg), cfg)
    outputs = _write_outputs(ctx, pred_dir, report)
    metrics = _stage_metrics(report)
    log_event(_log, EVENT_DONE, stage=STAGE_NAME, pred_run=report.pred_run, ref_run=report.ref_run,
              tiles=list(report.tiles), **metrics)
    _enforce(report, cfg, outputs[0])
    return StageResult(stage=STAGE_NAME, n_items=len(report.tiles), n_cached=0, n_failed=0, outputs=outputs,
                       metrics=metrics)


STAGE: Final = StageSpec(
    name=STAGE_NAME,
    version=STAGE_VERSION,
    scope="global",
    cfg_keys=CFG_KEYS,
    requires=(),
    run=run,
    description="official metrics of an AnnSet vs the reference on the example tiles (+ gates, baseline)",
)
