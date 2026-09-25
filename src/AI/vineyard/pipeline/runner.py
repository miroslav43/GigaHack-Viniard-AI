"""Stage runner: run_stages (run.json, timings.json, LATEST link, exit codes) and run_tile_stage
(per-tile cache check, spawn pool, per-tile failures). Only the main process writes cache keys and
pipeline.jsonl.

Exit codes: 0 ok, 1 failed tiles (unless allow_failures) or a stage error, 2 export blocked,
3 stage not implemented.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from vineyard.annset.io import ANNSET_JSON, update_latest_link
from vineyard.config import cfg_hash, resolved_config_dict
from vineyard.contracts.enums import Severity
from vineyard.contracts.qa import QaIssue
from vineyard.errors import StageError, VineyardError
from vineyard.logging_setup import get_logger, log_event, log_failure
from vineyard.pipeline.atomic import atomic_write_json
from vineyard.pipeline.cache import all_fresh, cache_key, read_key, remove_key, write_key
from vineyard.pipeline.context import RunContext, RunPaths, ensure_run_dirs
from vineyard.pipeline.parallel import ItemOutcome, parallel_map
from vineyard.pipeline.registry import StageSpec, load_stage
from vineyard.pipeline.tile_index import indexed_tile_ids
from vineyard.pipeline.timings import (
    EVENT_STAGE_END,
    EVENT_TILE_CACHED,
    EVENT_TILE_DONE,
    EVENT_TILE_FAILED,
    STATUS_CACHED,
    STATUS_DONE,
    STATUS_FAILED,
    ItemTiming,
    build_timings,
    stage_timing_block,
    summarize_items,
)

EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
QA_TILE_FAILED: Final = "tile_failed"
EVENT_STAGE_START: Final = "stage.start"
EVENT_STAGE_FAILED: Final = "stage.failed"
EVENT_RUN_END: Final = "run.end"
RUN_RUNNING: Final = "running"
RUN_OK: Final = "ok"
RUN_FAILED: Final = "failed"
TILE_FAILURES_FILE: Final = "tile_failures.json"
CONFIG_FILE: Final = "config.json"
TIMINGS_FILE: Final = "timings.json"
_MAX_LOGGED_STR: Final = 200
_RESERVED_EVENT_FIELDS: Final = frozenset(
    {"logger", "event", "level", "stage", "tile_id", "duration_s", "cpu_s", "peak_rss_mb"})
_ROUND: Final = 4

_log = get_logger("pipeline.runner")


@dataclass(frozen=True)
class StageResult:
    stage: str
    n_items: int
    n_cached: int
    n_failed: int
    failed: tuple[str, ...] = ()
    outputs: tuple[Path, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "n_items": self.n_items, "n_cached": self.n_cached, "n_failed": self.n_failed,
                "failed": list(self.failed), "outputs": [str(p) for p in self.outputs],
                "metrics": {k: self.metrics[k] for k in sorted(self.metrics)}}


@dataclass(frozen=True)
class RunReport:
    run_id: str
    results: tuple[StageResult, ...]
    exit_code: int


@dataclass(frozen=True)
class TileTask:
    """One tile of a tile stage. `key` is filled in by the runner; tile_fn must write every `outputs` path.
    Keys are written in `outputs` order, so put the completion-marker artifact last."""

    tile_id: str
    tif_path: Path
    key: str
    cfg: BaseModel
    inputs: Mapping[str, Path]
    outputs: Mapping[str, Path]


@dataclass(frozen=True)
class TileFailure:
    stage: str
    tile_id: str
    error: str

    def to_issue(self) -> QaIssue:
        return QaIssue(severity=Severity.ERROR, code=QA_TILE_FAILED, tile_id=self.tile_id, object_id=self.stage,
                       message=f"{self.stage}: {self.error}")


# ------------------------------------------------------------------ tile failures (for assemble / qa)


def _failures_path(paths: RunPaths) -> Path:
    return paths.qa_dir / TILE_FAILURES_FILE


def load_tile_failures(paths: RunPaths) -> tuple[TileFailure, ...]:
    """Every tile failure recorded in this run (sorted by tile_id, stage)."""
    path = _failures_path(paths)
    if not path.is_file():
        return ()
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = [TileFailure(stage=s, tile_id=t, error=e) for s, per_tile in doc.items() for t, e in per_tile.items()]
    return tuple(sorted(rows, key=lambda f: (f.tile_id, f.stage)))


def tile_failure_issues(paths: RunPaths) -> list[QaIssue]:
    """qa `tile_failed` errors of this run, one per (tile, stage)."""
    return [f.to_issue() for f in load_tile_failures(paths)]


def record_tile_failures(paths: RunPaths, stage: str, processed: Sequence[str],
                         failures: Sequence[TileFailure]) -> None:
    """Update the stage's entry in qa/tile_failures.json for the `processed` tiles (others are kept)."""
    path = _failures_path(paths)
    doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    done = frozenset(processed)
    per_tile = {t: e for t, e in doc.get(stage, {}).items() if t not in done}
    per_tile.update({f.tile_id: f.error for f in failures})
    per_tile = dict(sorted(per_tile.items()))
    new_doc = {k: v for k, v in doc.items() if k != stage} | ({stage: per_tile} if per_tile else {})
    if new_doc or path.is_file():
        atomic_write_json(path, new_doc)


# ------------------------------------------------------------------ tile stages


@dataclass(frozen=True)
class _Prepared:
    tile_id: str
    task: TileTask | None
    error: str | None


def _prepare(spec: StageSpec, cfg_digest: str, tile_id: str, make_task: Callable[[str], TileTask],
             input_keys: Callable[[str], Sequence[str]]) -> _Prepared:
    try:
        key = cache_key(spec.name, spec.version, cfg_digest, input_keys(tile_id))
        task = replace(make_task(tile_id), key=key)
    except Exception as exc:  # noqa: BLE001  (a tile whose inputs are missing fails alone, logged below)
        return _Prepared(tile_id, None, f"{type(exc).__name__}: {exc}")
    if task.tile_id != tile_id:
        return _Prepared(tile_id, None, f"make_task returned tile {task.tile_id!r} for {tile_id!r}")
    return _Prepared(tile_id, task, None)


def _loggable(value: object) -> dict[str, Any]:
    """Scalar entries of a tile_fn result that can ride on tile.done (never the event's own fields)."""
    if not isinstance(value, Mapping):
        return {}
    return {str(k): v for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if str(k) not in _RESERVED_EVENT_FIELDS
            and (isinstance(v, bool | int | float) or (isinstance(v, str) and len(v) <= _MAX_LOGGED_STR))}


def _fail_item(stage: str, tile_id: str, error: str, trace: str | None, duration_s: float) -> TileFailure:
    log_event(_log, EVENT_TILE_FAILED, level=logging.ERROR, stage=stage, tile_id=tile_id,
              duration_s=round(duration_s, _ROUND), error=error, traceback=trace or "")
    return TileFailure(stage=stage, tile_id=tile_id, error=error)


def _finish_item(stage: str, task: TileTask, outcome: ItemOutcome[Mapping[str, Any]]
                 ) -> tuple[ItemTiming, TileFailure | None]:
    timing = ItemTiming(stage, task.tile_id, STATUS_DONE, outcome.duration_s, outcome.cpu_s, outcome.peak_rss_mb)
    missing = sorted(n for n, p in task.outputs.items() if not Path(p).is_file()) if outcome.ok else []
    if not outcome.ok or missing:
        error = outcome.error if not outcome.ok else f"tile_fn did not write outputs {missing}"
        failure = _fail_item(stage, task.tile_id, error or "unknown error", outcome.traceback, outcome.duration_s)
        return replace(timing, status=STATUS_FAILED), failure
    for path in task.outputs.values():  # outputs order: the last output's key marks a complete tile
        write_key(path, task.key)
    log_event(_log, EVENT_TILE_DONE, stage=stage, tile_id=task.tile_id, duration_s=round(outcome.duration_s, _ROUND),
              cpu_s=round(outcome.cpu_s, _ROUND), peak_rss_mb=round(outcome.peak_rss_mb, 1),
              **_loggable(outcome.value))
    return timing, None


def _split(ctx: RunContext, spec: StageSpec, prepared: Sequence[_Prepared]
           ) -> tuple[list[TileTask], list[ItemTiming], list[TileFailure]]:
    todo: list[TileTask] = []
    timings: list[ItemTiming] = []
    failures: list[TileFailure] = []
    force = ctx.should_force(spec.name)
    for item in prepared:
        if item.task is None:
            failures.append(_fail_item(spec.name, item.tile_id, item.error or "task preparation failed", None, 0.0))
            timings.append(ItemTiming(spec.name, item.tile_id, STATUS_FAILED, 0.0))
        elif not force and all_fresh(list(item.task.outputs.values()), item.task.key):
            log_event(_log, EVENT_TILE_CACHED, stage=spec.name, tile_id=item.tile_id, duration_s=0.0)
            timings.append(ItemTiming(spec.name, item.tile_id, STATUS_CACHED, 0.0))
        else:
            todo.append(item.task)
    return todo, timings, failures


def _task_key(task: TileTask) -> str:
    return task.tile_id


def run_tile_stage(
    ctx: RunContext,
    spec: StageSpec,
    *,
    make_task: Callable[[str], TileTask],
    tile_fn: Callable[[TileTask], Mapping[str, Any]],
    input_keys: Callable[[str], Sequence[str]],
    tile_ids: Sequence[str] | None = None,
) -> StageResult:
    """Run `tile_fn` (top-level, picklable) on every selected tile whose outputs are not fresh.

    tile_ids defaults to the tile index, filtered by --tiles. A task with no outputs is never cached.
    """
    ids = ctx.selected_tiles(indexed_tile_ids(ctx) if tile_ids is None else tile_ids)
    digest = ctx.stage_cfg_digest(spec)
    todo, timings, failures = _split(ctx, spec, [_prepare(spec, digest, t, make_task, input_keys) for t in ids])
    for task in todo:  # a same-key recompute keeps its keys, so concurrent readers never see a gap
        for path in task.outputs.values():
            if read_key(path) != task.key:
                remove_key(path)
    by_id = {task.tile_id: task for task in todo}
    outcomes = parallel_map(tile_fn, todo, key=_task_key, workers=ctx.workers,
                            chunksize=ctx.cfg.runtime.chunksize, maxtasksperchild=ctx.cfg.runtime.maxtasksperchild)
    for outcome in outcomes:
        timing, failure = _finish_item(spec.name, by_id[outcome.key], outcome)
        timings.append(timing)
        failures.extend([failure] if failure is not None else [])
    record_tile_failures(ctx.paths, spec.name, ids, failures)
    failed = tuple(sorted(f.tile_id for f in failures))
    return StageResult(stage=spec.name, n_items=len(ids), n_cached=sum(t.status == STATUS_CACHED for t in timings),
                       n_failed=len(failed), failed=failed, metrics=summarize_items(timings))


# ------------------------------------------------------------------ run_stages


def _now(ctx: RunContext) -> str:
    return datetime.now(ZoneInfo(ctx.cfg.logging.tz)).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageError("unreadable run record", path=str(path), error=str(exc)) from exc


@dataclass(frozen=True)
class _RunState:
    started_at: str
    t0: float
    results: tuple[StageResult, ...] = ()
    walls: Mapping[str, float] = field(default_factory=dict)
    exit_code: int = EXIT_OK
    error: str | None = None
    status: str = RUN_RUNNING

    def with_result(self, result: StageResult, wall_s: float) -> _RunState:
        return replace(self, results=(*self.results, result), walls={**self.walls, result.stage: wall_s})


def _run_doc(ctx: RunContext, names: Sequence[str], state: _RunState, latest: str | None) -> dict[str, Any]:
    previous = _read_json(ctx.paths.run_json)
    stages = dict(previous.get("stages", {})) | {r.stage: r.to_dict() | {"wall_s": round(state.walls[r.stage],
                                                                                           _ROUND)}
                                                 for r in state.results}
    return {
        "run_id": ctx.run_id, "source": ctx.source.value, "annset_ref": ctx.annset_ref,
        "package_version": ctx.package_version, "git_sha": ctx.git_sha, "model_version": ctx.model_version(),
        "cfg_hash": cfg_hash(ctx.cfg, list(resolved_config_dict(ctx.cfg))), "requested": list(names),
        "tile_filter": list(ctx.tile_filter), "force": sorted(ctx.force), "force_all": ctx.force_all,
        "workers": ctx.workers, "allow_failures": ctx.allow_failures, "started_at": state.started_at,
        "finished_at": None if state.status == RUN_RUNNING else _now(ctx),
        "duration_s": round(time.perf_counter() - state.t0, _ROUND), "status": state.status,
        "exit_code": state.exit_code, "error": state.error, "latest_link": latest, "stages": stages,
    }


def _write_timings(ctx: RunContext, state: _RunState) -> Path:
    path = ctx.paths.metrics_dir / TIMINGS_FILE
    blocks = {r.stage: stage_timing_block(state.walls[r.stage], r.n_items, r.metrics) for r in state.results}
    return atomic_write_json(path, build_timings(ctx.run_id, ctx.workers, blocks, _read_json(path)))


def _run_one_stage(ctx: RunContext, name: str, state: _RunState) -> _RunState:
    try:
        spec = load_stage(name)
        log_event(_log, EVENT_STAGE_START, stage=name, version=spec.version, scope=spec.scope)
        t0 = time.perf_counter()
        result = spec.run(ctx)
        if not isinstance(result, StageResult):
            raise StageError("stage run() did not return a StageResult", stage=name, got=type(result).__name__)
    except VineyardError as exc:
        log_failure(_log, EVENT_STAGE_FAILED, exc, stage=name)
        return replace(state, exit_code=exc.exit_code, error=f"{name}: {exc}", status=RUN_FAILED)
    wall = time.perf_counter() - t0
    log_event(_log, EVENT_STAGE_END, stage=name, duration_s=round(wall, _ROUND), n_items=result.n_items,
              n_cached=result.n_cached, n_failed=result.n_failed)
    failed_code = EXIT_FAILED if result.n_failed and not ctx.allow_failures else EXIT_OK
    return replace(state.with_result(result, wall), exit_code=max(state.exit_code, failed_code))


def _latest_eligible(ctx: RunContext, state: _RunState) -> bool:
    return (state.exit_code == EXIT_OK and ctx.annset_ref is None and not ctx.tile_filter
            and (ctx.paths.annset_dir / ANNSET_JSON).is_file())


def _finalize(ctx: RunContext, names: Sequence[str], state: _RunState) -> _RunState:
    final = replace(state, status=RUN_OK if state.exit_code == EXIT_OK else RUN_FAILED)
    latest = None
    if _latest_eligible(ctx, final):
        update_latest_link(ctx.paths.work_dir, ctx.source, ctx.paths.run_dir)
        latest = f"LATEST_{ctx.source.value.upper()}"
    _write_timings(ctx, final)
    atomic_write_json(ctx.paths.run_json, _run_doc(ctx, names, final, latest))
    log_event(_log, EVENT_RUN_END, exit_code=final.exit_code, status=final.status, latest_link=latest,
              duration_s=round(time.perf_counter() - final.t0, _ROUND))
    return final


def run_stages(ctx: RunContext, names: Sequence[str]) -> RunReport:
    """Run the stages in order; a stage error stops the run, failed tiles do not."""
    ensure_run_dirs(ctx.paths)
    atomic_write_json(ctx.paths.run_dir / CONFIG_FILE, resolved_config_dict(ctx.cfg))
    state = _RunState(started_at=_now(ctx), t0=time.perf_counter())
    atomic_write_json(ctx.paths.run_json, _run_doc(ctx, names, state, None))
    try:
        for name in names:
            state = _run_one_stage(ctx, name, state)
            if state.status == RUN_FAILED:
                break
    except BaseException as exc:
        log_failure(_log, EVENT_STAGE_FAILED, exc)
        atomic_write_json(ctx.paths.run_json, _run_doc(ctx, names, replace(
            state, exit_code=EXIT_FAILED, error=f"{type(exc).__name__}: {exc}", status=RUN_FAILED), None))
        raise
    final = _finalize(ctx, names, state)
    return RunReport(run_id=ctx.run_id, results=final.results, exit_code=final.exit_code)
