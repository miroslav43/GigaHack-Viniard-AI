"""pipeline.runner: run_tile_stage (cache, failures, force) and run_stages (run.json, timings, LATEST, exit codes)."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.errors import ExportBlocked, StageError
from vineyard.pipeline import runner
from vineyard.pipeline.cache import read_key
from vineyard.pipeline.context import RunContext, new_run_context
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import (
    RunReport,
    StageResult,
    TileTask,
    load_tile_failures,
    run_stages,
    run_tile_stage,
    tile_failure_issues,
)

TILES = ("siret3_r006_c004", "siret3_r021_c012", "siret3_r018_c010")
BAD_TILE = "siret3_r021_c012"


def _make_ctx(work: Path, **kwargs: Any) -> RunContext:
    cfg = load_config()
    assert cfg.paths.work_dir == work
    opts = {"source": Source.MODEL, "run_id": "test-run", "workers": 1} | kwargs
    return new_run_context(cfg, **opts)


@pytest.fixture
def ctx(tmp_work: Path) -> RunContext:
    return _make_ctx(tmp_work)


def _write_tile(task: TileTask) -> Mapping[str, Any]:
    """Top-level (picklable) tile_fn: fails on BAD_TILE unless its marker input exists."""
    if task.tile_id == BAD_TILE and not task.inputs["allow"].exists():
        raise ValueError(f"synthetic failure on {task.tile_id}")
    out = task.outputs["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{task.tile_id}:{task.key}", encoding="utf-8")
    return {"n": 1, "label": "ok", "big": "x" * 1000, "arr": [1, 2]}


def _no_output(task: TileTask) -> Mapping[str, Any]:
    return {}


def _spec(name: str = "tile_prep", run: Any = None, version: str = "1") -> StageSpec:
    return StageSpec(name=name, version=version, scope="tile", cfg_keys=("veg",), requires=(),
                     run=run or (lambda c: StageResult(name, 0, 0, 0)))


def _task_factory(ctx: RunContext) -> Any:
    def make(tile_id: str) -> TileTask:
        return TileTask(tile_id=tile_id, tif_path=ctx.paths.tiles_dir / f"{tile_id}.tif", key="", cfg=ctx.cfg.veg,
                        inputs={"allow": ctx.paths.work_dir / "allow"},
                        outputs={"out": ctx.paths.cache_dir / "fake" / f"{tile_id}.txt"})
    return make


def _run(ctx: RunContext, fn: Any = _write_tile, **kwargs: Any) -> StageResult:
    return run_tile_stage(ctx, _spec(), make_task=_task_factory(ctx), tile_fn=fn,
                          input_keys=lambda t: [f"in-{t}"], tile_ids=TILES, **kwargs)


def _events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, Any]]:
    out = []
    for rec in caplog.records:
        payload = getattr(rec, "vineyard_payload", None)
        if payload and payload["event"] == name:
            out.append({"tile_id": payload["tile_id"], **payload["fields"]})
    return out


def test_failed_tile_then_cached_rerun_retries_only_the_failure(ctx: RunContext,
                                                                caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="vineyard")
    first = _run(ctx)
    assert (first.n_items, first.n_cached, first.n_failed, first.failed) == (3, 0, 1, (BAD_TILE,))
    failed = _events(caplog, "tile.failed")
    assert failed[0]["tile_id"] == BAD_TILE and "Traceback" in failed[0]["traceback"]
    done = _events(caplog, "tile.done")
    assert {d["tile_id"] for d in done} == set(TILES) - {BAD_TILE}
    assert done[0]["n"] == 1 and "big" not in done[0] and "arr" not in done[0]
    assert first.metrics["n_done"] == 2.0 and first.metrics["tile_s_max"] >= 0.0
    issues = tile_failure_issues(ctx.paths)
    assert [(i.code, i.tile_id, i.object_id) for i in issues] == [("tile_failed", BAD_TILE, "tile_prep")]

    (ctx.paths.work_dir / "allow").write_text("1")
    second = _run(ctx)
    assert (second.n_cached, second.n_failed) == (2, 0)
    assert second.metrics["n_done"] == 1.0
    assert load_tile_failures(ctx.paths) == ()


def test_keys_written_after_success_and_force_recomputes(ctx: RunContext) -> None:
    (ctx.paths.work_dir / "allow").write_text("1")
    _run(ctx)
    out = ctx.paths.cache_dir / "fake" / f"{TILES[0]}.txt"
    key = read_key(out)
    assert key is not None and out.read_text().endswith(key)
    mtime = os.stat(out).st_mtime_ns
    assert _run(ctx).n_cached == 3
    assert os.stat(out).st_mtime_ns == mtime
    forced = _run(_make_ctx(ctx.paths.work_dir, force=("tile_prep",)))
    assert forced.n_cached == 0 and forced.n_failed == 0


def _two_outputs(task: TileTask) -> Mapping[str, Any]:
    for path in task.outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(task.key, encoding="utf-8")
    return {}


def test_keys_follow_output_order_so_the_last_output_marks_completion(
        ctx: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[str] = []
    real = runner.write_key
    monkeypatch.setattr(runner, "write_key", lambda path, key: written.append(Path(path).name) or real(path, key))

    def make(tile_id: str) -> TileTask:
        return TileTask(tile_id=tile_id, tif_path=ctx.paths.tiles_dir / f"{tile_id}.tif", key="", cfg=ctx.cfg.veg,
                        inputs={}, outputs={"z": ctx.paths.cache_dir / "z" / f"{tile_id}.png",
                                            "a": ctx.paths.cache_dir / "a" / f"{tile_id}.json"})

    run_tile_stage(ctx, _spec(), make_task=make, tile_fn=_two_outputs, input_keys=lambda t: [t],
                   tile_ids=TILES[:1])
    assert written == [f"{TILES[0]}.png", f"{TILES[0]}.json"]


def _reserved_value_keys(task: TileTask) -> Mapping[str, Any]:
    _two_outputs(task)
    return {"tile_id": "other", "stage": "x", "event": "e", "logger": 1, "level": 99, "duration_s": 9.0,
            "cpu_s": 1.0, "peak_rss_mb": 1.0, "veg_frac": 0.5}


def test_tile_fn_values_never_clash_with_event_fields(ctx: RunContext, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="vineyard")
    result = run_tile_stage(ctx, _spec(), make_task=_task_factory(ctx), tile_fn=_reserved_value_keys,
                            input_keys=lambda t: [t], tile_ids=TILES[:1])
    assert result.n_failed == 0
    done = _events(caplog, "tile.done")[0]
    assert done["tile_id"] == TILES[0] and done["veg_frac"] == 0.5 and done["duration_s"] != 9.0


def _assert_key_still_there(task: TileTask) -> Mapping[str, Any]:
    """Top-level tile_fn for a forced rerun: a same-key artifact stays keyed while it is recomputed."""
    if read_key(task.outputs["out"]) != task.key:
        raise AssertionError("key removed during a same-key recompute")
    return _write_tile(task)


def test_forced_same_key_recompute_keeps_keys_readable(ctx: RunContext) -> None:
    (ctx.paths.work_dir / "allow").write_text("1")
    _run(ctx)
    forced = _run(_make_ctx(ctx.paths.work_dir, force=("tile_prep",)), fn=_assert_key_still_there)
    assert (forced.n_cached, forced.n_failed) == (0, 0)


def test_changed_key_removes_the_stale_key_before_recompute(ctx: RunContext) -> None:
    (ctx.paths.work_dir / "allow").write_text("1")
    _run(ctx)
    other = new_run_context(load_config(overrides=["veg.threshold=5.0"]), source=Source.MODEL, run_id="test-run",
                            workers=1)
    (ctx.paths.work_dir / "allow").unlink()
    result = _run(other)
    assert result.failed == (BAD_TILE,)
    assert read_key(ctx.paths.cache_dir / "fake" / f"{BAD_TILE}.txt") is None


def test_key_changes_with_config(ctx: RunContext) -> None:
    (ctx.paths.work_dir / "allow").write_text("1")
    _run(ctx)
    other = new_run_context(load_config(overrides=["veg.threshold=5.0"]), source=Source.MODEL, run_id="test-run",
                            workers=1)
    assert _run(other).n_cached == 0


def test_missing_outputs_is_a_failure(ctx: RunContext) -> None:
    result = _run(ctx, fn=_no_output)
    assert result.n_failed == 3
    assert "did not write outputs" in load_tile_failures(ctx.paths)[0].error


def test_input_key_error_fails_only_that_tile(ctx: RunContext) -> None:
    (ctx.paths.work_dir / "allow").write_text("1")

    def keys(tile_id: str) -> list[str]:
        if tile_id == TILES[0]:
            raise StageError("tile_prep cache key missing", tile_id=tile_id)
        return [tile_id]

    result = run_tile_stage(ctx, _spec(), make_task=_task_factory(ctx), tile_fn=_write_tile, input_keys=keys,
                            tile_ids=TILES)
    assert result.failed == (TILES[0],)
    assert "StageError" in load_tile_failures(ctx.paths)[0].error


def test_tile_filter_selects_tiles(tmp_work: Path) -> None:
    ctx = _make_ctx(tmp_work, tiles=("siret3_r00*",))
    (tmp_work / "allow").write_text("1")
    assert _run(ctx).n_items == 1


def test_spawn_workers_run_tile_fn(tmp_work: Path) -> None:
    ctx = _make_ctx(tmp_work, workers=2)
    (tmp_work / "allow").write_text("1")
    result = _run(ctx)
    assert (result.n_items, result.n_failed) == (3, 0)


def test_default_tile_ids_need_the_tile_index(ctx: RunContext) -> None:
    with pytest.raises(StageError, match="vineyard ingest"):
        run_tile_stage(ctx, _spec(), make_task=_task_factory(ctx), tile_fn=_write_tile, input_keys=lambda t: [])


# ------------------------------------------------------------------ run_stages


def _install(monkeypatch: pytest.MonkeyPatch, specs: Mapping[str, StageSpec]) -> None:
    def fake_load(name: str) -> StageSpec:
        if name not in specs:
            raise StageError("unknown stage", stage=name)
        return specs[name]
    monkeypatch.setattr(runner, "load_stage", fake_load)


def _result_stage(name: str, n_failed: int = 0) -> StageSpec:
    failed = tuple(f"t{k}" for k in range(n_failed))
    return _spec(name, run=lambda c: StageResult(name, 3, 1, n_failed, failed, metrics={"tile_s_mean": 0.1}))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_stages_ok_writes_run_json_and_timings(ctx: RunContext, monkeypatch: pytest.MonkeyPatch,
                                                   caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="vineyard")
    _install(monkeypatch, {"ingest": _result_stage("ingest"), "tile_prep": _result_stage("tile_prep")})
    report = run_stages(ctx, ("ingest", "tile_prep"))
    events = [getattr(r, "vineyard_payload", {}).get("event") for r in caplog.records]
    assert events.count("stage.start") == 2 and events.count("stage.end") == 2 and "run.end" in events
    assert isinstance(report, RunReport) and report.exit_code == 0
    assert [r.stage for r in report.results] == ["ingest", "tile_prep"]
    doc = _read(ctx.paths.run_json)
    assert doc["status"] == "ok" and doc["exit_code"] == 0 and set(doc["stages"]) == {"ingest", "tile_prep"}
    assert doc["latest_link"] is None  # no annset.json in this run
    timings = _read(ctx.paths.metrics_dir / "timings.json")
    assert set(timings["stages"]) == {"ingest", "tile_prep"}
    assert timings["stages"]["tile_prep"]["tile_s_mean"] == 0.1 and timings["hardware"]["cpu_count"]
    assert (ctx.paths.run_dir / "config.json").is_file()


def test_failed_tiles_exit_1_unless_allowed(tmp_work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"tile_prep": _result_stage("tile_prep", n_failed=1), "canopy": _result_stage("canopy")})
    report = run_stages(_make_ctx(tmp_work), ("tile_prep", "canopy"))
    assert report.exit_code == 1 and len(report.results) == 2  # failed tiles do not stop the run
    allowed = run_stages(_make_ctx(tmp_work, allow_failures=True, run_id="r2"), ("tile_prep",))
    assert allowed.exit_code == 0


def test_stage_error_stops_run_with_its_exit_code(ctx: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(c: RunContext) -> StageResult:
        raise ExportBlocked("zip too large", part=1)

    _install(monkeypatch, {"export_cvat": _spec("export_cvat", run=blocked), "canopy": _result_stage("canopy")})
    report = run_stages(ctx, ("export_cvat", "canopy"))
    assert report.exit_code == 2 and report.results == ()
    doc = _read(ctx.paths.run_json)
    assert doc["status"] == "failed" and "zip too large" in doc["error"]


def test_non_stageresult_is_an_error(ctx: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"canopy": _spec("canopy", run=lambda c: {"n": 1})})
    assert run_stages(ctx, ("canopy",)).exit_code == 1


def test_unexpected_exception_is_recorded_and_reraised(ctx: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def bug(c: RunContext) -> StageResult:
        raise ZeroDivisionError("bug")

    _install(monkeypatch, {"canopy": _spec("canopy", run=bug)})
    with pytest.raises(ZeroDivisionError):
        run_stages(ctx, ("canopy",))
    assert _read(ctx.paths.run_json)["status"] == "failed"


def _annset_stage(name: str) -> StageSpec:
    def run(c: RunContext) -> StageResult:
        c.paths.annset_dir.mkdir(parents=True, exist_ok=True)
        (c.paths.annset_dir / "annset.json").write_text("{}")
        return StageResult(name, 1, 0, 0)
    return _spec(name, run=run)


def test_latest_link_only_for_full_runs_with_an_annset(tmp_work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"assemble": _annset_stage("assemble")})
    filtered = run_stages(_make_ctx(tmp_work, tiles=("siret3_r006_c004",), run_id="filtered"), ("assemble",))
    assert filtered.exit_code == 0
    link = tmp_work / "runs" / "LATEST_MODEL"
    assert not link.exists()
    run_stages(_make_ctx(tmp_work, run_id="full"), ("assemble",))
    assert link.is_symlink() and link.resolve() == (tmp_work / "runs" / "full").resolve()
    assert _read(tmp_work / "runs" / "full" / "run.json")["latest_link"] == "LATEST_MODEL"


def test_rerun_same_run_id_merges_stage_records(ctx: RunContext, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"ingest": _result_stage("ingest"), "tile_prep": _result_stage("tile_prep")})
    run_stages(ctx, ("ingest",))
    run_stages(ctx, ("tile_prep",))
    assert set(_read(ctx.paths.run_json)["stages"]) == {"ingest", "tile_prep"}
    assert set(_read(ctx.paths.metrics_dir / "timings.json")["stages"]) == {"ingest", "tile_prep"}
