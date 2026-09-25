"""pipeline.timings: rusage normalisation, per-stage summaries, timings.json from JSONL events."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vineyard.errors import StageError
from vineyard.pipeline.timings import (
    ItemTiming,
    aggregate_timings,
    aggregate_timings_file,
    build_timings,
    peak_rss_mb,
    read_jsonl_events,
    stage_timing_block,
    summarize_items,
    usage_children,
    usage_self,
)


def test_peak_rss_is_normalised_per_platform() -> None:
    assert peak_rss_mb(1024 * 1024 * 100, "darwin") == pytest.approx(100.0)
    assert peak_rss_mb(1024 * 100, "linux") == pytest.approx(100.0)


def test_usage_snapshots_are_positive() -> None:
    me = usage_self()
    assert me.cpu_s > 0.0 and me.peak_rss_mb > 1.0
    assert usage_children().cpu_s >= 0.0


def test_summarize_items_counts_and_stats() -> None:
    items = [ItemTiming("s", "a", "done", 1.0, 0.9, 100.0), ItemTiming("s", "b", "done", 3.0, 2.5, 150.0),
             ItemTiming("s", "c", "cached", 0.0), ItemTiming("s", "d", "failed", 0.2)]
    out = summarize_items(items)
    assert (out["n_done"], out["n_cached"], out["n_failed"]) == (2.0, 1.0, 1.0)
    assert out["tile_s_mean"] == 2.0 and out["tile_s_max"] == 3.0 and out["tile_s_sum"] == 4.0
    assert out["tile_s_p50"] == 2.0 and out["cpu_s_sum"] == 3.4 and out["peak_rss_mb_max"] == 150.0


def test_summarize_no_done_items() -> None:
    out = summarize_items([ItemTiming("s", "a", "cached", 0.0)])
    assert out["n_done"] == 0.0 and out["tile_s_mean"] == 0.0


def test_build_timings_merges_previous_stages() -> None:
    first = build_timings("r", 2, {"a": stage_timing_block(1.0, 3, {"x": 1.0})})
    second = build_timings("r", 2, {"b": stage_timing_block(2.5, 3, {})}, previous=first)
    assert set(second["stages"]) == {"a", "b"}
    assert second["total_wall_s"] == 3.5 and second["workers"] == 2 and second["hardware"]["cpu_count"]


def _write_jsonl(path: Path, events: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n\n", encoding="utf-8")
    return path


def test_aggregate_from_jsonl(tmp_path: Path) -> None:
    events = [
        {"event": "stage.start", "stage": "tile_prep"},
        {"event": "tile.done", "stage": "tile_prep", "tile_id": "t1", "duration_s": 0.3, "cpu_s": 0.25,
         "peak_rss_mb": 200.0},
        {"event": "tile.cached", "stage": "tile_prep", "tile_id": "t2", "duration_s": 0.0},
        {"event": "tile.failed", "stage": "tile_prep", "tile_id": "t3", "duration_s": 0.1},
        {"event": "stage.end", "stage": "tile_prep", "duration_s": 5.0, "n_items": 3},
        {"event": "log", "message": "hello"},
    ]
    doc = aggregate_timings_file(_write_jsonl(tmp_path / "p.jsonl", events), run_id="r", workers=8)
    block = doc["stages"]["tile_prep"]
    assert block["wall_s"] == 5.0 and block["n_items"] == 3
    assert (block["n_done"], block["n_cached"], block["n_failed"]) == (1.0, 1.0, 1.0)
    assert doc["run_id"] == "r" and doc["workers"] == 8 and doc["total_wall_s"] == 5.0


def test_stage_without_end_is_not_reported() -> None:
    doc = aggregate_timings([{"event": "tile.done", "stage": "x", "tile_id": "t", "duration_s": 1.0}])
    assert doc["stages"] == {}


def test_unreadable_jsonl_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"event": "x"}\nnot json\n', encoding="utf-8")
    with pytest.raises(StageError, match="line"):
        read_jsonl_events(path)
