"""Resource usage and timing aggregation -> metrics/timings.json (arch §4.1, README timings)."""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from vineyard.errors import StageError

MB: Final = 1024.0 * 1024.0
_KIB: Final = 1024.0
_RSS_BYTES_PLATFORMS: Final = ("darwin",)  # ru_maxrss is bytes on macOS, KiB on Linux
EVENT_STAGE_END: Final = "stage.end"
EVENT_TILE_DONE: Final = "tile.done"
EVENT_TILE_CACHED: Final = "tile.cached"
EVENT_TILE_FAILED: Final = "tile.failed"
STATUS_DONE: Final = "done"
STATUS_CACHED: Final = "cached"
STATUS_FAILED: Final = "failed"
_STATUS_OF_EVENT: Final[Mapping[str, str]] = {
    EVENT_TILE_DONE: STATUS_DONE, EVENT_TILE_CACHED: STATUS_CACHED, EVENT_TILE_FAILED: STATUS_FAILED,
}
_P50: Final = 50.0
_P95: Final = 95.0
_ROUND: Final = 4


@dataclass(frozen=True)
class Usage:
    """CPU seconds (user + system) and peak RSS in MiB."""

    cpu_s: float
    peak_rss_mb: float


def peak_rss_mb(maxrss: float, platform_name: str = sys.platform) -> float:
    """ru_maxrss normalised to MiB (bytes on macOS, KiB on Linux)."""
    n_bytes = float(maxrss) if platform_name in _RSS_BYTES_PLATFORMS else float(maxrss) * _KIB
    return n_bytes / MB


def _usage(who: int) -> Usage:
    ru = resource.getrusage(who)
    return Usage(cpu_s=ru.ru_utime + ru.ru_stime, peak_rss_mb=peak_rss_mb(ru.ru_maxrss))


def usage_self() -> Usage:
    return _usage(resource.RUSAGE_SELF)


def usage_children() -> Usage:
    return _usage(resource.RUSAGE_CHILDREN)


@dataclass(frozen=True)
class ItemTiming:
    """One tile of one stage: status done|cached|failed; cpu/rss are worker-side (0 when unknown)."""

    stage: str
    tile_id: str
    status: str
    duration_s: float
    cpu_s: float = 0.0
    peak_rss_mb: float = 0.0


def hardware_info() -> dict[str, Any]:
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "python": platform.python_version()}


def _r(value: float) -> float:
    return round(float(value), _ROUND)


def summarize_items(items: Iterable[ItemTiming]) -> dict[str, float]:
    """Flat per-stage metrics: counts by status and per-tile seconds of the computed (done) tiles."""
    rows = list(items)
    done = np.asarray([i.duration_s for i in rows if i.status == STATUS_DONE], dtype=np.float64)
    out: dict[str, float] = {
        "n_done": float(done.size),
        "n_cached": float(sum(1 for i in rows if i.status == STATUS_CACHED)),
        "n_failed": float(sum(1 for i in rows if i.status == STATUS_FAILED)),
        "tile_s_sum": _r(done.sum()) if done.size else 0.0,
        "tile_s_mean": _r(done.mean()) if done.size else 0.0,
        "tile_s_p50": _r(np.percentile(done, _P50)) if done.size else 0.0,
        "tile_s_p95": _r(np.percentile(done, _P95)) if done.size else 0.0,
        "tile_s_max": _r(done.max()) if done.size else 0.0,
        "cpu_s_sum": _r(sum(i.cpu_s for i in rows if i.status == STATUS_DONE)),
        "peak_rss_mb_max": _r(max((i.peak_rss_mb for i in rows if i.status == STATUS_DONE), default=0.0)),
    }
    return out


def stage_timing_block(wall_s: float, n_items: int, metrics: Mapping[str, float]) -> dict[str, Any]:
    """timings.json entry of one stage: wall time, item count and the stage metrics."""
    return {"wall_s": _r(wall_s), "n_items": int(n_items), **{k: metrics[k] for k in sorted(metrics)}}


def build_timings(run_id: str, workers: int, stages: Mapping[str, Mapping[str, Any]],
                  previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """timings.json document; stages of this invocation replace the same stages of `previous`."""
    merged = dict((previous or {}).get("stages", {})) | {k: dict(v) for k, v in stages.items()}
    total = sum(float(b.get("wall_s") or 0.0) for b in merged.values())
    return {"run_id": run_id, "workers": int(workers), "stages": merged, "total_wall_s": _r(total),
            "hardware": hardware_info()}


def read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    """Every JSON object of a pipeline.jsonl file (blank lines skipped)."""
    source = Path(path)
    events: list[dict[str, Any]] = []
    for n, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise StageError("unreadable pipeline.jsonl line", path=str(source), line=n, error=str(exc)) from exc
    return events


def _item_from_event(event: Mapping[str, Any]) -> ItemTiming | None:
    status = _STATUS_OF_EVENT.get(str(event.get("event")))
    if status is None:
        return None
    return ItemTiming(stage=str(event.get("stage")), tile_id=str(event.get("tile_id")), status=status,
                      duration_s=float(event.get("duration_s") or 0.0), cpu_s=float(event.get("cpu_s") or 0.0),
                      peak_rss_mb=float(event.get("peak_rss_mb") or 0.0))


def aggregate_timings(events: Iterable[Mapping[str, Any]], *, run_id: str = "", workers: int = 0) -> dict[str, Any]:
    """timings.json from stage.end / tile.* events (e.g. a pipeline.jsonl); a later stage.end wins."""
    rows = list(events)
    walls: dict[str, tuple[float, int]] = {}
    items: dict[str, list[ItemTiming]] = {}
    for event in rows:
        if event.get("event") == EVENT_STAGE_END:
            walls[str(event.get("stage"))] = (float(event.get("duration_s") or 0.0), int(event.get("n_items") or 0))
        item = _item_from_event(event)
        if item is not None:
            items.setdefault(item.stage, []).append(item)
    stages = {name: stage_timing_block(wall, n, summarize_items(items.get(name, [])))
              for name, (wall, n) in walls.items()}
    return build_timings(run_id, workers, stages)


def aggregate_timings_file(jsonl_path: Path, *, run_id: str = "", workers: int = 0) -> dict[str, Any]:
    return aggregate_timings(read_jsonl_events(jsonl_path), run_id=run_id, workers=workers)
