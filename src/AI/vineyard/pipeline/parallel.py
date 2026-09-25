"""Process pool for per-tile work: spawn context, one OpenCV thread per worker, never torch.

`parallel_map` yields one ItemOutcome per item, in input order. A raising item becomes an
outcome with ok=False (error + traceback). A dead worker (e.g. OOM-killed) breaks the pool: every
chunk not finished by then fails with "worker crashed" (the cache makes a rerun retry only those).
workers <= 1 runs inline in this process (debuggable, no pickling).
"""

from __future__ import annotations

import multiprocessing
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Final

from vineyard.errors import StageError
from vineyard.pipeline.timings import usage_self

START_METHOD: Final = "spawn"
CV2_THREADS: Final = 0  # 0 = sequential; the macOS GCD backend ignores setNumThreads(1)


@dataclass(frozen=True)
class ItemOutcome[R]:
    key: str
    ok: bool
    value: R | None
    error: str | None
    traceback: str | None
    duration_s: float
    cpu_s: float = 0.0
    peak_rss_mb: float = 0.0


def init_worker() -> None:
    """Pool initializer: sequential OpenCV without OpenCL. The thread env is already set: unpickling this
    function imports `vineyard` (and so vineyard._env) before anything else in the worker."""
    import cv2

    cv2.setNumThreads(CV2_THREADS)
    cv2.ocl.setUseOpenCL(False)


def _run_one[T, R](fn: Callable[[T], R], item: T, key: str) -> ItemOutcome[R]:
    start_wall = time.perf_counter()
    start_cpu = usage_self().cpu_s
    try:
        value = fn(item)
    except Exception as exc:  # noqa: BLE001  (reported per item with its traceback, never swallowed)
        usage = usage_self()
        return ItemOutcome(key=key, ok=False, value=None, error=f"{type(exc).__name__}: {exc}",
                           traceback="".join(traceback.format_exception(exc)),
                           duration_s=time.perf_counter() - start_wall, cpu_s=usage.cpu_s - start_cpu,
                           peak_rss_mb=usage.peak_rss_mb)
    usage = usage_self()
    return ItemOutcome(key=key, ok=True, value=value, error=None, traceback=None,
                       duration_s=time.perf_counter() - start_wall, cpu_s=usage.cpu_s - start_cpu,
                       peak_rss_mb=usage.peak_rss_mb)


def _run_chunk[T, R](fn: Callable[[T], R], chunk: Sequence[tuple[T, str]]) -> list[ItemOutcome[R]]:
    return [_run_one(fn, item, key) for item, key in chunk]


def _chunks[T](pairs: list[tuple[T, str]], size: int) -> list[list[tuple[T, str]]]:
    return [pairs[i : i + size] for i in range(0, len(pairs), size)]


def _crashed(chunk: Sequence[tuple[object, str]], exc: BaseException) -> list[ItemOutcome[Any]]:
    trace = "".join(traceback.format_exception(exc))
    return [ItemOutcome(key=key, ok=False, value=None, error=f"worker crashed: {type(exc).__name__}: {exc}",
                        traceback=trace, duration_s=0.0) for _, key in chunk]


def _validate[T](items: Sequence[T], keys: list[str], workers: int, chunksize: int, maxtasksperchild: int) -> None:
    if workers < 1 or chunksize < 1 or maxtasksperchild < 1:
        raise StageError("parallel_map: workers, chunksize and maxtasksperchild must be >= 1",
                         workers=workers, chunksize=chunksize, maxtasksperchild=maxtasksperchild)
    if len(set(keys)) != len(keys):
        raise StageError("parallel_map: item keys must be unique", n_items=len(items), n_keys=len(set(keys)))


def _pooled[T, R](fn: Callable[[T], R], pairs: list[tuple[T, str]], workers: int, chunksize: int,
            maxtasksperchild: int) -> Iterator[ItemOutcome[R]]:
    chunks = _chunks(pairs, chunksize)
    context = multiprocessing.get_context(START_METHOD)
    n_procs = min(workers, len(chunks))
    with ProcessPoolExecutor(max_workers=n_procs, mp_context=context, initializer=init_worker,
                             max_tasks_per_child=maxtasksperchild) as pool:
        futures: list[Future[list[ItemOutcome[R]]]] = [pool.submit(_run_chunk, fn, c) for c in chunks]
        for chunk, future in zip(chunks, futures, strict=True):
            try:
                yield from future.result()
            except BrokenProcessPool as exc:
                yield from _crashed(chunk, exc)


def parallel_map[T, R](
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    key: Callable[[T], str],
    workers: int,
    chunksize: int = 1,
    maxtasksperchild: int = 50,
) -> Iterator[ItemOutcome[R]]:
    """Apply a top-level picklable `fn` to every item; yields outcomes in input order."""
    keys = [key(item) for item in items]
    _validate(items, keys, workers, chunksize, maxtasksperchild)
    pairs = list(zip(items, keys, strict=True))
    if not pairs:
        return iter(())
    if workers <= 1:
        return (_run_one(fn, item, k) for item, k in pairs)
    return _pooled(fn, pairs, workers, chunksize, maxtasksperchild)
