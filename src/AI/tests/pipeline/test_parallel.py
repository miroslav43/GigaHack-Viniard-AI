"""pipeline.parallel: inline and spawn-pool execution with per-item failures."""

from __future__ import annotations

import math
import operator
import sys

import pytest

from vineyard.errors import StageError
from vineyard.pipeline.parallel import ItemOutcome, init_worker, parallel_map


def _key(x: float) -> str:
    return f"k{x:+.1f}"


def test_inline_runs_in_order_and_captures_failures() -> None:
    outcomes = list(parallel_map(math.sqrt, [4.0, -1.0, 9.0], key=_key, workers=1))
    assert [o.key for o in outcomes] == ["k+4.0", "k-1.0", "k+9.0"]
    assert [o.ok for o in outcomes] == [True, False, True]
    assert outcomes[0].value == 2.0 and outcomes[2].value == 3.0
    bad = outcomes[1]
    assert bad.value is None
    assert bad.error is not None and bad.error.startswith("ValueError")
    assert bad.traceback is not None and "Traceback" in bad.traceback
    assert all(o.duration_s >= 0.0 for o in outcomes)


def test_empty_items_yield_nothing() -> None:
    assert list(parallel_map(math.sqrt, [], key=_key, workers=4)) == []


def test_spawn_pool_returns_all_outcomes_in_input_order() -> None:
    items = [float(v) for v in (1, 4, -4, 16, 25)]
    outcomes = list(parallel_map(math.sqrt, items, key=_key, workers=2, chunksize=2, maxtasksperchild=2))
    assert [o.key for o in outcomes] == [_key(v) for v in items]
    assert [o.ok for o in outcomes] == [True, True, False, True, True]
    assert [o.value for o in outcomes if o.ok] == [1.0, 2.0, 4.0, 5.0]
    assert all(isinstance(o, ItemOutcome) for o in outcomes)
    assert all(o.peak_rss_mb > 0.0 for o in outcomes if o.ok)


CRASH_EXIT_CODE = 3


def _crash_on_negative(x: float) -> float:
    """Top-level: a negative item kills its worker process (like an OOM kill)."""
    if x < 0:
        import os

        os._exit(CRASH_EXIT_CODE)
    return x


def test_dead_worker_fails_its_items_without_raising() -> None:
    items = [1.0, -1.0]
    outcomes = list(parallel_map(_crash_on_negative, items, key=_key, workers=2))
    assert [o.key for o in outcomes] == [_key(v) for v in items]
    crashed = outcomes[1]
    assert not crashed.ok and crashed.error is not None and crashed.error.startswith("worker crashed")


def test_duplicate_keys_rejected() -> None:
    with pytest.raises(StageError, match="unique"):
        list(parallel_map(operator.neg, [1, 1], key=str, workers=1))


@pytest.mark.parametrize("kwargs", [{"workers": 0}, {"workers": 2, "chunksize": 0},
                                    {"workers": 2, "maxtasksperchild": 0}])
def test_invalid_pool_settings_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(StageError, match=">= 1"):
        list(parallel_map(operator.neg, [1], key=str, **kwargs))


def test_init_worker_sets_single_cv2_thread_and_never_imports_torch() -> None:
    import cv2

    before = cv2.getNumThreads()
    try:
        init_worker()
        assert cv2.getNumThreads() == 1
        assert not cv2.ocl.useOpenCL()
    finally:
        cv2.setNumThreads(before)
    assert "torch" not in sys.modules
