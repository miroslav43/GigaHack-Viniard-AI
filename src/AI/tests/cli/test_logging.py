"""Logging: JSONL fields per contract §10, tz offsets, tracebacks, idempotent setup."""

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from vineyard import logging_setup as ls

RUN_ID = "20260926T0310-model-a1b2c3"


@pytest.fixture
def jsonl(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "logs" / "pipeline.jsonl"
    ls.setup_logging(level="INFO", console="plain", jsonl_path=path, run_id=RUN_ID, tz="Europe/Chisinau")
    yield path
    ls.teardown_logging()


def _lines(path: Path) -> list[dict]:
    for handler in logging.getLogger("vineyard").handlers:
        handler.flush()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_get_logger_namespaces_under_vineyard() -> None:
    assert ls.get_logger("canopy").name == "vineyard.canopy"
    assert ls.get_logger("vineyard.cli").name == "vineyard.cli"
    assert ls.get_logger("vineyard").name == "vineyard"


def test_event_line_has_contract_fields(jsonl: Path) -> None:
    log = ls.get_logger("canopy")
    ls.log_event(log, "tile.done", stage="canopy", tile_id="siret3_r021_c012",
                 duration_s=0.84, n_canopies=np.int64(402), veg_frac=np.float32(0.071))
    (line,) = _lines(jsonl)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+0[23]:00", line["ts"])
    assert line["level"] == "INFO"
    assert line["run_id"] == RUN_ID
    assert line["stage"] == "canopy"
    assert line["tile_id"] == "siret3_r021_c012"
    assert line["event"] == "tile.done"
    assert line["n_canopies"] == 402
    assert line["veg_frac"] == pytest.approx(0.071, abs=1e-6)
    assert line["logger"] == "vineyard.canopy"


def test_plain_log_record_goes_to_jsonl_as_log_event(jsonl: Path) -> None:
    ls.get_logger("x").warning("ceva %s", "simplu")
    (line,) = _lines(jsonl)
    assert line["event"] == "log"
    assert line["message"] == "ceva simplu"
    assert line["level"] == "WARNING"
    assert line["stage"] is None


def test_log_failure_includes_traceback(jsonl: Path) -> None:
    log = ls.get_logger("runner")
    try:
        raise ValueError("boom")
    except ValueError as exc:
        ls.log_failure(log, "tile.failed", exc, stage="canopy", tile_id="t1")
    (line,) = _lines(jsonl)
    assert line["event"] == "tile.failed"
    assert line["level"] == "ERROR"
    assert line["error"] == "ValueError: boom"
    assert "Traceback" in line["traceback"] and "boom" in line["traceback"]


def test_level_filters_debug(jsonl: Path) -> None:
    ls.log_event(ls.get_logger("x"), "dbg", level=logging.DEBUG)
    assert _lines(jsonl) == []


def test_setup_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    for _ in range(2):
        ls.setup_logging(level="INFO", console="rich", jsonl_path=path, run_id=RUN_ID, tz="UTC")
    try:
        ls.log_event(ls.get_logger("x"), "once")
        assert len(_lines(path)) == 1
        assert _lines(path)[0]["ts"].endswith("+00:00")
    finally:
        ls.teardown_logging()


def test_setup_without_jsonl_and_bad_values(tmp_path: Path) -> None:
    ls.setup_logging(level="WARNING", console="plain", jsonl_path=None, run_id=RUN_ID, tz="UTC")
    ls.teardown_logging()
    with pytest.raises(ValueError, match="console"):
        ls.setup_logging(level="INFO", console="fancy", jsonl_path=None, run_id=RUN_ID, tz="UTC")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tz"):
        ls.setup_logging(level="INFO", console="plain", jsonl_path=None, run_id=RUN_ID, tz="Mars/Olympus")
    with pytest.raises(ValueError, match="level"):
        ls.setup_logging(level="LOUD", console="plain", jsonl_path=None, run_id=RUN_ID, tz="UTC")


def test_unserialisable_values_become_strings(jsonl: Path) -> None:
    ls.log_event(ls.get_logger("x"), "paths", out=Path("/tmp/a b"), items=(1, 2), obj=object())
    (line,) = _lines(jsonl)
    assert line["out"] == "/tmp/a b"
    assert line["items"] == [1, 2]
    assert line["obj"].startswith("<object object")
