"""`status.json` of a demo job (web contract: src/Web/docs/design/2026-09-27-analiza-tif-spec.md).

The web page polls this file. `StageProgress` is a logging handler on the `vineyard` logger: every `stage.start`
event of the run rewrites the file with the stage name and its index, so the page can show live progress.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

from vineyard.logging_setup import ROOT_LOGGER
from vineyard.pipeline.atomic import atomic_write_json

STATUS_FILE: Final = "status.json"
PAYLOAD_ATTR: Final = "vineyard_payload"  # logging_setup._EXTRA_KEY
STAGE_START: Final = "stage.start"
State = Literal["queued", "running", "done", "error"]


def write_status(
    out_dir: Path,
    state: State,
    *,
    n_stages: int,
    stage: str | None = None,
    stage_index: int = 0,
    error: tuple[str, str] | None = None,
) -> Path:
    """Rewrite <out_dir>/status.json atomically; `error` is (code, message)."""
    doc = {
        "state": state,
        "stage": stage,
        "stage_index": stage_index,
        "n_stages": n_stages,
        "error": None if error is None else {"code": error[0], "message": error[1]},
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return atomic_write_json(out_dir / STATUS_FILE, doc)


class StageProgress(logging.Handler):
    """Writes status.json on every stage.start of `stages` (other stages and events are ignored)."""

    def __init__(self, out_dir: Path, stages: Sequence[str]) -> None:
        super().__init__(level=logging.INFO)
        self._out_dir = out_dir
        self._stages = tuple(stages)

    def emit(self, record: logging.LogRecord) -> None:
        payload = getattr(record, PAYLOAD_ATTR, None) or {}
        stage = payload.get("stage")
        if payload.get("event") != STAGE_START or stage not in self._stages:
            return
        try:
            write_status(self._out_dir, "running", n_stages=len(self._stages), stage=stage,
                         stage_index=self._stages.index(stage) + 1)
        except OSError:
            self.handleError(record)

    def __enter__(self) -> StageProgress:
        logging.getLogger(ROOT_LOGGER).addHandler(self)
        return self

    def __exit__(self, *exc: object) -> None:
        logging.getLogger(ROOT_LOGGER).removeHandler(self)
        self.close()
