"""Console (rich/plain) + JSONL logging. JSONL line fields follow contract §10."""

from __future__ import annotations

import json
import logging
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT_LOGGER: Final = "vineyard"
PLAIN_EVENT: Final = "log"
RESERVED_KEYS: Final = frozenset({"ts", "level", "run_id", "stage", "tile_id", "event", "logger", "message"})
_OWNED_ATTR: Final = "_vineyard_owned"
_EXTRA_KEY: Final = "vineyard_payload"

ConsoleKind = Literal["rich", "plain"]


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    return str(obj)


class JsonlFormatter(logging.Formatter):
    """One JSON object per record: ts, level, run_id, stage, tile_id, event, logger, extra fields."""

    def __init__(self, run_id: str, tz: ZoneInfo) -> None:
        super().__init__()
        self._run_id = run_id
        self._tz = tz

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = getattr(record, _EXTRA_KEY, None) or {}
        line: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, self._tz).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "run_id": self._run_id,
            "stage": payload.get("stage"),
            "tile_id": payload.get("tile_id"),
            "event": payload.get("event", PLAIN_EVENT),
            "logger": record.name,
        }
        if line["event"] == PLAIN_EVENT:
            line["message"] = record.getMessage()
        line.update(payload.get("fields", {}))
        if record.exc_info and "traceback" not in line:
            line["traceback"] = self.formatException(record.exc_info)
        return json.dumps(line, ensure_ascii=False, default=_json_default)


def _validate(level: str, console: str, tz: str) -> tuple[int, ZoneInfo]:
    levels = logging.getLevelNamesMapping()
    if level.upper() not in levels:
        raise ValueError(f"unknown log level: {level!r}")
    if console not in ("rich", "plain"):
        raise ValueError(f"unknown console kind: {console!r} (expected rich|plain)")
    try:
        zone = ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown tz: {tz!r}") from exc
    return levels[level.upper()], zone


def _console_handler(console: ConsoleKind) -> logging.Handler:
    if console == "rich":
        from rich.console import Console
        from rich.logging import RichHandler

        return RichHandler(console=Console(stderr=True), show_path=False, markup=False, rich_tracebacks=False)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    return handler


def teardown_logging() -> None:
    """Remove and close the handlers installed by `setup_logging`."""
    root = logging.getLogger(ROOT_LOGGER)
    for handler in [h for h in root.handlers if getattr(h, _OWNED_ATTR, False)]:
        root.removeHandler(handler)
        handler.close()
    root.propagate = True
    root.setLevel(logging.NOTSET)


def setup_logging(
    *, level: str, console: ConsoleKind, jsonl_path: Path | None, run_id: str, tz: str
) -> None:
    """Install console + optional JSONL handlers on the `vineyard` logger (idempotent)."""
    numeric, zone = _validate(level, console, tz)
    teardown_logging()
    handlers = [_console_handler(console)]
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(jsonl_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(JsonlFormatter(run_id, zone))
        handlers.append(file_handler)
    root = logging.getLogger(ROOT_LOGGER)
    for handler in handlers:
        setattr(handler, _OWNED_ATTR, True)
        root.addHandler(handler)
    root.setLevel(numeric)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Logger named `vineyard.<name>` (names already under `vineyard` are kept)."""
    if name == ROOT_LOGGER or name.startswith(ROOT_LOGGER + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def _safe_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {(f"field_{k}" if k in RESERVED_KEYS else k): v for k, v in fields.items()}


def _human(event: str, stage: str | None, tile_id: str | None, fields: dict[str, Any]) -> str:
    parts = [event]
    parts += [f"{key}={value}" for key, value in (("stage", stage), ("tile_id", tile_id)) if value is not None]
    parts += [f"{key}={value}" for key, value in fields.items() if key != "traceback"]
    return " ".join(parts)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    stage: str | None = None,
    tile_id: str | None = None,
    **fields: Any,
) -> None:
    """Structured event: human line on the console, full record in JSONL."""
    if not logger.isEnabledFor(level):
        return
    safe = _safe_fields(fields)
    payload = {"event": event, "stage": stage, "tile_id": tile_id, "fields": safe}
    logger.log(level, _human(event, stage, tile_id, safe), extra={_EXTRA_KEY: payload})


def log_failure(logger: logging.Logger, event: str, exc: BaseException, **fields: Any) -> None:
    """ERROR event carrying `error` and the full `traceback` of `exc`."""
    stage = fields.pop("stage", None)
    tile_id = fields.pop("tile_id", None)
    trace = "".join(traceback.format_exception(exc))
    log_event(
        logger,
        event,
        level=logging.ERROR,
        stage=stage,
        tile_id=tile_id,
        **fields,
        error=f"{type(exc).__name__}: {exc}",
        traceback=trace,
    )
