"""Atomic file writes: write to "<name>.tmp-<pid>" next to the target, then os.replace.

A reader never sees a half-written artifact, and a failure leaves the previous version
(or nothing) in place. Cache keys ("<artifact>.key") must be written after the artifact.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

JSON_INDENT = 2


def tmp_path_for(path: Path) -> Path:
    """Temporary sibling used while `path` is being written."""
    target = Path(path)
    return target.with_name(f"{target.name}.tmp-{os.getpid()}")


def _discard(tmp: Path) -> None:
    if tmp.exists():
        tmp.unlink()


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path; on normal exit it replaces `path`, on error it is removed.

    The body must create the yielded file; if it does not, os.replace raises FileNotFoundError.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path_for(target)
    try:
        yield tmp
        os.replace(tmp, target)
    except BaseException:
        _discard(tmp)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write `data` to `path` atomically and return `path`."""
    target = Path(path)
    with atomic_path(target) as tmp:
        tmp.write_bytes(data)
    return target


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write `text` to `path` atomically and return `path`."""
    return atomic_write_bytes(path, text.encode(encoding))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def dumps_json(obj: Any) -> str:
    """Canonical JSON text: sorted keys, 2-space indent, UTF-8, strict (no NaN), trailing newline."""
    text = json.dumps(
        obj,
        sort_keys=True,
        indent=JSON_INDENT,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )
    return text + "\n"


def atomic_write_json(path: Path, obj: Any) -> Path:
    """Serialize `obj` deterministically (see dumps_json) and write it atomically."""
    return atomic_write_text(path, dumps_json(obj))
