"""Content-addressed cache keys (contract §3.2): "<artifact>.key" holds the key, written last."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from vineyard.errors import StageError
from vineyard.pipeline.atomic import atomic_write_text

KEY_SUFFIX: Final = ".key"
_FIELD_SEP: Final = "\x1f"  # unit separator: no key part can contain it, so parts never run together
_ENCODING: Final = "utf-8"


def cache_key(stage: str, stage_version: str, cfg_digest: str, input_keys: Iterable[str]) -> str:
    """sha1 of `stage:version`, the stage config digest and the sorted input keys."""
    parts = [f"{stage}:{stage_version}", cfg_digest, *sorted(str(k) for k in input_keys)]
    return hashlib.sha1(_FIELD_SEP.join(parts).encode(_ENCODING)).hexdigest()


def key_path(artifact: Path) -> Path:
    """"<artifact>.key" next to the artifact."""
    target = Path(artifact)
    return target.with_name(target.name + KEY_SUFFIX)


def read_key(artifact: Path) -> str | None:
    """Stored key of `artifact`, or None when there is no key file."""
    path = key_path(artifact)
    if not path.is_file():
        return None
    return path.read_text(encoding=_ENCODING).strip()


def is_fresh(artifact: Path, key: str) -> bool:
    """True when the artifact exists and its key file holds exactly `key`."""
    return Path(artifact).is_file() and read_key(artifact) == key


def all_fresh(artifacts: Sequence[Path], key: str) -> bool:
    """True when there is at least one artifact and every one is fresh for `key`."""
    return bool(artifacts) and all(is_fresh(a, key) for a in artifacts)


def write_key(artifact: Path, key: str) -> Path:
    """Write "<artifact>.key" atomically; call only after the artifact itself is complete."""
    target = Path(artifact)
    if not key or not key.strip():
        raise StageError("empty cache key", artifact=str(target))
    if not target.is_file():
        raise StageError("cannot key a missing artifact: artifact missing", artifact=str(target))
    return atomic_write_text(key_path(target), key.strip() + "\n")


def write_keys(artifacts: Iterable[Path], key: str) -> tuple[Path, ...]:
    """write_key for each artifact in sorted path order."""
    return tuple(write_key(a, key) for a in sorted(Path(p) for p in artifacts))


def remove_key(artifact: Path) -> None:
    """Invalidate an artifact by deleting its key file (no error when absent)."""
    key_path(artifact).unlink(missing_ok=True)
