"""Human tile review list (`web.tile_review`, e.g. configs/tile_review.csv): tiles that need completion in Marcaj.

CSV with the exact header `tile_id,status,note`; status is missed | partial | verify, note is a short non-empty
text. Unknown tile ids, bad statuses, duplicates and malformed lines are ConfigErrors naming the file and line.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from vineyard.contracts.ids import tile_grid_ids
from vineyard.errors import ConfigError

REVIEW_HEADER: Final = ("tile_id", "status", "note")
REVIEW_STATUSES: Final = frozenset({"missed", "partial", "verify"})
HEADER_LINE: Final = 1


@dataclass(frozen=True)
class TileReview:
    status: str
    note: str


def _row(fields: list[str], line: int, path: Path, known: frozenset[str]) -> tuple[str, TileReview]:
    if len(fields) != len(REVIEW_HEADER):
        raise ConfigError(f"tile review line must have {len(REVIEW_HEADER)} columns", path=str(path), line=line,
                          found=len(fields))
    tile_id, status, note = (f.strip() for f in fields)
    if tile_id not in known:
        raise ConfigError("unknown tile id in the tile review file", path=str(path), line=line, tile_id=tile_id)
    if status not in REVIEW_STATUSES:
        raise ConfigError("tile review status must be missed, partial or verify", path=str(path), line=line,
                          status=status)
    if not note:
        raise ConfigError("tile review note is empty", path=str(path), line=line, tile_id=tile_id)
    return tile_id, TileReview(status=status, note=note)


def _lines(path: Path) -> list[list[str]]:
    if not path.is_file():
        raise ConfigError("tile review file not found", path=str(path), line=0)
    try:
        return list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ConfigError(f"tile review file is unreadable: {exc}", path=str(path), line=0) from exc


def read_tile_review(path: Path | None) -> Mapping[str, TileReview]:
    """tile_id -> TileReview in file order; {} when `path` is None (no review list configured)."""
    if path is None:
        return MappingProxyType({})
    source = Path(path)
    lines = _lines(source)
    if not lines or tuple(f.strip() for f in lines[0]) != REVIEW_HEADER:
        raise ConfigError("tile review header must be tile_id,status,note", path=str(source), line=HEADER_LINE)
    known = frozenset(tile_grid_ids())
    out: dict[str, TileReview] = {}
    for line, fields in enumerate(lines[1:], start=HEADER_LINE + 1):
        if not fields:
            continue
        tile_id, review = _row(fields, line, source, known)
        if tile_id in out:
            raise ConfigError("duplicate tile id in the tile review file", path=str(source), line=line,
                              tile_id=tile_id)
        out[tile_id] = review
    return MappingProxyType(out)
