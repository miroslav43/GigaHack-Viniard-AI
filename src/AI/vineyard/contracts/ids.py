"""Contract IDs (contract §2.4 + v1.1 amendments): regexes, formatters, parsers, validation.

Strict regexes apply to model output. Marcaj and reference data are validated in relaxed
mode (any non-blank text), because organizers score raw values and we never rename them.
"""

import re
from collections.abc import Mapping
from enum import StrEnum
from functools import cache
from importlib import resources
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Final, Literal

from vineyard.contracts.enums import TargetKind
from vineyard.errors import SchemaError


class IdKind(StrEnum):
    TILE = "tile"
    BLOCK = "block"
    ROW = "row"
    INTERROW = "interrow"
    ROW_PIECE = "row_piece"
    INTERROW_PIECE = "interrow_piece"
    CANOPY = "canopy"
    ROW_CANDIDATE = "row_candidate"
    WASTE = "waste"
    WASTE_CANDIDATE = "waste_candidate"
    TARGET = "target"
    ISSUE = "issue"
    CHAIN = "chain"


TARGET_CODES: Final[Mapping[TargetKind, str]] = MappingProxyType(
    {
        TargetKind.ROW_GAP: "GAP",
        TargetKind.ROW_END_SHORT: "END",
        TargetKind.MISSING_ROW: "MRW",
        TargetKind.MISSING_PLANT: "MSP",
        TargetKind.SPARSE: "SPR",
        TargetKind.WASTE: "WST",
        TargetKind.OTHER: "OTH",
    }
)
TARGET_KIND_OF_CODE: Final[Mapping[str, TargetKind]] = MappingProxyType(
    {code: kind for kind, code in TARGET_CODES.items()}
)

TILE_FILE_EXT: Final = ".tif"
TILE_GRID_FILE: Final = "tile_grid.txt"
MAX_BLOCKS_2_DIGITS: Final = 99

_T = r"siret3_r(?P<row>\d{3})_c(?P<col>\d{3})"
_T_PLAIN = r"siret3_r\d{3}_c\d{3}"
_V = r"V\d{2,3}"
_DUP = r"(?:#(?P<dup>\d+))?"
_TARGET_ALT = "|".join(TARGET_CODES.values())

ID_PATTERNS: Final[Mapping[IdKind, re.Pattern[str]]] = MappingProxyType(
    {
        IdKind.TILE: re.compile(rf"^{_T}$"),
        IdKind.BLOCK: re.compile(rf"^{_V}$"),
        IdKind.ROW: re.compile(rf"^{_V}-R\d{{3}}$"),
        IdKind.INTERROW: re.compile(rf"^{_V}-I\d{{3}}$"),
        IdKind.ROW_PIECE: re.compile(rf"^{_V}-R\d{{3}}@{_T_PLAIN}{_DUP}$"),
        IdKind.INTERROW_PIECE: re.compile(rf"^(?:{_V}-I\d{{3}}@{_T_PLAIN}{_DUP}|{_T_PLAIN}:I\d{{3}})$"),
        IdKind.CANOPY: re.compile(rf"^{_T_PLAIN}:C\d{{4}}$"),
        IdKind.ROW_CANDIDATE: re.compile(rf"^{_T_PLAIN}:K\d{{2,3}}$"),
        IdKind.WASTE: re.compile(r"^W\d{4}[ab]?$"),
        IdKind.WASTE_CANDIDATE: re.compile(rf"^{_T_PLAIN}:W\d{{4}}$"),
        IdKind.TARGET: re.compile(rf"^T-(?:{_TARGET_ALT})-\d{{4}}$"),
        IdKind.ISSUE: re.compile(r"^Q\d{5}$"),
        IdKind.CHAIN: re.compile(r"^L\d{5}$"),
    }
)

# Relaxed parsers: accept 2+ digit row numbers (example data uses V01-R01) and any row/interrow text.
_ROW_INDEX_RE: Final = re.compile(r"-R(\d+)$")
_INTERROW_INDEX_RE: Final = re.compile(r"-I(\d+)$")
_PIECE_RE: Final = re.compile(rf"^(?P<base>.+)@(?P<tile>{_T_PLAIN})(?:#(?P<dup>\d+))?$")
_SCOPED_RE: Final = re.compile(rf"^(?P<tile>{_T_PLAIN}):(?P<letter>[A-Z])(?P<num>\d+)$")
_WASTE_RE: Final = re.compile(r"^W(?P<num>\d{4})(?P<part>[ab]?)$")
_TARGET_RE: Final = re.compile(r"^T-(?P<code>[A-Z]{3})-(?P<num>\d{4})$")


def _check_range(what: str, value: int, lo: int, hi: int) -> None:
    if not (lo <= value <= hi):
        raise SchemaError(f"{what}={value} outside [{lo}, {hi}]")


def _require_match(kind: IdKind, value: str) -> None:
    if not is_valid_id(kind, value):
        raise SchemaError(f"invalid {kind} id {value!r} (expected {ID_PATTERNS[kind].pattern})")


def is_valid_id(kind: IdKind, value: str, *, strict: bool = True) -> bool:
    """Strict: full regex match. Relaxed: any non-blank string."""
    if not isinstance(value, str):
        return False
    if not strict:
        return value.strip() != ""
    return ID_PATTERNS[kind].fullmatch(value) is not None


# ---- tiles -------------------------------------------------------------------------------------


def format_tile_id(grid_row: int, grid_col: int) -> str:
    _check_range("grid_row", grid_row, 0, 999)
    _check_range("grid_col", grid_col, 0, 999)
    return f"siret3_r{grid_row:03d}_c{grid_col:03d}"


def parse_tile_id(tile_id: str) -> tuple[int, int]:
    m = ID_PATTERNS[IdKind.TILE].fullmatch(tile_id) if isinstance(tile_id, str) else None
    if m is None:
        raise SchemaError(f"invalid tile id {tile_id!r}")
    return int(m["row"]), int(m["col"])


def tile_id_from_file_name(file_name: str) -> str:
    """Basename without `.tif` (case-sensitive); `images/` prefixes are ignored."""
    base = PurePosixPath(file_name.replace("\\", "/")).name
    if not base.endswith(TILE_FILE_EXT):
        raise SchemaError(f"not a tile file name: {file_name!r}")
    tile_id = base[: -len(TILE_FILE_EXT)]
    _require_match(IdKind.TILE, tile_id)
    return tile_id


def file_name_from_tile_id(tile_id: str) -> str:
    _require_match(IdKind.TILE, tile_id)
    return f"{tile_id}{TILE_FILE_EXT}"


@cache
def tile_grid_ids() -> tuple[str, ...]:
    """The 311 existing tile ids (sorted), from the packaged `tile_grid.txt`."""
    text = resources.files("vineyard.contracts").joinpath(TILE_GRID_FILE).read_text(encoding="ascii")
    ids = tuple(line.strip() for line in text.splitlines() if line.strip())
    bad = [t for t in ids if not is_valid_id(IdKind.TILE, t)]
    if bad:
        raise SchemaError(f"{TILE_GRID_FILE}: invalid tile ids {bad[:5]}")
    return ids


# ---- blocks, rows, interrows --------------------------------------------------------------------


def format_vineyard_id(index: int, n_blocks: int) -> str:
    """V01..V99; 3 digits for every block when there are more than 99 blocks."""
    _check_range("vineyard index", index, 1, max(n_blocks, 1))
    width = 2 if n_blocks <= MAX_BLOCKS_2_DIGITS else 3
    _check_range("vineyard index", index, 1, 10**width - 1)
    return f"V{index:0{width}d}"


def format_row_id(vineyard_id: str, row_index: int) -> str:
    _require_match(IdKind.BLOCK, vineyard_id)
    _check_range("row_index", row_index, 1, 999)
    return f"{vineyard_id}-R{row_index:03d}"


def format_interrow_id(vineyard_id: str, k: int) -> str:
    _require_match(IdKind.BLOCK, vineyard_id)
    _check_range("interrow index", k, 1, 999)
    return f"{vineyard_id}-I{k:03d}"


def row_index_of(row_id: str) -> int | None:
    """Row number from `...-R<digits>` (accepts R01 and R001); None when absent."""
    m = _ROW_INDEX_RE.search(row_id) if isinstance(row_id, str) else None
    return int(m.group(1)) if m else None


def interrow_index_of(interrow_id: str) -> int | None:
    m = _INTERROW_INDEX_RE.search(interrow_id) if isinstance(interrow_id, str) else None
    return int(m.group(1)) if m else None


# ---- per-tile pieces ----------------------------------------------------------------------------


def _format_piece(base: str, tile_id: str, dup: int) -> str:
    _require_match(IdKind.TILE, tile_id)
    _check_range("piece dup", dup, 1, 10**6)
    suffix = f"#{dup}" if dup > 1 else ""
    return f"{base}@{tile_id}{suffix}"


def _parse_piece(piece_id: str, what: str) -> tuple[str, str, int]:
    m = _PIECE_RE.fullmatch(piece_id) if isinstance(piece_id, str) else None
    if m is None:
        raise SchemaError(f"invalid {what} id {piece_id!r}")
    return m["base"], m["tile"], int(m["dup"]) if m["dup"] else 1


def format_row_piece_id(row_id: str, tile_id: str, dup: int = 1) -> str:
    return _format_piece(row_id, tile_id, dup)


def parse_row_piece_id(piece_id: str) -> tuple[str, str, int]:
    """(row_id, tile_id, dup); relaxed on the row part (accepts V01-R01@tile)."""
    return _parse_piece(piece_id, "row piece")


def format_interrow_piece_id(interrow_id: str, tile_id: str, dup: int = 1) -> str:
    return _format_piece(interrow_id, tile_id, dup)


def parse_interrow_piece_id(piece_id: str) -> tuple[str, str, int]:
    return _parse_piece(piece_id, "interrow piece")


def _format_scoped(tile_id: str, letter: str, k: int, width: int) -> str:
    _require_match(IdKind.TILE, tile_id)
    _check_range(f"{letter} index", k, 1, 10**width - 1)
    return f"{tile_id}:{letter}{k:0{width}d}"


def _parse_scoped(value: str, kind: IdKind, letter: str) -> tuple[str, int]:
    m = _SCOPED_RE.fullmatch(value) if isinstance(value, str) else None
    if m is None or m["letter"] != letter or not is_valid_id(kind, value):
        raise SchemaError(f"invalid {kind} id {value!r}")
    return m["tile"], int(m["num"])


def format_marcaj_interrow_piece_id(tile_id: str, k: int) -> str:
    return _format_scoped(tile_id, "I", k, 3)


def format_canopy_id(tile_id: str, k: int) -> str:
    return _format_scoped(tile_id, "C", k, 4)


def parse_canopy_id(canopy_id: str) -> tuple[str, int]:
    return _parse_scoped(canopy_id, IdKind.CANOPY, "C")


def format_row_candidate_id(tile_id: str, k: int) -> str:
    """K01..K99, then K100..K999 (contract v1.1 allows 2-3 digits)."""
    _require_match(IdKind.TILE, tile_id)
    _check_range("row candidate index", k, 1, 999)
    return f"{tile_id}:K{k:02d}"


def parse_row_candidate_id(cand_id: str) -> tuple[str, int]:
    return _parse_scoped(cand_id, IdKind.ROW_CANDIDATE, "K")


def format_waste_candidate_id(tile_id: str, k: int) -> str:
    return _format_scoped(tile_id, "W", k, 4)


# ---- global ids ---------------------------------------------------------------------------------


def format_waste_id(k: int, part: Literal["", "a", "b"] = "") -> str:
    _check_range("waste index", k, 1, 9999)
    if part not in ("", "a", "b"):
        raise SchemaError(f"waste id part must be '', 'a' or 'b', got {part!r}")
    return f"W{k:04d}{part}"


def parse_waste_id(waste_id: str) -> tuple[int, str]:
    m = _WASTE_RE.fullmatch(waste_id) if isinstance(waste_id, str) else None
    if m is None:
        raise SchemaError(f"invalid waste id {waste_id!r}")
    return int(m["num"]), m["part"]


def format_target_id(kind: TargetKind, k: int) -> str:
    _check_range("target index", k, 1, 9999)
    return f"T-{TARGET_CODES[TargetKind(kind)]}-{k:04d}"


def parse_target_id(target_id: str) -> tuple[TargetKind, int]:
    m = _TARGET_RE.fullmatch(target_id) if isinstance(target_id, str) else None
    if m is None or m["code"] not in TARGET_KIND_OF_CODE:
        raise SchemaError(f"invalid target id {target_id!r}")
    return TARGET_KIND_OF_CODE[m["code"]], int(m["num"])


def format_issue_id(k: int) -> str:
    _check_range("issue index", k, 1, 99999)
    return f"Q{k:05d}"


def format_chain_id(k: int) -> str:
    _check_range("chain index", k, 1, 99999)
    return f"L{k:05d}"
