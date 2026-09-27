"""Read selected members of a remote ZIP through HTTP Range requests (no full download).

Only the central directory and the byte span covering the wanted members are fetched. Works for
non-ZIP64 archives with stored (0) or deflated (8) members, which covers the ICAERUS archive.
"""

from __future__ import annotations

import fnmatch
import struct
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from fte.data.http_range import fetch_range, remote_size

EOCD_SIG = b"PK\x05\x06"
CDIR_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"
EOCD_SEARCH = 65_557  # 22-byte EOCD + max 65 535-byte comment

Fetch = Callable[[str, int, int], bytes]


class ZipRangeError(RuntimeError):
    """The remote archive is not in a supported layout."""


@dataclass(frozen=True)
class Member:
    name: str
    method: int
    comp_size: int
    size: int
    header_offset: int


def parse_eocd(tail: bytes) -> tuple[int, int, int]:
    """(entries, cdir_size, cdir_offset) from the last bytes of the archive."""
    pos = tail.rfind(EOCD_SIG)
    if pos < 0:
        raise ZipRangeError("end-of-central-directory record not found")
    entries, cdir_size, cdir_offset = struct.unpack("<HII", tail[pos + 10 : pos + 20])
    if cdir_offset == 0xFFFFFFFF or entries == 0xFFFF:
        raise ZipRangeError("ZIP64 archives are not supported")
    return entries, cdir_size, cdir_offset


def parse_central_directory(cdir: bytes) -> list[Member]:
    members, pos = [], 0
    while pos + 46 <= len(cdir) and cdir[pos : pos + 4] == CDIR_SIG:
        method, = struct.unpack("<H", cdir[pos + 10 : pos + 12])
        comp_size, size = struct.unpack("<II", cdir[pos + 20 : pos + 28])
        name_len, extra_len, comment_len = struct.unpack("<HHH", cdir[pos + 28 : pos + 34])
        offset, = struct.unpack("<I", cdir[pos + 42 : pos + 46])
        name = cdir[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
        members.append(Member(name, method, comp_size, size, offset))
        pos += 46 + name_len + extra_len + comment_len
    return members


def list_members(url: str, fetch: Fetch = fetch_range, total: int | None = None) -> list[Member]:
    total = remote_size(url) if total is None else total
    tail_start = max(0, total - EOCD_SEARCH)
    _, cdir_size, cdir_offset = parse_eocd(fetch(url, tail_start, total - 1))
    return parse_central_directory(fetch(url, cdir_offset, cdir_offset + cdir_size - 1))


def select(members: Iterable[Member], patterns: Iterable[str]) -> list[Member]:
    pats = list(patterns)
    return [m for m in members if not m.name.endswith("/") and any(fnmatch.fnmatch(m.name, p) for p in pats)]


def _decompress(member: Member, raw: bytes) -> bytes:
    if member.method == 0:
        return raw
    if member.method == 8:
        return zlib.decompressobj(-15).decompress(raw)
    raise ZipRangeError(f"unsupported compression method {member.method} for {member.name}")


def extract_from_block(block: bytes, block_start: int, member: Member) -> bytes:
    """Decompress ``member`` from a fetched byte block starting at archive offset ``block_start``."""
    pos = member.header_offset - block_start
    if block[pos : pos + 4] != LOCAL_SIG:
        raise ZipRangeError(f"bad local header for {member.name}")
    name_len, extra_len = struct.unpack("<HH", block[pos + 26 : pos + 30])
    data_start = pos + 30 + name_len + extra_len
    data = _decompress(member, block[data_start : data_start + member.comp_size])
    if len(data) != member.size:
        raise ZipRangeError(f"size mismatch for {member.name}: {len(data)} != {member.size}")
    return data


def fetch_members(url: str, members: list[Member], total: int, fetch: Fetch = fetch_range,
                  slack: int = 1 << 16) -> dict[str, bytes]:
    """Fetch the smallest span holding every member (one request) and decompress each.

    ``slack`` covers the local header's extra field, which may differ from the central one.
    """
    if not members:
        return {}
    start = min(m.header_offset for m in members)
    end = max(m.header_offset + 30 + len(m.name.encode()) + slack + m.comp_size for m in members)
    block = fetch(url, start, min(end, total) - 1)
    return {m.name: extract_from_block(block, start, m) for m in members}
