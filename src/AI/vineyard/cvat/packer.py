"""Upload ZIP planning and writing (01 §3.9, arch §4.13 step 4).

Tiles are packed in sorted-name order into contiguous parts: the part count is the greedy minimum
at `max_bytes` (or `n_parts`), and `balance` evens the parts out. ZIPs are byte-deterministic:
annotations.xml first (DEFLATED), then images/<name> sorted (STORED), fixed timestamps and modes.
"""

from __future__ import annotations

import hashlib
import unicodedata
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from types import MappingProxyType
from typing import Final

from vineyard.errors import ExportBlocked
from vineyard.pipeline.atomic import atomic_path

ANNOTATIONS_NAME: Final = "annotations.xml"
IMAGES_DIR: Final = "images"
IMAGE_EXT: Final = ".tif"
ZIP_DATE_TIME: Final = (1980, 1, 1, 0, 0, 0)
ZIP_FILE_MODE: Final = 0o100644
ZIP_UNIX_SYSTEM: Final = 3
# PKZIP fixed sizes: local header 30 + central directory entry 46 (+ name twice); end record 22.
ZIP_ENTRY_OVERHEAD: Final = 30 + 46
ZIP_END_RECORD: Final = 22


@dataclass(frozen=True)
class PartPlan:
    index: int  # 1-based
    tile_ids: tuple[str, ...]
    est_bytes: int


def image_arcname(file_name: str) -> str:
    return f"{IMAGES_DIR}/{file_name}"


def estimate_tile_bytes(tif_bytes: int, file_name: str, xml_fragment_deflated: int) -> int:
    """Conservative bytes one tile adds to a ZIP: STORED data + headers + its deflated XML fragment."""
    return tif_bytes + ZIP_ENTRY_OVERHEAD + 2 * len(image_arcname(file_name)) + xml_fragment_deflated


def zip_base_bytes(xml_frame_deflated: int) -> int:
    """Per-ZIP fixed bytes: the deflated XML frame (header, meta, footer), its entry and the end record."""
    return xml_frame_deflated + ZIP_ENTRY_OVERHEAD + 2 * len(ANNOTATIONS_NAME) + ZIP_END_RECORD


def greedy_count(sizes: Sequence[int], budget: int) -> int:
    """Minimal number of contiguous parts with sum <= budget (greedy is optimal for contiguous packing)."""
    count, used = 0, 0
    for k, size in enumerate(sizes):
        if size > budget:
            raise ExportBlocked("a single tile exceeds the ZIP budget", index=k, size=size, budget=budget)
        if count == 0 or used + size > budget:
            count, used = count + 1, size
        else:
            used += size
    return count


def _greedy_cuts(sizes: Sequence[int], budget: int) -> list[int]:
    cuts, used = [], 0
    for k, size in enumerate(sizes):
        if k > 0 and used + size > budget:
            cuts.append(k)
            used = 0
        used += size
    return cuts


def _nearest_cuts(sizes: Sequence[int], n: int) -> list[int]:
    """Cut indices whose prefix sums are nearest to k*total/n; every part keeps >= 1 tile."""
    prefix = list(accumulate(sizes))
    total, m = prefix[-1], len(sizes)
    cuts: list[int] = []
    for k in range(1, n):
        lo = (cuts[-1] if cuts else 0) + 1
        hi = m - (n - k)
        ideal = total * k / n
        cuts.append(min(range(lo, hi + 1), key=lambda i: (abs(prefix[i - 1] - ideal), i)))
    return cuts


def _minmax_cuts(sizes: Sequence[int], n: int) -> list[int]:
    """Cuts minimising the largest part (binary search on capacity), split further to exactly n parts."""
    lo, hi = max(sizes), sum(sizes)
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if greedy_count(sizes, mid) <= n else (mid + 1, hi)
    bounds = [0, *_greedy_cuts(sizes, lo), len(sizes)]
    while len(bounds) - 1 < n:
        widest = max(range(len(bounds) - 1), key=lambda j: (bounds[j + 1] - bounds[j], -j))
        a, b = bounds[widest], bounds[widest + 1]
        bounds = sorted({*bounds, a + (b - a) // 2})
    return bounds[1:-1]


def _parts_from_cuts(ids: Sequence[str], sizes: Sequence[int], cuts: Sequence[int], base: int) -> tuple[PartPlan, ...]:
    bounds = [0, *cuts, len(ids)]
    return tuple(
        PartPlan(j + 1, tuple(ids[a:b]), base + sum(sizes[a:b]))
        for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
    )


def _check_sizes(sizes: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    ordered = sorted(sizes)
    ids = [t for t, _ in ordered]
    if len(set(ids)) != len(ids):
        raise ExportBlocked("duplicate tile in the packing plan", n=len(ids), n_unique=len(set(ids)))
    if any(s < 0 for _, s in ordered):
        raise ExportBlocked("negative tile size in the packing plan")
    return ordered


def _choose_cuts(sizes: list[int], budget: int, n: int, balance: bool, greedy: int) -> list[int]:
    if not balance and n == greedy:
        return _greedy_cuts(sizes, budget)
    cuts = _nearest_cuts(sizes, n)
    bounds = [0, *cuts, len(sizes)]
    if all(sum(sizes[a:b]) <= budget for a, b in zip(bounds[:-1], bounds[1:], strict=True)):
        return cuts
    return _minmax_cuts(sizes, n)


def plan_parts(
    sizes: Sequence[tuple[str, int]],
    *,
    max_bytes: int,
    n_parts: int | None,
    balance: bool,
    base_bytes: int = 0,
) -> tuple[PartPlan, ...]:
    """Contiguous parts over the sorted tile ids; each part's estimate (base + sizes) <= max_bytes."""
    ordered = _check_sizes(sizes)
    if not ordered:
        return ()
    ids, values = [t for t, _ in ordered], [s for _, s in ordered]
    budget = max_bytes - base_bytes
    greedy = greedy_count(values, budget)
    n = greedy if n_parts is None else n_parts
    if n < greedy:
        raise ExportBlocked("n_parts too small for max_bytes", n_parts=n, minimum=greedy, max_bytes=max_bytes)
    if n > len(ids):
        raise ExportBlocked("more parts than tiles", n_parts=n, n_tiles=len(ids))
    plans = _parts_from_cuts(ids, values, _choose_cuts(values, budget, n, balance, greedy), base_bytes)
    over = [p.index for p in plans if p.est_bytes > max_bytes]
    if over:
        raise ExportBlocked("packing plan exceeds max_bytes", parts=over, max_bytes=max_bytes)
    return plans


def _zip_info(arcname: str, compress_type: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(arcname, date_time=ZIP_DATE_TIME)
    info.compress_type = compress_type
    info.external_attr = ZIP_FILE_MODE << 16
    info.create_system = ZIP_UNIX_SYSTEM
    return info


def _check_image_names(images: Sequence[tuple[str, Path]], out: Path) -> list[tuple[str, Path]]:
    names = [name for name, _ in images]
    if len(set(names)) != len(names):
        raise ExportBlocked("duplicate image in an upload ZIP", zip=out.name)
    for name, src in images:
        bad = "/" in name or "\\" in name or not name.endswith(IMAGE_EXT)
        if bad or unicodedata.normalize("NFC", name) != name:
            raise ExportBlocked("invalid image file name for an upload ZIP", zip=out.name, name=name)
        if not Path(src).is_file():
            raise ExportBlocked("tile file not found", zip=out.name, name=name, path=str(src))
    return sorted(images)


def write_upload_zip(
    out: Path, xml_bytes: bytes, images: Sequence[tuple[str, Path]], *, deflate_level: int
) -> Mapping[str, str]:
    """Write the ZIP atomically; returns {file_name: sha256} of the images as written."""
    target = Path(out)
    ordered = _check_image_names(images, target)
    shas: dict[str, str] = {}
    with atomic_path(target) as tmp, zipfile.ZipFile(tmp, "w", allowZip64=False) as zf:
        zf.writestr(_zip_info(ANNOTATIONS_NAME, zipfile.ZIP_DEFLATED), xml_bytes,
                    compress_type=zipfile.ZIP_DEFLATED, compresslevel=deflate_level)
        for name, src in ordered:
            data = Path(src).read_bytes()
            shas[name] = hashlib.sha256(data).hexdigest()
            zf.writestr(_zip_info(image_arcname(name), zipfile.ZIP_STORED), data, compress_type=zipfile.ZIP_STORED)
    return MappingProxyType(shas)
