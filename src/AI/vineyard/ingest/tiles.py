"""Tile ingest: stream the challenge tiles out of their ZIPs into work/tiles (sha256 while copying),
verify names, count and GeoTIFF tags against the grid, and build the tile_index layer (01 §3.6).

A copy is skipped when "<tif>.key" equals sha1(name|CRC|size) of its ZIP member; "<tif>.sha256"
holds its digest so a cached ingest never re-hashes 385 MB.
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import rasterio

from vineyard.contracts.ids import tile_id_from_file_name
from vineyard.contracts.schemas import coerce_layer
from vineyard.errors import IngestError, SchemaError
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TileRef, existing_tile_ids, tile_box, tile_ref_from_tags
from vineyard.pipeline.atomic import atomic_path, atomic_write_text
from vineyard.pipeline.cache import is_fresh, remove_key, write_key

COPY_CHUNK_BYTES: Final = 1 << 20
SHA_SUFFIX: Final = ".sha256"
AREA_TAG: Final = "AREA_OR_POINT"
AREA_VALUE: Final = "Area"
RGB_BANDS: Final = 3
UINT8: Final = "uint8"
GRID_DECIMALS: Final = 6
TILE_INDEX_LAYER: Final = "tile_index"


@dataclass(frozen=True)
class ZipMember:
    zip_path: Path
    name: str
    tile_id: str
    crc: int
    file_size: int

    @property
    def key(self) -> str:
        return hashlib.sha1(f"{self.name}|{self.crc}|{self.file_size}".encode()).hexdigest()


@dataclass(frozen=True)
class IngestedTile:
    member: ZipMember
    path: Path
    sha256: str
    extracted: bool


@dataclass(frozen=True)
class IngestSummary:
    n_tiles: int
    n_extracted: int
    n_skipped: int
    total_bytes: int
    duration_s: float


# ------------------------------------------------------------------ scanning and name checks


def _member_tile_id(zip_path: Path, name: str) -> str:
    if "/" in name or "\\" in name:
        raise IngestError(f"unexpected path in tile ZIP: {name}", zip=zip_path.name, member=name)
    try:
        tile_id = tile_id_from_file_name(name)
    except SchemaError as exc:
        raise IngestError(f"not a tile file name: {name}", zip=zip_path.name, member=name) from exc
    if tile_id not in existing_tile_ids():
        raise IngestError(f"tile not in contracts/tile_grid.txt: {name}", zip=zip_path.name, member=name)
    return tile_id


def _scan_one(zip_path: Path) -> list[ZipMember]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
    except zipfile.BadZipFile as exc:
        raise IngestError(f"not a valid ZIP: {exc}", zip=str(zip_path)) from exc
    return [ZipMember(Path(zip_path), i.filename, _member_tile_id(Path(zip_path), i.filename), i.CRC, i.file_size)
            for i in infos]


def scan_zips(zip_paths: Iterable[Path]) -> tuple[ZipMember, ...]:
    """Every tile member of the ZIPs (sorted by tile_id); bad names and duplicates are IngestErrors."""
    members = [m for zp in sorted(Path(p) for p in zip_paths) for m in _scan_one(zp)]
    seen: dict[str, ZipMember] = {}
    for member in members:
        first = seen.setdefault(member.tile_id, member)
        if first is not member:
            raise IngestError(f"duplicate tile across ZIPs: {member.tile_id}", tile_id=member.tile_id,
                              zips=f"{first.zip_path.name}, {member.zip_path.name}")
    return tuple(sorted(members, key=lambda m: m.tile_id))


def _check_count(members: Sequence[ZipMember], expected_tiles: int) -> None:
    if len(members) != expected_tiles:
        raise IngestError(f"found {len(members)} tiles, expected {expected_tiles}", n_found=len(members),
                          expected=expected_tiles)


# ------------------------------------------------------------------ extraction


def _sha_path(dest: Path) -> Path:
    return dest.with_name(dest.name + SHA_SUFFIX)


def _cached(member: ZipMember, dest: Path) -> IngestedTile | None:
    sha_file = _sha_path(dest)
    if is_fresh(dest, member.key) and sha_file.is_file() and dest.stat().st_size == member.file_size:
        return IngestedTile(member, dest, sha_file.read_text(encoding="utf-8").strip(), extracted=False)
    return None


def _copy_member(zf: zipfile.ZipFile, member: ZipMember, dest: Path) -> str:
    digest = hashlib.sha256()
    with zf.open(member.name) as src, atomic_path(dest) as tmp, open(tmp, "wb") as out:
        while chunk := src.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def _extract_zip(zip_path: Path, members: Sequence[ZipMember], tiles_dir: Path) -> list[IngestedTile]:
    done: list[IngestedTile] = []
    todo: list[ZipMember] = []
    for member in members:
        cached = _cached(member, tiles_dir / member.name)
        done.extend([cached] if cached else [])
        todo.extend([] if cached else [member])
    if not todo:
        return done
    with zipfile.ZipFile(zip_path) as zf:
        for member in todo:
            dest = tiles_dir / member.name
            remove_key(dest)
            try:
                sha = _copy_member(zf, member, dest)
            except zipfile.BadZipFile as exc:  # CRC mismatch while streaming
                raise IngestError(f"corrupt ZIP member: {exc}", zip=zip_path.name, tile_id=member.tile_id) from exc
            atomic_write_text(_sha_path(dest), sha + "\n")
            write_key(dest, member.key)
            done.append(IngestedTile(member, dest, sha, extracted=True))
    return done


def extract_tiles(members: Sequence[ZipMember], tiles_dir: Path) -> tuple[IngestedTile, ...]:
    """Copy every member to tiles_dir/<name> (skipping fresh copies); sorted by tile_id."""
    Path(tiles_dir).mkdir(parents=True, exist_ok=True)
    zips = sorted({m.zip_path for m in members})
    tiles = [t for zp in zips for t in _extract_zip(zp, [m for m in members if m.zip_path == zp], Path(tiles_dir))]
    return tuple(sorted(tiles, key=lambda t: t.member.tile_id))


# ------------------------------------------------------------------ raster verification


def verify_tile_raster(path: Path, *, tile_px: int, tol_m: float) -> TileRef:
    """Contract tags: EPSG:32635, grid transform, tile_px², 3 x uint8, AREA_OR_POINT=Area."""
    ref = tile_ref_from_tags(path, tol_m=tol_m)
    with rasterio.open(path) as src:
        size, count, dtypes, area = (src.width, src.height), src.count, src.dtypes, src.tags().get(AREA_TAG)
    if size != (tile_px, tile_px):
        raise IngestError(f"{ref.tile_id}: size {size[0]}x{size[1]} != {tile_px}x{tile_px}", tile_id=ref.tile_id)
    if count != RGB_BANDS or any(dt != UINT8 for dt in dtypes):
        raise IngestError(f"{ref.tile_id}: expected {RGB_BANDS} uint8 bands, found {count} {dtypes}",
                          tile_id=ref.tile_id)
    if area != AREA_VALUE:
        raise IngestError(f"{ref.tile_id}: {AREA_TAG}={area!r}, expected {AREA_VALUE!r}", tile_id=ref.tile_id)
    return ref


# ------------------------------------------------------------------ tile_index


def _index_row(tile: IngestedTile, ref: TileRef, tile_px: int) -> dict[str, object]:
    minx, miny, maxx, maxy = ref.bounds
    return {
        "tile_id": ref.tile_id, "file_name": tile.member.name, "grid_row": ref.grid_row, "grid_col": ref.grid_col,
        "x0": ref.x0, "y0": ref.y0, "x1": round(maxx, GRID_DECIMALS), "y1": round(miny, GRID_DECIMALS),
        "gsd_m": GSD_M, "width_px": tile_px, "height_px": tile_px, "src_zip": tile.member.zip_path.name,
        "path": str(tile.path.resolve()), "sha256": tile.sha256, "file_size": tile.member.file_size,
        "nodata_frac": np.nan, "valid_area_m2": np.nan,
    }


def build_tile_index(tiles: Sequence[IngestedTile], refs: Sequence[TileRef], tile_px: int) -> gpd.GeoDataFrame:
    """tile_index frame (contract columns; nodata_frac / valid_area_m2 are NaN until tile_prep)."""
    rows = [_index_row(t, r, tile_px) for t, r in zip(tiles, refs, strict=True)]
    geoms = [tile_box(r) for r in refs]
    return coerce_layer(gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS_EPSG), TILE_INDEX_LAYER)


def ingest_tiles(zip_paths: Sequence[Path], tiles_dir: Path, *, expected_tiles: int, tile_px: int,
                 tol_m: float) -> tuple[gpd.GeoDataFrame, IngestSummary]:
    """Scan, count-check, extract and verify; returns the tile_index frame and a summary."""
    t0 = time.perf_counter()
    if not zip_paths:
        raise IngestError("no tile ZIPs found", tiles_dir=str(tiles_dir))
    members = scan_zips(zip_paths)
    _check_count(members, expected_tiles)
    tiles = extract_tiles(members, tiles_dir)
    refs = [verify_tile_raster(t.path, tile_px=tile_px, tol_m=tol_m) for t in tiles]
    n_new = sum(t.extracted for t in tiles)
    summary = IngestSummary(n_tiles=len(tiles), n_extracted=n_new, n_skipped=len(tiles) - n_new,
                            total_bytes=sum(t.member.file_size for t in tiles),
                            duration_s=time.perf_counter() - t0)
    return build_tile_index(tiles, refs, tile_px), summary

