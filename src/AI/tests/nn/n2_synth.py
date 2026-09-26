"""Synthetic data for the N2 NN tests (train / validate / infer / ablation / stage): one row across a real
tile position, rectangular canopies on it, and helpers to write the tile_prep cache and a reference AnnSet."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString

from vineyard.geo.raster import write_mask_png
from vineyard.geo.tiling import CRS_EPSG, TILE_PX, px_to_utm, tile_box, tile_ref
from vineyard.geo.vector_io import write_layer
from vineyard.pipeline.atomic import atomic_write_text

TILE_ID: Final = "siret3_r021_c012"
OTHER_TILE: Final = "siret3_r006_c004"
ROW_V_PX: Final = 1024.0
CANOPY_H_PX: Final = 16
CANOPY_W_PX: Final = 60
CANOPY_U0: Final = (100, 300, 500, 700, 900)
GRASS_BOX: Final = (1400, 400, 1500, 480)  # u0, v0, u1, v1: vegetation far from the row
VEG_RGB: Final = (40, 140, 40)
SOIL_RGB: Final = (150, 120, 90)


@dataclass(frozen=True, eq=False)
class SynthTile:
    tile_id: str
    veg: np.ndarray  # bool 2048², canopies on the row (+ grass when requested)
    canopy: np.ndarray  # bool 2048², canopies only
    valid: np.ndarray
    pieces: gpd.GeoDataFrame
    rgb: np.ndarray  # uint8 2048² x 3


def canopy_mask() -> np.ndarray:
    mask = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    v0 = int(ROW_V_PX) - CANOPY_H_PX // 2
    for u0 in CANOPY_U0:
        mask[v0 : v0 + CANOPY_H_PX, u0 : u0 + CANOPY_W_PX] = True
    return mask


def grass_mask() -> np.ndarray:
    mask = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    u0, v0, u1, v1 = GRASS_BOX
    mask[v0:v1, u0:u1] = True
    return mask


def row_pieces(tile_id: str = TILE_ID) -> gpd.GeoDataFrame:
    t = tile_ref(tile_id)
    uv = np.array([[0.0, ROW_V_PX], [float(TILE_PX), ROW_V_PX]])
    line = LineString(px_to_utm(t, uv))
    return gpd.GeoDataFrame({"row_id": ["V01-R01"], "vineyard_id": ["V01"], "tile_id": [tile_id],
                             "piece_id": [f"V01-R01@{tile_id}"]}, geometry=[line], crs=CRS_EPSG)


def rgb_from(veg: np.ndarray) -> np.ndarray:
    rgb = np.empty((*veg.shape, 3), dtype=np.uint8)
    rgb[...] = SOIL_RGB
    rgb[veg] = VEG_RGB
    return rgb


def synth_tile(tile_id: str = TILE_ID, *, with_grass: bool = True) -> SynthTile:
    canopy = canopy_mask()
    veg = canopy | grass_mask() if with_grass else canopy
    valid = np.ones_like(veg)
    return SynthTile(tile_id, veg, canopy, valid, row_pieces(tile_id), rgb_from(veg))


def write_tile_prep_cache(cache_dir: Path, tile: SynthTile) -> None:
    write_mask_png(cache_dir / "veg" / f"{tile.tile_id}.png", tile.veg)
    write_mask_png(cache_dir / "valid" / f"{tile.tile_id}.png", tile.valid)
    stats = cache_dir / "stats" / f"{tile.tile_id}.json"
    atomic_write_text(stats, "{}")
    atomic_write_text(stats.with_name(stats.name + ".key"), f"prepkey-{tile.tile_id}\n")


def write_tile_valid(path: Path, tile_ids: tuple[str, ...]) -> Path:
    frame = gpd.GeoDataFrame({"tile_id": list(tile_ids), "valid_frac": [1.0] * len(tile_ids)},
                             geometry=[tile_box(tile_ref(t)) for t in tile_ids], crs=CRS_EPSG)
    return write_layer(frame, "tile_valid", path)


def reference_canopies(tile_id: str, polys: list[shapely.Polygon]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"tile_id": [tile_id] * len(polys)}, geometry=polys, crs=CRS_EPSG)


def _record(cls: type, **known: object) -> object:
    """Dataclass instance with the given fields; every other (count / timing) field is 0."""
    return cls(**{f.name: known.get(f.name, 0) for f in fields(cls)})


def write_tiny_store(root: Path, *, patch_px: int = 32, n_per_tile: int = 3,
                     tiles: tuple[str, ...] = ("siret3_r030_c001", "siret3_r030_c002")) -> Path:
    """A patch store in the nn.patch_store format: green squares labelled 1, soil 0, one ignored row."""
    from vineyard.nn.patch_store import (
        IMG_NAME,
        LBL_NAME,
        PatchRecord,
        StoreManifest,
        TileEntry,
        write_manifest,
    )

    root.mkdir(parents=True, exist_ok=True)
    tile_entries, records = [], []
    for shard, tile_id in enumerate(tiles):
        lbl = np.zeros((n_per_tile, patch_px, patch_px), dtype=np.uint8)
        lbl[:, 8:20, 4 + shard : 16 + shard] = 1
        lbl[:, -1, :] = 255
        img = np.empty((n_per_tile, patch_px, patch_px, 3), dtype=np.uint8)
        img[...] = SOIL_RGB
        img[lbl == 1] = VEG_RGB
        np.save(root / IMG_NAME.format(shard), img)
        np.save(root / LBL_NAME.format(shard), lbl)
        n_pos, n_ign = int((lbl[0] == 1).sum()), int((lbl[0] == 255).sum())
        tile_entries.append(_record(TileEntry, tile_id=tile_id, kind="positive", shard=shard, n_patches=n_per_tile,
                                    n_pos=n_pos * n_per_tile, n_ignore=n_ign * n_per_tile))
        records += [_record(PatchRecord, tile_id=tile_id, shard=shard, index=i, n_pos=n_pos,
                            n_neg=patch_px * patch_px - n_pos - n_ign, n_ignore=n_ign) for i in range(n_per_tile)]
    write_manifest(root, StoreManifest(key="tiny", format=1, run_id="synthetic", created_at="now", patch_px=patch_px,
                                       gsd_m=0.05, holdout=(TILE_ID, OTHER_TILE), tiles=tuple(tile_entries),
                                       patches=tuple(records), selection={}, params={}, failed=(), build_s=0.0))
    return root
