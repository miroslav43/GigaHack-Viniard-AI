"""Files the web page shows for a finished demo job (web contract: src/Web/docs/design/2026-09-27-analiza-tif-spec.md).

preview.jpg (the tile) and veg_mask.png (tile_prep vegetation, coloured) at WEB_PX; canopies / rows / interrows /
waste GeoJSON in EPSG:4326 from the run's AnnSet (waste = the candidates the decision rule flags for review:
nothing is confirmed on a fresh tile); result.json with the corners, counts, areas and per-stage timings.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer

from vineyard.geo.raster import read_tile
from vineyard.geo.tiling import CRS_EPSG, TileRef, tile_ref
from vineyard.geo.vector_io import write_geojson_4326
from vineyard.pipeline.atomic import atomic_write_bytes, atomic_write_json
from vineyard.pipeline.context import RunPaths
from vineyard.pipeline.tile_cache import load_veg_mask
from vineyard.web.masks_export import downsample_mask

WEB_PX: Final = 1024
JPEG_QUALITY: Final = 85
LONLAT_DECIMALS: Final = 7
M2_DECIMALS: Final = 2
S_DECIMALS: Final = 2
WGS84: Final = 4326
PREVIEW: Final = "preview.jpg"
VEG_MASK: Final = "veg_mask.png"
RESULT: Final = "result.json"
WASTE_CANDIDATES: Final = "waste_candidates.parquet"

# output file -> (AnnSet parquet, properties kept)
LAYERS: Final[Mapping[str, tuple[str, tuple[str, ...]]]] = {
    "canopies": ("canopies.parquet", ("canopy_id", "vineyard_id", "row_id", "area_m2")),
    "rows": ("row_pieces.parquet", ("piece_id", "row_id", "vineyard_id", "row_structure", "length_m", "max_gap_m")),
    "interrows": ("interrow_pieces.parquet", ("piece_id", "interrow_id", "vineyard_id", "interrow_cover", "area_m2")),
}
WASTE_COLUMNS: Final = ("waste_id", "rank_score", "probe_p", "area_m2", "category")


@dataclass(frozen=True)
class MaskColour:
    rgb: tuple[int, int, int]
    alpha: float

    @classmethod
    def from_hex(cls, hex_colour: str, alpha: float) -> MaskColour:
        text = hex_colour.lstrip("#")
        if len(text) != 6 or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"expected #RRGGBB and alpha in [0, 1], got {hex_colour!r}, {alpha}")
        r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
        return cls((r, g, b), alpha)


def tile_corners(t: TileRef) -> list[list[float]]:
    """[[lon, lat] x 4] in MapLibre image order: TL, TR, BR, BL."""
    minx, miny, maxx, maxy = t.bounds
    to_wgs = Transformer.from_crs(CRS_EPSG, WGS84, always_xy=True)
    pts = ((minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny))
    return [[round(v, LONLAT_DECIMALS) for v in to_wgs.transform(x, y)] for x, y in pts]


def preview_jpeg(rgb: np.ndarray, px: int = WEB_PX) -> bytes:
    small = cv2.resize(rgb, (px, px), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def mask_rgba_png(mask: np.ndarray, colour: MaskColour, px: int = WEB_PX) -> bytes:
    """Coloured, semi-transparent where vegetation (any-pixel max-pool to px), fully transparent elsewhere."""
    small = downsample_mask(mask, px)
    bgra = np.zeros((px, px, 4), np.uint8)
    r, g, b = colour.rgb
    bgra[small] = (b, g, r, round(colour.alpha * 255))
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def _read(path: Path) -> gpd.GeoDataFrame | None:
    return gpd.read_parquet(path) if path.is_file() else None


def _subset(frame: gpd.GeoDataFrame | None, columns: Sequence[str]) -> gpd.GeoDataFrame:
    if frame is None:
        return gpd.GeoDataFrame({c: [] for c in columns}, geometry=gpd.GeoSeries([], crs=CRS_EPSG), crs=CRS_EPSG)
    keep = [c for c in columns if c in frame.columns]
    return gpd.GeoDataFrame(frame[keep], geometry=frame.geometry, crs=frame.crs)


def review_waste(candidates: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    """Candidates the decision rule puts in the review list, best rank first."""
    if candidates is None or "review" not in candidates.columns:
        return _subset(None, WASTE_COLUMNS)
    flagged = candidates[candidates["review"].fillna(False).astype(bool)]
    return _subset(flagged.sort_values("rank_score", ascending=False), WASTE_COLUMNS)


def _union_area(frame: gpd.GeoDataFrame) -> float:
    return round(float(shapely.union_all(frame.geometry.to_numpy()).area), M2_DECIMALS) if len(frame) else 0.0


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def stage_timings(paths: RunPaths) -> dict[str, float]:
    stages = _json(paths.metrics_dir / "timings.json").get("stages", {})
    return {name: round(float(v["wall_s"]), S_DECIMALS) for name, v in stages.items() if "wall_s" in v}


def model_version(run_doc: Mapping[str, Any], candidates: gpd.GeoDataFrame | None) -> str | None:
    """The waste stage's version string (it names the waste verifier); run.json's is fixed before that stage."""
    if candidates is not None and len(candidates) and "model_version" in candidates.columns:
        return str(candidates["model_version"].iloc[0])
    return run_doc.get("model_version")


def export_job(out_dir: Path, paths: RunPaths, tile_id: str, colour: MaskColour, total_s: float) -> Path:
    """Write every web file of the job and return result.json."""
    t = tile_ref(tile_id)
    atomic_write_bytes(out_dir / PREVIEW, preview_jpeg(read_tile(paths.tiles_dir / f"{tile_id}.tif")))
    atomic_write_bytes(out_dir / VEG_MASK, mask_rgba_png(load_veg_mask(paths.cache_dir, tile_id), colour))
    frames = {name: _subset(_read(paths.annset_dir / src), cols) for name, (src, cols) in LAYERS.items()}
    candidates = _read(paths.layers_dir / WASTE_CANDIDATES)
    frames["waste"] = review_waste(candidates)
    for name, frame in frames.items():
        write_geojson_4326(frame, out_dir / f"{name}.geojson", decimals=LONLAT_DECIMALS)
    run_doc = _json(paths.run_dir / "run.json")
    result = {
        "tile_id": tile_id,
        "corners": tile_corners(t),
        "counts": {name: len(frame) for name, frame in frames.items()},
        "canopy_area_m2": _union_area(frames["canopies"]),
        "interrow_area_m2": _union_area(frames["interrows"]),
        "row_length_m": round(float(frames["rows"]["length_m"].sum()) if len(frames["rows"]) else 0.0, M2_DECIMALS),
        "timings_s": stage_timings(paths),
        "total_s": round(total_s, S_DECIMALS),
        "model_version": model_version(run_doc, candidates),
        "waste_verify_level": _json(paths.metrics_dir / "waste.json").get("level"),
    }
    return atomic_write_json(out_dir / RESULT, result)
