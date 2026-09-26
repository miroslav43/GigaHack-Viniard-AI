"""Stage `canopy` (tile): rows ∩ tile (+rows margin) and the canopy mask -> cache/canopy/<t>.parquet + .json
(+ <t>.evidence.parquet: plant pieces below the canopy minimum, read by row_attrs for the gaps).

The canopy mask is ExG from the tile RGB (canopy.mask_method = exg) or the tile_prep a* veg mask (veg).
The cache key covers the tile_prep key (which chains the tif key), the NN weights and a sha1 of the local
row pieces (WKB + ids), so a row change re-runs only the tiles it touches. Provenance is inherited from the
rows layer; assemble stamps the final run provenance.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import FusionVariant, Source
from vineyard.contracts.schemas import empty_layer
from vineyard.errors import StageError
from vineyard.geo.raster import read_tile
from vineyard.geo.tiling import TILE_PX, TileRef, tile_box, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.nn.fusion import fuse_masks
from vineyard.nn.probs import load_canopy_prob, prob_png_path, upsample_prob
from vineyard.perception.canopy import EVIDENCE_COLUMNS, CanopyOptions, extract_canopies
from vineyard.perception.corridor import clip_rows_to_tile
from vineyard.perception.trees import tree_mask
from vineyard.perception.vegmask import erode_mask, exg_mask
from vineyard.pipeline.atomic import atomic_path, atomic_write_json
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult, TileTask, run_tile_stage
from vineyard.pipeline.tile_cache import (
    load_valid_mask,
    load_veg_mask,
    tile_prep_key,
    valid_mask_path,
    veg_mask_path,
)
from vineyard.pipeline.tile_index import tile_path

if TYPE_CHECKING:
    from vineyard.config import AppConfig
    from vineyard.pipeline.context import RunContext, RunPaths

NAME: Final = "canopy"
VERSION: Final = "3"
LAYER: Final = "canopies"
EVIDENCE_EXT: Final = "evidence.parquet"
MASK_EXG: Final = "exg"
ROWS_FILE: Final = "rows.parquet"
PROB_CHANNEL: Final = "canopy_prob"
PROVENANCE: Final = ("source", "run_id", "model_version")
DIGEST_COLUMNS: Final = ("row_id", "vineyard_id", "row_index", "qa_flags", "interp_tile_ids")
TILE_VALID_LAYER: Final = "tile_valid"
CFG_KEYS: Final = ("canopy", "nn.enabled", "nn.version", "nn.weights_sha256", "nn.fusion", "nn.prob_threshold",
                   "orchard.tree_blob_area_m2", "orchard.tree_blob_axis_ratio_max", "nodata.veg_erode_px")


# ------------------------------------------------------------------ shared helpers (also used by interrow)


def geometry_digest(frame: gpd.GeoDataFrame, columns: Sequence[str]) -> str:
    """sha1 over the WKB and the given columns of every row, in frame order."""
    h = hashlib.sha1()
    cols = [c for c in columns if c in frame.columns]
    for i, geom in enumerate(frame.geometry):
        h.update(shapely.to_wkb(geom, hex=False, output_dimension=2))
        h.update("\x1f".join(str(frame[c].iloc[i]) for c in cols).encode("utf-8"))
    return h.hexdigest()


def tile_clip(tile_valid_path: Path, tile: TileRef) -> BaseGeometry:
    """clip = tile_box ∩ tile_valid (contract §1.6); fails loudly when tile_valid lacks the tile."""
    valid = read_layer(tile_valid_path, TILE_VALID_LAYER)
    match = valid[valid["tile_id"] == tile.tile_id]
    if match.empty:
        raise StageError("tile missing from tile_valid", stage=NAME, tile_id=tile.tile_id, path=str(tile_valid_path))
    geom = match.geometry.iloc[0]
    box = tile_box(tile)
    if geom is None or geom.is_empty:
        return shapely.Polygon()
    return geom.intersection(box)


def local_pieces(rows: gpd.GeoDataFrame, tile: TileRef, margin_m: float) -> gpd.GeoDataFrame:
    """Row pieces of the tile grown by margin_m (clip = tile box: the valid clip is applied later)."""
    near = rows[rows.intersects(tile_box(tile).buffer(margin_m, join_style="mitre"))]
    return clip_rows_to_tile(near, tile, tile_box(tile), margin_m=margin_m)


def with_provenance(frame: gpd.GeoDataFrame, source: gpd.GeoDataFrame, index: Sequence[int]) -> gpd.GeoDataFrame:
    """New frame with source / run_id / model_version copied from `source` rows `index`."""
    values = {c: [str(source[c].iloc[i]) for i in index] if c in source.columns else [_default(c)] * len(index)
              for c in PROVENANCE}
    return frame.assign(**values)


def _default(column: str) -> str:
    return Source.MODEL.value if column == "source" else f"{NAME}@{VERSION}"


def evidence_path(paths: RunPaths, tile_id: str) -> Path:
    """work/cache/canopy/<tile>.evidence.parquet (internal: gap evidence below the canopy minimum)."""
    return paths.tile_cache(NAME, tile_id, EVIDENCE_EXT)


def empty_evidence(crs: object) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({c: [] for c in EVIDENCE_COLUMNS}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def write_evidence(frame: gpd.GeoDataFrame, path: Path) -> Path:
    """Internal evidence layer (no contract schema): GeoParquet written atomically."""
    with atomic_path(Path(path)) as tmp:
        frame.to_parquet(tmp, index=False)
    return Path(path)


def read_evidence(path: Path) -> gpd.GeoDataFrame:
    """The evidence layer of one tile; fails loudly when the canopy stage did not write it."""
    if not Path(path).is_file():
        raise StageError("canopy gap evidence missing; re-run canopy", stage=NAME, path=str(path))
    return gpd.read_parquet(path)


# ------------------------------------------------------------------ tile function (worker)


def canopy_source_mask(task: TileTask, cfg: AppConfig, veg: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """ExG > threshold on the tile RGB ∧ valid eroded like the veg mask (exg), or the a* veg mask (veg)."""
    can = cfg.canopy
    if can.mask_method != MASK_EXG:
        return veg
    return exg_mask(read_tile(task.tif_path), erode_mask(valid, cfg.nodata.veg_erode_px),
                    blur_sigma_px=can.exg_blur_sigma_px, threshold=can.exg_threshold)


def _fused_mask(task: TileTask, base: np.ndarray, cfg: AppConfig) -> tuple[np.ndarray, str, bool]:
    prob_path = task.inputs.get("prob")
    prob = upsample_prob(load_canopy_prob(prob_path), TILE_PX) if prob_path is not None else None
    fused = fuse_masks(base, prob, None, FusionVariant(cfg.nn.fusion), cfg.nn.prob_threshold)
    return np.asarray(fused.mask), fused.variant_used.value, fused.fell_back


def _empty_outputs(task: TileTask, stats: Mapping[str, Any]) -> Mapping[str, Any]:
    empty = empty_layer(LAYER)
    write_layer(empty, LAYER, task.outputs["canopy"])
    write_evidence(empty_evidence(empty.crs), task.outputs["evidence"])
    atomic_write_json(task.outputs["stats"], dict(stats))
    return {"n_canopies": 0, "n_pieces": 0}


def canopy_tile(task: TileTask) -> Mapping[str, Any]:
    """Worker: extract the canopies of one tile and write the parquet + stats JSON."""
    cfg: AppConfig = task.cfg  # type: ignore[assignment]
    tile = tile_ref(task.tile_id)
    rows = read_layer(task.inputs["rows"], "rows")
    pieces = local_pieces(rows, tile, cfg.canopy.rows_margin_m)
    if pieces.empty:
        return _empty_outputs(task, {"tile_id": task.tile_id, "n_pieces": 0, "n_canopies": 0})
    clip = tile_clip(task.inputs["tile_valid"], tile)
    veg = load_veg_mask(task.inputs["veg"].parents[1], task.tile_id)
    valid = load_valid_mask(task.inputs["valid"].parents[1], task.tile_id)
    mask, variant, fell_back = _fused_mask(task, canopy_source_mask(task, cfg, veg, valid), cfg)
    tree = tree_mask(veg, pieces, tile, cfg.canopy, cfg.orchard) if cfg.canopy.tree_filter_enabled else None
    res = extract_canopies(mask, pieces, tile, clip, CanopyOptions.from_config(cfg.canopy), valid=valid, tree=tree)
    src_index = [int(np.flatnonzero(pieces["row_id"].to_numpy() == r)[0]) for r in res.canopies["row_id"]]
    out = with_provenance(res.canopies, pieces, src_index)
    write_layer(out, LAYER, task.outputs["canopy"])
    write_evidence(res.evidence, task.outputs["evidence"])
    stats = {"tile_id": task.tile_id, "n_canopies": len(out), "n_evidence": len(res.evidence),
             "mask_method": cfg.canopy.mask_method, "fusion_variant": variant, "fusion_fell_back": fell_back,
             "tree_filter": tree is not None, **vars(res.stats)}
    atomic_write_json(task.outputs["stats"], stats)
    return {"n_canopies": len(out), "n_pieces": res.stats.n_pieces, "corridor_veg_frac": res.stats.corridor_veg_frac}


# ------------------------------------------------------------------ stage (main process)


def _prob_path(ctx: RunContext, tile_id: str) -> Path | None:
    nn = ctx.cfg.nn
    return prob_png_path(ctx.paths.cache_dir, nn.version, PROB_CHANNEL, tile_id) if nn.enabled else None


def _make_task(ctx: RunContext, tile_id: str) -> TileTask:
    cache = ctx.paths.cache_dir
    inputs = {"rows": ctx.paths.layers_dir / ROWS_FILE, "veg": veg_mask_path(cache, tile_id),
              "valid": valid_mask_path(cache, tile_id), "tile_valid": ctx.paths.tile_valid}
    prob = _prob_path(ctx, tile_id)
    inputs = {**inputs, "prob": prob} if prob is not None else inputs
    outputs = {"canopy": ctx.paths.tile_cache(NAME, tile_id, "parquet"), "evidence": evidence_path(ctx.paths, tile_id),
               "stats": ctx.paths.tile_cache(NAME, tile_id, "json")}
    return TileTask(tile_id=tile_id, tif_path=tile_path(ctx, tile_id), key="", cfg=ctx.cfg, inputs=inputs,
                    outputs=outputs)


def _input_keys(ctx: RunContext, rows: gpd.GeoDataFrame, tile_id: str) -> list[str]:
    pieces = local_pieces(rows, tile_ref(tile_id), ctx.cfg.canopy.rows_margin_m)
    keys = [f"tile_prep:{tile_prep_key(ctx.paths.cache_dir, tile_id)}",
            f"rows:{geometry_digest(pieces, DIGEST_COLUMNS)}"]
    nn = ctx.cfg.nn
    return keys + ([f"nn:{nn.version}:{nn.weights_sha256}"] if nn.enabled else [])


def run(ctx: RunContext) -> StageResult:
    rows_path = ctx.paths.layers_dir / ROWS_FILE
    if not rows_path.is_file():
        raise StageError("rows layer missing; run blocks first", stage=NAME, path=str(rows_path))
    if not ctx.paths.tile_valid.is_file():
        raise StageError("tile_valid layer missing; run tile_prep first", stage=NAME, path=str(ctx.paths.tile_valid))
    rows = read_layer(rows_path, "rows")
    return run_tile_stage(ctx, STAGE, make_task=partial(_make_task, ctx), tile_fn=canopy_tile,
                          input_keys=partial(_input_keys, ctx, rows))


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="tile", cfg_keys=CFG_KEYS, requires=("tile_prep", "blocks"), run=run,
    description="canopy polygons per tile: ExG (or a* / NN-fused) mask ∩ refined row corridors -> raw contours",
)
