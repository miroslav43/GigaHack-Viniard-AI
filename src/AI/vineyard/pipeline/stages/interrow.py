"""Stage `interrow` (block -> tile): global bands -> layers/interrows.parquet (route domain, web), then
per-tile pieces from the tile's own row pairs (perception.interrow_local, annotation rule 5.1) with cover
classification -> cache/interrow/<t>.parquet (contract `interrow_pieces`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.schemas import empty_layer
from vineyard.errors import StageError
from vineyard.geo.tiling import GSD_M, tile_box, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.corridor import clip_rows_to_tile, extend_censored_ends
from vineyard.perception.interrow import build_interrow_bands, classify_pieces, row_positions
from vineyard.perception.interrow_local import LocalPairOptions, tile_interrow_pieces
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult, TileTask, run_tile_stage
from vineyard.pipeline.stages.canopy import geometry_digest, tile_clip
from vineyard.pipeline.tile_cache import load_veg_mask, load_vis, tile_prep_key, veg_mask_path, vis_path
from vineyard.pipeline.tile_index import indexed_tile_ids, tile_path

if TYPE_CHECKING:
    from vineyard.config import AppConfig
    from vineyard.pipeline.context import RunContext

NAME: Final = "interrow"
VERSION: Final = "4"
BANDS_LAYER: Final = "interrows"
PIECES_LAYER: Final = "interrow_pieces"
ROWS_FILE: Final = "rows.parquet"
PAIRS_FILE: Final = "row_pairs.parquet"
FORBIDDEN_FILE: Final = "in_forbidden.parquet"
PROVENANCE: Final = ("source", "run_id", "model_version", "confidence", "qa_flags")
DIGEST_COLUMNS: Final = ("interrow_id", "row_left_id", "row_right_id", "n_notches", "qa_flags", "source")
ROW_DIGEST_COLUMNS: Final = ("row_id", "vineyard_id", "row_index")
BORDERLINE_CONFIDENCE: Final = 0.5
FULL_CONFIDENCE: Final = 1.0
CFG_KEYS: Final = ("interrow", "export.min_interrow_piece_m2", "export.cvat.notch_width_px", "export.min_row_piece_m",
                   "blocks.neighbour_max_m", "blocks.min_overlap_frac")
EVENT_NO_FORBIDDEN: Final = "interrow.no_forbidden_layer"
SEAM_CLOSE_M: Final = 0.01  # closing radius of the tile-clip union (float seams are ~1e-10 m wide)
_MITRE: Final = "mitre"

_log = get_logger("pipeline.stages.interrow")


# ------------------------------------------------------------------ global bands


def read_forbidden(path: Path, near: BaseGeometry | None = None) -> BaseGeometry | None:
    """Union of the in_forbidden polygons (those meeting `near` when given); None without the layer."""
    if not path.is_file():
        log_event(_log, EVENT_NO_FORBIDDEN, path=str(path))
        return None
    layer = read_layer(path, "in_forbidden")
    geoms = [g for g in layer.geometry if g is not None and (near is None or g.intersects(near))]
    return shapely.union_all(geoms) if geoms else None


def _forbidden(ctx: RunContext) -> BaseGeometry | None:
    return read_forbidden(ctx.paths.static_layers_dir / FORBIDDEN_FILE)


def _pair_flags(pairs: pd.DataFrame, bands: gpd.GeoDataFrame) -> list[str]:
    if "qa_flags" not in pairs.columns:
        return [""] * len(bands)
    flags: dict[frozenset[str], str] = {}
    for a, b, f in zip(pairs["row_a"], pairs["row_b"], pairs["qa_flags"], strict=True):
        flags[frozenset((str(a), str(b)))] = "" if f is None or pd.isna(f) else str(f)
    return [flags.get(frozenset((str(a), str(b))), "") for a, b in zip(bands["row_left_id"], bands["row_right_id"],
                                                                        strict=True)]


def bands_with_provenance(ctx: RunContext, bands: gpd.GeoDataFrame, pairs: pd.DataFrame) -> gpd.GeoDataFrame:
    """Contract provenance for the global `interrows` layer (qa_flags from the row pair, e.g. skip-one)."""
    return bands.assign(source=ctx.source.value, run_id=ctx.run_id, model_version=ctx.model_version(),
                        confidence=FULL_CONFIDENCE, qa_flags=_pair_flags(pairs, bands))


def clips_boundary(clips: list[BaseGeometry]) -> BaseGeometry | None:
    """Boundary of the union of tile clips, closed over the float seams between adjacent tiles (their shared
    edges differ by ~1e-10 m, which would leave every interior tile edge on the 'data boundary')."""
    if not clips:
        return None
    grown = shapely.union_all([c.buffer(SEAM_CLOSE_M, join_style=_MITRE) for c in clips])
    return grown.buffer(-SEAM_CLOSE_M, join_style=_MITRE).boundary


def data_boundary(ctx: RunContext) -> BaseGeometry | None:
    """Boundary of the union of this run's tile clips: where rows stop because the data stops."""
    selected = set(ctx.selected_tiles(indexed_tile_ids(ctx)))
    valid = read_layer(ctx.paths.tile_valid, "tile_valid")
    return clips_boundary([g.intersection(tile_box(tile_ref(t)))
                           for t, g in zip(valid["tile_id"].astype(str), valid.geometry, strict=True)
                           if t in selected and g is not None and not g.is_empty])


def build_bands_layer(ctx: RunContext) -> tuple[gpd.GeoDataFrame, Path]:
    """Read rows / row_pairs / in_forbidden, build the bands and write layers/interrows.parquet."""
    layers = ctx.paths.layers_dir
    for path in (layers / ROWS_FILE, layers / PAIRS_FILE):
        if not path.is_file():
            raise StageError("input layer missing; run blocks first", stage=NAME, path=str(path))
    ir = ctx.cfg.interrow
    rows = extend_censored_ends(read_layer(layers / ROWS_FILE, "rows"), data_boundary(ctx),
                                extend_m=ir.edge_extend_m, tol_m=ir.edge_tol_m)
    pairs = read_layer(layers / PAIRS_FILE, "row_pairs")
    notch_m = ctx.cfg.export.cvat.notch_width_px * GSD_M
    bands = build_interrow_bands(rows, pairs, _forbidden(ctx), ir, notch_m)
    out = bands_with_provenance(ctx, bands, pairs)
    path = layers / f"{BANDS_LAYER}.parquet"
    write_layer(out, BANDS_LAYER, path)
    return out, path


# ------------------------------------------------------------------ tile function (worker)


def _local_bands(bands: gpd.GeoDataFrame, tile_id: str) -> gpd.GeoDataFrame:
    if bands.empty:
        return bands
    return bands[bands.intersects(tile_box(tile_ref(tile_id)))].reset_index(drop=True)


def _piece_provenance(pieces: gpd.GeoDataFrame, bands: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Provenance and pair flags of each piece's global band (a local pair without a band: the run's
    provenance from any band, no pair flags); `cover_borderline` lowers the confidence."""
    by_id = bands.drop_duplicates("interrow_id").set_index("interrow_id")
    run = bands.iloc[0]
    src = [by_id.loc[i] if i in by_id.index else None for i in pieces["interrow_id"]]
    band_flags = [str(b["qa_flags"]) if b is not None else "" for b in src]
    flags = [";".join(f for f in (bf, str(own)) if f) for bf, own in zip(band_flags, pieces["qa_flags"], strict=True)]
    conf = [BORDERLINE_CONFIDENCE if own else FULL_CONFIDENCE for own in pieces["qa_flags"]]
    return pieces.assign(**{c: [str((b if b is not None else run)[c]) for b in src]
                            for c in ("source", "run_id", "model_version")}, confidence=conf, qa_flags=flags)


def local_rows(rows: gpd.GeoDataFrame, tile_id: str, min_piece_m: float) -> gpd.GeoDataFrame:
    """The tile's row pieces as exported (tile-box cut, >= min_piece_m), with row_id / vineyard_id / row_index."""
    box = tile_box(tile_ref(tile_id))
    pieces = clip_rows_to_tile(rows[rows.intersects(box).to_numpy()], tile_ref(tile_id), box, margin_m=0.0)
    return pieces[(pieces.geometry.length >= min_piece_m).to_numpy()].reset_index(drop=True)


def _write_empty(task: TileTask) -> Mapping[str, Any]:
    write_layer(empty_layer(PIECES_LAYER), PIECES_LAYER, task.outputs["pieces"])
    return {"n_pieces": 0}


def interrow_tile(task: TileTask) -> Mapping[str, Any]:
    """Worker: pair the tile's own rows, cut the bands, classify the cover, write the interrow_pieces parquet."""
    cfg: AppConfig = task.cfg  # type: ignore[assignment]
    bands = read_layer(task.inputs["interrows"], BANDS_LAYER)
    rows = read_layer(task.inputs["rows"], "rows")
    local = local_rows(rows, task.tile_id, cfg.export.min_row_piece_m)
    if bands.empty or len(local) < 2:
        return _write_empty(task)
    tile = tile_ref(task.tile_id)
    clip = tile_clip(task.inputs["tile_valid"], tile)
    forbidden = read_forbidden(task.inputs["forbidden"], tile_box(tile))
    pieces = tile_interrow_pieces(local, bands, tile, clip, forbidden, row_positions(rows),
                                  LocalPairOptions.from_config(cfg))
    if pieces.empty:
        return _write_empty(task)
    cache = task.inputs["veg"].parents[1]
    classified = classify_pieces(pieces, load_veg_mask(cache, task.tile_id), load_vis(cache, task.tile_id), tile,
                                 cfg.interrow)
    out = _piece_provenance(classified, bands)
    write_layer(out, PIECES_LAYER, task.outputs["pieces"])
    return {"n_pieces": len(out)}


# ------------------------------------------------------------------ stage (main process)


def _make_task(ctx: RunContext, bands_path: Path, tile_id: str) -> TileTask:
    cache = ctx.paths.cache_dir
    inputs = {"interrows": bands_path, "rows": ctx.paths.layers_dir / ROWS_FILE, "veg": veg_mask_path(cache, tile_id),
              "vis": vis_path(cache, tile_id), "tile_valid": ctx.paths.tile_valid,
              "forbidden": ctx.paths.static_layers_dir / FORBIDDEN_FILE}
    return TileTask(tile_id=tile_id, tif_path=tile_path(ctx, tile_id), key="", cfg=ctx.cfg, inputs=inputs,
                    outputs={"pieces": ctx.paths.tile_cache(NAME, tile_id, "parquet")})


def _input_keys(ctx: RunContext, bands: gpd.GeoDataFrame, rows: gpd.GeoDataFrame, tile_id: str) -> list[str]:
    local = _local_bands(bands, tile_id)
    pieces = local_rows(rows, tile_id, ctx.cfg.export.min_row_piece_m)
    return [f"tile_prep:{tile_prep_key(ctx.paths.cache_dir, tile_id)}",
            f"bands:{geometry_digest(local, DIGEST_COLUMNS)}", f"rows:{geometry_digest(pieces, ROW_DIGEST_COLUMNS)}"]


def run(ctx: RunContext) -> StageResult:
    if not ctx.paths.tile_valid.is_file():
        raise StageError("tile_valid layer missing; run tile_prep first", stage=NAME, path=str(ctx.paths.tile_valid))
    bands, bands_path = build_bands_layer(ctx)
    rows = read_layer(ctx.paths.layers_dir / ROWS_FILE, "rows")
    result = run_tile_stage(ctx, STAGE, make_task=partial(_make_task, ctx, bands_path), tile_fn=interrow_tile,
                            input_keys=partial(_input_keys, ctx, bands, rows))
    return replace(result, outputs=(bands_path, *result.outputs))


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="block", cfg_keys=CFG_KEYS, requires=("tile_prep", "blocks"), run=run,
    description="interrow bands between neighbour rows (global); per-tile pieces from local row pairs + cover",
)
