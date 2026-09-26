"""Validation of canopy probabilities with the official vector metric (design 03 N5, A§4.6), torch-free.

A probability raster goes the same way as in the pipeline: uint8 quantisation (cache PNG), INTER_LINEAR
upsample to the tile, fusion variant (nn.fusion), canopy extraction on the row corridors (the export
path), then eval's 0.6·IoU + 0.4·F1@0.5 against the reference canopies. Early stopping uses variant B
with the reference axes, so the score isolates the quality of the mask itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from vineyard.annset.io import ANNSET_DIRNAME, read_annset, resolve_run_dir
from vineyard.contracts.enums import FusionVariant
from vineyard.contracts.ids import file_name_from_tile_id
from vineyard.errors import StageError
from vineyard.eval.matching import clean_polygonal
from vineyard.eval.metrics import CanopyScore, canopy_metrics
from vineyard.geo.ops import make_valid_polygonal
from vineyard.geo.raster import mask_to_polygons, read_tile
from vineyard.geo.tiling import TILE_PX, tile_box, tile_ref
from vineyard.geo.vector_io import read_layer
from vineyard.nn.fusion import fuse_masks
from vineyard.nn.probs import PROB_PX, PROB_SCALE, upsample_prob
from vineyard.nn.pseudolabels import downsample_rgb
from vineyard.perception.canopy import CanopyOptions, extract_canopies
from vineyard.perception.trees import tree_mask
from vineyard.pipeline.context import make_run_paths
from vineyard.pipeline.stages.canopy import local_pieces, tile_clip
from vineyard.pipeline.tile_cache import load_valid_mask, load_veg_mask

if TYPE_CHECKING:
    from vineyard.config import AppConfig

TILE_COLUMN: Final = "tile_id"
REF_PIECES_LAYER: Final = "row_pieces"
REF_CANOPY_LAYER: Final = "canopies"
LAYERS_DIRNAME: Final = "layers"
ROWS_FILE: Final = "rows.parquet"
ROWS_LAYER: Final = "rows"
VAL_RUN_ID: Final = "nn-validate"  # only used to derive the work-dir layout; nothing is written there

__all__ = [
    "TileScore", "ValResult", "ValTile", "input_factor", "load_model_rows", "load_reference",
    "load_val_tiles", "make_val_tile", "model_axes", "network_input", "predict_canopies", "prob_to_tile",
    "reference_val_tiles", "score_polygons", "score_tile", "tile_reader", "val_variant", "validate_probs",
    "variant_mask", "with_pieces",
]


@dataclass(frozen=True, eq=False)
class ValTile:
    """One validation tile: network input, tile masks, corridors (row pieces) and reference canopies (UTM)."""

    tile_id: str
    net_input: np.ndarray  # uint8 (PROB_PX, PROB_PX, 3), nodata = 0
    valid: np.ndarray  # bool (TILE_PX, TILE_PX)
    veg: np.ndarray  # bool (TILE_PX, TILE_PX)
    pieces: gpd.GeoDataFrame
    clip: BaseGeometry
    ref: tuple[BaseGeometry, ...]


@dataclass(frozen=True)
class TileScore:
    tile_id: str
    variant: str
    iou: float
    f1: float
    score: float
    n_pred: int
    n_ref: int


@dataclass(frozen=True)
class ValResult:
    variant: str
    scores: tuple[TileScore, ...]

    def _mean(self, attr: str) -> float:
        return float(np.mean([getattr(s, attr) for s in self.scores])) if self.scores else 0.0

    @property
    def mean_score(self) -> float:
        return self._mean("score")

    @property
    def mean_iou(self) -> float:
        return self._mean("iou")

    @property
    def mean_f1(self) -> float:
        return self._mean("f1")

    def to_dict(self) -> dict[str, Any]:
        tiles = {s.tile_id: {"iou": s.iou, "f1": s.f1, "score": s.score, "n_pred": s.n_pred, "n_ref": s.n_ref}
                 for s in self.scores}
        return {"variant": self.variant, "tiles": tiles,
                "mean": {"iou": self.mean_iou, "f1": self.mean_f1, "score": self.mean_score}}


# ------------------------------------------------------------------ network input and probabilities


def input_factor(cfg: AppConfig) -> int:
    """Tile px per NN px (0.05 / 0.025 = 2); the NN raster must be PROB_PX² (contract §6)."""
    factor = round(cfg.nn.in_gsd_m / cfg.grid.gsd_m)
    if factor < 1 or cfg.grid.tile_px != factor * PROB_PX:
        raise StageError("nn.in_gsd_m does not map the tile onto the contract raster", in_gsd_m=cfg.nn.in_gsd_m,
                         gsd_m=cfg.grid.gsd_m, tile_px=cfg.grid.tile_px, prob_px=PROB_PX)
    return factor


def network_input(rgb: np.ndarray, valid: np.ndarray, factor: int) -> np.ndarray:
    """uint8 RGB at 1/factor: nodata set to 0, then INTER_AREA (contract §6), exactly as the patch store."""
    return downsample_rgb(rgb, valid, factor)


def prob_to_tile(prob: np.ndarray) -> np.ndarray:
    """PROB_PX² probability (float [0, 1] or uint8) -> float32 TILE_PX², quantised like the cache PNG."""
    if prob.shape != (PROB_PX, PROB_PX):
        raise ValueError(f"probability raster must have shape {(PROB_PX, PROB_PX)}, got {prob.shape}")
    as_u8 = prob if prob.dtype == np.uint8 else np.rint(np.clip(prob, 0.0, 1.0) * PROB_SCALE).astype(np.uint8)
    up = upsample_prob(as_u8, TILE_PX)
    if up is None:  # pragma: no cover - upsample_prob only returns None for None
        raise ValueError("upsample failed")
    return up


# ------------------------------------------------------------------ masks -> canopies -> score


def variant_mask(veg: np.ndarray, prob_tile: np.ndarray | None, variant: FusionVariant, threshold: float) -> np.ndarray:
    """Fusion before the corridor (canopy extraction applies the corridor; E skips it)."""
    return np.asarray(fuse_masks(veg, prob_tile, None, FusionVariant(variant), threshold).mask)


def _uncorridored(mask: np.ndarray, tile: ValTile, opts: CanopyOptions) -> list[Polygon]:
    polys = mask_to_polygons(mask & tile.valid, tile_ref(tile.tile_id), approx_eps_px=opts.approx_eps_px,
                             min_area_px=opts.min_component_px)
    clipped = [p for poly in polys for p in make_valid_polygonal(poly.intersection(tile.clip))]
    return [p for p in clipped if p.area >= opts.min_area_m2]


def predict_canopies(tile: ValTile, prob: np.ndarray | None, variant: FusionVariant, cfg: AppConfig) -> list[Polygon]:
    """Canopy polygons (UTM) of the tile for one fusion variant, as the canopy stage would export them."""
    prob_tile = None if prob is None else prob_to_tile(prob)
    mask = variant_mask(tile.veg, prob_tile, variant, cfg.nn.prob_threshold)
    opts = CanopyOptions.from_config(cfg.canopy)
    if FusionVariant(variant) is FusionVariant.E and prob_tile is not None:
        return _uncorridored(mask, tile, opts)
    tref = tile_ref(tile.tile_id)
    tree = tree_mask(tile.veg, tile.pieces, tref, cfg.canopy, cfg.orchard) if cfg.canopy.tree_filter_enabled else None
    res = extract_canopies(mask, tile.pieces, tref, tile.clip, opts, valid=tile.valid, tree=tree)
    return list(res.canopies.geometry)


def score_polygons(pred: Sequence[BaseGeometry], ref: Sequence[BaseGeometry], cfg: AppConfig) -> CanopyScore:
    ev = cfg.eval
    return canopy_metrics(pred, ref, match_iou=ev.canopy_match_iou, w_iou=ev.canopy_w_iou, w_f1=ev.canopy_w_f1)


def score_tile(tile: ValTile, prob: np.ndarray | None, variant: FusionVariant, cfg: AppConfig) -> TileScore:
    s = score_polygons(predict_canopies(tile, prob, variant, cfg), tile.ref, cfg)
    return TileScore(tile.tile_id, FusionVariant(variant).value, s.iou, s.f1, s.score, s.n_pred, s.n_ref)


def val_variant(cfg: AppConfig) -> FusionVariant:
    """The fusion variant whose official score drives early stopping (nn.train.val_variant)."""
    return FusionVariant(cfg.nn.train.val_variant)


def validate_probs(tiles: Sequence[ValTile], probs: Mapping[str, np.ndarray], cfg: AppConfig,
                   variant: FusionVariant | None = None) -> ValResult:
    """Per-tile official canopy score of `variant` (default nn.train.val_variant) for probabilities keyed by
    tile id, plus the means."""
    chosen = val_variant(cfg) if variant is None else FusionVariant(variant)
    missing = [t.tile_id for t in tiles if t.tile_id not in probs]
    if missing:
        raise StageError("no probability raster for validation tiles", tiles=", ".join(missing))
    return ValResult(chosen.value, tuple(score_tile(t, probs[t.tile_id], chosen, cfg) for t in tiles))


# ------------------------------------------------------------------ loading


def _clean_ref(geoms: Sequence[BaseGeometry], tile_id: str) -> tuple[BaseGeometry, ...]:
    """Reference canopies cleaned and clipped to the tile, like `vineyard eval-examples` does."""
    box = tile_box(tile_ref(tile_id))
    cleaned = (clean_polygonal(g) for g in geoms)
    clipped = (g if box.covers(g) else clean_polygonal(g.intersection(box)) for g in cleaned if not g.is_empty)
    return tuple(g for g in clipped if not g.is_empty)


def make_val_tile(tile_id: str, rgb: np.ndarray, valid: np.ndarray, veg: np.ndarray, pieces: gpd.GeoDataFrame,
                  clip: BaseGeometry, ref: Sequence[BaseGeometry], factor: int) -> ValTile:
    if pieces.empty:
        raise StageError("validation tile has no row pieces (corridors)", tile_id=tile_id)
    for name, arr in (("valid", valid), ("veg", veg)):
        if arr.shape != (TILE_PX, TILE_PX):
            raise StageError(f"{name} mask has the wrong shape", tile_id=tile_id, shape=arr.shape)
    net = network_input(rgb, valid, factor)
    return ValTile(tile_id, net, valid.astype(bool), veg.astype(bool), pieces.reset_index(drop=True), clip,
                   _clean_ref(ref, tile_id))


def _of_tile(frame: gpd.GeoDataFrame, tile_id: str) -> gpd.GeoDataFrame:
    return frame[frame[TILE_COLUMN] == tile_id] if TILE_COLUMN in frame.columns else frame


def load_val_tiles(cfg: AppConfig, cache_dir: Path, tile_valid_path: Path, tile_ids: Sequence[str],
                   pieces: gpd.GeoDataFrame, ref_canopies: gpd.GeoDataFrame,
                   read_rgb: Callable[[str], np.ndarray]) -> tuple[ValTile, ...]:
    """Validation tiles from the tile_prep cache, `pieces` (reference or model row pieces with tile_id)
    and the reference canopies; `read_rgb(tile_id)` returns the 2048² RGB tile."""
    factor = input_factor(cfg)
    out = []
    for tile_id in tile_ids:
        clip = tile_clip(tile_valid_path, tile_ref(tile_id))
        out.append(make_val_tile(tile_id, read_rgb(tile_id), load_valid_mask(cache_dir, tile_id),
                                 load_veg_mask(cache_dir, tile_id), _of_tile(pieces, tile_id), clip,
                                 list(_of_tile(ref_canopies, tile_id).geometry), factor))
    return tuple(out)


def with_pieces(tile: ValTile, pieces: gpd.GeoDataFrame) -> ValTile:
    """Same tile with other corridors (e.g. model rows instead of reference axes)."""
    if pieces.empty:
        raise StageError("validation tile has no row pieces (corridors)", tile_id=tile.tile_id)
    return replace(tile, pieces=pieces.reset_index(drop=True))


# ------------------------------------------------------------------ work-dir glue (CLI, train, ablation)


def tile_reader(tiles_dir: Path) -> Callable[[str], np.ndarray]:
    """tile_id -> 2048² RGB of the ingested tile under work/tiles."""
    return lambda tile_id: read_tile(Path(tiles_dir) / file_name_from_tile_id(tile_id))


def load_reference(work_dir: Path, ref: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """(row_pieces, canopies) of a reference AnnSet (run id, LATEST_REFERENCE or path)."""
    run_dir = resolve_run_dir(work_dir, ref)
    annset = read_annset(run_dir / ANNSET_DIRNAME)
    return annset.layer(REF_PIECES_LAYER), annset.layer(REF_CANOPY_LAYER)


def model_axes(tiles: Sequence[ValTile], rows: gpd.GeoDataFrame, margin_m: float) -> tuple[ValTile, ...]:
    """The same tiles with the corridors of a model run's rows layer (as the canopy stage clips them)."""
    return tuple(with_pieces(t, local_pieces(rows, tile_ref(t.tile_id), margin_m)) for t in tiles)


def load_model_rows(work_dir: Path, run_ref: str) -> gpd.GeoDataFrame:
    path = resolve_run_dir(work_dir, run_ref) / LAYERS_DIRNAME / ROWS_FILE
    if not path.is_file():
        raise StageError("model run has no rows layer", run=run_ref, path=str(path))
    return read_layer(path, ROWS_LAYER)


def reference_val_tiles(cfg: AppConfig, tile_ids: Sequence[str], reference_ref: str) -> tuple[ValTile, ...]:
    """Validation tiles with the reference axes, read from the configured work dir."""
    paths = make_run_paths(cfg, VAL_RUN_ID)
    pieces, canopies = load_reference(paths.work_dir, reference_ref)
    return load_val_tiles(cfg, paths.cache_dir, paths.tile_valid, tile_ids, pieces, canopies,
                          tile_reader(paths.tiles_dir))
