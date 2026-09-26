"""NN pseudo-labels (A§4.6, design 03 §3 N2), torch-free.

Whole-tile labels at nn.in_gsd_m (1024² for 0.05 m): vine = veg ∧ row corridor after the tree filter
-> 1; every other pixel, including grass and trees outside the corridors -> 0, explicitly; 255 (ignored
by the loss) on nodata, on a band of edge_band_px on both sides of the label edges, on vegetation
within ignore_band_px outside a corridor (vine canopy overhanging the ±corridor_half_m corridor is
neither clearly 1 nor 0), and on in-corridor vegetation the rules cannot call (trees, small or clump
components, interpolated rows without vegetation evidence).

Band geometry (measured on 8 viz42 tiles): a ±3 px band on both the label and the 12 px wide
corridor edges ignored 92.5 % of the vine pixels; ±1 px on the label edges plus the vegetation-only
overhang band keeps 60 % of them with 18 % of all pixels ignored.
Training-tile selection never returns a holdout (reference) tile.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import cv2
import numpy as np
import pandas as pd

from vineyard.contracts.enums import TileStatus
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.errors import VineyardError
from vineyard.geo.tiling import GSD_M, TileRef
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.canopy import eligible_labels, interpolated_mask, label_veg_fractions
from vineyard.perception.corridor import corridor_label_raster
from vineyard.perception.trees import tree_mask
from vineyard.perception.types import I32, U8, BoolMask
from vineyard.qa.review import CODE_SEP, PRIORITY_WARN

if TYPE_CHECKING:
    import geopandas as gpd

    from vineyard.config import AppConfig

NEGATIVE: Final = 0
POSITIVE: Final = 1
IGNORE: Final = 255
LABEL_VALUES: Final = (NEGATIVE, POSITIVE, IGNORE)
MAJORITY: Final = 0.5  # a label pixel is positive / in the corridor when half of its native pixels are
FULLY_VALID: Final = 1.0 - 1e-6  # a label pixel with any nodata native pixel is ignored
CONNECTIVITIES: Final = (4, 8)
FACTOR_TOL: Final = 1e-6
CAP_EPS: Final = 1e-9
COMMENT: Final = "#"
STATUS_COLUMNS: Final = ("tile_id", "status", "has_vineyard", "n_row_pieces", "review_priority", "issues")
EMPTY_ORDER_COLUMN: Final = "veg_frac"

_log = get_logger("nn.pseudolabels")


class PseudoLabelError(VineyardError):
    """Pseudo-labels or the training-tile selection cannot be built."""


# ------------------------------------------------------------------ parameters


@dataclass(frozen=True)
class LabelParams:
    factor: int  # native px per label px (0.05 m / 0.025 m = 2)
    ignore_band_px: int  # label px: vegetation this close outside a corridor is ignored (canopy overhang)
    edge_band_px: int  # label px ignored on each side of the positive-label edges
    ignore_small_in_corridor: bool
    ignore_clumps: bool
    tree_filter: bool
    min_component_px: int  # native px (canopy.min_area_m2)
    clump_min_px: int  # native px; a component larger than this is a clump (canopy.clump_area_m2)
    connectivity: int

    def __post_init__(self) -> None:
        if self.factor < 1:
            raise ValueError(f"factor must be >= 1, got {self.factor}")
        if self.ignore_band_px < 0 or self.edge_band_px < 0:
            raise ValueError(f"band widths must be >= 0, got ignore_band_px={self.ignore_band_px} "
                             f"edge_band_px={self.edge_band_px}")
        if self.connectivity not in CONNECTIVITIES:
            raise ValueError(f"connectivity must be one of {CONNECTIVITIES}, got {self.connectivity}")

    @classmethod
    def from_config(cls, cfg: AppConfig) -> LabelParams:
        pl, canopy = cfg.nn.pseudolabels, cfg.canopy
        px_area = GSD_M * GSD_M
        return cls(
            factor=label_factor(cfg.nn.in_gsd_m), ignore_band_px=pl.ignore_band_px, edge_band_px=pl.edge_band_px,
            ignore_small_in_corridor=pl.ignore_small_in_corridor, ignore_clumps=pl.ignore_clumps,
            tree_filter=pl.tree_filter, min_component_px=int(math.ceil(canopy.min_area_m2 / px_area - CAP_EPS)),
            clump_min_px=int(math.floor(canopy.clump_area_m2 / px_area + CAP_EPS)), connectivity=canopy.connectivity,
        )


def label_factor(in_gsd_m: float) -> int:
    """Integer downsampling factor from the tile GSD to nn.in_gsd_m (contract §6)."""
    ratio = in_gsd_m / GSD_M
    factor = round(ratio)
    if factor < 1 or abs(ratio - factor) > FACTOR_TOL:
        raise PseudoLabelError("nn.in_gsd_m must be an integer multiple of the tile GSD", in_gsd_m=in_gsd_m,
                               tile_gsd_m=GSD_M)
    return factor


# ------------------------------------------------------------------ rasters


@dataclass(frozen=True, eq=False)
class CorridorInputs:
    labels: I32  # piece k+1 inside its corridor, 0 outside (canopy.corridor_label_convention)
    eligible: np.ndarray  # (n_pieces + 1,) bool lookup, index 0 = background
    tree: BoolMask | None


@dataclass(frozen=True, eq=False)
class NativeMasks:
    vine: BoolMask
    ignore: BoolMask
    corridor: BoolMask
    n_small_px: int
    n_clump_px: int
    n_tree_px: int
    n_ineligible_px: int
    n_overhang_px: int


@dataclass(frozen=True)
class LabelCounts:
    n_pos: int
    n_neg: int
    n_ignore: int


def corridor_inputs(pieces: gpd.GeoDataFrame, tile: TileRef, veg: BoolMask, valid: BoolMask, cfg: AppConfig,
                    p: LabelParams) -> CorridorInputs:
    """Corridor labels, piece eligibility (S4) and the tree mask, exactly as the canopy stage builds them."""
    canopy = cfg.canopy
    labels = corridor_label_raster(pieces, tile, canopy.corridor_half_m, convention=canopy.corridor_label_convention)
    veg_frac = label_veg_fractions(veg, labels, len(pieces), valid)
    eligible = eligible_labels(interpolated_mask(pieces), veg_frac, canopy.interpolated_min_veg_frac)
    tree = tree_mask(veg, pieces, tile, canopy, cfg.orchard) if p.tree_filter else None
    return CorridorInputs(labels=labels, eligible=eligible, tree=tree)


def _component_masks(rest: BoolMask, p: LabelParams) -> tuple[BoolMask, BoolMask]:
    n, comp, stats, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=p.connectivity)
    area = stats[:, cv2.CC_STAT_AREA].copy()
    area[0] = 0
    small = (area < p.min_component_px) & (np.arange(n) > 0)
    clump = area > p.clump_min_px
    return small[comp], clump[comp]


def _near(mask: BoolMask, radius_px: int) -> BoolMask:
    if radius_px == 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def classify_corridor_veg(veg: BoolMask, labels: I32, eligible: np.ndarray, tree: BoolMask | None,
                          p: LabelParams) -> NativeMasks:
    """Split vegetation (native px) into vine (label 1) and ignored pixels (in-corridor rules and the
    overhang band just outside the corridors); everything else is 0."""
    if eligible.shape[0] < int(labels.max(initial=0)) + 1:
        raise ValueError(f"eligible lookup has {eligible.shape[0]} entries for {int(labels.max())} corridor labels")
    corridor = labels > 0
    cand = veg & corridor
    tree_px = cand & tree if (p.tree_filter and tree is not None) else np.zeros_like(cand)
    inelig = cand & ~eligible[labels]
    rest = cand & ~tree_px & ~inelig
    small, clump = _component_masks(rest, p)
    vine = rest & ~small & ~(clump if p.ignore_clumps else np.zeros_like(clump))
    overhang = veg & ~corridor & _near(corridor, p.ignore_band_px * p.factor)
    ignore = tree_px | inelig | overhang
    ignore = ignore | small if p.ignore_small_in_corridor else ignore
    ignore = ignore | clump if p.ignore_clumps else ignore
    return NativeMasks(vine=vine, ignore=ignore, corridor=corridor, n_small_px=int(small.sum()),
                       n_clump_px=int(clump.sum()), n_tree_px=int(tree_px.sum()), n_ineligible_px=int(inelig.sum()),
                       n_overhang_px=int(overhang.sum()))


def edge_band(mask: BoolMask, band_px: int) -> BoolMask:
    """Pixels within band_px of the mask edge, on both sides (dilate XOR erode); image borders are no edge."""
    if band_px < 0:
        raise ValueError(f"band width must be >= 0, got {band_px}")
    if band_px == 0:
        return np.zeros(mask.shape, dtype=bool)
    kernel = np.ones((2 * band_px + 1, 2 * band_px + 1), dtype=np.uint8)
    m = mask.astype(np.uint8)
    return cv2.dilate(m, kernel) != cv2.erode(m, kernel)


def _area_fraction(mask: np.ndarray, factor: int) -> np.ndarray:
    h, w = mask.shape
    return cv2.resize(mask.astype(np.float32), (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def _check_native(arrays: Sequence[np.ndarray], factor: int) -> None:
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays) or len(shape) != 2:
        raise ValueError(f"native masks must share one 2-D shape, got {[a.shape for a in arrays]}")
    if shape[0] % factor or shape[1] % factor:
        raise ValueError(f"native shape {shape} is not divisible by the factor {factor}")


def build_pseudolabel(vine: BoolMask, ignore: BoolMask, valid: BoolMask, p: LabelParams) -> U8:
    """Native masks -> uint8 label {0, 1, 255} at the NN resolution (INTER_AREA fractions, contract §6).

    A label pixel is 1 when half of its native pixels are vine, 255 when any of them is ignored or
    nodata or when it lies within edge_band_px of a positive edge, else 0.
    """
    _check_native((vine, ignore, valid), p.factor)
    pos = _area_fraction(vine & valid, p.factor) >= MAJORITY
    unsure = (_area_fraction(ignore, p.factor) > 0) | (_area_fraction(valid, p.factor) < FULLY_VALID)
    label = np.where(pos, np.uint8(POSITIVE), np.uint8(NEGATIVE))
    return np.where(edge_band(pos, p.edge_band_px) | unsure, np.uint8(IGNORE), label).astype(np.uint8)


def empty_tile_label(valid: BoolMask, p: LabelParams) -> U8:
    """All-negative label of a confirmed empty tile; nodata stays ignored."""
    zero = np.zeros(valid.shape, dtype=bool)
    return build_pseudolabel(zero, zero, valid, p)


def downsample_rgb(rgb: U8, valid: BoolMask, factor: int) -> U8:
    """Tile RGB -> NN input (contract §6): nodata set to 0, then cv2.INTER_AREA by `factor`."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8 RGB, got {rgb.shape} {rgb.dtype}")
    if valid.shape != rgb.shape[:2]:
        raise ValueError(f"valid shape {valid.shape} != RGB shape {rgb.shape[:2]}")
    masked = np.where(valid[..., None], rgb, np.uint8(0)).astype(np.uint8)
    if factor == 1:
        return masked
    h, w = valid.shape
    return cv2.resize(masked, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def label_counts(label: U8) -> LabelCounts:
    """Pixel counts per class; any value outside {0, 1, 255} is an error."""
    counts = np.bincount(label.ravel(), minlength=IGNORE + 1)
    stray = int(counts.sum() - counts[list(LABEL_VALUES)].sum())
    if stray:
        raise ValueError(f"label holds {stray} pixels with values outside {LABEL_VALUES}")
    return LabelCounts(n_pos=int(counts[POSITIVE]), n_neg=int(counts[NEGATIVE]), n_ignore=int(counts[IGNORE]))


# ------------------------------------------------------------------ training-tile selection


@dataclass(frozen=True)
class SelectionPolicy:
    min_row_pieces: int
    blocking_codes: frozenset[str]
    fill_from_urgent: bool  # add review-priority-1 tiles only when the clean ones are too few
    include_empty_tiles: bool
    max_empty_tile_frac: float
    min_train_tiles: int

    @classmethod
    def from_config(cls, cfg: AppConfig) -> SelectionPolicy:
        pl = cfg.nn.pseudolabels
        return cls(min_row_pieces=pl.min_row_pieces, blocking_codes=frozenset(pl.blocking_codes),
                   fill_from_urgent=pl.fill_from_urgent,
                   include_empty_tiles=pl.include_empty_tiles, max_empty_tile_frac=pl.max_empty_tile_frac,
                   min_train_tiles=pl.min_train_tiles)


@dataclass(frozen=True)
class TileSelection:
    positives: tuple[str, ...]
    empties: tuple[str, ...]
    source: Literal["qa_file", "auto"]
    n_clean: int
    n_urgent_added: int
    holdout_removed: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...]  # (tile_id, reason), sorted

    def summary(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for _, reason in self.excluded:
            reasons[reason] = reasons.get(reason, 0) + 1
        return {"source": self.source, "n_positives": len(self.positives), "n_empties": len(self.empties),
                "n_clean": self.n_clean, "n_urgent_added": self.n_urgent_added,
                "holdout_removed": sorted(self.holdout_removed), "excluded_by_reason": dict(sorted(reasons.items()))}


def _codes(issues: object) -> frozenset[str]:
    text = "" if issues is None or (isinstance(issues, float) and math.isnan(issues)) else str(issues)
    return frozenset(c.strip() for c in text.split(CODE_SEP) if c.strip())


def _exclusion(row: Any, p: SelectionPolicy, error_tiles: frozenset[str]) -> str | None:
    if str(row.status) != TileStatus.OK.value:
        return f"status:{row.status}"
    if not bool(row.has_vineyard) or int(row.n_row_pieces) < p.min_row_pieces:
        return "few_rows"
    if str(row.tile_id) in error_tiles:
        return "qa_error"
    blocked = sorted(_codes(row.issues) & p.blocking_codes)
    return f"blocked:{blocked[0]}" if blocked else None


def _auto_positives(status: pd.DataFrame, p: SelectionPolicy,
                    error_tiles: frozenset[str]) -> tuple[list[str], int, int, list[tuple[str, str]]]:
    clean: list[str] = []
    urgent: list[str] = []
    excluded: list[tuple[str, str]] = []
    for row in status.itertuples(index=False):
        reason = _exclusion(row, p, error_tiles)
        if reason is not None:
            excluded.append((str(row.tile_id), reason))
        elif int(row.review_priority) >= PRIORITY_WARN:
            clean.append(str(row.tile_id))
        else:
            urgent.append(str(row.tile_id))
    fill = p.fill_from_urgent and len(clean) < p.min_train_tiles
    excluded += [] if fill else [(t, "urgent_not_needed") for t in urgent]
    chosen = clean + (urgent if fill else [])
    return sorted(chosen), len(clean), len(urgent) if fill else 0, excluded


def _qa_positives(status: pd.DataFrame, approved: Sequence[str]) -> tuple[list[str], list[tuple[str, str]]]:
    rows = {str(t): int(n) for t, n in zip(status["tile_id"], status["n_row_pieces"], strict=True)}
    chosen = sorted({t for t in approved if rows.get(t, 0) > 0})
    excluded = [(t, "not_in_run" if t not in rows else "no_rows") for t in sorted(set(approved)) if rows.get(t, 0) <= 0]
    return chosen, excluded


def _empties(status: pd.DataFrame, n_pos: int, p: SelectionPolicy, holdout: frozenset[str]) -> tuple[str, ...]:
    if not p.include_empty_tiles or p.max_empty_tile_frac <= 0:
        return ()
    empty = status[(status["status"] == TileStatus.NO_VINEYARD.value) & ~status["tile_id"].isin(holdout)]
    order = empty.assign(_g=empty[EMPTY_ORDER_COLUMN] if EMPTY_ORDER_COLUMN in empty.columns else 0.0)
    ranked = order.sort_values(["_g", "tile_id"], ascending=[False, True], kind="mergesort")["tile_id"]
    frac = min(p.max_empty_tile_frac, 1.0 - CAP_EPS)
    cap = int(math.floor(frac * n_pos / (1.0 - frac) + CAP_EPS))
    return tuple(str(t) for t in ranked.head(cap))


def select_training_tiles(status: pd.DataFrame, qa_approved: Sequence[str] | None, holdout: Sequence[str],
                          p: SelectionPolicy, error_tiles: frozenset[str] = frozenset()) -> TileSelection:
    """Positive (vineyard) and empty tiles for the patch store; holdout tiles are always removed."""
    missing = [c for c in STATUS_COLUMNS if c not in status.columns]
    if missing:
        raise PseudoLabelError("tile_status lacks required columns", missing=missing)
    hold = frozenset(holdout)
    removed = sorted((hold & set(qa_approved or ())) | (hold & set(status["tile_id"].astype(str))))
    if removed:
        log_event(_log, "nn.holdout_removed", level=logging.WARNING, tiles=removed)
    kept = status[~status["tile_id"].isin(hold)]
    if qa_approved is not None:
        positives, excluded = _qa_positives(kept, [t for t in qa_approved if t not in hold])
        n_clean, n_urgent, source = len(positives), 0, "qa_file"
    else:
        positives, n_clean, n_urgent, excluded = _auto_positives(kept, p, error_tiles)
        source = "auto"
        log_event(_log, "nn.train_tiles.auto", n_clean=n_clean, n_urgent_added=n_urgent)
    if len(positives) < p.min_train_tiles:
        raise PseudoLabelError("too few training tiles (nn.pseudolabels.min_train_tiles)", n_tiles=len(positives),
                               min_train_tiles=p.min_train_tiles, source=source)
    return TileSelection(
        positives=tuple(positives), empties=_empties(kept, len(positives), p, hold), source=source,  # type: ignore[arg-type]
        n_clean=n_clean, n_urgent_added=n_urgent, holdout_removed=tuple(removed), excluded=tuple(sorted(excluded)),
    )


def read_train_tiles_file(path: Path) -> tuple[str, ...]:
    """QA list: one tile id per line; blank lines and `#` comments are skipped."""
    source = Path(path)
    if not source.is_file():
        raise PseudoLabelError("training-tile list file missing", path=str(source))
    ids: list[str] = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        tile_id = line.split(COMMENT, 1)[0].strip()
        if not tile_id:
            continue
        if not is_valid_id(IdKind.TILE, tile_id):
            raise PseudoLabelError(f"invalid tile id on line {line_no}", path=str(source), value=tile_id)
        ids.append(tile_id)
    return tuple(dict.fromkeys(ids))
