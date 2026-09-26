"""NN section (design 03 §4, X4: nn.fusion is the only fusion key, nn.use_in_* the only NN switches)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from vineyard.config.sections_core import (
    Frac,
    NonNegFloat,
    NonNegInt,
    PosFloat,
    PosInt,
    Section,
    TileIdTuple,
)

FusionName = Literal["A", "B", "C", "D", "E"]
AblationVariant = Literal["A", "B", "C", "D", "E", "F"]
Device = Literal["mps", "cpu", "cuda"]


class PseudoLabelConfig(Section):
    ignore_band_px: NonNegInt  # label px: vegetation this close OUTSIDE a corridor is ignored (canopy overhang)
    edge_band_px: NonNegInt  # label px ignored on each side of the positive-label edges
    ignore_small_in_corridor: bool
    ignore_clumps: bool
    tree_filter: bool  # trees inside a corridor become ignore
    min_row_pieces: PosInt  # a vineyard tile needs at least this many row pieces to be selected
    blocking_codes: tuple[str, ...]  # qa issue codes that exclude a tile from training
    fill_from_urgent: bool  # add review-priority-1 tiles when the clean ones are fewer than min_train_tiles
    min_labeled_frac: Frac  # patches with less labelled (not ignore) area are skipped
    train_tiles_file: Path | None
    include_empty_tiles: bool
    max_empty_tile_frac: Frac
    min_train_tiles: PosInt
    patch_px: PosInt
    patch_stride_px: PosInt


class DropNoisyConfig(Section):
    enabled: bool
    after_epoch: NonNegInt
    frac: Frac


class TrainConfig(Section):
    epochs: PosInt
    lr: PosFloat
    weight_decay: NonNegFloat
    label_smoothing: Frac
    patience: PosInt
    bce_weight: NonNegFloat
    dice_weight: NonNegFloat
    num_workers: NonNegInt
    colour_jitter: Frac
    encoder_weights: Literal["imagenet"] | None
    smoke_max_batches: PosInt
    val_variant: FusionName  # fusion variant scored on the reference tiles for early stopping
    drop_noisy: DropNoisyConfig


class AblationConfig(Section):
    variants: tuple[AblationVariant, ...]
    promotable: tuple[FusionName, ...]
    min_gain: NonNegFloat
    axes_sources: tuple[Literal["model", "reference"], ...]
    panel_tiles: tuple[str, ...]
    n_auto_panels: NonNegInt


class NnConfig(Section):
    enabled: bool
    name: str
    version: str
    arch: Literal["unet"]
    encoder: str
    in_gsd_m: PosFloat
    out_channels: PosInt
    axis_sigma_m: PosFloat
    device: Device
    fallback_device: Device
    batch_size: PosInt
    infer_batch_size: PosInt
    prob_threshold: Frac
    weights_sha256: str | None
    weights_url: str | None
    fusion: FusionName
    use_in_rows: bool
    use_in_interrow: bool
    holdout_tiles: TileIdTuple
    mps_high_watermark_ratio: Frac
    pseudolabels: PseudoLabelConfig
    train: TrainConfig
    ablation: AblationConfig
