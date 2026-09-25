"""Waste section (design 03 §4; A§4.9 decision rule, auto-accept disabled before Publish)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from vineyard.config.sections_core import (
    Frac,
    IntRange,
    NonNegFloat,
    NonNegInt,
    PosFloat,
    PosInt,
    Section,
)

Hsv = Annotated[int, Field(ge=0, le=255)]
WasteCategory = Literal["bag", "bottle", "tyre", "debris", "heap", "unknown"]


class WasteLatticeConfig(Section):
    pitch_range_m: tuple[PosFloat, PosFloat]
    min_points: PosInt


class WasteColourConfig(Section):
    vivid_s_min: Hsv
    vivid_v_min: Hsv
    veg_hue_range: IntRange
    bright_v_min: Hsv
    bright_s_max: Hsv
    open_px: NonNegInt


class WasteDecideConfig(Section):
    auto_accept_enabled: bool
    candidate_min: Frac
    auto_probe_min: Frac
    auto_sam3_min: Frac
    auto_clip_margin_min: float
    rule_only_max_review: PosInt


class ProbePositivesConfig(Section):
    mode: Literal["paste", "plain", "both"]
    long_side_px: IntRange
    max_n: PosInt
    jpeg_quality: Annotated[int, Field(ge=1, le=100)]
    paste_blur_sigma_px: NonNegFloat
    zenodo_record: PosInt
    repo_url: str
    licence: str
    max_download_gb: PosFloat


class ProbeNegativesConfig(Section):
    random_per_tile: NonNegInt


class ProbeConfig(Section):
    enabled: bool
    name: str
    version: str
    clip_model: str
    clip_pretrained: str
    device: Literal["mps", "cpu", "cuda"]
    embed_batch: PosInt
    crop_context: NonNegFloat
    crop_min_px: PosInt
    crop_resize_px: PosInt
    lr_c: PosFloat
    calibration_margin: NonNegFloat
    min_recall: Frac
    positives: ProbePositivesConfig
    negatives: ProbeNegativesConfig
    clip_pos_prompts: tuple[str, ...]
    clip_neg_prompts: tuple[str, ...]


class Sam3Config(Section):
    enabled: bool
    model_ids: tuple[str, ...]
    local_checkpoint: Path | None
    device: Literal["mps", "cpu", "cuda"]
    crop_px: PosInt
    text_prompts: tuple[str, ...]
    prompt_categories: dict[str, WasteCategory]
    max_negative_boxes: NonNegInt
    threshold: Frac
    mask_threshold: Frac
    min_overlap_frac: Frac
    time_budget_s: NonNegFloat
    min_probe_to_verify: Frac


class WasteReviewConfig(Section):
    crop_px: PosInt
    confirm_match_iou: Frac
    edge_tol_px: NonNegFloat


class WasteConfig(Section):
    enabled: bool
    min_area_m2: NonNegFloat
    max_area_m2: PosFloat
    white_min_area_m2: NonNegFloat
    axis_exclusion_m: NonNegFloat
    axis_exclusion_mode: Literal["all", "periodic"]
    plant_phase_tol: Frac
    lattice: WasteLatticeConfig
    tube_max_aspect: PosFloat
    tube_max_area_m2: PosFloat
    hose_min_len_m: PosFloat
    hose_max_width_m: PosFloat
    nms_iou: Frac
    box_pad: NonNegFloat
    block_assign_max_m: NonNegFloat
    max_candidates_total: PosInt
    colour: WasteColourConfig
    decide: WasteDecideConfig
    probe: ProbeConfig
    sam3: Sam3Config
    review: WasteReviewConfig
