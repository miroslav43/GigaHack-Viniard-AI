"""Perception sections: rows, orchard, blocks, canopy, interrow, row_structure, qa (design 02 §4)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from vineyard.config.sections_core import (
    Frac,
    NonNegFloat,
    NonNegInt,
    PosFloat,
    PosInt,
    Range,
    Section,
)

Bgr = tuple[
    Annotated[int, Field(ge=0, le=255)], Annotated[int, Field(ge=0, le=255)], Annotated[int, Field(ge=0, le=255)]
]


class RowsDetectConfig(Section):
    angle_step_deg: PosFloat
    angle_coarse_step_deg: PosFloat
    angle_bin_m: PosFloat
    max_sample_points: PosInt
    profile_bin_m: PosFloat
    smooth_sigma_bins: NonNegFloat
    spacing_estimate: Literal["autocorr"]
    spacing_min_m: PosFloat
    spacing_max_m: PosFloat
    peak_min_dist_factor: PosFloat
    periodicity_min_snr: NonNegFloat
    snr_band_m: Range
    onlattice_tol_factor: PosFloat
    band_init_m: PosFloat
    band_m: PosFloat
    band_max_drift_m: PosFloat
    min_band_area_m2: NonNegFloat
    fit_iterations: PosInt
    angle_gate_deg: PosFloat
    end_percentiles: Range
    snap_to_edge_m: NonNegFloat
    min_row_len_m: NonNegFloat
    residual_split_m: PosFloat
    track_window_px: PosInt
    track_vertex_m: PosFloat
    dp_tolerance_m: PosFloat
    tracking_enabled: bool
    max_orientations: PosInt
    multi_min_residual_frac: Frac
    multi_min_angle_sep_deg: PosFloat
    gap_record_min_m: PosFloat
    mask_forbidden: bool
    snr_peak_rel_width: PosFloat
    snap_ray_max_factor: PosFloat
    angle_fallback_enabled: bool
    angle_fallback_min_rows: PosInt

    @model_validator(mode="after")
    def _spacing_ordered(self) -> RowsDetectConfig:
        if self.spacing_min_m >= self.spacing_max_m:
            raise ValueError("rows.detect.spacing_min_m must be < spacing_max_m")
        return self


class RowsLinkConfig(Section):
    link_angle_max_deg: PosFloat
    link_offset_max_m: PosFloat
    link_gap_max_tiles: NonNegInt
    angle_min_len_m: NonNegFloat
    passage_erode_m: NonNegFloat
    dup_overlap_m: NonNegFloat
    rescue_soft_rejects: bool
    hole_support_min_frac: Frac
    refit_sample_m: PosFloat


class RowsFilterConfig(Section):
    width_p80_max_m: PosFloat
    width_station_m: PosFloat
    occupancy_min: Frac
    min_vine_score: Frac
    wide_min_rel_contrast: NonNegFloat


class RowsConfig(Section):
    detect: RowsDetectConfig
    link: RowsLinkConfig
    filter: RowsFilterConfig


class RowsGuidedConfig(Section):
    """Neighbour-guided second pass (rows_link): tiles with few accepted rows retried at a neighbour's lattice."""

    enabled: bool
    target_max_accepted: NonNegInt
    prior_min_rows: PosInt
    prior_angle_merge_deg: PosFloat
    max_priors_per_cluster: PosInt
    angle_search_deg: NonNegFloat
    angle_step_deg: PosFloat
    spacing_rel_tol: Frac
    min_snr: NonNegFloat
    lattice_tol_frac: Frac
    phase_tol_frac: Frac
    occupancy_min: Frac
    min_rel_contrast: NonNegFloat
    width_p80_max_m: PosFloat
    wide_min_rel_contrast: NonNegFloat
    orchard_along_period_m: Range
    min_row_len_m: NonNegFloat
    min_rows: PosInt
    min_sep_m: NonNegFloat
    continue_max_m: NonNegFloat
    cut_gap_m: PosFloat
    self_prior: bool                       # the tile's own accepted lattice is a prior too (partial tiles)
    lateral_enabled: bool                  # a guided row may also run beside an accepted row of its tile
    lateral_max_factor: PosFloat           # ... at <= this x spacing (block grows row by row)
    lateral_min_rows: PosInt               # rows a cluster needs when the tile already holds >= min_rows rows
    rounds: PosInt                         # round k uses the rows of round k-1 as priors/anchors


class OrchardConfig(Section):
    width_spacing_ratio_max: PosFloat
    along_period_m: Range
    along_duty_max: Frac
    tree_blob_area_m2: PosFloat
    tree_blob_axis_ratio_max: PosFloat
    block_reject_enabled: bool
    block_majority_frac: Frac


class BlocksConfig(Section):
    neighbour_max_m: PosFloat
    parallel_max_deg: PosFloat
    min_overlap_frac: Frac
    collinear_gap_max_m: PosFloat
    skip_one_enabled: bool
    skip_one_spacing_factor: Range
    spacing_cut_enabled: bool
    spacing_jump_max_m: PosFloat
    phase_tol_factor: PosFloat
    headland_min_rows: PosInt
    transverse_band_min_m: PosFloat
    transverse_full_width: bool
    min_rows_per_block: PosInt
    cut_by_passages: bool
    outline_buffer_m: NonNegFloat
    id_prefix: Literal["V"]
    id_width: Literal[2]
    row_id_width: Literal[3]
    garden_max_row_length_m: PosFloat
    band_bin_m: PosFloat
    band_pad_m: NonNegFloat
    spacing_cut_side_rows: PosInt
    too_few_rows_issue_min: PosInt
    override_match_tol_m: PosFloat


class CanopyConfig(Section):
    corridor_half_m: PosFloat
    min_area_m2: NonNegFloat
    connectivity: Literal[4, 8]
    simplify_px: NonNegFloat
    vector_offset_px: float
    vector_outset_px: NonNegFloat
    clump_area_m2: PosFloat
    split_clumps: bool
    max_perp_width_tree_m: PosFloat
    tree_filter_enabled: bool
    rows_margin_m: NonNegFloat
    corridor_label_convention: Literal["index", "continuous"]
    clip_to_corridor: bool
    interpolated_min_veg_frac: Frac
    mask_method: Literal["exg", "veg"]
    exg_blur_sigma_px: NonNegFloat
    exg_threshold: float
    axis_refine_band_m: NonNegFloat
    axis_refine_iters: PosInt
    axis_refine_max_m: NonNegFloat
    axis_refine_min_px: PosInt
    gap_evidence_min_area_m2: NonNegFloat
    neck_split_enabled: bool = False
    neck_split_min_along_m: PosFloat = 1.6
    neck_split_ratio: Frac = 0.5
    neck_split_min_piece_m: PosFloat = 0.5
    neck_split_window_m: PosFloat = 0.6


class InterrowConfig(Section):
    offset_m: NonNegFloat
    cover_bare_max: Frac
    cover_veg_min: Frac
    shadow_unassessable: Frac
    min_width_assessable_m: NonNegFloat
    cover_borderline_margin: Frac
    hole_min_area_m2: NonNegFloat
    hole_sources: tuple[Literal["forbidden", "tree_mask"], ...]
    edge_extend_m: NonNegFloat
    edge_tol_m: NonNegFloat
    canopy_clearance_m: NonNegFloat

    @model_validator(mode="after")
    def _cover_ordered(self) -> InterrowConfig:
        if self.cover_bare_max > self.cover_veg_min:
            raise ValueError("interrow.cover_bare_max must be <= cover_veg_min")
        return self


class RowStructureConfig(Section):
    gap_disrupted_m: PosFloat
    include_end_gaps: bool
    borderline_m: Range
    unassessable_visible_min: Frac
    occ_bin_m: PosFloat
    occ_close_m: NonNegFloat
    occ_window_m: PosFloat
    occ_min: Frac
    occ_smoothing_enabled: bool
    gap_fill_max_width_m: NonNegFloat
    visible_bin_max_hidden: Frac


class QaConfig(Section):
    preview_px: PosInt
    preview_jpeg_quality: Annotated[int, Field(ge=1, le=100)]
    overview_px_per_tile: PosInt
    colors_bgr: dict[str, Bgr]
    canopy_fill_alpha: Frac
    snr_review_max: NonNegFloat
