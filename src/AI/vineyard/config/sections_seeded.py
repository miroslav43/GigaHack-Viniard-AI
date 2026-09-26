"""`rows_seeded` section: row detection inside reviewed tile seeds (configs/row_seeds.csv), rows_link stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from vineyard.config.sections_core import Frac, NonNegFloat, PosFloat, PosInt, Range, Section

SeedConfidence = Literal["high", "medium", "low"]


class RowsSeededConfig(Section):
    """Seeded rows: per reviewed seed (tile, polygon, row angle, optional spacing) the lattice is fitted inside
    the polygon only; the rows then flow through linking/blocks/canopy/interrow like detected rows."""

    enabled: bool
    path: Path                              # relative to project_root; a missing file = no seeds
    use_confidence: tuple[SeedConfidence, ...]   # seeds with vines=yes and one of these confidences are used
    angle_search_deg: NonNegFloat           # +- around the seed angle
    angle_step_deg: PosFloat
    spacing_range_m: Range                  # autocorrelation window without a seed spacing
    spacing_rel_tol: Frac                   # ... or seed spacing +- this fraction
    lattice_tol_frac: Frac                  # a profile peak within this x spacing of a lattice line refines it
    fit_max_angle_dev_deg: NonNegFloat      # a band fit further than this from the region angle is not used
    occupancy_min: Frac                     # per-row vegetation occupancy along the row (reviewed: lower)
    min_row_len_m: NonNegFloat
    min_rows: PosInt                        # parallel rows a region needs
    min_sep_m: NonNegFloat                  # seeded pieces within this of an accepted row are dropped there
    cut_gap_m: PosFloat                     # rows split at empty runs >= this (road, headland)
    min_points: PosInt                      # vegetation pixels a region needs
