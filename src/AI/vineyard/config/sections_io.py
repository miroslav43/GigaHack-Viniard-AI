"""I/O sections: export (+cvat), import, eval (foundation design 01 §4, X4 for ImportConfig)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from vineyard.config.sections_core import Frac, NonNegFloat, PosFloat, PosInt, Section, TileIdTuple


class CvatExportConfig(Section):
    coord_decimals: Annotated[int, Field(ge=0, le=6)]
    shape_source: Literal["manual", "auto", "semi-auto", "file"]
    max_zip_bytes: PosInt
    n_parts: PosInt | None
    balance_parts: bool
    zip_name: str
    verify_roundtrip: bool
    min_polygon_px2: NonNegFloat
    min_polyline_px: NonNegFloat
    simplify_px: NonNegFloat
    notch_width_px: PosFloat
    max_canopy_interrow_overlap_m2: NonNegFloat
    roundtrip_max_dev_px: NonNegFloat
    selfcheck_min_union_iou: Frac
    xml_deflate_level: Annotated[int, Field(ge=0, le=9)]
    manual_row_start: PosInt
    allow_qa_errors: bool


class ExportConfig(Section):
    min_row_piece_m: NonNegFloat
    min_interrow_piece_m2: NonNegFloat
    cvat: CvatExportConfig


class ImportConfig(Section):
    files: tuple[Path, ...]
    accept_enum_synonyms: bool
    duplicate_policy: Literal["last_wins", "error"]
    require_all_tiles: bool
    enum_synonyms: dict[str, str]
    canopy_row_assign_max_m: NonNegFloat
    min_line_length_m: NonNegFloat
    accept_masks: bool


class EvalGatesConfig(Section):
    canopy_score: Frac
    row_f1: Frac
    interrow_iou: Frac
    attributes: Frac


class EvalConfig(Section):
    example_tiles: TileIdTuple
    reference_annset: str
    canopy_match_iou: Frac
    canopy_w_iou: Frac
    canopy_w_f1: Frac
    row_tol_m: PosFloat
    row_min_cover: Frac
    waste_match_iou: Frac
    count_tol: PosFloat
    length_tol: PosFloat
    fp_canopy_factor: NonNegFloat
    interrow_match_iou: Frac
    regression_max_drop: NonNegFloat
    enforce_gates: bool
    baseline_file: Path | None
    write_baseline: bool
    gates: EvalGatesConfig
