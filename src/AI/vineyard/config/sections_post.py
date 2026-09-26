"""Post-Marcaj sections: derive, targets, route, measure, publish, web (design 04 §4, I14 single keys)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import AfterValidator, Field, model_validator

from vineyard.config.sections_core import Frac, NonNegFloat, NonNegInt, PosFloat, PosInt, Section

# The values of vineyard.contracts.enums.TargetKind (config stays free of contract imports; a test pins both).
TargetKindName = Literal["row_gap", "row_end_short", "missing_row", "missing_plant", "sparse", "waste", "other"]

# Survey identity the web platform can register: the public.survey CHECKs (src/Web/supabase/migrations/
# 20260926000100_platform_admin.sql) and the super-admin KEY_RE. Shared by WebConfig and web.manifest.SurveyInfo.
SURVEY_ID_PATTERN: Final = r"^[a-z0-9][a-z0-9-]{1,39}$"
SURVEY_NAME_MIN_LEN: Final = 2
SURVEY_NAME_MAX_LEN: Final = 160


def _unique(kinds: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(kinds)) != len(kinds):
        raise ValueError(f"duplicate target kinds in {list(kinds)}")
    return kinds


class DeriveConfig(Section):
    join_lateral_max_m: PosFloat
    row_missing_min_len_m: NonNegFloat
    vertex_dedupe_m: PosFloat
    blocks_merge_warn_m: NonNegFloat
    interrow_link_max_m: PosFloat
    garden_forbidden_dist_m: NonNegFloat
    garden_max_rows: NonNegInt
    interrow_overlap_min_m2: NonNegFloat
    interrow_overlap_support_m: NonNegFloat
    interrow_overlap_min_width_m: NonNegFloat


class TargetsConfig(Section):
    gap_min_m: PosFloat
    long_gap_m: PosFloat
    gap_step_m: PosFloat
    gap_sample_step_m: PosFloat
    missing_min_m: PosFloat
    include_missing: bool
    include_sparse: bool
    sparse_window_m: PosFloat
    sparse_step_m: PosFloat
    sparse_occ_max: Frac
    sparse_known_min: Frac
    end_short_min_m: PosFloat
    missing_row_spacing_factor: PosFloat
    include_waste: bool
    priority_long_gap_m: PosFloat
    dedupe_m: NonNegFloat
    must_kinds: Annotated[tuple[TargetKindName, ...], AfterValidator(_unique)]
    edge_margin_m: PosFloat
    end_skip_outer_rows: bool
    end_neighbour_min_offset_m: NonNegFloat


class CrossPathsConfig(Section):
    """Tracks across the rows of a block (vineyard.route.cross_paths): detection, targets, walking."""

    enabled: bool
    min_gap_m: PosFloat
    max_gap_m: PosFloat
    max_width_m: PosFloat
    min_rows: Annotated[int, Field(ge=2)]
    min_segment_rows: Annotated[int, Field(ge=2)]
    link_min_m: PosFloat
    link_max_m: PosFloat
    overlap_tol_m: NonNegFloat
    max_residual_m: NonNegFloat
    end_extend_spacing: PosFloat
    drop_targets: bool
    route: bool
    passable: bool

    @model_validator(mode="after")
    def _ordered(self) -> CrossPathsConfig:
        if self.min_gap_m > self.max_gap_m:
            raise ValueError(f"cross_paths.min_gap_m ({self.min_gap_m}) must not exceed max_gap_m ({self.max_gap_m})")
        if self.link_min_m >= self.link_max_m:
            raise ValueError(f"cross_paths.link_min_m ({self.link_min_m}) must be below link_max_m ({self.link_max_m})")
        if self.min_segment_rows > self.min_rows:
            raise ValueError(f"cross_paths.min_segment_rows ({self.min_segment_rows}) must not exceed min_rows "
                             f"({self.min_rows})")
        return self


class RouteDomainConfig(Section):
    grid_size_m: PosFloat
    inner_buffer_m: NonNegFloat
    eroded_buffer_m: NonNegFloat
    seam_close_m: NonNegFloat
    subtract_canopies: bool


class RouteGraphConfig(Section):
    centerline_step_m: PosFloat
    centerline_min_len_m: NonNegFloat
    skeleton_res_m: PosFloat
    skeleton_erode_m: NonNegFloat
    spur_min_m: NonNegFloat
    simplify_tol_m: NonNegFloat
    connector_max_m: PosFloat
    connector_outside_penalty: NonNegFloat
    snap_join_m: NonNegFloat
    pull_lookahead: PosInt
    target_spur_standoff_m: NonNegFloat


class RouteSolverConfig(Section):
    method: Literal["auto", "ortools", "fallback"]
    time_limit_s: PosInt
    time_limit_final_s: PosInt
    final: bool
    first_solution: str
    metaheuristic: str
    cost_per_m: PosInt
    cover_iterations: NonNegInt
    fallback_time_s: PosInt
    max_tsp_nodes: PosInt
    merge_node_m: NonNegFloat
    dijkstra_chunk: PosInt
    include_optional: bool
    optional_max_detour_m: NonNegFloat
    probe_time_limit_s: PosInt
    budget_rounds: NonNegInt


class RouteValidateConfig(Section):
    coord_decimals: Annotated[int, Field(ge=0, le=6)]
    closure_max_m: NonNegFloat
    length_tol_m: NonNegFloat
    zero_len_eps_m: NonNegFloat


class RouteConfig(Section):
    start_file: Path
    start_tolerance_m: NonNegFloat
    visit_radius_m: PosFloat
    candidate_radius_m: PosFloat
    max_candidates_per_side: PosInt
    max_snap_m: PosFloat
    max_outside_frac_official: Frac
    max_outside_frac_publish: Frac
    plan_outside_margin: NonNegFloat
    walking_speed_kmh: PosFloat
    domain: RouteDomainConfig
    graph: RouteGraphConfig
    solver: RouteSolverConfig
    validate_: RouteValidateConfig = Field(alias="validate")

    @model_validator(mode="after")
    def _outside_limits(self) -> RouteConfig:
        if self.max_outside_frac_publish > self.max_outside_frac_official:
            raise ValueError(f"route.max_outside_frac_publish ({self.max_outside_frac_publish}) must not exceed "
                             f"route.max_outside_frac_official ({self.max_outside_frac_official})")
        if self.plan_outside_margin >= self.max_outside_frac_publish:
            raise ValueError(f"route.plan_outside_margin ({self.plan_outside_margin}) must be below "
                             f"route.max_outside_frac_publish ({self.max_outside_frac_publish})")
        return self

    @property
    def plan_outside_frac(self) -> float:
        """Outside share the planner accepts: the publish/validator limit minus the planning margin."""
        return self.max_outside_frac_publish - self.plan_outside_margin


class MeasureConfig(Section):
    round_m: PosFloat
    round_ha: PosFloat
    total_label: str
    area_union_sum_warn_frac: NonNegFloat


class PublishConfig(Section):
    require_source: Literal["model", "marcaj", "reference"] | None
    sum_check_tol_m: NonNegFloat
    sum_check_tol_m2: NonNegFloat


class FarmsConfig(Section):
    """Farms (groups of neighbouring blocks) and road classes (vineyard.farms): web map only, never CVAT."""

    enabled: bool
    gap_max_m: PosFloat
    touch_m: NonNegFloat
    link_width_m: PosFloat
    public_road_buffer_m: NonNegFloat
    outline_simplify_m: NonNegFloat
    osm_highways: Path | None = None
    public_highways: Annotated[tuple[str, ...], Field(min_length=1)]
    internal_min_len_m: NonNegFloat
    osm_fetch_pad_m: NonNegFloat

    @model_validator(mode="after")
    def _touch_below_gap(self) -> FarmsConfig:
        if self.touch_m > self.gap_max_m:
            raise ValueError(f"farms.touch_m ({self.touch_m}) must not exceed gap_max_m ({self.gap_max_m})")
        return self


class Gdal2TilesConfig(Section):
    zoom: str
    resampling: str
    tiledriver: Literal["WEBP", "PNG", "JPEG"]
    webp_quality: Annotated[int, Field(ge=1, le=100)]
    processes: PosInt


class WebConfig(Section):
    out_dir: Path
    # Bundle directory name and manifest identity (`surveys/<survey_id>/pipeline/`, web contract §6.1).
    survey_id: Annotated[str, Field(pattern=SURVEY_ID_PATTERN)]
    survey_name: Annotated[str, Field(min_length=SURVEY_NAME_MIN_LEN, max_length=SURVEY_NAME_MAX_LEN)]
    geojson_decimals_4326: Annotated[int, Field(ge=0, le=9)]
    utm_decimals: Annotated[int, Field(ge=0, le=6)]
    ortho_mode: Literal["auto", "xyz", "tiles_jpeg"]
    ortho_px: PosInt
    ortho_quality: Annotated[int, Field(ge=1, le=100)]
    canopy_minzoom: NonNegInt
    gdal2tiles: Gdal2TilesConfig
    canopy_mvt: Literal["auto", "off"]
    utm_copies: bool
    # tiles.geojson / masks/ (web contract §6.3): mask side in px (divides grid.tile_px) and the optional
    # human review list (tile_id,status,note; status missed | partial | verify), null = none.
    mask_px: PosInt
    tile_review: Path | None = None
