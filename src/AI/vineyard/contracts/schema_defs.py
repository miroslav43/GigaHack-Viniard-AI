"""Layer schema table (contract §2.5 + v1.1 additive layers from the four subsystem designs).

Column order here is the on-disk column order; provenance columns follow, geometry is last.
Float columns use NaN as their null value and are never null-checked.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal

from vineyard.contracts.enums import (
    EdgeKind,
    InterrowCover,
    RowStructure,
    Severity,
    Source,
    TargetKind,
    TileStatus,
)
from vineyard.contracts.ids import IdKind

ColumnKind = Literal["str", "int8", "int16", "int32", "int64", "float32", "float64", "bool"]

GEOMETRY_COLUMN: Final = "geometry"
# Contract §2.7: every row-like LineString has >= 2 distinct vertices and length >= 0.05 m.
MIN_LINE_LENGTH_M: Final = 0.05


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: ColumnKind
    nullable: bool = False
    enum: type[StrEnum] | None = None
    id_kind: IdKind | None = None
    blank_ok: bool = False  # "" is a legal value (e.g. waste vineyard_id outside every block)


@dataclass(frozen=True)
class LayerSchema:
    name: str
    geom_types: tuple[str, ...]
    pk: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    provenance: bool = True
    annset: bool = False
    allow_extra: bool = False
    allow_empty_geom: bool = False
    line_rules: bool = True

    @property
    def all_columns(self) -> tuple[ColumnSpec, ...]:
        return self.columns + (PROVENANCE_COLUMNS if self.provenance else ())

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.all_columns)

    def column(self, name: str) -> ColumnSpec:
        for col in self.all_columns:
            if col.name == name:
                return col
        raise KeyError(f"{self.name}: no column {name!r}")


C = ColumnSpec

PROVENANCE_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    C("source", "str", enum=Source),
    C("run_id", "str"),
    C("model_version", "str"),
    C("confidence", "float32"),
    C("qa_flags", "str", blank_ok=True),
)

POLY: Final = ("Polygon",)
POLY_MULTI: Final = ("Polygon", "MultiPolygon")
LINE: Final = ("LineString",)
POINT: Final = ("Point",)


def _tile(name: str = "tile_id", *, blank_ok: bool = False) -> ColumnSpec:
    return C(name, "str", id_kind=IdKind.TILE, blank_ok=blank_ok)


def _block(name: str = "vineyard_id", *, blank_ok: bool = False) -> ColumnSpec:
    return C(name, "str", id_kind=IdKind.BLOCK, blank_ok=blank_ok)


def _row(name: str = "row_id", *, nullable: bool = False) -> ColumnSpec:
    return C(name, "str", nullable=nullable, id_kind=IdKind.ROW)


def _f32(*names: str) -> tuple[ColumnSpec, ...]:
    return tuple(C(n, "float32") for n in names)


def _f64(*names: str) -> tuple[ColumnSpec, ...]:
    return tuple(C(n, "float64") for n in names)


def _i32(*names: str) -> tuple[ColumnSpec, ...]:
    return tuple(C(n, "int32") for n in names)


def _input_layer(name: str, geom: tuple[str, ...]) -> LayerSchema:
    cols = (C("fid", "int64"), C("type", "str", nullable=True), C("name", "str", nullable=True),
            C("source", "str", nullable=True))
    return LayerSchema(name, geom, ("fid",), cols, provenance=False)


_STATIC: Final = (
    LayerSchema(
        "tile_index", POLY, ("tile_id",),
        (_tile(), C("file_name", "str"), C("grid_row", "int16"), C("grid_col", "int16"),
         *_f64("x0", "y0", "x1", "y1", "gsd_m"), C("width_px", "int16"), C("height_px", "int16"),
         C("src_zip", "str"), C("path", "str"), C("sha256", "str"), C("file_size", "int64"),
         C("nodata_frac", "float32"), C("valid_area_m2", "float64")),
        provenance=False,
    ),
    LayerSchema("tile_valid", POLY_MULTI, ("tile_id",), (_tile(), C("valid_frac", "float32")),
                provenance=False, allow_empty_geom=True),
    _input_layer("in_passages", POLY_MULTI),
    _input_layer("in_forbidden", POLY_MULTI),
    _input_layer("in_study_area", POLY_MULTI),
    _input_layer("in_start", POINT),
)

_PERCEPTION: Final = (
    LayerSchema(
        "tile_status", POLY, ("tile_id",),
        (_tile(), C("status", "str", enum=TileStatus), C("has_vineyard", "bool"),
         *_i32("n_canopies", "n_row_pieces", "n_interrow_pieces", "n_waste"),
         *_f32("veg_frac", "vineyard_score"), C("upload_zip", "str", nullable=True), C("image_id", "int32"),
         C("review_priority", "int8"), C("issues", "str", blank_ok=True)),
    ),
    LayerSchema(
        "row_candidates", LINE, ("cand_id",),
        (C("cand_id", "str", id_kind=IdKind.ROW_CANDIDATE), _tile(),
         *_f32("angle_deg", "offset_m", "length_m", "support_frac", "width_med_m", "local_spacing_m",
               "vine_score"),
         C("rejected_reason", "str", nullable=True)),
        allow_extra=True,
    ),
    LayerSchema(
        "rows_raw", LINE, ("chain_id",),
        (C("chain_id", "str", id_kind=IdKind.CHAIN), C("member_cand_ids", "str"), C("tile_ids", "str"),
         C("n_tiles", "int16"), C("angle_deg", "float32"), C("extent_m", "float64"),
         *_f32("support_frac", "width_p80_m", "along_duty", "harmonic_frac", "rescued_frac"),
         C("is_curved", "bool"), C("gaps_json", "str", blank_ok=True)),
        allow_extra=True,
    ),
    LayerSchema(
        "rows", LINE, ("row_id",),
        (_row(), _block(), C("row_index", "int16"), *_f64("length_m", "extent_m"), C("n_pieces", "int16"),
         C("tile_ids", "str"), *_f32("angle_deg", "spacing_prev_m", "spacing_next_m", "max_gap_m"),
         C("n_gaps_ge5", "int16", nullable=True), C("structure_any", "str", nullable=True, enum=RowStructure)),
        allow_extra=True,
    ),
    LayerSchema(
        "row_pairs", LINE, ("vineyard_id", "row_a", "row_b"),
        (_block(), _row("row_a"), _row("row_b"), C("k_a", "int16"),
         *_f32("spacing_m", "overlap_from_m", "overlap_to_m", "angle_diff_deg")),
        allow_extra=True,
    ),
    LayerSchema("rows_rejected", LINE, ("chain_id",),
                (C("chain_id", "str", id_kind=IdKind.CHAIN), C("reason", "str")), allow_extra=True),
    LayerSchema(
        "blocks", POLY_MULTI, ("vineyard_id",),
        (_block(), *_i32("n_rows", "n_row_pieces"), C("n_canopies", "int32", nullable=True),
         C("n_tiles", "int32"),
         *_f64("row_length_m", "canopy_area_m2", "interrow_area_m2", "outline_area_m2"),
         *_f32("angle_deg", "spacing_med_m"), C("is_garden", "bool")),
        allow_extra=True,
    ),
    LayerSchema(
        "interrows", POLY_MULTI, ("interrow_id",),
        (C("interrow_id", "str", id_kind=IdKind.INTERROW), _block(), _row("row_left_id"), _row("row_right_id"),
         C("area_m2", "float64"), C("width_mean_m", "float32"), C("length_m", "float64")),
        allow_extra=True,
    ),
)

_INTERROW_PIECE_COLUMNS: Final = (
    C("piece_id", "str", id_kind=IdKind.INTERROW_PIECE),
    C("interrow_id", "str", nullable=True, id_kind=IdKind.INTERROW),
    _block(), _tile(), _row("row_left_id", nullable=True), _row("row_right_id", nullable=True),
    C("interrow_cover", "str", enum=InterrowCover), *_f32("veg_frac", "shadow_frac"),
    C("area_m2", "float64"), C("width_mean_m", "float32"), C("n_notches", "int8"),
)

_WASTE_BOX_COLUMNS: Final = (
    _tile(), _block(blank_ok=True), C("dist_block_m", "float32"),
    *_f32("px_xtl", "px_ytl", "px_xbr", "px_ybr", "area_m2"),
    C("category", "str"), C("detector", "str"), C("exported", "bool"),
)

_ANNSET: Final = (
    LayerSchema(
        "canopies", POLY, ("canopy_id",),
        (C("canopy_id", "str", id_kind=IdKind.CANOPY), _tile(), _block(), _row(nullable=True),
         C("area_m2", "float64"), C("n_vertices", "int16"), C("along_m", "float32"), C("is_clump", "bool"),
         C("touches_edge", "bool")),
        annset=True,
    ),
    LayerSchema(
        "row_pieces", LINE, ("piece_id",),
        (C("piece_id", "str", id_kind=IdKind.ROW_PIECE), _row(), _block(), _tile(),
         C("row_structure", "str", enum=RowStructure), C("length_m", "float64"), C("max_gap_m", "float32"),
         C("n_vertices", "int16")),
        annset=True,
    ),
    LayerSchema("interrow_pieces", POLY, ("piece_id",), _INTERROW_PIECE_COLUMNS, annset=True),
    LayerSchema("interrow_pieces_linked", POLY, ("piece_id",), _INTERROW_PIECE_COLUMNS, allow_extra=True),
    LayerSchema("waste", POLY, ("waste_id",), (C("waste_id", "str", id_kind=IdKind.WASTE), *_WASTE_BOX_COLUMNS),
                annset=True),
    LayerSchema(
        "waste_candidates", POLY, ("waste_id",),
        (C("waste_id", "str", id_kind=IdKind.WASTE_CANDIDATE), *_WASTE_BOX_COLUMNS,
         C("reject_reason", "str", nullable=True)),
        allow_extra=True,
    ),
)

_POST: Final = (
    LayerSchema(
        "targets", POINT, ("target_id",),
        (C("target_id", "str", id_kind=IdKind.TARGET), C("kind", "str", enum=TargetKind),
         _block(blank_ok=True), _tile(), _row(nullable=True),
         C("interrow_id", "str", nullable=True, id_kind=IdKind.INTERROW),
         C("waste_id", "str", nullable=True, id_kind=IdKind.WASTE), *_f64("x", "y"),
         C("gap_length_m", "float32"), C("priority", "int8"), C("reachable", "bool"),
         C("reach_note", "str", blank_ok=True), C("snap_dist_m", "float32")),
        allow_extra=True,
    ),
    LayerSchema(
        "target_extents", LINE, ("target_id",),
        (C("target_id", "str", id_kind=IdKind.TARGET), C("kind", "str", enum=TargetKind), C("length_m", "float64")),
    ),
    LayerSchema(
        "target_visits", POINT, ("target_id",),
        (C("target_id", "str", id_kind=IdKind.TARGET), C("reachable_final", "bool"),
         C("reach_note", "str", blank_ok=True), C("n_candidates", "int32"), C("covered", "bool"),
         C("visit_dist_m", "float32"), C("route_role", "str")),
        allow_empty_geom=True,
    ),
    LayerSchema(
        "cross_paths", POLY, ("path_id",),
        (C("path_id", "str"), _block(), C("n_rows", "int32"), *_f64("width_m", "length_m", "residual_m"),
         C("row_ids", "str")),
    ),
    LayerSchema(
        "cross_path_lines", LINE, ("path_id",),
        (C("path_id", "str"), _block(), C("length_m", "float64")),
    ),
    LayerSchema(
        "passable_parts", POLY_MULTI, ("part_id",),
        (C("part_id", "str"), C("kind", "str"), C("ref_id", "str", blank_ok=True), C("area_m2", "float64"),
         C("erosion_m", "float32")),
    ),
    LayerSchema(
        "passable_domain", POLY_MULTI, ("domain_id",),
        (C("domain_id", "str"), C("area_m2", "float64"), C("erosion_m", "float32"), C("n_components", "int32")),
    ),
    LayerSchema("walk_nodes", POINT, ("node_id",),
                (C("node_id", "int64"), C("kind", "str"), C("ref_id", "str", blank_ok=True))),
    LayerSchema(
        "walk_edges", LINE, ("edge_id",),
        (C("edge_id", "int64"), C("u", "int64"), C("v", "int64"), C("length_m", "float64"),
         C("kind", "str", enum=EdgeKind), C("ref_id", "str", blank_ok=True), C("inside_frac", "float32"),
         C("cost", "float64")),
        line_rules=False,
    ),
    LayerSchema(
        "route", LINE, ("route_id",),
        (C("route_id", "str"), C("length_m", "float64"), *_i32("n_targets", "n_visited_est"),
         *_f32("coverage_est", "outside_frac_est", "closure_m", "walking_time_min"), C("solver", "str"),
         C("solve_time_s", "float32")),
    ),
    LayerSchema(
        "route_stops", POINT, ("route_id", "seq"),
        (C("route_id", "str"), C("seq", "int32"), C("target_id", "str", nullable=True, id_kind=IdKind.TARGET),
         *_f64("cum_dist_m", "leg_m")),
    ),
    LayerSchema(
        "qa_issues", POINT, ("issue_id",),
        (C("issue_id", "str", id_kind=IdKind.ISSUE), C("severity", "str", enum=Severity), C("code", "str"),
         _tile(blank_ok=True), C("object_id", "str", blank_ok=True), C("message", "str", blank_ok=True)),
        allow_empty_geom=True,
    ),
)

LAYER_SCHEMAS: Final[Mapping[str, LayerSchema]] = MappingProxyType(
    {s.name: s for s in _STATIC + _PERCEPTION + _ANNSET + _POST}
)
