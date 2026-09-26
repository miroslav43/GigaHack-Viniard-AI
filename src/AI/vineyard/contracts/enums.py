"""Contract enumerations (contract §2.3 as amended in v1.1) and label tables.

Values are written exactly as they appear in data files: lowercase, except the
fusion variants, which are the ablation letters A-E.
"""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class Label(StrEnum):
    VINEYARD = "vineyard"
    WASTE = "waste"
    ROW = "row"
    INTERROW_AREA = "interrow_area"


class RowStructure(StrEnum):
    REGULAR = "regular"
    DISRUPTED = "disrupted"
    UNASSESSABLE = "unassessable"


class InterrowCover(StrEnum):
    BARE_SOIL = "bare_soil"
    VEGETATION = "vegetation"
    MIXED = "mixed"
    UNASSESSABLE = "unassessable"


class Source(StrEnum):
    MODEL = "model"
    MARCAJ = "marcaj"
    REFERENCE = "reference"


class TargetKind(StrEnum):
    ROW_GAP = "row_gap"
    ROW_END_SHORT = "row_end_short"
    MISSING_ROW = "missing_row"
    MISSING_PLANT = "missing_plant"
    SPARSE = "sparse"
    WASTE = "waste"
    OTHER = "other"


class TileStatus(StrEnum):
    OK = "ok"
    EMPTY_NODATA = "empty_nodata"
    NO_VINEYARD = "no_vineyard"
    FAILED = "failed"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class EdgeKind(StrEnum):
    INTERROW_CENTERLINE = "interrow_centerline"
    PASSAGE_CENTERLINE = "passage_centerline"
    CONNECTOR = "connector"
    TARGET_SPUR = "target_spur"
    CROSS_PATH = "cross_path"


class GeomKind(StrEnum):
    POLYGON = "polygon"
    POLYLINE = "polyline"
    BOX = "box"


class FusionVariant(StrEnum):
    """Canopy mask variants: A=veg∩corridor (classic), B=nn∩corridor, C=(veg∧nn)∩corridor,
    D=(veg∨nn)∩corridor, E=nn without corridor (ablation only)."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


LABEL_GEOMETRY: Final[Mapping[Label, GeomKind]] = MappingProxyType(
    {
        Label.VINEYARD: GeomKind.POLYGON,
        Label.WASTE: GeomKind.BOX,
        Label.ROW: GeomKind.POLYLINE,
        Label.INTERROW_AREA: GeomKind.POLYGON,
    }
)

# Attribute order exactly as in the example annotations.xml <meta> block.
LABEL_ATTRIBUTES: Final[Mapping[Label, tuple[str, ...]]] = MappingProxyType(
    {
        Label.VINEYARD: ("vineyard_id",),
        Label.WASTE: ("vineyard_id",),
        Label.ROW: ("vineyard_id", "row_id", "row_structure"),
        Label.INTERROW_AREA: ("vineyard_id", "interrow_cover"),
    }
)

ANNSET_LAYER_OF_LABEL: Final[Mapping[Label, str]] = MappingProxyType(
    {
        Label.VINEYARD: "canopies",
        Label.ROW: "row_pieces",
        Label.INTERROW_AREA: "interrow_pieces",
        Label.WASTE: "waste",
    }
)

_CONTRACT_QA_CODES: Final = (
    "missing_attr", "bad_enum", "id_case_collision", "row_multi_block", "row_piece_misaligned",
    "row_missing_in_tile", "dup_row_in_tile", "canopy_interrow_overlap", "canopy_on_non_vineyard_tile",
    "waste_block_unknown", "blocks_should_merge", "block_split", "structure_borderline",
    "cover_borderline", "empty_tile_confirm", "tile_failed",
)
_FOUNDATION_QA_CODES: Final = ("invalid_after_rounding", "schema_violation", "geom_dropped")
_PERCEPTION_QA_CODES: Final = (
    "row_interpolated", "missing_row_suspect", "row_offlattice", "row_rescued_low_snr",
    "orchard_rejected", "tree_removed", "tree_hole", "otsu_fallback", "texture_fallback",
    "transverse_band_cut", "curved_row", "override_applied", "override_unmatched", "low_snr",
)
_POST_QA_CODES: Final = (
    "interrow_unlinked", "enum_normalized", "id_whitespace", "domain_disconnected",
    "target_unreachable", "connector_outside", "block_too_few_rows", "images_missing", "geom_repaired",
    "area_union_sum_mismatch",
)
_WASTE_NN_QA_CODES: Final = ("confirm_unmatched", "nn_fallback_classic")

QA_CODES: Final[frozenset[str]] = frozenset(
    _CONTRACT_QA_CODES + _FOUNDATION_QA_CODES + _PERCEPTION_QA_CODES + _POST_QA_CODES + _WASTE_NN_QA_CODES
)

__all__ = [
    "ANNSET_LAYER_OF_LABEL",
    "LABEL_ATTRIBUTES",
    "LABEL_GEOMETRY",
    "QA_CODES",
    "EdgeKind",
    "FusionVariant",
    "GeomKind",
    "InterrowCover",
    "Label",
    "RowStructure",
    "Severity",
    "Source",
    "TargetKind",
    "TileStatus",
]
