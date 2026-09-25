from vineyard.contracts import CONTRACT_VERSION
from vineyard.contracts.enums import (
    ANNSET_LAYER_OF_LABEL,
    LABEL_ATTRIBUTES,
    LABEL_GEOMETRY,
    QA_CODES,
    EdgeKind,
    FusionVariant,
    GeomKind,
    InterrowCover,
    Label,
    RowStructure,
    Severity,
    Source,
    TargetKind,
    TileStatus,
)


def _values(enum_cls: type) -> list[str]:
    return [m.value for m in enum_cls]


def test_contract_version() -> None:
    assert CONTRACT_VERSION == "1.1"


def test_enum_values_exact_lowercase() -> None:
    assert _values(Label) == ["vineyard", "waste", "row", "interrow_area"]
    assert _values(RowStructure) == ["regular", "disrupted", "unassessable"]
    assert _values(InterrowCover) == ["bare_soil", "vegetation", "mixed", "unassessable"]
    assert _values(Source) == ["model", "marcaj", "reference"]
    assert _values(TargetKind) == [
        "row_gap", "row_end_short", "missing_row", "missing_plant", "sparse", "waste", "other",
    ]
    assert _values(TileStatus) == ["ok", "empty_nodata", "no_vineyard", "failed"]
    assert _values(Severity) == ["error", "warning", "info"]
    assert _values(EdgeKind) == ["interrow_centerline", "passage_centerline", "connector", "target_spur"]
    assert _values(GeomKind) == ["polygon", "polyline", "box"]


def test_fusion_variants() -> None:
    assert _values(FusionVariant) == ["A", "B", "C", "D", "E"]
    assert FusionVariant("C") is FusionVariant.C


def test_enums_are_str() -> None:
    assert Label.ROW == "row"
    assert f"{RowStructure.DISRUPTED}" == "disrupted"


def test_label_geometry_and_layers() -> None:
    assert LABEL_GEOMETRY[Label.VINEYARD] is GeomKind.POLYGON
    assert LABEL_GEOMETRY[Label.WASTE] is GeomKind.BOX
    assert LABEL_GEOMETRY[Label.ROW] is GeomKind.POLYLINE
    assert LABEL_GEOMETRY[Label.INTERROW_AREA] is GeomKind.POLYGON
    assert dict(ANNSET_LAYER_OF_LABEL) == {
        Label.VINEYARD: "canopies",
        Label.ROW: "row_pieces",
        Label.INTERROW_AREA: "interrow_pieces",
        Label.WASTE: "waste",
    }


def test_label_attributes_ordered() -> None:
    assert LABEL_ATTRIBUTES[Label.VINEYARD] == ("vineyard_id",)
    assert LABEL_ATTRIBUTES[Label.WASTE] == ("vineyard_id",)
    assert LABEL_ATTRIBUTES[Label.ROW] == ("vineyard_id", "row_id", "row_structure")
    assert LABEL_ATTRIBUTES[Label.INTERROW_AREA] == ("vineyard_id", "interrow_cover")


def test_qa_codes_cover_all_designs() -> None:
    contract = {
        "missing_attr", "bad_enum", "id_case_collision", "row_multi_block", "row_piece_misaligned",
        "row_missing_in_tile", "dup_row_in_tile", "canopy_interrow_overlap", "canopy_on_non_vineyard_tile",
        "waste_block_unknown", "blocks_should_merge", "block_split", "structure_borderline",
        "cover_borderline", "empty_tile_confirm", "tile_failed",
    }
    extra = {
        "invalid_after_rounding", "row_interpolated", "missing_row_suspect", "row_offlattice",
        "row_rescued_low_snr", "orchard_rejected", "tree_removed", "tree_hole", "otsu_fallback",
        "texture_fallback", "transverse_band_cut", "curved_row", "override_applied", "override_unmatched",
        "interrow_unlinked", "enum_normalized", "id_whitespace", "domain_disconnected", "target_unreachable",
        "connector_outside", "block_too_few_rows", "images_missing", "geom_repaired", "confirm_unmatched",
    }
    assert contract | extra <= QA_CODES
    assert all(code == code.lower() and " " not in code for code in QA_CODES)
