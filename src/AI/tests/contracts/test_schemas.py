from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from vineyard.contracts.enums import Source
from vineyard.contracts.schema_defs import LAYER_SCHEMAS as DEFS_SCHEMAS
from vineyard.contracts.schemas import (
    LAYER_SCHEMAS,
    PROVENANCE_COLUMNS,
    STR_DTYPE,
    coerce_layer,
    column_dtype,
    empty_layer,
    get_schema,
    validate_layer,
)
from vineyard.errors import SchemaError

TILE = "siret3_r021_c012"

EXPECTED_LAYERS = {
    "tile_index", "tile_valid", "tile_status", "in_passages", "in_forbidden", "in_study_area", "in_start",
    "row_candidates", "rows_raw", "rows", "row_pairs", "rows_rejected", "blocks", "canopies", "row_pieces",
    "interrows", "interrow_pieces", "interrow_pieces_linked", "waste_candidates", "waste", "targets",
    "target_extents", "target_visits", "cross_paths", "cross_path_lines", "passable_parts", "passable_domain", "walk_nodes", "walk_edges",
    "route", "route_stops", "qa_issues", "farms", "farm_blocks", "roads",
}


def _prov(n: int, source: str = "model") -> dict[str, list[Any]]:
    return {
        "source": [source] * n,
        "run_id": ["r1"] * n,
        "model_version": ["pipe@0.1"] * n,
        "confidence": [0.9] * n,
        "qa_flags": [""] * n,
    }


def _row_pieces(src: str = "model", **overrides: list[Any]) -> gpd.GeoDataFrame:
    data: dict[str, list[Any]] = {
        "piece_id": [f"V01-R001@{TILE}", f"V01-R002@{TILE}"],
        "row_id": ["V01-R001", "V01-R002"],
        "vineyard_id": ["V01", "V01"],
        "tile_id": [TILE, TILE],
        "row_structure": ["regular", "disrupted"],
        "length_m": [10.0, 12.5],
        "max_gap_m": [1.0, float("nan")],
        "n_vertices": [2, 2],
        **_prov(2, src),
    }
    default_geoms = [LineString([(0, 0), (10, 0)]), LineString([(0, 3), (12.5, 3)])]
    geoms = overrides.get("geometry", default_geoms)
    data.update({k: v for k, v in overrides.items() if k != "geometry"})
    return coerce_layer(gpd.GeoDataFrame(data, geometry=geoms, crs=32635), "row_pieces")


def _canopies(source: str = "model") -> gpd.GeoDataFrame:
    data = {
        "canopy_id": [f"{TILE}:C0001"],
        "tile_id": [TILE],
        "vineyard_id": ["V01"],
        "row_id": [None],
        "area_m2": [1.0],
        "n_vertices": [4],
        "along_m": [1.0],
        "is_clump": [False],
        "touches_edge": [True],
        **_prov(1, source),
    }
    return gpd.GeoDataFrame(data, geometry=[box(0, 0, 1, 1)], crs=32635)


def test_every_layer_is_defined() -> None:
    assert set(LAYER_SCHEMAS) == EXPECTED_LAYERS
    assert DEFS_SCHEMAS is LAYER_SCHEMAS
    annset = {n for n, s in LAYER_SCHEMAS.items() if s.annset}
    assert annset == {"canopies", "row_pieces", "interrow_pieces", "waste"}


def test_provenance_columns() -> None:
    assert [c.name for c in PROVENANCE_COLUMNS] == ["source", "run_id", "model_version", "confidence", "qa_flags"]
    assert column_dtype(PROVENANCE_COLUMNS[3]) == np.dtype(np.float32)
    assert LAYER_SCHEMAS["canopies"].column_names[-5:] == tuple(c.name for c in PROVENANCE_COLUMNS)
    assert "source" not in LAYER_SCHEMAS["tile_index"].column_names


def test_schema_column_lookup() -> None:
    schema = get_schema("row_pieces")
    assert schema.column("row_structure").kind == "str"
    with pytest.raises(KeyError):
        schema.column("nope")


@pytest.mark.parametrize("name", sorted(EXPECTED_LAYERS))
def test_empty_layer_validates(name: str) -> None:
    gdf = empty_layer(name)
    assert list(gdf.columns) == [*LAYER_SCHEMAS[name].column_names, "geometry"]
    assert gdf.crs.to_epsg() == 32635
    validate_layer(gdf, name)


def test_unknown_layer() -> None:
    with pytest.raises(SchemaError, match="unknown layer"):
        empty_layer("nope")


def test_valid_row_pieces_pass() -> None:
    validate_layer(_row_pieces(), "row_pieces")


def test_missing_column_raises() -> None:
    gdf = _row_pieces().drop(columns=["row_structure"])
    with pytest.raises(SchemaError, match="missing columns") as exc:
        validate_layer(gdf, "row_pieces")
    assert "row_structure" in str(exc.value)


def test_bad_enum_strict_for_model_relaxed_for_reference() -> None:
    with pytest.raises(SchemaError, match="RowStructure"):
        validate_layer(_row_pieces(row_structure=["Regular", "regular"]), "row_pieces")
    validate_layer(_row_pieces("reference", row_structure=["Regular", "regular"]), "row_pieces")
    validate_layer(_row_pieces("marcaj", row_structure=["Regular", "regular"]), "row_pieces")


def test_row_id_regex_strict_only_for_model() -> None:
    ids = {"row_id": ["V01-R01", "V01-R02"], "piece_id": [f"V01-R01@{TILE}", f"V01-R02@{TILE}"]}
    with pytest.raises(SchemaError, match="invalid"):
        validate_layer(_row_pieces(**ids), "row_pieces")
    validate_layer(_row_pieces("reference", **ids), "row_pieces")
    with pytest.raises(SchemaError):
        validate_layer(_row_pieces("reference", **ids), "row_pieces", strict_ids=True)
    validate_layer(_row_pieces(**ids), "row_pieces", strict_ids=False)


def test_bad_source_always_rejected() -> None:
    gdf = _row_pieces("reference", source=["reference", "Model"])
    with pytest.raises(SchemaError, match="Source"):
        validate_layer(gdf, "row_pieces")


def test_duplicate_pk_raises() -> None:
    gdf = _row_pieces(piece_id=[f"V01-R001@{TILE}", f"V01-R001@{TILE}"])
    with pytest.raises(SchemaError, match="duplicate primary key"):
        validate_layer(gdf, "row_pieces")


def test_wrong_crs_raises() -> None:
    gdf = _row_pieces().set_crs(4326, allow_override=True)
    with pytest.raises(SchemaError, match="EPSG:32635"):
        validate_layer(gdf, "row_pieces")
    with pytest.raises(SchemaError, match="EPSG:32635"):
        validate_layer(_row_pieces().set_crs(None, allow_override=True), "row_pieces")


def test_bowtie_polygon_raises() -> None:
    gdf = _canopies()
    gdf = gdf.set_geometry([Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])], crs=32635)
    with pytest.raises(SchemaError, match="invalid geometry"):
        validate_layer(coerce_layer(gdf, "canopies"), "canopies")


def test_short_line_raises() -> None:
    gdf = _row_pieces(geometry=[LineString([(0, 0), (0.04, 0)]), LineString([(0, 3), (12.5, 3)])])
    with pytest.raises(SchemaError, match="shorter"):
        validate_layer(gdf, "row_pieces")


def test_wrong_geometry_type_and_empty() -> None:
    gdf = _canopies().set_geometry([Point(0, 0)], crs=32635)
    with pytest.raises(SchemaError, match="geometry type"):
        validate_layer(coerce_layer(gdf, "canopies"), "canopies")
    gdf = _canopies().set_geometry([Polygon()], crs=32635)
    with pytest.raises(SchemaError, match="empty geometry"):
        validate_layer(coerce_layer(gdf, "canopies"), "canopies")


def test_3d_geometry_raises() -> None:
    gdf = _canopies().set_geometry([Polygon([(0, 0, 1), (1, 0, 1), (1, 1, 1)])], crs=32635)
    with pytest.raises(SchemaError, match="3D"):
        validate_layer(coerce_layer(gdf, "canopies"), "canopies")


def test_nulls_in_non_nullable_column() -> None:
    gdf = coerce_layer(_canopies(), "canopies")
    gdf = gdf.assign(vineyard_id=pd.array([None], dtype=STR_DTYPE))
    with pytest.raises(SchemaError, match="nulls"):
        validate_layer(gdf, "canopies")


def test_wrong_dtype_raises_and_coerce_fixes() -> None:
    raw = _canopies()
    with pytest.raises(SchemaError, match="wrong dtype"):
        validate_layer(raw, "canopies")
    fixed = coerce_layer(raw, "canopies")
    assert fixed["n_vertices"].dtype == np.int16
    assert fixed["area_m2"].dtype == np.float64
    assert fixed["confidence"].dtype == np.float32
    assert fixed["canopy_id"].dtype == STR_DTYPE
    assert fixed["row_id"].isna().all()
    validate_layer(fixed, "canopies")
    assert raw["n_vertices"].dtype == np.int64  # input not mutated


def test_coerce_column_order_and_extras() -> None:
    raw = _canopies()
    shuffled = raw[["geometry", *reversed([c for c in raw.columns if c != "geometry"])]]
    shuffled = shuffled.assign(debug=[1])
    out = coerce_layer(shuffled, "canopies")
    assert list(out.columns) == [*LAYER_SCHEMAS["canopies"].column_names, "debug", "geometry"]
    with pytest.raises(SchemaError, match="unexpected columns"):
        validate_layer(out, "canopies")


def test_coerce_adds_missing_nullable_and_rejects_missing_required() -> None:
    out = coerce_layer(_canopies().drop(columns=["row_id"]), "canopies")
    assert out["row_id"].isna().all()
    with pytest.raises(SchemaError, match="missing column"):
        coerce_layer(_canopies().drop(columns=["tile_id"]), "canopies")


def test_coerce_rejects_bad_values() -> None:
    with pytest.raises(SchemaError, match="out of range"):
        coerce_layer(_canopies().assign(n_vertices=[40000]), "canopies")
    with pytest.raises(SchemaError, match="non-integral"):
        coerce_layer(_canopies().assign(n_vertices=[2.5]), "canopies")
    with pytest.raises(SchemaError, match="nulls in non-nullable int"):
        coerce_layer(_canopies().assign(n_vertices=[None]), "canopies")
    with pytest.raises(SchemaError, match="non-numeric"):
        coerce_layer(_canopies().assign(area_m2=["big"]), "canopies")
    with pytest.raises(SchemaError, match="non-string"):
        coerce_layer(_canopies().assign(tile_id=[5]), "canopies")
    with pytest.raises(SchemaError, match="non-boolean"):
        coerce_layer(_canopies().assign(is_clump=[2]), "canopies")
    with pytest.raises(SchemaError, match="nulls in non-nullable bool"):
        coerce_layer(_canopies().assign(is_clump=[None]), "canopies")
    with pytest.raises(SchemaError, match="GeoDataFrame"):
        coerce_layer(pd.DataFrame({"a": [1]}), "canopies")  # type: ignore[arg-type]


def test_coerce_nullable_int_and_enum_members() -> None:
    raw = _row_pieces(Source.MODEL)
    assert raw["source"].iloc[0] == "model"
    rows = {
        "row_id": ["V01-R001"], "vineyard_id": ["V01"], "row_index": [1], "length_m": [5.0], "extent_m": [5.0],
        "n_pieces": [1], "tile_ids": [TILE], "angle_deg": [10.0], "spacing_prev_m": [np.nan],
        "spacing_next_m": [2.5], "max_gap_m": [np.nan], "n_gaps_ge5": [None], "structure_any": [""],
        **_prov(1),
    }
    out = coerce_layer(gpd.GeoDataFrame(rows, geometry=[LineString([(0, 0), (5, 0)])], crs=32635), "rows")
    assert str(out["n_gaps_ge5"].dtype) == "Int16"
    validate_layer(out, "rows")


def test_waste_blank_vineyard_and_extra_columns() -> None:
    data = {
        "waste_id": [f"{TILE}:W0001"], "tile_id": [TILE], "vineyard_id": [""], "dist_block_m": [12.0],
        "px_xtl": [1.0], "px_ytl": [1.0], "px_xbr": [5.0], "px_ybr": [5.0], "area_m2": [0.01],
        "category": ["unknown"], "detector": ["rule"], "exported": [False], "reject_reason": [None],
        "rank_score": [0.3], **_prov(1),
    }
    gdf = coerce_layer(gpd.GeoDataFrame(data, geometry=[box(0, 0, 0.1, 0.1)], crs=32635), "waste_candidates")
    validate_layer(gdf, "waste_candidates")
    bad = coerce_layer(gdf.assign(waste_id=["W0001"]), "waste_candidates")
    with pytest.raises(SchemaError, match="waste_candidate"):
        validate_layer(bad, "waste_candidates")


def test_qa_issues_allow_empty_geometry() -> None:
    data = {
        "issue_id": ["Q00001"], "severity": ["warning"], "code": ["bad_enum"], "tile_id": [""],
        "object_id": [""], "message": ["x"], **_prov(1),
    }
    gdf = coerce_layer(gpd.GeoDataFrame(data, geometry=[Point()], crs=32635), "qa_issues")
    validate_layer(gdf, "qa_issues")


def test_static_layer_without_provenance() -> None:
    data = {"fid": [0], "type": ["passage"], "name": [None], "source": ["osm"]}
    gdf = coerce_layer(gpd.GeoDataFrame(data, geometry=[box(0, 0, 5, 5)], crs=32635), "in_passages")
    validate_layer(gdf, "in_passages")
    tv = coerce_layer(gpd.GeoDataFrame({"tile_id": ["bad"], "valid_frac": [1.0]},
                                       geometry=[box(0, 0, 1, 1)], crs=32635), "tile_valid")
    with pytest.raises(SchemaError, match="tile id"):
        validate_layer(tv, "tile_valid")


def test_geometry_column_name_enforced() -> None:
    gdf = _row_pieces().rename_geometry("geom")
    with pytest.raises(SchemaError, match="'geometry'"):
        validate_layer(gdf, "row_pieces")
    validate_layer(coerce_layer(gdf, "row_pieces"), "row_pieces")


def test_parquet_round_trip_validates(tmp_path: Path) -> None:
    gdf = _row_pieces()
    path = tmp_path / "row_pieces.parquet"
    gdf.to_parquet(path)
    back = gpd.read_parquet(path)
    validate_layer(back, "row_pieces")
    assert list(back.columns) == list(gdf.columns)
    assert back.dtypes.to_dict() == gdf.dtypes.to_dict()
