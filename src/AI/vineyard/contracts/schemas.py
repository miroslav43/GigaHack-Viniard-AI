"""Layer validation and coercion at every stage boundary (contract §2.7, v1.1 relaxations).

Geometry, CRS, primary key, column presence and dtypes are always strict. ID regexes and enum
values are strict only for rows whose `source` is `model` (or when `strict_ids=True`); marcaj and
reference values are kept raw and turned into qa warnings by the caller.
"""

from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pandas.api import types as ptypes

from vineyard.contracts.enums import Source
from vineyard.contracts.ids import ID_PATTERNS
from vineyard.contracts.schema_defs import (
    GEOMETRY_COLUMN,
    LAYER_SCHEMAS,
    MIN_LINE_LENGTH_M,
    PROVENANCE_COLUMNS,
    ColumnKind,
    ColumnSpec,
    LayerSchema,
)
from vineyard.errors import SchemaError
from vineyard.geo.tiling import CRS_EPSG

STR_DTYPE = pd.StringDtype(na_value=np.nan)
_NUMPY_KINDS: dict[ColumnKind, type[np.generic]] = {
    "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
    "float32": np.float32, "float64": np.float64, "bool": np.bool_,
}
_NULLABLE_EXT: dict[ColumnKind, str] = {
    "int8": "Int8", "int16": "Int16", "int32": "Int32", "int64": "Int64", "bool": "boolean",
}
_NUMERIC_INFERRED = frozenset({"integer", "floating", "mixed-integer-float", "empty", "boolean"})
_MAX_EXAMPLES = 5

__all__ = [
    "LAYER_SCHEMAS",
    "PROVENANCE_COLUMNS",
    "STR_DTYPE",
    "ColumnSpec",
    "LayerSchema",
    "coerce_layer",
    "column_dtype",
    "empty_layer",
    "get_schema",
    "validate_layer",
]


def get_schema(name: str) -> LayerSchema:
    try:
        return LAYER_SCHEMAS[name]
    except KeyError:
        raise SchemaError("unknown layer", layer=name, known=sorted(LAYER_SCHEMAS)) from None


def column_dtype(col: ColumnSpec) -> object:
    """The exact pandas/numpy dtype a coerced column has."""
    if col.kind == "str":
        return STR_DTYPE
    if col.nullable and col.kind in _NULLABLE_EXT:
        return pd.api.types.pandas_dtype(_NULLABLE_EXT[col.kind])
    return np.dtype(_NUMPY_KINDS[col.kind])


def _examples(values: Sequence[object] | pd.Series | np.ndarray) -> list[object]:
    return list(pd.Series(values).head(_MAX_EXAMPLES))


# ---- coercion -------------------------------------------------------------------------------


def _coerce_str(series: pd.Series, col: ColumnSpec, layer: str) -> pd.api.extensions.ExtensionArray:
    if series.dtype == STR_DTYPE:
        return series.array.copy()
    values = series.to_numpy(dtype=object)
    missing = pd.isna(values)
    bad = [v for v, m in zip(values, missing, strict=True) if not m and not isinstance(v, str)]
    if bad:
        raise SchemaError("non-string values in str column", layer=layer, column=col.name,
                          examples=bad[:_MAX_EXAMPLES])
    clean = [None if m else str(v) for v, m in zip(values, missing, strict=True)]
    return pd.array(clean, dtype=STR_DTYPE)


def _as_float64(series: pd.Series, col: ColumnSpec, layer: str) -> np.ndarray:
    if ptypes.infer_dtype(series, skipna=True) not in _NUMERIC_INFERRED:
        raise SchemaError("non-numeric values in numeric column", layer=layer, column=col.name,
                          examples=_examples(series))
    return series.to_numpy(dtype=np.float64, na_value=np.nan)


def _coerce_int(series: pd.Series, col: ColumnSpec, layer: str) -> object:
    target = column_dtype(col)
    if ptypes.is_integer_dtype(series.dtype) and not series.isna().any():
        values = series.to_numpy(dtype=np.int64)
    else:
        floats = _as_float64(series, col, layer)
        missing = np.isnan(floats)
        if missing.any() and not col.nullable:
            raise SchemaError("nulls in non-nullable int column", layer=layer, column=col.name)
        if np.any(floats[~missing] != np.round(floats[~missing])):
            raise SchemaError("non-integral values in int column", layer=layer, column=col.name)
        if missing.any():
            return pd.array([None if m else int(v) for v, m in zip(floats, missing, strict=True)], dtype=target)
        values = floats.astype(np.int64)
    info = np.iinfo(_NUMPY_KINDS[col.kind])
    if values.size and (values.min() < info.min or values.max() > info.max):
        raise SchemaError("int values out of range", layer=layer, column=col.name, kind=col.kind)
    return pd.array(values.astype(_NUMPY_KINDS[col.kind]), dtype=target)


def _coerce_bool(series: pd.Series, col: ColumnSpec, layer: str) -> object:
    floats = _as_float64(series, col, layer)
    missing = np.isnan(floats)
    if not np.isin(floats[~missing], (0.0, 1.0)).all():
        raise SchemaError("non-boolean values in bool column", layer=layer, column=col.name)
    if missing.any() and not col.nullable:
        raise SchemaError("nulls in non-nullable bool column", layer=layer, column=col.name)
    values = [None if m else bool(v) for v, m in zip(floats, missing, strict=True)]
    return pd.array(values, dtype=column_dtype(col))


def _coerce_column(series: pd.Series, col: ColumnSpec, layer: str) -> object:
    if col.kind == "str":
        return _coerce_str(series, col, layer)
    if col.kind == "bool":
        return _coerce_bool(series, col, layer)
    if col.kind.startswith("int"):
        return _coerce_int(series, col, layer)
    return _as_float64(series, col, layer).astype(_NUMPY_KINDS[col.kind])


def _null_column(col: ColumnSpec, n: int) -> object:
    if col.kind.startswith("float"):
        return np.full(n, np.nan, dtype=_NUMPY_KINDS[col.kind])
    return pd.array([None] * n, dtype=column_dtype(col))


def _require_geodataframe(gdf: object, layer: str) -> gpd.GeoDataFrame:
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise SchemaError("expected a GeoDataFrame", layer=layer, got=type(gdf).__name__)
    if gdf.active_geometry_name is None or gdf.active_geometry_name not in gdf.columns:
        raise SchemaError("frame has no active geometry column", layer=layer)
    return gdf


def coerce_layer(gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """New frame with exact dtypes and contract column order (extras kept after, geometry last).

    Missing nullable columns are added as nulls; a missing non-nullable column is an error.
    """
    schema = get_schema(name)
    frame = _require_geodataframe(gdf, name)
    geom_name = frame.geometry.name
    data: dict[str, object] = {}
    for col in schema.all_columns:
        if col.name in frame.columns and col.name != geom_name:
            data[col.name] = _coerce_column(frame[col.name], col, name)
        elif col.nullable:
            data[col.name] = _null_column(col, len(frame))
        else:
            raise SchemaError("missing column", layer=name, column=col.name)
    for extra in frame.columns:
        if extra not in data and extra != geom_name:
            data[extra] = frame[extra].reset_index(drop=True).copy()
    index = pd.RangeIndex(len(frame))
    geometry = gpd.GeoSeries(frame.geometry.to_numpy(copy=True), crs=frame.crs, index=index,
                             name=GEOMETRY_COLUMN)
    return gpd.GeoDataFrame(data, index=index, geometry=geometry)


def empty_layer(name: str) -> gpd.GeoDataFrame:
    """Zero-row frame of `name` with exact dtypes and EPSG:32635."""
    schema = get_schema(name)
    data = {col.name: pd.Series(_null_column(col, 0)) for col in schema.all_columns}
    geometry = gpd.GeoSeries([], crs=CRS_EPSG, name=GEOMETRY_COLUMN)
    return gpd.GeoDataFrame(data, geometry=geometry)


# ---- validation -----------------------------------------------------------------------------


def _check_frame(gdf: gpd.GeoDataFrame, name: str) -> None:
    _require_geodataframe(gdf, name)
    if gdf.geometry.name != GEOMETRY_COLUMN:
        raise SchemaError("active geometry column must be 'geometry'", layer=name, got=gdf.geometry.name)
    epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
    if epsg != CRS_EPSG:
        raise SchemaError(f"CRS must be EPSG:{CRS_EPSG}", layer=name, got=str(gdf.crs)[:60])


def _dtype_ok(series: pd.Series, col: ColumnSpec) -> bool:
    if col.kind == "str":
        return ptypes.is_string_dtype(series)
    if series.dtype == np.dtype(_NUMPY_KINDS[col.kind]):
        return True
    return col.nullable and col.kind in _NULLABLE_EXT and series.dtype == column_dtype(col)


def _check_columns(gdf: gpd.GeoDataFrame, schema: LayerSchema) -> None:
    names = set(schema.column_names)
    missing = [n for n in schema.column_names if n not in gdf.columns]
    if missing:
        raise SchemaError("missing columns", layer=schema.name, columns=missing)
    extra = [c for c in gdf.columns if c not in names and c != GEOMETRY_COLUMN]
    if extra and not schema.allow_extra:
        raise SchemaError("unexpected columns", layer=schema.name, columns=extra)
    for col in schema.all_columns:
        if not _dtype_ok(gdf[col.name], col):
            raise SchemaError("wrong dtype", layer=schema.name, column=col.name, kind=col.kind,
                              got=str(gdf[col.name].dtype))
        if col.kind.startswith("float") or col.nullable:
            continue
        if gdf[col.name].isna().any():
            raise SchemaError("nulls in non-nullable column", layer=schema.name, column=col.name)


def _strict_mask(gdf: gpd.GeoDataFrame, schema: LayerSchema, strict_ids: bool | None) -> np.ndarray:
    if strict_ids is not None:
        return np.full(len(gdf), strict_ids, dtype=bool)
    if not schema.provenance:
        return np.ones(len(gdf), dtype=bool)
    return (gdf["source"] == Source.MODEL.value).to_numpy(dtype=bool, na_value=False)


def _checked_values(series: pd.Series, col: ColumnSpec, rows: np.ndarray) -> pd.Series:
    mask = rows & series.notna().to_numpy()
    if col.nullable or col.blank_ok:
        mask &= (series != "").to_numpy(dtype=bool, na_value=False)
    return series[mask]


def _raise_bad(schema: LayerSchema, col: ColumnSpec, problem: str, bad: pd.Series) -> None:
    if len(bad):
        raise SchemaError(problem, layer=schema.name, column=col.name, n_bad=len(bad), examples=_examples(bad))


def _check_values(gdf: gpd.GeoDataFrame, schema: LayerSchema, strict: np.ndarray) -> None:
    always = np.ones(len(gdf), dtype=bool)
    for col in schema.all_columns:
        if col.enum is not None:
            rows = always if (schema.provenance and col.name == "source") else strict
            values = _checked_values(gdf[col.name], col, rows)
            allowed = [m.value for m in col.enum]
            _raise_bad(schema, col, f"value not in {col.enum.__name__}", values[~values.isin(allowed)])
        if col.id_kind is not None:
            values = _checked_values(gdf[col.name], col, strict)
            ok = values.str.fullmatch(ID_PATTERNS[col.id_kind]).to_numpy(dtype=bool, na_value=False)
            _raise_bad(schema, col, f"invalid {col.id_kind} id", values[~ok])


def _check_pk(gdf: gpd.GeoDataFrame, schema: LayerSchema) -> None:
    keys = gdf[list(schema.pk)]
    if keys.isna().any().any():
        raise SchemaError("null primary key", layer=schema.name, pk=schema.pk)
    dup = keys[keys.duplicated(keep=False)]
    if len(dup):
        raise SchemaError("duplicate primary key", layer=schema.name, pk=schema.pk,
                          examples=dup.head(_MAX_EXAMPLES).to_dict("records"))


def _check_geometry(gdf: gpd.GeoDataFrame, schema: LayerSchema) -> None:
    geoms = gdf.geometry.to_numpy()
    blank = shapely.is_missing(geoms) | shapely.is_empty(geoms)
    if blank.any() and not schema.allow_empty_geom:
        raise SchemaError("null or empty geometry", layer=schema.name, n_bad=int(blank.sum()))
    present = geoms[~blank]
    types = shapely.get_type_id(present)
    kinds = np.array([g.geom_type for g in present], dtype=object) if len(present) else np.array([])
    wrong = ~np.isin(kinds, schema.geom_types)
    if wrong.any():
        raise SchemaError("wrong geometry type", layer=schema.name, expected=schema.geom_types,
                          examples=sorted(set(kinds[wrong]))[:_MAX_EXAMPLES])
    if shapely.has_z(present).any():
        raise SchemaError("3D geometry", layer=schema.name)
    invalid = ~shapely.is_valid(present)
    if invalid.any():
        reasons = shapely.is_valid_reason(present[invalid])
        raise SchemaError("invalid geometry", layer=schema.name, n_bad=int(invalid.sum()),
                          examples=list(reasons[:_MAX_EXAMPLES]))
    lines = types == shapely.GeometryType.LINESTRING
    if schema.line_rules and lines.any():
        short = shapely.length(present[lines]) < MIN_LINE_LENGTH_M
        if short.any():
            raise SchemaError(f"LineString shorter than {MIN_LINE_LENGTH_M} m", layer=schema.name,
                              n_bad=int(short.sum()))


def validate_layer(gdf: gpd.GeoDataFrame, name: str, *, strict_ids: bool | None = None) -> None:
    """Raise SchemaError on the first contract violation.

    strict_ids: None = per row (strict where source == model), True/False = force for all rows.
    """
    schema = get_schema(name)
    _check_frame(gdf, name)
    _check_columns(gdf, schema)
    _check_values(gdf, schema, _strict_mask(gdf, schema, strict_ids))
    _check_pk(gdf, schema)
    _check_geometry(gdf, schema)
