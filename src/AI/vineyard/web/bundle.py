"""Web data bundle writer: `<web.out_dir>/surveys/<survey_id>/pipeline/` (src/Web/CLAUDE.md §6.1-6.4).

EPSG:32635 GeoJSON FeatureCollections with the `crs` member, canopies as GeoJSONSeq, measurements.csv and
manifest.json. Everything is computed before the first file is written; each file is written atomically,
manifest.json last. Bundle files this build does not produce (e.g. route.geojson without a route) are removed,
anything else in the directory is left alone. Output bytes depend only on the inputs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, mapping

from vineyard.annset.model import AnnSet
from vineyard.errors import ConfigError, SchemaError
from vineyard.geo.ops import make_valid_polygonal, orient_ccw
from vineyard.geo.vector_io import round_geometry, write_geojson
from vineyard.measure.csv_format import (
    check_measurements_bytes,
    decimals_for,
    describe_failed,
    format_measurements_csv,
)
from vineyard.measure.measurements import MeasureInputs, compute_measurements
from vineyard.perception.block_overlap import BlockOverlapParams, remove_block_overlap
from vineyard.pipeline.atomic import atomic_write_bytes, atomic_write_json, atomic_write_text
from vineyard.web.manifest import SurveyInfo, build_manifest, manifest_stage, survey_info
from vineyard.web.objects_export import (
    RouteInfo,
    canopy_features,
    interrow_features,
    route_features,
    target_features,
    waste_features,
)
from vineyard.web.rows_export import features_frame, natural_key, physical_rows, text_or_none

if TYPE_CHECKING:
    from vineyard.config import AppConfig

SURVEYS_DIRNAME: Final = "surveys"
PIPELINE_DIRNAME: Final = "pipeline"
MANIFEST_FILE: Final = "manifest.json"
MEASUREMENTS_FILE: Final = "measurements.csv"
LAYER_FILES: Final[Mapping[str, str]] = MappingProxyType({
    "blocks": "blocks.geojson", "rows": "rows.geojson", "canopies": "canopies.geojsonl",
    "interrows": "interrows.geojson", "waste": "waste.geojson", "targets": "targets.geojson",
    "route": "route.geojson",
})
SEQ_LAYERS: Final = frozenset({"canopies"})
# canopies.geojson is the non-sequence spelling of the same layer: a stale copy would shadow ours.
BUNDLE_FILES: Final = frozenset({*LAYER_FILES.values(), "canopies.geojson", MEASUREMENTS_FILE, MANIFEST_FILE})
BLOCK_COLUMNS: Final = ("vineyard_id", "area_m2")
ROW_LINK_TOL_M: Final = 1.0  # CONFIG-REQUEST: web.row_link_tol_m = 1.0
WASTE_BLOCK_MAX_M: Final = 10.0  # CONFIG-REQUEST: web.waste_block_max_m = 10.0 (contract §6.3)
SEQ_SEPARATORS: Final = (",", ":")  # compact GeoJSONSeq: canopies are the bulk of the bundle
# A share of 0.00467 must not be written as 0.005 (the publish limit): same precision as the route export.
FRACTION_DECIMALS: Final[Mapping[str, int]] = MappingProxyType({"outside_share": 5})


@dataclass(frozen=True)
class WebParams:
    survey: SurveyInfo
    utm_decimals: int
    m_decimals: int
    ha_decimals: int
    sum_tol_m: float
    sum_tol_m2: float
    overlap: BlockOverlapParams  # computed measurements without derive's pieces remove the overlap here
    visit_radius_m: float
    speed_kmh: float
    row_link_tol_m: float
    waste_block_max_m: float
    block_buffer_m: float
    tz: str


def decimals_of(step: float, key: str) -> int:
    """2 for a rounding step of 0.01 (measure.csv_format.decimals_for, reported as a config error)."""
    try:
        return decimals_for(step)
    except ValueError as exc:
        raise ConfigError("rounding step must be a power of ten <= 1", key=key, step=step) from exc


def web_params(cfg: AppConfig) -> WebParams:
    return WebParams(
        survey=survey_info(cfg), utm_decimals=cfg.web.utm_decimals,
        m_decimals=decimals_of(cfg.measure.round_m, "measure.round_m"),
        ha_decimals=decimals_of(cfg.measure.round_ha, "measure.round_ha"),
        sum_tol_m=cfg.publish.sum_check_tol_m, sum_tol_m2=cfg.publish.sum_check_tol_m2,
        overlap=BlockOverlapParams.from_config(cfg), visit_radius_m=cfg.route.visit_radius_m,
        speed_kmh=cfg.route.walking_speed_kmh,
        row_link_tol_m=ROW_LINK_TOL_M, waste_block_max_m=WASTE_BLOCK_MAX_M,
        block_buffer_m=cfg.blocks.outline_buffer_m, tz=cfg.logging.tz,
    )


@dataclass(frozen=True)
class WebInputs:
    """What the bundle is built from; only the AnnSet is required (derive/targets/route layers are optional)."""

    annset: AnnSet
    rows: pd.DataFrame | None = None  # derive `rows` (whole-row max_gap_m)
    blocks: gpd.GeoDataFrame | None = None  # derive `blocks`
    interrows: gpd.GeoDataFrame | None = None  # derive `interrow_pieces_linked`
    targets: gpd.GeoDataFrame | None = None
    stops: pd.DataFrame | None = None  # route `route_stops`
    visits: pd.DataFrame | None = None  # route `target_visits`
    route: RouteInfo | None = None
    measurements_csv: bytes | None = None  # measure's exports/measurements.csv, copied verbatim


@dataclass(frozen=True)
class BundleResult:
    out_dir: Path
    written: tuple[Path, ...]
    removed: tuple[Path, ...]
    counts: Mapping[str, int]


def bundle_dir(web_root: Path, survey_id: str) -> Path:
    return Path(web_root) / SURVEYS_DIRNAME / survey_id / PIPELINE_DIRNAME


# ------------------------------------------------------------------ layers


def _polygonal(geom: Any) -> Any:
    parts = sorted((orient_ccw(p) for p in make_valid_polygonal(geom)),
                   key=lambda p: (-p.area, p.representative_point().coords[0]))
    if not parts:
        raise SchemaError("block outline has no polygonal part")
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _closing(geoms: list[Any], buffer_m: float) -> Any:
    # Buffering the members before the union is ~20x faster than buffering the union (same outline).
    return shapely.union_all(shapely.buffer(geoms, buffer_m)).buffer(-buffer_m)


def _derived_outlines(annset: AnnSet, buffer_m: float, skip: frozenset[str]) -> dict[str, Any]:
    """Closing (buffer +b then -b) of each block's canopies, row pieces and interrow pieces (blocks in `skip`
    are left out)."""
    members: dict[str, list[Any]] = {}
    for frame in (annset.canopies, annset.row_pieces, annset.interrow_pieces):
        for vid, geom in zip(frame["vineyard_id"], frame.geometry, strict=True):
            key = text_or_none(vid)
            if key is not None and key not in skip:
                members.setdefault(key, []).append(geom)
    return {vid: _closing(geoms, buffer_m) for vid, geoms in members.items()}


def block_features(inputs: WebInputs, buffer_m: float) -> gpd.GeoDataFrame:
    """derive's blocks, completed with an outline for every other vineyard_id of the AnnSet (so every
    `vineyard_id` the bundle references exists in blocks.geojson); area_m2 = outline area."""
    blocks = inputs.blocks
    given = {} if blocks is None else {str(v): g for v, g in zip(blocks["vineyard_id"], blocks.geometry, strict=True)}
    outlines = given | _derived_outlines(inputs.annset, buffer_m, frozenset(given))
    ids = sorted(outlines, key=natural_key)
    geoms = [_polygonal(outlines[vid]) for vid in ids]
    records = [{"vineyard_id": vid, "area_m2": float(g.area)} for vid, g in zip(ids, geoms, strict=True)]
    return features_frame(records, geoms, BLOCK_COLUMNS)


def _ids(frame: gpd.GeoDataFrame, column: str) -> frozenset[str]:
    return frozenset(str(v) for v in frame[column])


def build_layers(inputs: WebInputs, params: WebParams) -> dict[str, gpd.GeoDataFrame | None]:
    """Every bundle layer in file order; `route` is None without a route."""
    ann = inputs.annset
    rows = physical_rows(ann.row_pieces, ann.canopies, rows=inputs.rows)
    waste = waste_features(ann.waste, max_block_dist_m=params.waste_block_max_m)
    linked = inputs.interrows
    pieces = linked if linked is not None and not linked.empty else ann.interrow_pieces
    targets = target_features(inputs.targets, route=inputs.route, visit_radius_m=params.visit_radius_m,
                              stops=inputs.stops, visits=inputs.visits, known_rows=_ids(rows, "row_id"),
                              known_waste=_ids(waste, "waste_id"))
    route = None if inputs.route is None else route_features(inputs.route, speed_kmh=params.speed_kmh,
                                                             decimals=params.utm_decimals)
    return {"blocks": block_features(inputs, params.block_buffer_m), "rows": rows,
            "canopies": canopy_features(ann.canopies),
            "interrows": interrow_features(pieces, ann.row_pieces, link_tol_m=params.row_link_tol_m),
            "waste": waste, "targets": targets, "route": route}


def tiles_with_objects(layers: Mapping[str, gpd.GeoDataFrame | None]) -> int:
    """Distinct tiles holding a row piece (rows' `tile_structures`), a canopy or an interrow piece."""
    tiles: set[str] = set()
    for name, column in (("canopies", "tile"), ("interrows", "tile")):
        frame = layers.get(name)
        if frame is not None:
            tiles.update(t for t in map(text_or_none, frame[column]) if t is not None)
    rows = layers.get("rows")
    if rows is not None:
        tiles.update(str(t) for structures in rows["tile_structures"] for t in (structures or {}))
    return len(tiles)


def bundle_bbox(layers: Mapping[str, gpd.GeoDataFrame | None], decimals: int) -> list[float] | None:
    """[minx, miny, maxx, maxy] (EPSG:32635) of every geometry of the bundle; None when it has none."""
    arrays = [frame.geometry.to_numpy() for frame in layers.values() if frame is not None and len(frame)]
    bounds = shapely.total_bounds(np.concatenate(arrays)) if arrays else np.full(4, np.nan)
    if not np.isfinite(bounds).all():
        return None  # no layer, or only empty / null geometries
    return [round(float(v), decimals) for v in bounds]


def manifest_extras(inputs: WebInputs, layers: Mapping[str, gpd.GeoDataFrame | None], params: WebParams, *,
                    run_id: str) -> dict[str, Any]:
    """Informative manifest fields: provenance (runs, AnnSet source, model version), layer counts, extent."""
    meta = inputs.annset.meta
    counts = {name: 0 if frame is None else len(frame) for name, frame in layers.items()}
    return {"run_id": run_id, "annset_run_id": meta.run_id, "annset_source": meta.source.value,
            "model_version": meta.model_version, "counts": counts, "tiles_with_objects": tiles_with_objects(layers),
            "bbox_32635": bundle_bbox(layers, params.utm_decimals)}


def overlap_free_interrows(inputs: WebInputs, params: WebParams) -> gpd.GeoDataFrame:
    """derive's overlap-free pieces when given, else the AnnSet's pieces without their cross-block overlap
    (like `measure`): the block interrow areas then add up to the survey line."""
    linked = inputs.interrows
    if linked is not None and not linked.empty:
        return linked
    ann = inputs.annset
    return remove_block_overlap(ann.interrow_pieces, ann.canopies, params.overlap).pieces


def measurements_bytes(inputs: WebInputs, params: WebParams) -> bytes:
    """measure's CSV when given (copied verbatim), else computed by the same `vineyard.measure` writer on
    overlap-free interrow pieces; either way it must pass the checks `publish` applies (SchemaError
    otherwise)."""
    data = inputs.measurements_csv
    if data is None:
        ann = inputs.annset
        m = compute_measurements(MeasureInputs(canopies=ann.canopies, row_pieces=ann.row_pieces,
                                               interrow_pieces=overlap_free_interrows(inputs, params),
                                               waste=ann.waste))
        text = format_measurements_csv(m.records(), m_decimals=params.m_decimals, ha_decimals=params.ha_decimals)
        data = text.encode("utf-8")
    failed = check_measurements_bytes(data, sum_tol_m=params.sum_tol_m, sum_tol_m2=params.sum_tol_m2)
    if failed:
        raise SchemaError("measurements.csv fails the web contract checks (§6.4)", failed=describe_failed(failed))
    return data


# ------------------------------------------------------------------ writing


def _rounded_value(value: Any, decimals: int) -> Any:
    return round(value, decimals) if isinstance(value, float) and math.isfinite(value) else value


def _rounded_column(values: pd.Series, decimals: int) -> pd.Series:
    # Built as object dtype: Series.map would infer float64 and write the ints of `[1, 2, None]` as 1.0, 2.0.
    return pd.Series([_rounded_value(v, decimals) for v in values], index=values.index, dtype=object)


def rounded_properties(frame: gpd.GeoDataFrame, decimals: int) -> gpd.GeoDataFrame:
    """New frame whose float properties are rounded to `decimals` (hides float32 / summation noise); fractions
    in FRACTION_DECIMALS keep their own precision, ints (and nulls next to them) stay as they are."""
    places = {c: FRACTION_DECIMALS.get(c, decimals) for c in frame.columns if c != frame.geometry.name}
    return frame.assign(**{c: _rounded_column(frame[c], n) for c, n in places.items()})


def _json_ready(value: Any) -> Any:
    if value is None or value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value.item() if hasattr(value, "item") and not isinstance(value, dict | list | str) else value


def write_geojsonseq(frame: gpd.GeoDataFrame, path: Path, *, decimals: int) -> Path:
    """GeoJSONSeq (one Feature per line, no crs member: the bundle is EPSG:32635 by contract §6.2)."""
    columns = [c for c in frame.columns if c != frame.geometry.name]
    geoms = round_geometry(frame.geometry.to_numpy(), decimals) if len(frame) else []
    lines = [json.dumps({"type": "Feature", "properties": {c: _json_ready(v) for c, v in zip(columns, rec,
                                                                                               strict=True)},
                         "geometry": mapping(geom)}, ensure_ascii=False, allow_nan=False, separators=SEQ_SEPARATORS)
             for rec, geom in zip(frame[columns].itertuples(index=False, name=None), geoms, strict=True)]
    return atomic_write_text(Path(path), "".join(f"{line}\n" for line in lines))


def _write_layer(name: str, frame: gpd.GeoDataFrame, out: Path, decimals: int) -> Path:
    path = out / LAYER_FILES[name]
    rounded = rounded_properties(frame, decimals)
    if name in SEQ_LAYERS:
        return write_geojsonseq(rounded, path, decimals=decimals)
    return write_geojson(rounded, path, decimals=decimals)


def _remove_stale(out: Path, keep: frozenset[str]) -> tuple[Path, ...]:
    stale = sorted(name for name in BUNDLE_FILES - keep if (out / name).is_file())
    for name in stale:
        (out / name).unlink()
    return tuple(out / name for name in stale)


def build_web_bundle(inputs: WebInputs, out_dir: Path, params: WebParams, *, generated_at: str,
                     pipeline_version: str, run_id: str) -> BundleResult:
    """Write the whole bundle into `out_dir` (see the module docstring). Without derive's interrows, the
    interrow features and the computed measurements both use the AnnSet's pieces without their cross-block
    overlap (resolved once)."""
    resolved = replace(inputs, interrows=overlap_free_interrows(inputs, params))
    layers = build_layers(resolved, params)
    csv_bytes = measurements_bytes(resolved, params)
    extras = manifest_extras(resolved, layers, params, run_id=run_id)
    counts = extras["counts"]
    manifest = build_manifest(params.survey, stage=manifest_stage(inputs.annset.meta.source),
                              generated_at=generated_at, pipeline_version=pipeline_version, extras=extras)
    out = Path(out_dir)
    written = [_write_layer(name, frame, out, params.utm_decimals) for name, frame in layers.items()
               if frame is not None]
    written.append(atomic_write_bytes(out / MEASUREMENTS_FILE, csv_bytes))
    removed = _remove_stale(out, frozenset({p.name for p in written} | {MANIFEST_FILE}))
    written.append(atomic_write_json(out / MANIFEST_FILE, manifest))
    return BundleResult(out_dir=out, written=tuple(written), removed=removed, counts=MappingProxyType(counts))
