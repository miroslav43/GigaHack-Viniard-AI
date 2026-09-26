"""Cross-path I/O of the post chain (not a stage itself).

`passable` detects the tracks across the rows (it runs right after derive and needs them for the walk graph)
and writes layers/cross_paths.parquet (strips) + layers/cross_path_lines.parquet (centrelines) +
metrics/cross_paths.json; `targets` and `web_bundle` read the strips back from the same post run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import geopandas as gpd
import shapely
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from vineyard.geo.tiling import CRS_EPSG
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.pipeline.atomic import atomic_write_json
from vineyard.pipeline.stages._post_io import coverage, layer_path
from vineyard.route.cross_paths import CrossPath, CrossPathParams, detect_cross_paths, paths_frame
from vineyard.route.target_gaps import engine_gap_fn
from vineyard.route.targets import TargetInputs, TargetSettings, row_contexts

if TYPE_CHECKING:
    from vineyard.annset.model import AnnSet
    from vineyard.pipeline.context import RunContext, RunPaths

STRIPS_LAYER: Final = "cross_paths"
LINES_LAYER: Final = "cross_path_lines"
METRICS_FILE: Final = "cross_paths.json"
PATH_CONFIDENCE: Final = 1.0


def detect_for_run(ctx: RunContext, annset: AnnSet, rows: gpd.GeoDataFrame) -> tuple[CrossPath, ...]:
    """Cross-paths of the run's global rows (the same gap engine and coverage as the targets stage)."""
    if not ctx.cfg.cross_paths.enabled or rows.empty:
        return ()
    covered, _ = coverage(ctx, sorted(annset.tile_ids()))
    inputs = TargetInputs(rows=rows, canopies=annset.canopies, waste=annset.waste, coverage=covered)
    gap_fn = engine_gap_fn(ctx.cfg.row_structure, ctx.cfg.canopy.corridor_half_m)
    contexts = row_contexts(inputs, TargetSettings.from_config(ctx.cfg), gap_fn)
    return detect_cross_paths(contexts, CrossPathParams.from_config(ctx.cfg.cross_paths))


def _provenance(ctx: RunContext) -> dict[str, Any]:
    return {"source": ctx.source.value, "run_id": ctx.run_id, "model_version": ctx.model_version(),
            "confidence": PATH_CONFIDENCE, "qa_flags": ""}


def lines_frame(paths: tuple[CrossPath, ...], provenance: dict[str, Any]) -> gpd.GeoDataFrame:
    data = {"path_id": [p.path_id for p in paths], "vineyard_id": [p.vineyard_id for p in paths],
            "length_m": [p.length_m for p in paths], **{k: [v] * len(paths) for k, v in provenance.items()}}
    return gpd.GeoDataFrame(data, geometry=gpd.GeoSeries([p.centerline for p in paths], crs=CRS_EPSG), crs=CRS_EPSG)


def metrics(paths: tuple[CrossPath, ...], enabled: bool) -> dict[str, Any]:
    return {"enabled": enabled, "n_paths": len(paths), "n_rows_crossed": sum(p.n_rows for p in paths),
            "length_m": round(sum(p.length_m for p in paths), 2),
            "area_m2": round(sum(p.polygon.area for p in paths), 2),
            "paths": [{"path_id": p.path_id, "vineyard_id": p.vineyard_id, "n_rows": p.n_rows,
                       "width_m": round(p.width_m, 2), "length_m": round(p.length_m, 2),
                       "residual_m": round(p.residual_m, 3)} for p in paths]}


def write_cross_paths(ctx: RunContext, paths: tuple[CrossPath, ...]) -> tuple[Path, ...]:
    prov = _provenance(ctx)
    return (write_layer(paths_frame(paths, prov), STRIPS_LAYER, layer_path(ctx.paths, STRIPS_LAYER)),
            write_layer(lines_frame(paths, prov), LINES_LAYER, layer_path(ctx.paths, LINES_LAYER)),
            atomic_write_json(ctx.paths.metrics_dir / METRICS_FILE, metrics(paths, ctx.cfg.cross_paths.enabled)))


def read_strips(paths: RunPaths) -> gpd.GeoDataFrame | None:
    """The run's `cross_paths` layer (None when the run has none, e.g. written before the feature)."""
    path = layer_path(paths, STRIPS_LAYER)
    return read_layer(path, STRIPS_LAYER) if path.is_file() else None


def strips_union(paths: RunPaths) -> BaseGeometry | None:
    frame = read_strips(paths)
    if frame is None or frame.empty:
        return None
    return shapely.union_all(list(frame.geometry))


def route_lines(paths: tuple[CrossPath, ...]) -> tuple[tuple[str, LineString], ...]:
    return tuple((p.path_id, p.centerline) for p in paths)


__all__ = [
    "LINES_LAYER", "METRICS_FILE", "STRIPS_LAYER", "detect_for_run", "lines_frame", "metrics", "read_strips",
    "route_lines", "strips_union", "write_cross_paths",
]
