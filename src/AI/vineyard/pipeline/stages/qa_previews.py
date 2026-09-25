"""Stage `qa_previews` (tile + global tail): one JPEG per tile under runs/<id>/qa/previews/, then
qa/overview.jpg (block-coloured mosaic of the previews) and qa/review_queue.csv (priority order).

Reads the AnnSet, tile_status and qa_issues written by `assemble`. Each task carries its tile's
objects and header, so workers only read the tile GeoTIFF and its rejected row candidates (grey,
possible missed rows). empty_tiles.csv belongs to the export.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy as np
import pandas as pd
import shapely
from pydantic import BaseModel, ConfigDict
from shapely.geometry import LineString

from vineyard.annset.io import ANNSET_JSON, read_annset, resolve_run_dir
from vineyard.annset.model import AnnSet
from vineyard.config import QaConfig
from vineyard.errors import StageError
from vineyard.geo.raster import read_tile
from vineyard.geo.tiling import tile_ref
from vineyard.geo.vector_io import read_layer
from vineyard.pipeline.cache import read_key
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult, TileTask, run_tile_stage
from vineyard.pipeline.tile_index import tile_path
from vineyard.qa.overview import build_overview
from vineyard.qa.render import PreviewHeader, TileObjects, render_preview, write_preview
from vineyard.qa.review import (
    TileIssueSummary,
    preview_relpath,
    review_queue,
    summaries_from_frame,
    write_review_queue,
)

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext, RunPaths

NAME: Final = "qa_previews"
VERSION: Final = "1"
OVERVIEW_FILE: Final = "overview.jpg"
REVIEW_QUEUE_FILE: Final = "review_queue.csv"
TILE_STATUS_FILE: Final = "tile_status.parquet"
QA_ISSUES_FILE: Final = "qa_issues.parquet"
COUNT_LABELS: Final[Mapping[str, str]] = {"canopies": "C", "row_pieces": "R", "interrow_pieces": "I", "waste": "W"}
BLOCK_LAYERS: Final = ("row_pieces", "canopies")
CANDIDATES_STAGE: Final = "rows_detect"
CANDIDATES_LAYER: Final = "row_candidates"
NO_CANDIDATES: Final = "none"
THUMB_READ_FLAG: Final = cv2.IMREAD_REDUCED_COLOR_8  # decode previews at 1/8 size for the overview
CFG_KEYS: Final = ("qa",)
RUN_LEVEL_DIRS: Final = ("layers_dir", "annset_dir", "exports_dir", "qa_dir", "metrics_dir", "logs_dir")
_SEP: Final = "\x1f"


class PreviewTaskCfg(BaseModel):
    """TileTask.cfg of this stage: qa settings plus the tile's objects and header."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    qa: QaConfig
    objects: TileObjects
    header: PreviewHeader


@dataclass(frozen=True, eq=False)
class QaInputs:
    annset: AnnSet
    tile_status: pd.DataFrame  # tile_status without geometry, one row per AnnSet tile
    summaries: Mapping[str, TileIssueSummary]


# ------------------------------------------------------------------ helpers


def annset_paths(ctx: RunContext) -> RunPaths:
    """Paths of the AnnSet's run: --annset (`vineyard qa`) or this run (`vineyard run`)."""
    if ctx.annset_ref is None:
        return ctx.paths
    run_dir = resolve_run_dir(ctx.paths.work_dir, ctx.annset_ref)
    here = ctx.paths
    return replace(here, run_dir=run_dir, **{name: run_dir / getattr(here, name).relative_to(here.run_dir)
                                            for name in RUN_LEVEL_DIRS})


def preview_path(paths: RunPaths, tile_id: str) -> Path:
    """runs/<id>/qa/previews/<tile>.jpg"""
    return paths.qa_dir / preview_relpath(tile_id)


def _require(path: Path, what: str) -> Path:
    if not path.is_file():
        raise StageError(f"{what} missing; run assemble first", stage=NAME, path=str(path))
    return path


def load_qa_inputs(paths: RunPaths) -> QaInputs:
    """AnnSet + tile_status + per-tile issue summaries written by assemble."""
    _require(paths.annset_dir / ANNSET_JSON, "annset")
    status = read_layer(_require(paths.layers_dir / TILE_STATUS_FILE, "tile_status"), "tile_status")
    issues = read_layer(_require(paths.qa_dir / QA_ISSUES_FILE, "qa_issues"), "qa_issues")
    return QaInputs(read_annset(paths.annset_dir), pd.DataFrame(status.drop(columns=status.geometry.name)),
                    summaries_from_frame(issues))


def header_for(tile_id: str, inputs: QaInputs) -> PreviewHeader:
    rows = inputs.tile_status[inputs.tile_status["tile_id"] == tile_id]
    summary = inputs.summaries.get(tile_id)
    codes = summary.codes if summary is not None else ()
    if rows.empty:
        return PreviewHeader(tile_id=tile_id, flags=codes)
    row = rows.iloc[0]
    counts = {label: int(row[f"n_{name}"]) for name, label in COUNT_LABELS.items()}
    return PreviewHeader(tile_id=tile_id, status=str(row["status"]), counts=counts, flags=codes,
                         priority=int(row["review_priority"]))


def objects_digest(objects: TileObjects, header: PreviewHeader) -> str:
    """sha1 over the WKB of every drawn geometry, the row labels and the header text."""
    h = hashlib.sha1()
    geoms = [*objects.canopies, *(line for line, _, _ in objects.rows), *objects.interrows, *objects.waste]
    for geom in geoms:
        h.update(shapely.to_wkb(geom, hex=False, output_dimension=2))
    h.update(_SEP.join(f"{rid}:{s}" for _, rid, s in objects.rows).encode("utf-8"))
    h.update(header.text().encode("utf-8"))
    return h.hexdigest()


def rejected_candidates(path: Path | None) -> tuple[LineString, ...]:
    """Rejected row candidates of a tile (rows_detect cache); none when the cache is absent."""
    if path is None or not Path(path).is_file():
        return ()
    cands = read_layer(path, CANDIDATES_LAYER)
    rejected = cands[cands["rejected_reason"].notna().to_numpy()]
    return tuple(g for g in rejected.geometry if isinstance(g, LineString) and not g.is_empty)


def dominant_blocks(annset: AnnSet) -> dict[str, str]:
    """tile -> its most frequent vineyard_id over row pieces and canopies (ties: smallest id)."""
    counts: dict[str, Counter[str]] = {}
    for name in BLOCK_LAYERS:
        frame = annset.layer(name)
        for tile_id, vid in zip(frame["tile_id"], frame["vineyard_id"], strict=True):
            if isinstance(vid, str) and vid:
                counts.setdefault(str(tile_id), Counter())[vid] += 1
    return {t: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for t, c in sorted(counts.items())}


# ------------------------------------------------------------------ tile function (worker)


def _task_cfg(task: TileTask) -> PreviewTaskCfg:
    if not isinstance(task.cfg, PreviewTaskCfg):
        raise StageError("qa_previews task needs a PreviewTaskCfg", stage=NAME, tile_id=task.tile_id,
                         got=type(task.cfg).__name__)
    return task.cfg


def preview_tile(task: TileTask) -> Mapping[str, Any]:
    """Worker: render the tile's preview and write it as JPEG."""
    tcfg = _task_cfg(task)
    objects = replace(tcfg.objects, rejected=rejected_candidates(task.inputs.get("candidates")))
    img = render_preview(read_tile(task.inputs["tif"]), tile_ref(task.tile_id), objects, tcfg.header, tcfg.qa)
    write_preview(task.outputs["preview"], img, tcfg.qa.preview_jpeg_quality)
    return {"n_canopies": len(objects.canopies), "n_rows": len(objects.rows), "n_rejected": len(objects.rejected)}


# ------------------------------------------------------------------ stage (main process)


def _candidates_path(ctx: RunContext, tile_id: str) -> Path:
    return ctx.paths.tile_cache(CANDIDATES_STAGE, tile_id, "parquet")


def _make_task(ctx: RunContext, paths: RunPaths, cfgs: Mapping[str, PreviewTaskCfg], tile_id: str) -> TileTask:
    inputs = {"tif": tile_path(ctx, tile_id), "candidates": _candidates_path(ctx, tile_id)}
    return TileTask(tile_id=tile_id, tif_path=tile_path(ctx, tile_id), key="", cfg=cfgs[tile_id], inputs=inputs,
                    outputs={"preview": preview_path(paths, tile_id)})


def _input_keys(ctx: RunContext, cfgs: Mapping[str, PreviewTaskCfg], tile_id: str) -> list[str]:
    tif = tile_path(ctx, tile_id)
    tcfg = cfgs[tile_id]
    candidates = read_key(_candidates_path(ctx, tile_id)) or NO_CANDIDATES
    return [f"objects:{objects_digest(tcfg.objects, tcfg.header)}", f"tif:{tif.name}:{tif.stat().st_size}",
            f"candidates:{candidates}"]


def _read_thumb(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    img = cv2.imread(str(path), THUMB_READ_FLAG)
    if img is None:
        raise StageError("unreadable preview JPEG", stage=NAME, path=str(path))
    return img


def write_overview(ctx: RunContext, annset: AnnSet, tile_ids: Sequence[str], paths: RunPaths | None = None) -> Path:
    """qa/overview.jpg from the previews on disk (tiles without one are drawn grey)."""
    paths = ctx.paths if paths is None else paths
    thumbs = {t: img for t in tile_ids if (img := _read_thumb(preview_path(paths, t))) is not None}
    mosaic = build_overview(thumbs, dominant_blocks(annset), ctx.cfg.qa, tile_ids=tile_ids)
    return write_preview(paths.qa_dir / OVERVIEW_FILE, mosaic, ctx.cfg.qa.preview_jpeg_quality)


def run(ctx: RunContext) -> StageResult:
    paths = annset_paths(ctx)
    inputs = load_qa_inputs(paths)
    tile_ids = ctx.selected_tiles(inputs.annset.meta.tile_ids)
    if not tile_ids:
        raise StageError("no AnnSet tile matches the --tiles filter", stage=NAME, filter=list(ctx.tile_filter))
    cfgs = {}
    for tile_id in tile_ids:
        header = header_for(tile_id, inputs)
        cfgs[tile_id] = PreviewTaskCfg(qa=ctx.cfg.qa, objects=TileObjects.from_annset(inputs.annset, tile_id),
                                       header=header)
    result = run_tile_stage(ctx, STAGE, make_task=partial(_make_task, ctx, paths, cfgs), tile_fn=preview_tile,
                            input_keys=partial(_input_keys, ctx, cfgs), tile_ids=tile_ids)
    overview = write_overview(ctx, inputs.annset, tile_ids, paths)
    queue = write_review_queue(review_queue(inputs.tile_status, inputs.summaries), paths.qa_dir / REVIEW_QUEUE_FILE)
    return replace(result, outputs=(overview, queue))


STAGE: Final = StageSpec(
    name=NAME, version=VERSION, scope="tile", cfg_keys=CFG_KEYS, requires=("assemble",), run=run,
    description="per-tile QA preview JPEGs, the block-coloured overview mosaic and review_queue.csv",
)
