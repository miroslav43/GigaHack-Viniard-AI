"""assemble (02 §3.10, plan S5/S8/S9): AnnSet + tile_status + qa_issues from the per-tile stage outputs.

Steps: waste layer (empty + warning when absent) -> drop forced-empty tiles -> subtract canopies from
interrows only in tiles over the overlap limit -> validate every layer -> contract §2.7 invariants,
object qa_flags, vine-evidence checks and empty-tile confirmations -> tile_status + review priority.
Invariant errors never stop assemble; the CVAT export refuses to run on them.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from vineyard.annset.io import ANNSET_JSON, write_annset
from vineyard.annset.model import ANNSET_LAYERS, AnnSet, AnnSetMeta
from vineyard.config import AppConfig
from vineyard.contracts.enums import Severity, TileStatus
from vineyard.contracts.qa import QaIssue, issues_to_gdf
from vineyard.contracts.schemas import coerce_layer, empty_layer, validate_layer
from vineyard.geo.tiling import CRS_EPSG, tile_box, tile_ref
from vineyard.geo.vector_io import write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.topology import check_invariants, remove_canopy_overlap
from vineyard.pipeline.atomic import atomic_write_text
from vineyard.qa.review import (
    CODE_SEP,
    PRIORITY_CLEAN,
    TileIssueSummary,
    review_queue,
    summarize_issues,
    write_review_queue,
)

LAYER_IDS: Final[Mapping[str, str]] = MappingProxyType(
    {"canopies": "canopy_id", "row_pieces": "piece_id", "interrow_pieces": "piece_id", "waste": "waste_id"}
)
EMPTY_CONFIRM_CODE: Final = "empty_tile_confirm"
NON_VINEYARD_CODE: Final = "canopy_on_non_vineyard_tile"
LOW_SNR_CODE: Final = "low_snr"
OVERRIDE_APPLIED_CODE: Final = "override_applied"
INTERPOLATED_FLAG: Final = "row_interpolated"
OVERRIDE_FLAG_BASE: Final = "override"
FLAG_ID_SEP: Final = ":"
FORCE_EMPTY_OBJECT: Final = "force_empty_tiles"
NO_IMAGE_ID: Final = -1
STATUS_CONFIDENCE: Final = 1.0
TILE_STATUS_LAYER: Final = "tile_status"
QA_ISSUES_LAYER: Final = "qa_issues"
TILE_STATUS_CSV: Final = "tile_status.csv"
REVIEW_QUEUE_CSV: Final = "review_queue.csv"
_SNAKE: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_CLEAN: Final = TileIssueSummary(n_errors=0, n_warnings=0, codes=(), priority=PRIORITY_CLEAN)
_log = get_logger("annset.assemble")


@dataclass(frozen=True)
class TileEvidence:
    """Per-tile signals from tile_prep / rows_detect / canopy (NaN = unknown, never flagged)."""

    veg_frac: float = math.nan
    snr: float = math.nan  # rows_detect periodicity SNR
    vineyard_score: float = math.nan  # NN p95 inside the corridors
    empty_nodata: bool = False


@dataclass(frozen=True)
class AssembleSettings:
    max_overlap_m2: float
    min_interrow_piece_m2: float
    canopy_clearance_m: float
    snr_min: float  # below it a tile has no row periodicity: canopies there are suspect
    snr_review_max: float  # rows on a tile below it go to priority-1 review
    min_vine_score: float

    @classmethod
    def from_config(cls, cfg: AppConfig) -> AssembleSettings:
        return cls(
            max_overlap_m2=cfg.export.cvat.max_canopy_interrow_overlap_m2,
            min_interrow_piece_m2=cfg.export.min_interrow_piece_m2,
            canopy_clearance_m=cfg.interrow.canopy_clearance_m,
            snr_min=cfg.rows.detect.periodicity_min_snr,
            snr_review_max=cfg.qa.snr_review_max,
            min_vine_score=cfg.rows.filter.min_vine_score,
        )


@dataclass(frozen=True, eq=False)
class AssembleInputs:
    canopies: gpd.GeoDataFrame
    row_pieces: gpd.GeoDataFrame
    interrow_pieces: gpd.GeoDataFrame
    waste: gpd.GeoDataFrame | None = None
    clips: Mapping[str, BaseGeometry] = field(default_factory=dict)  # tile_box ∩ tile_valid
    evidence: Mapping[str, TileEvidence] = field(default_factory=dict)
    failed_tiles: tuple[str, ...] = ()
    force_empty_tiles: tuple[str, ...] = ()
    upstream_issues: tuple[QaIssue, ...] = ()
    rows: gpd.GeoDataFrame | None = None  # global rows layer, for the interrow side invariant


@dataclass(frozen=True, eq=False)
class AssembleResult:
    annset: AnnSet
    tile_status: gpd.GeoDataFrame
    qa_issues: gpd.GeoDataFrame
    issues: tuple[QaIssue, ...]
    summaries: Mapping[str, TileIssueSummary]
    fixed_overlap_tiles: tuple[str, ...]

    @property
    def n_errors(self) -> int:
        return sum(i.severity == Severity.ERROR for i in self.issues)


# ------------------------------------------------------------------ layers


def _without_tiles(frame: gpd.GeoDataFrame, tiles: tuple[str, ...]) -> gpd.GeoDataFrame:
    return frame[~frame["tile_id"].isin(tiles).to_numpy()].reset_index(drop=True) if tiles else frame


def _waste_layer(waste: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    if waste is None:
        log_event(_log, "assemble.waste_missing", level=logging.WARNING, stage="assemble",
                  detail="layers/waste.parquet absent: empty waste layer")
        return empty_layer("waste")
    return waste


def _prepare_layers(inputs: AssembleInputs, settings: AssembleSettings
                    ) -> tuple[dict[str, gpd.GeoDataFrame], tuple[str, ...]]:
    raw = {"canopies": inputs.canopies, "row_pieces": inputs.row_pieces,
           "interrow_pieces": inputs.interrow_pieces, "waste": _waste_layer(inputs.waste)}
    layers = {name: coerce_layer(_without_tiles(frame, inputs.force_empty_tiles), name)
              for name, frame in raw.items()}
    irs, fixed = remove_canopy_overlap(layers["interrow_pieces"], layers["canopies"],
                                       max_overlap_m2=settings.max_overlap_m2,
                                       min_piece_m2=settings.min_interrow_piece_m2,
                                       clearance_m=settings.canopy_clearance_m)
    layers["interrow_pieces"] = coerce_layer(irs, "interrow_pieces")
    for name in ANNSET_LAYERS:
        validate_layer(layers[name], name)
    return layers, fixed


# ------------------------------------------------------------------ issues


def _flag_code(flag: str) -> str | None:
    base = flag.split(FLAG_ID_SEP, 1)[0].strip()
    code = OVERRIDE_APPLIED_CODE if base == OVERRIDE_FLAG_BASE else base
    return code if _SNAKE.fullmatch(code) else None


def _flag_issues(annset: AnnSet) -> list[QaIssue]:
    """Every object qa_flag becomes a located warning (e.g. structure_borderline, row_interpolated)."""
    issues = []
    for name in ANNSET_LAYERS:
        frame = annset.layer(name)
        flagged = frame[(frame["qa_flags"].fillna("") != "").to_numpy()]
        for oid, tile_id, flags, geom in zip(flagged[LAYER_IDS[name]], flagged["tile_id"], flagged["qa_flags"],
                                             flagged.geometry, strict=True):
            point = geom.representative_point()
            for flag in dict.fromkeys(f.strip() for f in flags.split(CODE_SEP) if f.strip()):
                code = _flag_code(flag)
                if code is not None:
                    issues.append(QaIssue(Severity.WARNING, code, tile_id, oid, f"{name}: {flag}", point.x, point.y))
    return issues


def _counts(annset: AnnSet) -> dict[str, dict[str, int]]:
    return {name: annset.layer(name)["tile_id"].value_counts().to_dict() for name in ANNSET_LAYERS}


def _n(counts: dict[str, dict[str, int]], layer: str, tile_id: str) -> int:
    return int(counts[layer].get(tile_id, 0))


def _interpolated_tiles(annset: AnnSet) -> frozenset[str]:
    rows = annset.row_pieces
    has = rows["qa_flags"].fillna("").str.split(CODE_SEP).apply(lambda codes: INTERPOLATED_FLAG in codes)
    return frozenset(rows["tile_id"][has.to_numpy(dtype=bool)])


def _is_below(value: float, limit: float) -> bool:
    return math.isfinite(value) and value < limit


def _evidence_issues(annset: AnnSet, inputs: AssembleInputs, settings: AssembleSettings,
                     tile_ids: tuple[str, ...]) -> list[QaIssue]:
    counts, interpolated = _counts(annset), _interpolated_tiles(annset)
    issues = []
    for tile_id in tile_ids:
        ev = inputs.evidence.get(tile_id, TileEvidence())
        weak = (_is_below(ev.snr, settings.snr_min) or _is_below(ev.vineyard_score, settings.min_vine_score)
                or tile_id in interpolated)
        if _n(counts, "canopies", tile_id) and weak:
            msg = f"canopies without vine evidence (snr={ev.snr:.2f}, score={ev.vineyard_score:.2f})"
            issues.append(QaIssue(Severity.WARNING, NON_VINEYARD_CODE, tile_id, "", msg))
        if _n(counts, "row_pieces", tile_id) and _is_below(ev.snr, settings.snr_review_max):
            issues.append(QaIssue(Severity.WARNING, LOW_SNR_CODE, tile_id, "", f"rows on a low-SNR tile ({ev.snr:.2f})"))
    return issues


def _tile_issues(annset: AnnSet, inputs: AssembleInputs, tile_ids: tuple[str, ...]) -> list[QaIssue]:
    counts = _counts(annset)
    forced = [QaIssue(Severity.INFO, OVERRIDE_APPLIED_CODE, t, FORCE_EMPTY_OBJECT, "tile forced empty")
              for t in tile_ids if t in inputs.force_empty_tiles]
    empty = [QaIssue(Severity.INFO, EMPTY_CONFIRM_CODE, t, "", "no objects: tick 'No objects' in Marcaj")
             for t in tile_ids
             if t not in inputs.failed_tiles and not any(_n(counts, name, t) for name in ANNSET_LAYERS)]
    return forced + empty


# ------------------------------------------------------------------ tile_status


def _status(tile_id: str, n_objects: int, inputs: AssembleInputs) -> TileStatus:
    if tile_id in inputs.failed_tiles:
        return TileStatus.FAILED
    if inputs.evidence.get(tile_id, TileEvidence()).empty_nodata:
        return TileStatus.EMPTY_NODATA
    return TileStatus.OK if n_objects else TileStatus.NO_VINEYARD


def _status_record(tile_id: str, annset: AnnSet, counts: dict[str, dict[str, int]], inputs: AssembleInputs,
                   summary: TileIssueSummary) -> dict[str, object]:
    n = {name: _n(counts, name, tile_id) for name in ANNSET_LAYERS}
    ev = inputs.evidence.get(tile_id, TileEvidence())
    return {
        "tile_id": tile_id, "status": _status(tile_id, sum(n.values()), inputs).value,
        "has_vineyard": bool(n["canopies"] or n["row_pieces"]), "n_canopies": n["canopies"],
        "n_row_pieces": n["row_pieces"], "n_interrow_pieces": n["interrow_pieces"], "n_waste": n["waste"],
        "veg_frac": ev.veg_frac, "vineyard_score": ev.vineyard_score, "upload_zip": None, "image_id": NO_IMAGE_ID,
        "review_priority": summary.priority, "issues": CODE_SEP.join(summary.codes),
        "source": annset.meta.source.value, "run_id": annset.meta.run_id,
        "model_version": annset.meta.model_version, "confidence": STATUS_CONFIDENCE, "qa_flags": "",
        "geometry": tile_box(tile_ref(tile_id)),
    }


def tile_status_frame(annset: AnnSet, inputs: AssembleInputs,
                      summaries: Mapping[str, TileIssueSummary]) -> gpd.GeoDataFrame:
    """Contract §2.5.3 tile_status, one row per meta tile, sorted by tile_id."""
    counts = _counts(annset)
    records = [_status_record(t, annset, counts, inputs, summaries.get(t, _CLEAN)) for t in annset.meta.tile_ids]
    if not records:
        return empty_layer(TILE_STATUS_LAYER)
    return coerce_layer(gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_EPSG), TILE_STATUS_LAYER)



def assemble_annset(inputs: AssembleInputs, settings: AssembleSettings, meta: AnnSetMeta) -> AssembleResult:
    """Build AnnSet(meta.source) and its QA products; raises SchemaError on a contract violation."""
    layers, fixed = _prepare_layers(inputs, settings)
    empty = AnnSet(meta=meta, **{name: empty_layer(name) for name in ANNSET_LAYERS})
    annset = empty
    for name in ANNSET_LAYERS:
        annset = annset.with_layer(name, layers[name])
    tile_ids = annset.meta.tile_ids
    issues = tuple(sorted(
        [*check_invariants(annset, inputs.clips, max_overlap_m2=settings.max_overlap_m2, rows=inputs.rows),
         *_flag_issues(annset), *_evidence_issues(annset, inputs, settings, tile_ids),
         *_tile_issues(annset, inputs, tile_ids), *inputs.upstream_issues],
        key=lambda i: i.sort_key,
    ))
    summaries = summarize_issues(issues)
    qa = issues_to_gdf(issues, source=meta.source, run_id=meta.run_id, model_version=meta.model_version)
    return AssembleResult(annset, tile_status_frame(annset, inputs, summaries), qa, issues, summaries, fixed)


def write_assemble_outputs(result: AssembleResult, *, annset_dir: Path, layers_dir: Path,
                           qa_dir: Path) -> tuple[Path, ...]:
    """annset/*, layers/tile_status.parquet, qa/{qa_issues.parquet, tile_status.csv, review_queue.csv}."""
    status = result.tile_status
    csv_frame = pd.DataFrame(status.drop(columns=status.geometry.name))
    queue = review_queue(csv_frame, result.summaries)
    return (
        write_annset(result.annset, Path(annset_dir)) / ANNSET_JSON,
        write_layer(status, TILE_STATUS_LAYER, Path(layers_dir) / f"{TILE_STATUS_LAYER}.parquet"),
        write_layer(result.qa_issues, QA_ISSUES_LAYER, Path(qa_dir) / f"{QA_ISSUES_LAYER}.parquet"),
        atomic_write_text(Path(qa_dir) / TILE_STATUS_CSV, csv_frame.to_csv(index=False, lineterminator="\n")),
        write_review_queue(queue, Path(qa_dir) / REVIEW_QUEUE_CSV),
    )

