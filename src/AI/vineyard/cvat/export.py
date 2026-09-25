"""export_upload: AnnSet(model) -> validated Marcaj upload ZIPs, published atomically (01 §3.10).

Flow: one CvatImage per tile (failed tiles become empty images) -> in-memory validation ->
packing plan -> ZIPs in `<out>.staging-<pid>/` (repacked with n+1 parts, at most twice, when a
written ZIP exceeds max_zip_bytes) -> validate_zip + validate_upload_set -> self-check ->
manifests -> os.replace into `<out>/`. Any error renames the staging dir to `<out>.FAILED-<ts>/`
(with validation_report.json) and raises ExportBlocked.
"""

from __future__ import annotations

import os
import shutil
import time
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

import geopandas as gpd

from vineyard.annset.model import AnnSet
from vineyard.config import AppConfig
from vineyard.config.sections_io import CvatExportConfig
from vineyard.contracts import CONTRACT_VERSION
from vineyard.contracts.enums import Severity
from vineyard.cvat.model import CvatDocument, CvatImage
from vineyard.cvat.packer import PartPlan, estimate_tile_bytes, plan_parts, write_upload_zip, zip_base_bytes
from vineyard.cvat.report import (
    EMPTY_TILES_COLUMNS,
    EMPTY_TILES_CSV,
    ID_REGISTRY_JSON,
    MANIFEST_COLUMNS,
    MANIFEST_CSV,
    SUMMARY_JSON,
    VALIDATION_JSON,
    Issue,
    PartInfo,
    ValidationReport,
    empty_tile_rows,
    error,
    id_registry,
    label_totals,
    manifest_rows,
    warning,
    write_csv,
)
from vineyard.cvat.selfcheck import selfcheck_upload
from vineyard.cvat.to_cvat import TileWriteStats, annset_to_images
from vineyard.cvat.validator_doc import validate_image
from vineyard.cvat.validator_zip import validate_upload_set
from vineyard.cvat.writer import serialize_document, serialize_image
from vineyard.errors import ExportBlocked
from vineyard.geo.tiling import TILE_PX, tile_ref
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.atomic import atomic_write_json

MAX_REPACKS: Final = 2
InvalidTilePolicy = Literal["block", "empty"]
REASON_FAILED: Final = "failed"
REASON_INVALID: Final = "invalid"
STAGING_SUFFIX: Final = ".staging-{pid}"
FAILED_SUFFIX: Final = ".FAILED-{stamp}"
OLD_SUFFIX: Final = ".old-{pid}"

_log = get_logger("cvat.export")


@dataclass(frozen=True)
class TileSource:
    tile_id: str
    file_name: str
    path: Path
    sha256: str
    file_size: int


@dataclass(frozen=True)
class UploadResult:
    out_dir: Path
    zips: tuple[Path, ...]
    report: ValidationReport
    manifest: Path
    empty_tiles: Path
    id_registry: Path
    summary: Path
    validation_report: Path
    parts: tuple[PartInfo, ...]
    failed_tiles: tuple[str, ...]


@dataclass(frozen=True)
class _Written:
    plan: PartPlan
    path: Path
    doc: CvatDocument
    size: int


def tile_sources(tile_index: gpd.GeoDataFrame, tile_ids: Sequence[str] | None = None) -> tuple[TileSource, ...]:
    """Tile files to upload (sorted), from tile_index columns tile_id/file_name/path/sha256/file_size."""
    frame = tile_index if tile_ids is None else tile_index[tile_index["tile_id"].isin(set(tile_ids))]
    sources = tuple(sorted(
        (TileSource(str(r.tile_id), str(r.file_name), Path(str(r.path)), str(r.sha256), int(r.file_size))
         for r in frame.itertuples(index=False)), key=lambda s: s.tile_id))
    missing = [s.tile_id for s in sources if not s.path.is_file()]
    if missing:
        raise ExportBlocked("tile files missing; run `vineyard ingest`", n_missing=len(missing), examples=missing[:5])
    if tile_ids is not None and len(sources) != len(set(tile_ids)):
        unknown = sorted(set(tile_ids) - {s.tile_id for s in sources})
        raise ExportBlocked("tiles not in tile_index", examples=unknown[:5])
    return sources


# ---------------------------------------------------------------- images and in-memory validation


def _waste_distances(annset: AnnSet) -> dict[str, float]:
    waste = annset.waste
    return {str(w): float(d) for w, d in zip(waste["waste_id"], waste["dist_block_m"], strict=True)}


def _validate_images(images: Sequence[CvatImage], cfg: AppConfig, dist: Mapping[str, float]
                     ) -> dict[str, ValidationReport]:
    return {img.name[:-4]: validate_image(img, cfg=cfg.export.cvat, tile_px=TILE_PX, waste_block_dist_m=dist,
                                          waste_min_dist_m=cfg.waste.block_assign_max_m) for img in images}


def _apply_policy(images: Sequence[CvatImage], reports: Mapping[str, ValidationReport],
                  policy: InvalidTilePolicy) -> tuple[list[CvatImage], dict[str, str], list[Issue]]:
    """Returns (images, {tile: reason} for emptied tiles, issues to report)."""
    kept: list[CvatImage] = []
    emptied: dict[str, str] = {}
    issues: list[Issue] = []
    for img in images:
        rep = reports[img.name[:-4]]
        issues += list(rep.warnings)
        if rep.ok:
            kept.append(img)
        elif policy == "empty":
            emptied[img.name[:-4]] = REASON_INVALID
            issues += [warning("tile_emptied", str(i), tile_id=i.tile_id, object_ref=i.object_ref) for i in rep.errors]
            kept.append(img.with_shapes(()))
        else:
            issues += list(rep.errors)
            kept.append(img)
    return kept, emptied, issues


def build_images(annset: AnnSet, sources: Sequence[TileSource], cfg: AppConfig, policy: InvalidTilePolicy
                 ) -> tuple[list[CvatImage], tuple[TileWriteStats, ...], dict[str, str], ValidationReport]:
    images, stats = annset_to_images(annset, [tile_ref(s.tile_id) for s in sources], cfg.export.cvat,
                                     on_tile_error="empty")
    reports = _validate_images(images, cfg, _waste_distances(annset))
    kept, emptied, issues = _apply_policy(images, reports, policy)
    failed = {s.tile_id: REASON_FAILED for s in stats if s.failed}
    issues += [i for s in stats for i in s.issues]
    return kept, stats, failed | emptied, ValidationReport(tuple(issues))


# ---------------------------------------------------------------- packing


def _estimates(images: Sequence[CvatImage], sources: Mapping[str, TileSource], cvat: CvatExportConfig
               ) -> tuple[list[tuple[str, int]], int]:
    level = cvat.xml_deflate_level
    frame = len(zlib.compress(serialize_document(CvatDocument(())), level))
    sizes = []
    for img in images:
        src = sources[img.name[:-4]]
        fragment = serialize_image(img, decimals=cvat.coord_decimals).encode("utf-8")
        sizes.append((src.tile_id, estimate_tile_bytes(src.file_size, src.file_name,
                                                       len(zlib.compress(fragment, level)))))
    return sizes, zip_base_bytes(frame)


def _write_part(plan: PartPlan, n: int, images: Mapping[str, CvatImage], sources: Mapping[str, TileSource],
                cvat: CvatExportConfig, staging: Path) -> _Written:
    doc = CvatDocument(tuple(images[t].with_id(k) for k, t in enumerate(plan.tile_ids)))
    path = staging / cvat.zip_name.format(i=plan.index, n=n)
    xml = serialize_document(doc, decimals=cvat.coord_decimals)
    files = [(sources[t].file_name, sources[t].path) for t in plan.tile_ids]
    write_upload_zip(path, xml, files, deflate_level=cvat.xml_deflate_level)
    return _Written(plan, path, doc, path.stat().st_size)


def pack_parts(images: Sequence[CvatImage], sources: Sequence[TileSource], cvat: CvatExportConfig,
               staging: Path) -> tuple[_Written, ...]:
    """Plan, write, and repack with one more part (at most MAX_REPACKS times) if a ZIP is over the limit."""
    by_tile = {img.name[:-4]: img for img in images}
    src = {s.tile_id: s for s in sources}
    sizes, base = _estimates(images, src, cvat)
    n_parts = cvat.n_parts
    for attempt in range(MAX_REPACKS + 1):
        plans = plan_parts(sizes, max_bytes=cvat.max_zip_bytes, n_parts=n_parts, balance=cvat.balance_parts,
                           base_bytes=base)
        written = tuple(_write_part(p, len(plans), by_tile, src, cvat, staging) for p in plans)
        over = [w.path.name for w in written if w.size > cvat.max_zip_bytes]
        if not over:
            return written
        log_event(_log, "export.repack", level=30, attempt=attempt + 1, over=over, n_parts=len(plans) + 1)
        for w in written:
            w.path.unlink()
        n_parts = len(plans) + 1
    raise ExportBlocked("ZIP over max_zip_bytes after repacking", max_zip_bytes=cvat.max_zip_bytes,
                        repacks=MAX_REPACKS)


# ---------------------------------------------------------------- staging, publish


def _fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _publish(staging: Path, out_dir: Path) -> Path:
    old = out_dir.with_name(out_dir.name + OLD_SUFFIX.format(pid=os.getpid()))
    if out_dir.exists():
        if old.exists():
            shutil.rmtree(old)
        os.replace(out_dir, old)
    os.replace(staging, out_dir)
    if old.exists():
        shutil.rmtree(old)
    return out_dir


def _fail(staging: Path, out_dir: Path, report: ValidationReport, what: str) -> ExportBlocked:
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-%f")  # two failures in one second must not collide
    failed = out_dir.with_name(out_dir.name + FAILED_SUFFIX.format(stamp=stamp))
    staging.mkdir(parents=True, exist_ok=True)
    atomic_write_json(staging / VALIDATION_JSON, report.to_json_dict())
    os.replace(staging, failed)
    for line in report.summary_lines():
        log_event(_log, "export.issue", level=40, detail=line)
    return ExportBlocked(f"export blocked: {what}", failed_dir=str(failed), n_errors=report.n_errors,
                         first="; ".join(str(i) for i in report.errors[:3]))


# ---------------------------------------------------------------- orchestration


def _validate_written(written: Sequence[_Written], sources: Sequence[TileSource], cfg: AppConfig,
                      annset: AnnSet, emptied: frozenset[str]) -> tuple[ValidationReport, ValidationReport]:
    cvat = cfg.export.cvat
    shas = {s.tile_id: s.sha256 for s in sources}
    expected = frozenset(shas)
    zips = validate_upload_set([w.path for w in written], expected_tiles=expected, expected_sha256=shas, cfg=cvat,
                               tile_px=TILE_PX, waste_min_dist_m=cfg.waste.block_assign_max_m)
    check = selfcheck_upload([(w.path, w.doc) for w in written], expected_sha256=shas, expected_tiles=expected,
                             cfg=cvat, annset=annset, skip_geometry=emptied)
    return zips, check


def _summary(annset: AnnSet, cfg: AppConfig, run_id: str, written: Sequence[_Written], parts: Sequence[PartInfo],
             stats: Sequence[TileWriteStats], empty_rows: Sequence[Mapping[str, object]], report: ValidationReport,
             elapsed_s: float) -> dict[str, object]:
    docs = [(w.path.name, w.doc) for w in written]
    reasons = Counter(str(r["reason"]) for r in empty_rows)
    dropped = Counter({k: v for s in stats for k, v in s.n_dropped.items()})
    return {
        "contract_version": CONTRACT_VERSION, "run_id": run_id, "annset_run_id": annset.meta.run_id,
        "annset_source": annset.meta.source.value, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_tiles": sum(p.n_tiles for p in parts), "n_zips": len(parts), "max_zip_bytes": cfg.export.cvat.max_zip_bytes,
        "parts": [asdict(p) for p in parts], "totals": label_totals(docs), "n_empty_tiles": len(empty_rows),
        "empty_reasons": dict(sorted(reasons.items())), "n_dropped": dict(sorted(dropped.items())),
        "n_split": sum(s.n_split for s in stats), "failed_tiles": sorted(s.tile_id for s in stats if s.failed),
        "ok": report.ok, "n_errors": report.n_errors, "n_warnings": report.n_warnings,
        "warnings_by_code": report.count_by_code(), "elapsed_s": round(elapsed_s, 2),
    }


def _write_files(staging: Path, written: Sequence[_Written], sources: Sequence[TileSource],
                 reasons: Mapping[str, str], tile_status: gpd.GeoDataFrame | None
                 ) -> tuple[tuple[PartInfo, ...], list[dict[str, object]]]:
    parts = tuple(PartInfo.of(w.path, w.plan.tile_ids, w.size) for w in written)
    docs = [(w.path.name, w.doc) for w in written]
    write_csv(staging / MANIFEST_CSV, MANIFEST_COLUMNS, manifest_rows(docs, {s.tile_id: s.sha256 for s in sources}))
    empty_rows = empty_tile_rows(docs, reasons, tile_status)
    write_csv(staging / EMPTY_TILES_CSV, EMPTY_TILES_COLUMNS, empty_rows)
    return parts, empty_rows


def export_upload(
    annset: AnnSet,
    tile_index: gpd.GeoDataFrame,
    cfg: AppConfig,
    out_dir: Path,
    tile_status: gpd.GeoDataFrame | None = None,
    *,
    run_id: str = "",
    tile_ids: Sequence[str] | None = None,
    invalid_tile_policy: InvalidTilePolicy | None = None,
) -> UploadResult:
    """Build, validate, self-check and atomically publish the upload set; raises ExportBlocked on any error.
    invalid_tile_policy None = export.cvat.invalid_tile_policy."""
    started = time.monotonic()
    policy = cfg.export.cvat.invalid_tile_policy if invalid_tile_policy is None else invalid_tile_policy
    out_dir = Path(out_dir)
    sources = tile_sources(tile_index, tile_ids)
    staging = _fresh_dir(out_dir.with_name(out_dir.name + STAGING_SUFFIX.format(pid=os.getpid())))
    images, stats, reasons, pre = build_images(annset, sources, cfg, policy)
    if not pre.ok:
        raise _fail(staging, out_dir, pre, "in-memory validation")
    try:
        written = pack_parts(images, sources, cfg.export.cvat, staging)
    except ExportBlocked as exc:
        raise _fail(staging, out_dir, pre.with_issues(error("packing_failed", str(exc))), "packing") from exc
    zips, check = _validate_written(written, sources, cfg, annset, frozenset(reasons))
    report = pre.merge(zips).merge(check)
    if not report.ok:
        raise _fail(staging, out_dir, report, "ZIP validation / self-check")
    parts, empty_rows = _write_files(staging, written, sources, reasons, tile_status)
    registry = id_registry(annset, CONTRACT_VERSION, run_id, cfg.export.cvat.manual_row_start)
    atomic_write_json(staging / ID_REGISTRY_JSON, registry)
    atomic_write_json(staging / VALIDATION_JSON, report.to_json_dict() | {"selfcheck": check.to_json_dict()})
    failed = tuple(sorted(t for t, r in reasons.items() if r == REASON_FAILED))
    summary = _summary(annset, cfg, run_id, written, parts, stats, empty_rows, report, time.monotonic() - started)
    atomic_write_json(staging / SUMMARY_JSON, summary)
    final = _publish(staging, out_dir)
    log_event(_log, "export.published", out_dir=str(final), n_zips=len(parts), sizes=[p.size for p in parts],
              n_tiles=len(sources), n_empty=summary["n_empty_tiles"], warnings=report.n_warnings)
    return UploadResult(final, tuple(final / p.zip_name for p in parts), report, final / MANIFEST_CSV,
                        final / EMPTY_TILES_CSV, final / ID_REGISTRY_JSON, final / SUMMARY_JSON,
                        final / VALIDATION_JSON, parts, failed)


def qa_error_count(qa_issues: gpd.GeoDataFrame | None) -> int:
    if qa_issues is None or qa_issues.empty:
        return 0
    return int((qa_issues["severity"] == Severity.ERROR.value).sum())


def require_no_qa_errors(qa_issues: gpd.GeoDataFrame | None, *, allow: bool) -> None:
    """Contract §2.7: AnnSet(model) with qa errors is not exported unless --allow-qa-errors."""
    n = qa_error_count(qa_issues)
    if n and not allow:
        codes = sorted(set(qa_issues.loc[qa_issues["severity"] == Severity.ERROR.value, "code"]))  # type: ignore[union-attr]
        raise ExportBlocked("AnnSet has qa errors (use --allow-qa-errors)", n_errors=n, codes=codes[:10])
    if n:
        log_event(_log, "export.qa_errors_allowed", level=30, n_errors=n)


__all__ = ["TileSource", "UploadResult", "export_upload", "pack_parts", "require_no_qa_errors", "tile_sources"]
