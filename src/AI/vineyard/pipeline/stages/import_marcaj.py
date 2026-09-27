"""Stage import_marcaj: Marcaj CVAT 1.1 export(s) -> AnnSet(marcaj) (design 04 §3.1, A§4.13.8).

Reads every `import.files` path (a `.zip` with annotations.xml at any depth, images ignored, or a `.xml`),
merges all images into one document (a repeated tile: `import.duplicate_policy`, the last copy wins with
a warning), drops images that are not contract tiles (qa error), requires the 311-tile contract grid
(`import.require_all_tiles`; `--partial` lowers a missing tile to the `images_missing` warning), runs the
import checks, writes runs/<run_id>/annset/ + qa/qa_issues.parquet and points runs/LATEST_MARCAJ at the run.
Never reuses an older run: `vineyard final` runs the post stages on this very run id.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

import geopandas as gpd

from vineyard.annset.import_checks import MAX_LISTED, ImportCheckParams, check_annset
from vineyard.annset.io import update_latest_link, write_annset
from vineyard.annset.model import ANNSET_LAYERS, AnnSet
from vineyard.config import AppConfig
from vineyard.contracts.enums import Severity, Source
from vineyard.contracts.qa import QaIssue, issues_to_gdf
from vineyard.cvat.model import CvatDocument, CvatImage
from vineyard.cvat.reader import image_key, load_annotation_bytes, merge_documents, parse_xml
from vineyard.cvat.to_annset import document_to_annset
from vineyard.errors import CvatFormatError, StageError
from vineyard.geo.tiling import TILE_PX, existing_tile_ids, tile_ref
from vineyard.geo.vector_io import write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult
from vineyard.qa.review import issues_from_frame

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext

STAGE_NAME: Final = "import_marcaj"
STAGE_VERSION: Final = "1"
CFG_KEYS: Final = ("import", "canopy", "export.cvat.max_canopy_interrow_overlap_m2")
QA_FILE: Final = "qa_issues.parquet"
MODEL_VERSION_PREFIX: Final = "marcaj-export@"
SHA_SHORT: Final = 8
TILE_SUFFIX: Final = ".tif"
# Marcaj may serve a tile under another raster extension; the stem is still the tile id.
IMAGE_SUFFIXES: Final = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})
CODE_UNKNOWN_IMAGE: Final = "schema_violation"
EVENT_WRITTEN: Final = "import_marcaj.written"

_log = get_logger("pipeline.stages.import_marcaj")


@dataclass(frozen=True)
class MarcajFile:
    """One export as read: its path, the annotations.xml bytes and their sha256."""

    path: Path
    xml: bytes
    sha256: str


def read_export(path: Path) -> MarcajFile:
    """annotations.xml of one `.zip` / `.xml` export; a missing or unreadable file is a StageError."""
    try:
        xml = load_annotation_bytes(Path(path))
    except CvatFormatError as exc:
        raise StageError(f"cannot read Marcaj export {path}: {exc}", stage=STAGE_NAME, path=str(path)) from exc
    return MarcajFile(Path(path), xml, hashlib.sha256(xml).hexdigest())


def export_model_version(files: Sequence[MarcajFile]) -> str:
    """marcaj-export@<sha8>: sha256 of the files' annotations.xml digests, in the order given."""
    digest = hashlib.sha256("\n".join(f.sha256 for f in files).encode("ascii")).hexdigest()
    return f"{MODEL_VERSION_PREFIX}{digest[:SHA_SHORT]}"


def tile_id_of(name: str) -> str:
    """Tile id of an image name: basename without its raster extension (`images/x.tif` -> `x`)."""
    key = PurePosixPath(image_key(name))
    return key.stem if key.suffix.lower() in IMAGE_SUFFIXES else key.name


def _normalized(doc: CvatDocument) -> CvatDocument:
    """Every image renamed `<tile_id>.tif`, so one tile under two spellings is caught as a duplicate."""
    images = tuple(dataclasses.replace(img, name=f"{tile_id_of(img.name)}{TILE_SUFFIX}") for img in doc.images)
    return dataclasses.replace(doc, images=images)


def _parse(file: MarcajFile, accept_masks: bool) -> tuple[CvatDocument, tuple[QaIssue, ...]]:
    try:
        doc, issues = parse_xml(file.xml, source_name=str(file.path), accept_masks=accept_masks)
    except CvatFormatError as exc:
        raise StageError(f"unreadable annotations.xml in {file.path}: {exc}", stage=STAGE_NAME,
                         path=str(file.path)) from exc
    return _normalized(doc), issues


def merge_exports(files: Sequence[MarcajFile], cfg: AppConfig) -> tuple[CvatDocument, tuple[QaIssue, ...]]:
    """All images of all files in one document (duplicates per `import.duplicate_policy`) + reader issues."""
    parsed: list[tuple[str, CvatDocument]] = []
    issues: list[QaIssue] = []
    for file in files:
        doc, file_issues = _parse(file, cfg.import_.accept_masks)
        parsed.append((str(file.path), doc))
        issues.extend(file_issues)
    try:
        merged, merge_issues = merge_documents(parsed, cfg.import_.duplicate_policy)
    except CvatFormatError as exc:
        raise StageError(f"duplicate images across Marcaj exports: {exc}", stage=STAGE_NAME) from exc
    return merged, (*issues, *merge_issues)


def _is_contract_tile(img: CvatImage, known: frozenset[str]) -> bool:
    return tile_id_of(img.name) in known and (img.width, img.height) == (TILE_PX, TILE_PX)


def split_unknown(doc: CvatDocument) -> tuple[CvatDocument, tuple[QaIssue, ...]]:
    """Drop images that are not 2048 px contract tiles; each one becomes a qa error."""
    known = existing_tile_ids()
    keep = tuple(img for img in doc.images if _is_contract_tile(img, known))
    issues = tuple(
        QaIssue(Severity.ERROR, CODE_UNKNOWN_IMAGE, "", img.name,
                f"imaginea {img.name} ({img.width}x{img.height}) nu e un tile al contractului; ignorată")
        for img in doc.images if not _is_contract_tile(img, known))
    return dataclasses.replace(doc, images=keep), issues


def check_coverage(tile_ids: Sequence[str], params: ImportCheckParams) -> None:
    """Missing contract tiles raise when `require_all_tiles` (else check_annset warns images_missing)."""
    missing = sorted(set(params.expected_tile_ids) - set(tile_ids))
    if not missing or not params.require_all_tiles:
        return
    raise StageError(
        f"Marcaj export has {len(set(tile_ids))} of {len(params.expected_tile_ids)} tiles, {len(missing)} missing "
        f"(first: {', '.join(missing[:MAX_LISTED])}); pass every export ZIP, or --partial for an intermediate one",
        stage=STAGE_NAME, n_missing=len(missing))


def build_marcaj_annset(cfg: AppConfig, files: Sequence[MarcajFile], *,
                        run_id: str) -> tuple[AnnSet, gpd.GeoDataFrame]:
    """AnnSet(marcaj) + qa_issues (reader, merge, unknown images, conversion, import checks)."""
    merged, read_issues = merge_exports(files, cfg)
    doc, unknown_issues = split_unknown(merged)
    tile_ids = [tile_id_of(img.name) for img in doc.images]
    params = ImportCheckParams.from_config(cfg)
    check_coverage(tile_ids, params)
    model_version = export_model_version(files)
    annset, qa = document_to_annset(
        doc, tile_refs={t: tile_ref(t) for t in tile_ids}, source=Source.MARCAJ, run_id=run_id,
        model_version=model_version, cfg=cfg.import_, canopy_cfg=cfg.canopy,
        inputs=tuple(str(f.path) for f in files), extra_issues=(*read_issues, *unknown_issues),
    )
    issues = (*issues_from_frame(qa), *check_annset(annset, params))
    return annset, issues_to_gdf(issues, source=Source.MARCAJ, run_id=run_id, model_version=model_version)


def _files(cfg: AppConfig) -> tuple[MarcajFile, ...]:
    if not cfg.import_.files:
        raise StageError("no Marcaj export given (import.files is empty)", stage=STAGE_NAME)
    return tuple(read_export(Path(p)) for p in cfg.import_.files)


def _metrics(annset: AnnSet, qa: gpd.GeoDataFrame, n_files: int) -> dict[str, float]:
    counts = annset.counts()
    severities = qa["severity"].value_counts()
    return {**{name: float(counts.get(name, 0)) for name in ANNSET_LAYERS}, "files": float(n_files),
            "qa_errors": float(severities.get(Severity.ERROR.value, 0)),
            "qa_warnings": float(severities.get(Severity.WARNING.value, 0))}


def run(ctx: RunContext) -> StageResult:
    files = _files(ctx.cfg)
    annset, qa = build_marcaj_annset(ctx.cfg, files, run_id=ctx.run_id)
    annset_dir = write_annset(annset, ctx.paths.annset_dir)
    qa_path = write_layer(qa, "qa_issues", ctx.paths.qa_dir / QA_FILE)
    update_latest_link(ctx.paths.work_dir, Source.MARCAJ, ctx.paths.run_dir)
    metrics = _metrics(annset, qa, len(files))
    log_event(_log, EVENT_WRITTEN, stage=STAGE_NAME, n_tiles=annset.meta.n_tiles, n_qa=len(qa), **annset.counts())
    return StageResult(stage=STAGE_NAME, n_items=annset.meta.n_tiles, n_cached=0, n_failed=0,
                       outputs=(annset_dir, qa_path), metrics=metrics)


STAGE: Final = StageSpec(
    name=STAGE_NAME,
    version=STAGE_VERSION,
    scope="global",
    cfg_keys=CFG_KEYS,
    requires=(),
    run=run,
    description="Marcaj CVAT 1.1 export(s) -> AnnSet(marcaj) + qa_issues + LATEST_MARCAJ",
)
