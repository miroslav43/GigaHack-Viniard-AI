"""Stage import_reference: organizers' examples annotations.xml -> AnnSet(reference) (design 01 §2.7).

Writes runs/<run_id>/annset/ + qa/qa_issues.parquet and points runs/LATEST_REFERENCE at the run.
Idempotent: annset.json carries a content key (stage version, config, XML sha256); a later run with
the same key reuses that run (LATEST_REFERENCE re-pointed) unless the stage is forced.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

import geopandas as gpd

from vineyard.annset.io import ANNSET_JSON, read_annset, update_latest_link, write_annset
from vineyard.annset.model import ANNSET_LAYERS, AnnSet
from vineyard.config import AppConfig
from vineyard.contracts.enums import Source
from vineyard.cvat.reader import image_key, parse_xml
from vineyard.cvat.to_annset import document_to_annset
from vineyard.errors import StageError
from vineyard.geo.tiling import TileRef, existing_tile_ids, tile_ref
from vineyard.geo.vector_io import write_layer
from vineyard.logging_setup import get_logger, log_event
from vineyard.pipeline.cache import cache_key, read_key, write_key
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.runner import StageResult

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext

STAGE_NAME: Final = "import_reference"
STAGE_VERSION: Final = "1"
CFG_KEYS: Final = ("paths.examples_dir", "import", "canopy")
EXAMPLES_XML: Final = "annotations.xml"
QA_FILE: Final = "qa_issues.parquet"
MODEL_VERSION_PREFIX: Final = "reference-examples@"
SHA_SHORT: Final = 8
RUN_GLOB: Final = f"*-{Source.REFERENCE.value}-*"
EVENT_REUSED: Final = "import_reference.reused"
EVENT_WRITTEN: Final = "import_reference.written"

_log = get_logger("pipeline.stages.import_reference")


def examples_xml_path(cfg: AppConfig) -> Path:
    return Path(cfg.paths.examples_dir) / EXAMPLES_XML


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tile_refs(tile_ids: list[str]) -> Mapping[str, TileRef]:
    known = existing_tile_ids()
    return {t: tile_ref(t) for t in tile_ids if t in known}


def build_reference_annset(cfg: AppConfig, xml: bytes, *, run_id: str,
                           source_name: str) -> tuple[AnnSet, gpd.GeoDataFrame]:
    """AnnSet(reference) + qa_issues from the examples XML bytes (tiles located on the contract grid)."""
    doc, issues = parse_xml(xml, source_name=source_name, accept_masks=cfg.import_.accept_masks)
    tile_ids = [image_key(img.name).removesuffix(".tif") for img in doc.images]
    return document_to_annset(
        doc, tile_refs=_tile_refs(tile_ids), source=Source.REFERENCE, run_id=run_id,
        model_version=f"{MODEL_VERSION_PREFIX}{_sha256(xml)[:SHA_SHORT]}", cfg=cfg.import_,
        canopy_cfg=cfg.canopy, inputs=(source_name,), extra_issues=issues,
    )


def find_reusable_run(runs_dir: Path, key: str) -> Path | None:
    """Newest reference run whose annset.json key equals `key`."""
    if not Path(runs_dir).is_dir():
        return None
    hits = [p for p in sorted(Path(runs_dir).glob(RUN_GLOB)) if read_key(p / "annset" / ANNSET_JSON) == key]
    return hits[-1] if hits else None


def _read_examples(cfg: AppConfig) -> tuple[Path, bytes]:
    path = examples_xml_path(cfg)
    if not path.is_file():
        raise StageError("examples annotations.xml not found", stage=STAGE_NAME, path=str(path))
    return path, path.read_bytes()


def _metrics(counts: dict[str, int]) -> dict[str, float]:
    return {name: float(counts.get(name, 0)) for name in ANNSET_LAYERS}


def _reuse(ctx: RunContext, run_dir: Path) -> StageResult:
    update_latest_link(ctx.paths.work_dir, Source.REFERENCE, run_dir)
    annset = read_annset(run_dir / "annset", validate=False)
    log_event(_log, EVENT_REUSED, stage=STAGE_NAME, run_dir=str(run_dir))
    return StageResult(stage=STAGE_NAME, n_items=annset.meta.n_tiles, n_cached=1, n_failed=0,
                       outputs=(run_dir / "annset",), metrics=_metrics(annset.counts()))


def run(ctx: RunContext) -> StageResult:
    path, xml = _read_examples(ctx.cfg)
    key = cache_key(STAGE_NAME, STAGE_VERSION, ctx.stage_cfg_digest(STAGE), [_sha256(xml)])
    previous = None if ctx.should_force(STAGE_NAME) else find_reusable_run(ctx.paths.runs_dir, key)
    if previous is not None:
        return _reuse(ctx, previous)
    annset, qa = build_reference_annset(ctx.cfg, xml, run_id=ctx.run_id, source_name=str(path))
    annset_dir = write_annset(annset, ctx.paths.annset_dir)
    qa_path = write_layer(qa, "qa_issues", ctx.paths.qa_dir / QA_FILE)
    write_key(annset_dir / ANNSET_JSON, key)
    update_latest_link(ctx.paths.work_dir, Source.REFERENCE, ctx.paths.run_dir)
    log_event(_log, EVENT_WRITTEN, stage=STAGE_NAME, n_qa=len(qa), **annset.counts())
    return StageResult(stage=STAGE_NAME, n_items=annset.meta.n_tiles, n_cached=0, n_failed=0,
                       outputs=(annset_dir, qa_path), metrics=_metrics(annset.counts()))


STAGE: Final = StageSpec(
    name=STAGE_NAME,
    version=STAGE_VERSION,
    scope="global",
    cfg_keys=CFG_KEYS,
    requires=(),
    run=run,
    description="05_examples annotations.xml -> AnnSet(reference) + LATEST_REFERENCE",
)
