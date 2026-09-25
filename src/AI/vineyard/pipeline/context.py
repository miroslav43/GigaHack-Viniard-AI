"""Run identity and directory layout (contract §3.4). `tile_valid` + `in_*` are static in work/layers."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final
from zoneinfo import ZoneInfo

from vineyard import __version__
from vineyard.config import AppConfig, canonical_json, cfg_hash, resolved_config_dict
from vineyard.contracts.enums import Source
from vineyard.errors import ConfigError
from vineyard.pipeline.registry import STAGE_MODULES

if TYPE_CHECKING:
    from vineyard.pipeline.registry import StageSpec

POST_KIND: Final = "post"
RUN_KINDS: Final[tuple[str, ...]] = (*(s.value for s in Source), POST_KIND)
RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RESERVED_RUN_PREFIX: Final = "LATEST_"
RUN_DIGEST_LEN: Final = 6
GIT_SHA_SHORT: Final = 7
GIT_UNKNOWN: Final = "nogit00"
GIT_TIMEOUT_S: Final = 10.0
_HEX_RE: Final = re.compile(r"^[0-9a-f]+$")
_REF_SOURCE_RE: Final = re.compile(r"\d{8}T\d{4}-(model|marcaj|reference)-")


@dataclass(frozen=True)
class RunPaths:
    project_root: Path
    work_dir: Path
    tiles_dir: Path
    cache_dir: Path
    static_layers_dir: Path
    runs_dir: Path
    run_dir: Path
    layers_dir: Path
    annset_dir: Path
    exports_dir: Path
    qa_dir: Path
    metrics_dir: Path
    logs_dir: Path

    def tile_cache(self, stage: str, tile_id: str, ext: str) -> Path:
        """work/cache/<stage>/<tile_id>.<ext>"""
        return self.cache_dir / stage / f"{tile_id}.{ext.lstrip('.')}"

    @property
    def tile_index(self) -> Path:
        return self.work_dir / "tile_index.parquet"

    @property
    def tile_valid(self) -> Path:
        return self.static_layers_dir / "tile_valid.parquet"

    @property
    def run_json(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def pipeline_log(self) -> Path:
        return self.logs_dir / "pipeline.jsonl"


def make_run_paths(cfg: AppConfig, run_id: str) -> RunPaths:
    work = cfg.paths.work_dir
    run_dir = work / "runs" / run_id
    return RunPaths(
        project_root=cfg.paths.project_root,
        work_dir=work,
        tiles_dir=work / "tiles",
        cache_dir=work / "cache",
        static_layers_dir=work / "layers",
        runs_dir=work / "runs",
        run_dir=run_dir,
        layers_dir=run_dir / "layers",
        annset_dir=run_dir / "annset",
        exports_dir=run_dir / "exports",
        qa_dir=run_dir / "qa",
        metrics_dir=run_dir / "metrics",
        logs_dir=run_dir / "logs",
    )


def ensure_run_dirs(paths: RunPaths) -> None:
    """Create every directory of the layout (idempotent)."""
    for directory in (paths.tiles_dir, paths.cache_dir, paths.static_layers_dir, paths.layers_dir,
                      paths.annset_dir, paths.exports_dir, paths.qa_dir, paths.metrics_dir, paths.logs_dir):
        directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    source: Source
    cfg: AppConfig
    paths: RunPaths
    tile_filter: tuple[str, ...]
    force: frozenset[str]
    force_all: bool
    workers: int
    allow_failures: bool
    git_sha: str
    package_version: str
    annset_ref: str | None

    def selected_tiles(self, all_ids: Iterable[str]) -> tuple[str, ...]:
        """Unique ids matching any `--tiles` glob (all ids when no filter), sorted."""
        ids = sorted(set(all_ids))
        if not self.tile_filter:
            return tuple(ids)
        return tuple(t for t in ids if any(fnmatch.fnmatchcase(t, pat) for pat in self.tile_filter))

    def should_force(self, stage: str) -> bool:
        return self.force_all or stage in self.force

    def stage_cfg_digest(self, spec: StageSpec) -> str:
        return cfg_hash(self.cfg, spec.cfg_keys)

    def model_version(self, *, nn: str = "none", waste: str = "none") -> str:
        """pipe@<pkg>+<git7>;nn=<...>;waste=<...> (contract §2.2)."""
        return f"pipe@{self.package_version}+{self.git_sha[:GIT_SHA_SHORT]};nn={nn};waste={waste}"


def make_run_id(now: datetime, source: Source | str, digest: str) -> str:
    """YYYYMMDDTHHMM-<kind>-<hash6>, e.g. 20260926T0310-model-a1b2c3 (kind: a Source or "post")."""
    kind = str(source)
    if kind not in RUN_KINDS:
        raise ConfigError("unknown run kind", kind=kind, allowed=", ".join(RUN_KINDS))
    if len(digest) < RUN_DIGEST_LEN or not _HEX_RE.match(digest):
        raise ConfigError("run digest must be >= 6 lowercase hex chars", digest=digest)
    return f"{now:%Y%m%dT%H%M}-{kind}-{digest[:RUN_DIGEST_LEN]}"


def source_from_annset_ref(ref: str | None) -> Source:
    """Best-effort source of an --annset reference (LATEST_<SOURCE> or a run id / run dir)."""
    if not ref:
        return Source.MODEL
    if ref.startswith(RESERVED_RUN_PREFIX):
        name = ref.removeprefix(RESERVED_RUN_PREFIX).lower()
        return Source(name) if name in {s.value for s in Source} else Source.MODEL
    match = _REF_SOURCE_RE.search(Path(ref).name)
    return Source(match.group(1)) if match else Source.MODEL


def _git_sha(project_root: Path) -> str:
    git = shutil.which("git")
    if git is None:
        return GIT_UNKNOWN
    result = subprocess.run([git, "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True,
                            timeout=GIT_TIMEOUT_S, check=False)
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and _HEX_RE.match(sha) else GIT_UNKNOWN


def _resolve_workers(cfg: AppConfig, workers: int | None) -> int:
    if workers is None:
        configured = cfg.runtime.n_workers
        workers = max(1, (os.cpu_count() or 1) - cfg.runtime.workers_auto_reserve) if configured == "auto" \
            else configured
    if workers < 1:
        raise ConfigError("workers must be >= 1", workers=workers)
    return workers


def _validate_inputs(run_id: str | None, tiles: Sequence[str], force: Sequence[str], kind: str) -> None:
    if run_id is not None and (not RUN_ID_RE.match(run_id) or run_id.startswith(RESERVED_RUN_PREFIX)):
        raise ConfigError("invalid run id (letters, digits, . _ - only; not LATEST_*)", run_id=run_id)
    if any(not pattern.strip() for pattern in tiles):
        raise ConfigError("empty --tiles pattern", tiles=list(tiles))
    unknown = sorted(set(force) - set(STAGE_MODULES))
    if unknown:
        raise ConfigError("unknown stage in --force", stages=", ".join(unknown))
    if kind not in RUN_KINDS:
        raise ConfigError("unknown run kind", kind=kind, allowed=", ".join(RUN_KINDS))


def _run_digest(cfg: AppConfig, kind: str, tiles: Sequence[str], annset_ref: str | None) -> str:
    payload = {"cfg": resolved_config_dict(cfg), "kind": kind, "tiles": sorted(tiles), "annset": annset_ref}
    return hashlib.sha1(canonical_json(payload).encode("utf-8")).hexdigest()


def new_run_context(
    cfg: AppConfig,
    *,
    source: Source,
    run_id: str | None = None,
    tiles: Sequence[str] = (),
    force: Sequence[str] = (),
    force_all: bool = False,
    workers: int | None = None,
    allow_failures: bool | None = None,
    annset_ref: str | None = None,
    kind: str | None = None,
    now: datetime | None = None,
) -> RunContext:
    """Build a RunContext; no directory is created (see `ensure_run_dirs`)."""
    source = Source(source)
    run_kind = kind or source.value
    _validate_inputs(run_id, tiles, force, run_kind)
    if run_id is None:
        stamp = now or datetime.now(ZoneInfo(cfg.logging.tz))
        run_id = make_run_id(stamp, run_kind, _run_digest(cfg, run_kind, tiles, annset_ref))
    return RunContext(
        run_id=run_id,
        source=source,
        cfg=cfg,
        paths=make_run_paths(cfg, run_id),
        tile_filter=tuple(tiles),
        force=frozenset(force),
        force_all=force_all,
        workers=_resolve_workers(cfg, workers),
        allow_failures=cfg.runtime.allow_failures if allow_failures is None else allow_failures,
        git_sha=_git_sha(cfg.paths.project_root),
        package_version=__version__,
        annset_ref=annset_ref,
    )
