"""Copy a src/AI run folder to a new run id (no exports/, no heavy QA sub-folders).

The frozen fallback run (complete-v4) is never written: it can only be a source. An existing target is
replaced only when it was created by this module (marker file FTE_MARKER).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Final

from fte.paths import AI_RUNS

FROZEN_RUNS: Final = frozenset({"complete-v4"})
RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FTE_MARKER: Final = "fte_integrate.json"
COPY_DIRS: Final = ("layers", "annset", "metrics")
COPY_FILES: Final = ("run.json", "config.json")
QA_TOP_LEVEL_ONLY: Final = "qa"  # previews/ and waste_crops/ are large and stale after edits
LOGS_DIR: Final = "logs"
LOGS_MAX_BYTES: Final = 50 * 1024 * 1024

_log = logging.getLogger(__name__)


class RunCopyError(RuntimeError):
    pass


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _check_target(out_run: str, runs_dir: Path, overwrite: bool) -> Path:
    if not RUN_ID_RE.match(out_run) or out_run.startswith("LATEST"):
        raise RunCopyError(f"invalid output run id {out_run!r}")
    if out_run in FROZEN_RUNS:
        raise RunCopyError(f"refusing to write into the frozen run {out_run!r}")
    target = runs_dir / out_run
    if target.is_symlink():
        raise RunCopyError(f"target {target} is a symlink; refusing")
    if target.exists():
        if not overwrite:
            raise RunCopyError(f"target run {target} exists (use --overwrite)")
        if not (target / FTE_MARKER).is_file():
            raise RunCopyError(f"target run {target} was not created by fte.integrate; refusing to replace it")
        shutil.rmtree(target)
    return target


def _copy_qa(src: Path, dst: Path) -> None:
    qa = src / QA_TOP_LEVEL_ONLY
    if not qa.is_dir():
        return
    (dst / QA_TOP_LEVEL_ONLY).mkdir(parents=True, exist_ok=True)
    for item in qa.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / QA_TOP_LEVEL_ONLY / item.name)


def copy_run(base_run: str, out_run: str, *, runs_dir: Path = AI_RUNS, overwrite: bool = False) -> Path:
    """Copy runs/<base_run> to runs/<out_run>; returns the new run directory."""
    src = (runs_dir / base_run).resolve()
    if not (src / "annset" / "annset.json").is_file():
        raise RunCopyError(f"base run {src} has no complete annset/")
    if src.name == out_run:
        raise RunCopyError("output run equals the base run")
    target = _check_target(out_run, runs_dir, overwrite)
    tmp = runs_dir / f".{out_run}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name in COPY_DIRS:
        if (src / name).is_dir():
            shutil.copytree(src / name, tmp / name)
    for name in COPY_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, tmp / name)
    _copy_qa(src, tmp)
    logs = src / LOGS_DIR
    if logs.is_dir() and _dir_size(logs) <= LOGS_MAX_BYTES:
        shutil.copytree(logs, tmp / LOGS_DIR)
    marker = {"base_run": src.name, "out_run": out_run, "created_at": datetime.now().astimezone().isoformat()}
    (tmp / FTE_MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    tmp.rename(target)
    _log.info("copied run %s -> %s", src, target)
    return target


def update_marker(run_dir: Path, **fields: object) -> dict[str, object]:
    """Merge `fields` into the run's marker file; returns the new marker content."""
    path = Path(run_dir) / FTE_MARKER
    if not path.is_file():
        raise RunCopyError(f"{path} missing: not an fte.integrate run")
    doc = json.loads(path.read_text(encoding="utf-8")) | {k: v for k, v in fields.items()}
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return doc
