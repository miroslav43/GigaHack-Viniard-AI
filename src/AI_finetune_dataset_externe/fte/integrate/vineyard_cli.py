"""Subprocess wrapper around the `vineyard` CLI of src/AI, with the Makefile environment."""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from fte.paths import AI_ROOT

VINEYARD_BIN: Final = AI_ROOT / ".venv" / "bin" / "vineyard"
UNSET_ENV: Final = ("PROJ_DATA", "PROJ_LIB", "GDAL_DATA", "CONDA_PREFIX")
MAKE_ENV: Final[Mapping[str, str]] = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "GDAL_NUM_THREADS": "1", "GDAL_CACHEMAX": "256", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
}
DEFAULT_TIMEOUT_S: Final = 3600
TAIL_CHARS: Final = 4000

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CliResult:
    args: tuple[str, ...]
    returncode: int
    stdout_tail: str
    stderr_tail: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class VineyardCliError(RuntimeError):
    pass


def vineyard_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in (os.environ if base is None else base).items() if k not in UNSET_ENV}
    return env | dict(MAKE_ENV)


def run_vineyard(args: Sequence[str], *, timeout_s: int = DEFAULT_TIMEOUT_S, check: bool = True,
                 log_path: Path | None = None) -> CliResult:
    """Run `vineyard <args>` with cwd=src/AI; raises VineyardCliError on failure when check."""
    if not VINEYARD_BIN.is_file():
        raise VineyardCliError(f"vineyard CLI not found at {VINEYARD_BIN}")
    cmd = [str(VINEYARD_BIN), *args]
    _log.info("running: vineyard %s", " ".join(args))
    proc = subprocess.run(cmd, cwd=AI_ROOT, env=vineyard_env(), capture_output=True, text=True,
                          timeout=timeout_s, check=False)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"$ vineyard {' '.join(args)}\n{proc.stdout}\n{proc.stderr}\n[exit {proc.returncode}]\n")
    result = CliResult(tuple(args), proc.returncode, proc.stdout[-TAIL_CHARS:], proc.stderr[-TAIL_CHARS:])
    if check and not result.ok:
        raise VineyardCliError(f"vineyard {' '.join(args)} failed (exit {proc.returncode}):\n"
                               f"{result.stderr_tail or result.stdout_tail}")
    return result
