"""AnnSet on disk: `<run>/annset/{canopies,row_pieces,interrow_pieces,waste}.parquet + annset.json`.

annset.json is written last, so a directory with annset.json is always complete. Run references
(`--annset`) resolve through `work/runs/LATEST_<SOURCE>` symlinks, a run id or a path.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from vineyard.annset.model import ANNSET_LAYERS, TILE_COLUMN, AnnSet, AnnSetMeta
from vineyard.contracts.enums import Source
from vineyard.errors import SchemaError, StageError
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.pipeline.atomic import atomic_write_json, tmp_path_for

ANNSET_JSON: Final = "annset.json"
ANNSET_DIRNAME: Final = "annset"
RUNS_DIRNAME: Final = "runs"
LATEST_LINKS: Final[Mapping[Source, str]] = MappingProxyType({s: f"LATEST_{s.value.upper()}" for s in Source})


def _layer_path(annset_dir: Path, name: str) -> Path:
    return annset_dir / f"{name}.parquet"


def _check_tiles_covered(annset: AnnSet) -> None:
    covered = frozenset(annset.meta.tile_ids)
    for name in ANNSET_LAYERS:
        extra = sorted(set(annset.layer(name)[TILE_COLUMN].dropna()) - covered)
        if extra:
            raise SchemaError("objects on tiles not in meta.tile_ids", layer=name, run_id=annset.meta.run_id,
                              examples=extra[:5])


def write_annset(annset: AnnSet, annset_dir: Path) -> Path:
    """Validate and write every layer, then annset.json (with recomputed counts). Returns annset_dir."""
    target = Path(annset_dir)
    _check_tiles_covered(annset)
    for name in ANNSET_LAYERS:
        write_layer(annset.layer(name), name, _layer_path(target, name))
    meta = annset.meta.to_json_dict() | {"counts": annset.counts()}
    atomic_write_json(target / ANNSET_JSON, meta)
    return target


def _read_meta(annset_dir: Path) -> AnnSetMeta:
    path = annset_dir / ANNSET_JSON
    if not path.is_file():
        raise SchemaError("annset.json not found", annset_dir=str(annset_dir))
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError("annset.json is not valid JSON", path=str(path), error=str(exc)) from exc
    return AnnSetMeta.from_json_dict(doc)


def read_annset(annset_dir: Path, *, validate: bool = True) -> AnnSet:
    source = Path(annset_dir)
    meta = _read_meta(source)
    layers = {}
    for name in ANNSET_LAYERS:
        path = _layer_path(source, name)
        if not path.is_file():
            raise SchemaError("AnnSet layer file missing", layer=name, annset_dir=str(source))
        layers[name] = read_layer(path, name, validate=validate)
    return AnnSet(meta=meta, **layers)


def _runs_dir(work_dir: Path) -> Path:
    return Path(work_dir) / RUNS_DIRNAME


def resolve_run_dir(work_dir: Path, ref: str) -> Path:
    """Run directory for `ref`: LATEST_MODEL | LATEST_MARCAJ | LATEST_REFERENCE | run id | path.

    A path to a run's annset/ directory resolves to the run itself.
    """
    if not ref:
        raise StageError("empty run reference", work_dir=str(work_dir))
    as_path = Path(ref)
    candidates = [as_path] if as_path.is_absolute() or os.sep in ref else [_runs_dir(work_dir) / ref, as_path]
    for cand in candidates:
        if cand.is_dir():
            run = cand.resolve()
            return run.parent if run.name == ANNSET_DIRNAME and (run / ANNSET_JSON).is_file() else run
    raise StageError("run not found", ref=ref, runs_dir=str(_runs_dir(work_dir)))


def update_latest_link(work_dir: Path, source: Source, run_dir: Path) -> None:
    """Point work/runs/LATEST_<SOURCE> at run_dir (relative when inside runs/), replaced atomically."""
    run = Path(run_dir)
    if not run.is_dir():
        raise StageError("cannot link a missing run directory", run_dir=str(run), source=str(source))
    runs = _runs_dir(work_dir)
    runs.mkdir(parents=True, exist_ok=True)
    link = runs / LATEST_LINKS[Source(source)]
    resolved = run.resolve()
    target = resolved.relative_to(runs.resolve()) if resolved.is_relative_to(runs.resolve()) else resolved
    tmp = tmp_path_for(link)
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target, tmp, target_is_directory=True)
    os.replace(tmp, link)
