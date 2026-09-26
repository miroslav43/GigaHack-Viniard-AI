"""Stage protocol and registry. Stage modules are imported lazily, so importing this never loads torch."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, get_args

from vineyard.errors import StageError, StageNotImplemented

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext
    from vineyard.pipeline.runner import StageResult

StageRun = Callable[["RunContext"], "StageResult"]
Scope = Literal["global", "tile", "block"]
STAGES_PACKAGE: Final = "vineyard.pipeline.stages"

PRE_STAGES: Final[tuple[str, ...]] = (
    "ingest", "tile_prep", "nn_infer", "rows_detect", "rows_link", "blocks", "canopy",
    "interrow", "row_attrs", "waste", "assemble", "qa_previews",
)
POST_STAGES: Final[tuple[str, ...]] = (
    "import_marcaj", "derive", "passable", "targets", "route", "measure", "farms", "web_bundle",
)
STANDALONE_STAGES: Final[tuple[str, ...]] = ("export_cvat", "import_reference", "evaluate", "publish")
ALL_STAGES: Final[tuple[str, ...]] = (*PRE_STAGES, *POST_STAGES, *STANDALONE_STAGES)
STAGE_MODULES: Mapping[str, str] = MappingProxyType({name: f"{STAGES_PACKAGE}.{name}" for name in ALL_STAGES})


@dataclass(frozen=True)
class StageSpec:
    """What every `vineyard.pipeline.stages.<name>` module exports as `STAGE`."""

    name: str
    version: str
    scope: Scope
    cfg_keys: tuple[str, ...]
    requires: tuple[str, ...]
    run: StageRun
    description: str = ""

    def __post_init__(self) -> None:
        if self.scope not in get_args(Scope):
            raise ValueError(f"stage {self.name!r}: scope must be one of {get_args(Scope)}, got {self.scope!r}")
        if not self.name or not self.version:
            raise ValueError(f"stage {self.name!r}: name and version must be non-empty")
        object.__setattr__(self, "cfg_keys", tuple(self.cfg_keys))
        object.__setattr__(self, "requires", tuple(self.requires))


def module_available(module_name: str) -> bool:
    """True if `module_name` is importable (a missing parent package counts as missing)."""
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _module_of(name: str) -> str:
    modules = STAGE_MODULES
    if name not in modules:
        raise StageError("unknown stage", stage=name, known=", ".join(modules))
    return modules[name]


def stage_available(name: str) -> bool:
    """True if `name` is a registered stage whose module exists."""
    return name in STAGE_MODULES and module_available(STAGE_MODULES[name])


def load_stage(name: str) -> StageSpec:
    """Import `vineyard.pipeline.stages.<name>` and return its `STAGE`."""
    module_name = _module_of(name)
    if not module_available(module_name):
        raise StageNotImplemented("stage module not implemented yet", stage=name, module=module_name)
    module = importlib.import_module(module_name)
    spec = getattr(module, "STAGE", None)
    if spec is None:
        raise StageError("stage module defines no STAGE", stage=name, module=module_name)
    if not isinstance(spec, StageSpec):
        raise StageError("STAGE is not a StageSpec", stage=name, module=module_name, got=type(spec).__name__)
    if spec.name != name:
        raise StageError("STAGE.name does not match the registry name", stage=name, module=module_name,
                         got=spec.name)
    return spec


def select_stages(order: Sequence[str], *, from_: str | None = None, until: str | None = None) -> tuple[str, ...]:
    """Contiguous slice of `order` from `from_` to `until`, both inclusive."""
    names = tuple(order)
    for bound in (from_, until):
        if bound is not None and bound not in names:
            raise StageError("stage not in this sequence", stage=bound, order=", ".join(names))
    start = names.index(from_) if from_ is not None else 0
    stop = names.index(until) if until is not None else len(names) - 1
    if start > stop:
        raise StageError("--from comes after --until", from_=from_, until=until)
    return names[start : stop + 1]
