"""Stage registry: names/orders per the plan (R2), lazy loading, selection."""

import os
import subprocess
import sys
import types
from pathlib import Path
from types import MappingProxyType

import pytest

from vineyard.errors import StageError, StageNotImplemented
from vineyard.pipeline import registry
from vineyard.pipeline.registry import (
    POST_STAGES,
    PRE_STAGES,
    STAGE_MODULES,
    STANDALONE_STAGES,
    StageSpec,
    load_stage,
    select_stages,
    stage_available,
)


def test_stage_orders_exactly_as_plan() -> None:
    assert PRE_STAGES == (
        "ingest", "tile_prep", "nn_infer", "rows_detect", "rows_link", "blocks", "canopy",
        "interrow", "row_attrs", "waste", "assemble", "qa_previews",
    )
    assert POST_STAGES == ("import_marcaj", "derive", "passable", "targets", "route", "measure", "farms",
                           "web_bundle")
    assert STANDALONE_STAGES == ("export_cvat", "import_reference", "evaluate", "publish")


def test_every_stage_maps_to_its_module() -> None:
    names = (*PRE_STAGES, *POST_STAGES, *STANDALONE_STAGES)
    assert tuple(STAGE_MODULES) == names
    assert all(STAGE_MODULES[n] == f"vineyard.pipeline.stages.{n}" for n in names)
    with pytest.raises(TypeError):
        STAGE_MODULES["x"] = "y"  # type: ignore[index]


def test_registry_import_is_lazy_and_torch_free() -> None:
    code = (
        "import sys, vineyard.pipeline.registry as r; "
        "print('torch' in sys.modules, any(m.startswith('vineyard.pipeline.stages.') for m in sys.modules))"
    )
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root)}
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"from_": "canopy", "until": "assemble"}, ("canopy", "interrow", "row_attrs", "waste", "assemble")),
        ({"from_": None, "until": "tile_prep"}, ("ingest", "tile_prep")),
        ({"from_": "assemble", "until": None}, ("assemble", "qa_previews")),
        ({"from_": None, "until": None}, PRE_STAGES),
        ({"from_": "waste", "until": "waste"}, ("waste",)),
    ],
)
def test_select_stages(kwargs: dict, expected: tuple[str, ...]) -> None:
    assert select_stages(PRE_STAGES, **kwargs) == expected


@pytest.mark.parametrize(("from_", "until"), [("derive", None), (None, "nope"), ("assemble", "canopy")])
def test_select_stages_rejects_bad_bounds(from_: str | None, until: str | None) -> None:
    with pytest.raises(StageError):
        select_stages(PRE_STAGES, from_=from_, until=until)


def _fake_run(ctx: object) -> object:
    return ctx


@pytest.fixture
def fake_stage(monkeypatch: pytest.MonkeyPatch) -> str:
    module = types.ModuleType("vineyard.pipeline.stages._fake_ok")
    module.STAGE = StageSpec(name="derive", version="1", scope="global", cfg_keys=("derive",),
                             requires=("assemble",), run=_fake_run, description="test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(registry, "STAGE_MODULES", MappingProxyType({**STAGE_MODULES, "derive": module.__name__}))
    return module.__name__


def test_load_stage_returns_spec(fake_stage: str) -> None:
    spec = load_stage("derive")
    assert spec.name == "derive" and spec.scope == "global" and spec.cfg_keys == ("derive",)
    assert stage_available("derive")


def test_load_stage_missing_module_raises_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = "vineyard.pipeline.stages._definitely_missing"
    monkeypatch.setattr(registry, "STAGE_MODULES", MappingProxyType({**STAGE_MODULES, "derive": missing}))
    assert not stage_available("derive")
    with pytest.raises(StageNotImplemented) as info:
        load_stage("derive")
    assert info.value.context["module"] == missing
    assert info.value.exit_code == 3


def test_missing_parent_package_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = "vineyard._no_such_pkg.stage"
    monkeypatch.setattr(registry, "STAGE_MODULES", MappingProxyType({**STAGE_MODULES, "derive": missing}))
    with pytest.raises(StageNotImplemented):
        load_stage("derive")


def test_load_unknown_stage_fails() -> None:
    with pytest.raises(StageError, match="unknown stage"):
        load_stage("frobnicate")
    assert not stage_available("frobnicate")


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({}, "STAGE"),
        ({"STAGE": "not a spec"}, "StageSpec"),
        ({"STAGE": StageSpec("route", "1", "global", (), (), _fake_run)}, "name"),
    ],
)
def test_load_stage_rejects_malformed_modules(monkeypatch: pytest.MonkeyPatch, attrs: dict, match: str) -> None:
    module = types.ModuleType("vineyard.pipeline.stages._fake_bad")
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(registry, "STAGE_MODULES", MappingProxyType({**STAGE_MODULES, "derive": module.__name__}))
    with pytest.raises(StageError, match=match) as info:
        load_stage("derive")
    assert not isinstance(info.value, StageNotImplemented)


def test_stage_spec_validation() -> None:
    with pytest.raises(ValueError, match="scope"):
        StageSpec("x", "1", "planet", (), (), _fake_run)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="version"):
        StageSpec("x", "", "tile", (), (), _fake_run)
    spec = StageSpec("x", "1", "tile", ["canopy"], ["blocks"], _fake_run)  # type: ignore[arg-type]
    assert spec.cfg_keys == ("canopy",) and spec.requires == ("blocks",)
