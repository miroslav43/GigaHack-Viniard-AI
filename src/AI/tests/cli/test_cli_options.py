"""cli_options helpers: option injection, YAML-safe values, lazy sub-app mounting."""

import sys
import types
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from vineyard import cli_options
from vineyard.cli_options import CommonOptions, mount_subapp, with_common_options, yaml_list, yaml_value

runner = CliRunner()


def test_yaml_values_round_trip_awkward_paths() -> None:
    paths = ["/a/data & info/x.zip", "/b/ÎnălțimeȘ.xml", "&anchor", "*ref: x"]
    assert yaml.safe_load(yaml_list(paths)) == paths
    assert yaml.safe_load(yaml_value("/x/y & z")) == "/x/y & z"


def test_common_options_with_sets_is_immutable() -> None:
    base = CommonOptions(sets=("a=1",))
    extended = base.with_sets("b=2")
    assert base.sets == ("a=1",) and extended.sets == ("a=1", "b=2")


def test_injected_options_reach_the_command() -> None:
    seen: list[CommonOptions] = []
    app = typer.Typer()

    @app.command("x")
    @with_common_options
    def x(opts: CommonOptions, name: str = "n") -> None:
        seen.append(opts)
        typer.echo(name)

    @app.command("y")
    def y() -> None:
        """second command so `x` is not the default."""

    result = runner.invoke(app, ["x", "--tiles", "a*", "--tiles", "b*", "--force-all", "--set", "k=v",
                                 "--config", "c.yaml", "--annset", "LATEST_MODEL", "--name", "zz"])
    assert result.exit_code == 0, result.output
    assert seen[0] == CommonOptions(configs=(Path("c.yaml"),), sets=("k=v",), tiles=("a*", "b*"),
                                    force_all=True, annset="LATEST_MODEL")
    assert "zz" in result.output


def _parent() -> typer.Typer:
    parent = typer.Typer()

    @parent.command("noop")
    def noop() -> None:
        """keeps the parent a group."""

    return parent


def test_mount_missing_module_is_placeholder() -> None:
    parent = _parent()
    assert mount_subapp(parent, "nnx", "vineyard._no_such_module.cli", "help") is False
    result = runner.invoke(parent, ["nnx", "train", "--smoke"])
    assert result.exit_code == 3
    assert "comanda nu e implementată încă: vineyard._no_such_module.cli" in result.output


def test_mount_broken_module_is_placeholder_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = types.ModuleType("vineyard._broken_cli_for_test")  # no `app` attribute
    monkeypatch.setitem(sys.modules, broken.__name__, broken)
    parent = _parent()
    assert mount_subapp(parent, "brk", broken.__name__, "help") is False
    result = runner.invoke(parent, ["brk"])
    assert result.exit_code == 3 and "import eșuat" in result.output


def test_mount_real_subapp(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("vineyard._good_cli_for_test")
    sub = typer.Typer()

    @sub.command("hello")
    def hello() -> None:
        typer.echo("salut")

    @sub.command("bye")
    def bye() -> None:
        typer.echo("pa")

    module.app = sub
    monkeypatch.setitem(sys.modules, module.__name__, module)
    parent = _parent()
    assert mount_subapp(parent, "good", module.__name__, "help") is True
    assert "salut" in runner.invoke(parent, ["good", "hello"]).output


def test_missing_modules_lists_runner_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_options, "RUNNER_MODULE", "vineyard._missing_runner_x")
    missing = cli_options.missing_modules(("ingest",))
    assert missing[0] == "vineyard._missing_runner_x"
