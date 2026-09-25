"""CLI wiring: --help completeness, lazy dispatch, exit codes, Romanian messages."""

import re
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vineyard import cli_options, cli_post
from vineyard.cli import app
from vineyard.contracts.enums import Source
from vineyard.errors import ExportBlocked
from vineyard.logging_setup import teardown_logging
from vineyard.pipeline import registry
from vineyard.pipeline.context import RunContext
from vineyard.pipeline.registry import PRE_STAGES

runner = CliRunner()
FAKE_RUNNER = "vineyard.pipeline._fake_runner_for_cli_tests"
EXPECTED_COMMANDS = (
    "ingest", "run", "all", "export-cvat", "import-marcaj", "from-marcaj", "import-reference", "derive",
    "passable", "targets", "route", "measure", "post", "final", "publish", "eval-examples", "qa", "nn", "waste",
    "web", "bench", "cvat", "config", "doctor",
)


@dataclass
class FakeReport:
    exit_code: int


@dataclass
class Recorder:
    calls: list[tuple[RunContext, tuple[str, ...]]] = field(default_factory=list)
    codes: list[int] = field(default_factory=list)
    error: Exception | None = None

    def run_stages(self, ctx: RunContext, names: tuple[str, ...]) -> FakeReport:
        self.calls.append((ctx, names))
        if self.error is not None:
            raise self.error
        return FakeReport(self.codes.pop(0) if self.codes else 0)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("VINEYARD_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.delenv("VINEYARD_DATA_ROOT", raising=False)
    yield
    teardown_logging()


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    module = types.ModuleType(FAKE_RUNNER)
    module.run_stages = recorder.run_stages
    monkeypatch.setitem(sys.modules, FAKE_RUNNER, module)
    monkeypatch.setattr(cli_options, "RUNNER_MODULE", FAKE_RUNNER)
    monkeypatch.setattr(registry, "module_available", lambda name: True)
    return recorder


def _invoke(*args: str) -> object:
    return runner.invoke(app, list(args))


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"], env={"TERMINAL_WIDTH": "200"})
    assert result.exit_code == 0
    listed = set(re.findall(r"^│ ([a-z][a-z-]*)\s", result.output, flags=re.MULTILINE))
    assert set(EXPECTED_COMMANDS) <= listed, set(EXPECTED_COMMANDS) - listed


def test_version() -> None:
    result = _invoke("--version")
    assert result.exit_code == 0 and "vineyard 0.1.0" in result.output


def test_missing_stage_module_exits_3(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = "vineyard.pipeline.stages._missing_for_cli_test"
    monkeypatch.setattr(registry, "STAGE_MODULES", {**registry.STAGE_MODULES, "derive": missing})
    result = _invoke("derive", "--annset", "LATEST_MODEL")
    assert result.exit_code == 3
    assert "comanda nu e implementată încă: " in result.output and missing in result.output


def test_missing_runner_exits_3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_options, "RUNNER_MODULE", "vineyard.pipeline._missing_runner_for_cli_test")
    result = _invoke("ingest")
    assert result.exit_code == 3 and "nu e implementată" in result.output


def test_run_dispatches_selected_pre_stages(fake_runner: Recorder, tmp_path: Path) -> None:
    result = _invoke("run", "--until", "tile_prep", "--tiles", "siret3_r0*", "--workers", "2",
                     "--force", "tile_prep", "--run-id", "t1", "--allow-failures")
    assert result.exit_code == 0, result.output
    ((ctx, names),) = fake_runner.calls
    assert names == ("ingest", "tile_prep")
    assert (ctx.run_id, ctx.source, ctx.workers, ctx.allow_failures) == ("t1", Source.MODEL, 2, True)
    assert ctx.tile_filter == ("siret3_r0*",) and ctx.force == frozenset({"tile_prep"})
    assert (tmp_path / "work" / "runs" / "t1" / "logs" / "pipeline.jsonl").exists()


def test_run_rejects_unknown_stage(fake_runner: Recorder) -> None:
    result = _invoke("run", "--from", "nope")
    assert result.exit_code == 1 and "eroare" in result.output and fake_runner.calls == []


def test_bad_set_is_a_config_error(fake_runner: Recorder) -> None:
    result = _invoke("ingest", "--set", "canopy.min_area_mm=1")
    assert result.exit_code == 1 and "eroare" in result.output


def test_config_overlay_option(fake_runner: Recorder, tmp_path: Path) -> None:
    overlay = tmp_path / "local.yaml"
    overlay.write_text("runtime:\n  n_workers: 3\n", encoding="utf-8")
    assert _invoke("ingest", "--config", str(overlay)).exit_code == 0
    assert fake_runner.calls[0][0].workers == 3


@pytest.mark.parametrize(
    ("args", "names"),
    [
        (("all",), (*PRE_STAGES, "export_cvat")),
        (("ingest",), ("ingest",)),
        (("import-reference",), ("import_reference",)),
        (("qa",), ("qa_previews",)),
        (("export-cvat",), ("export_cvat",)),
        (("eval-examples",), ("evaluate",)),
    ],
)
def test_simple_commands(fake_runner: Recorder, args: tuple[str, ...], names: tuple[str, ...]) -> None:
    result = _invoke(*args)
    assert result.exit_code == 0, result.output
    assert fake_runner.calls[0][1] == names


def test_bench_forces_everything(fake_runner: Recorder) -> None:
    assert _invoke("bench").exit_code == 0
    ctx, names = fake_runner.calls[0]
    assert names == PRE_STAGES and ctx.force_all


def test_export_defaults_to_latest_model(fake_runner: Recorder) -> None:
    assert _invoke("export-cvat", "--allow-qa-errors").exit_code == 0
    ctx, _ = fake_runner.calls[0]
    assert ctx.annset_ref == "LATEST_MODEL" and "-post-" in ctx.run_id
    assert ctx.cfg.export.cvat.allow_qa_errors is True


def test_eval_flags_map_to_config(fake_runner: Recorder, tmp_path: Path) -> None:
    baseline = tmp_path / "base line & x.json"
    assert _invoke("eval-examples", "--annset", "r1", "--gates", "--baseline", str(baseline),
                   "--write-baseline").exit_code == 0
    cfg = fake_runner.calls[0][0].cfg
    assert cfg.eval.enforce_gates and cfg.eval.write_baseline and cfg.eval.baseline_file == baseline


def test_import_marcaj_passes_files_and_partial(fake_runner: Recorder, tmp_path: Path) -> None:
    export = tmp_path / "data & info" / "job 1.zip"
    for command in ("import-marcaj", "from-marcaj"):
        assert _invoke(command, str(export), "--partial").exit_code == 0
    for ctx, names in fake_runner.calls:
        assert names == ("import_marcaj",)
        assert ctx.source is Source.MARCAJ and "-marcaj-" in ctx.run_id
        assert ctx.cfg.import_.files == (export,)
        assert ctx.cfg.import_.require_all_tiles is False


def test_import_marcaj_relative_file_is_made_absolute(fake_runner: Recorder, monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert _invoke("import-marcaj", "x.zip").exit_code == 0
    assert fake_runner.calls[0][0].cfg.import_.files == (tmp_path / "x.zip",)


def test_post_runs_derive_to_web_in_post_run(fake_runner: Recorder) -> None:
    assert _invoke("post", "--annset", "LATEST_MARCAJ", "--final").exit_code == 0
    ctx, names = fake_runner.calls[0]
    assert names == ("derive", "passable", "targets", "route", "measure", "web_bundle")
    assert ctx.source is Source.MARCAJ and "-post-" in ctx.run_id and ctx.annset_ref == "LATEST_MARCAJ"
    assert ctx.cfg.route.solver.final is True


@pytest.mark.parametrize(
    ("args", "names"),
    [
        (("derive",), ("derive",)),
        (("passable",), ("passable",)),
        (("targets",), ("targets",)),
        (("route", "--final"), ("route",)),
        (("measure",), ("measure",)),
        (("publish", "--require-source", "marcaj"), ("publish",)),
        (("web", "build"), ("web_bundle",)),
    ],
)
def test_post_commands_need_annset(fake_runner: Recorder, args: tuple[str, ...], names: tuple[str, ...]) -> None:
    missing = _invoke(*args)
    assert missing.exit_code == 1 and "lipsește --annset" in missing.output
    assert _invoke(*args, "--annset", "20260926T0310-model-a1b2c3").exit_code == 0
    assert fake_runner.calls[-1][1] == names


def test_publish_require_source(fake_runner: Recorder) -> None:
    assert _invoke("publish", "--annset", "LATEST_MARCAJ", "--require-source", "marcaj").exit_code == 0
    assert fake_runner.calls[0][0].cfg.publish.require_source == "marcaj"


def test_final_chains_import_then_post_and_publish(fake_runner: Recorder, tmp_path: Path) -> None:
    result = _invoke("final", str(tmp_path / "final.zip"), "--run-id", "m1")
    assert result.exit_code == 0, result.output
    (imp_ctx, imp_names), (post_ctx, post_names) = fake_runner.calls
    assert imp_names == ("import_marcaj",) and imp_ctx.run_id == "m1"
    assert post_names == (*cli_post.DERIVE_TO_WEB, "publish")
    assert post_ctx.annset_ref == "m1" and post_ctx.run_id != "m1"
    assert post_ctx.cfg.route.solver.final and post_ctx.cfg.publish.require_source == "marcaj"


def test_final_stops_when_import_fails(fake_runner: Recorder, tmp_path: Path) -> None:
    fake_runner.codes.extend([1])
    assert _invoke("final", str(tmp_path / "f.zip")).exit_code == 1
    assert len(fake_runner.calls) == 1


def test_runner_exit_code_and_errors_propagate(fake_runner: Recorder) -> None:
    fake_runner.codes.extend([1])
    assert _invoke("ingest").exit_code == 1
    fake_runner.error = ExportBlocked("ZIP prea mare", zip_name="p1.zip")
    result = _invoke("export-cvat")
    assert result.exit_code == 2 and "eroare: ZIP prea mare" in result.output


def test_config_show(tmp_path: Path) -> None:
    result = _invoke("config", "show")
    assert result.exit_code == 0 and "contract_version: '1.1'" in result.output
    sub = _invoke("config", "show", "--key", "rows.detect", "--hash")
    assert sub.exit_code == 0 and "spacing_min_m: 1.8" in sub.output and "# cfg_hash = " in sub.output
    assert _invoke("config", "show", "--key", "bogus").exit_code == 1
    assert _invoke("config", "show", "--set", "x.y=1").exit_code == 1


def test_doctor_command_uses_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    from vineyard import doctor

    monkeypatch.setattr(doctor, "run_checks", lambda cfg: (doctor.Check("x", False, True, "d"),))
    result = _invoke("doctor")
    assert result.exit_code == 1 and "vineyard doctor" in result.output
    monkeypatch.setattr(doctor, "run_checks", lambda cfg: (doctor.Check("x", False, False, "d"),))
    assert _invoke("doctor").exit_code == 0
    assert _invoke("doctor", "--set", "nope=1").exit_code == 1
