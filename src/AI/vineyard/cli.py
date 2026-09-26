"""`vineyard` CLI (contract §8). Commands dispatch lazily to the stage runner; sub-apps load lazily."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated, Final

import typer
import yaml

from vineyard import __version__, cli_post
from vineyard.cli_options import (
    CommonOptions,
    dispatch,
    fail,
    load_cli_config,
    mount_subapp,
    run_pipeline,
    with_common_options,
    with_config_options,
    yaml_value,
)
from vineyard.config import cfg_hash, cfg_subtree, resolved_config_dict
from vineyard.contracts.enums import Source
from vineyard.errors import VineyardError
from vineyard.pipeline.context import POST_KIND, source_from_annset_ref
from vineyard.pipeline.registry import PRE_STAGES, select_stages

LATEST_MODEL: Final = "LATEST_MODEL"
SET_ALLOW_QA_ERRORS: Final = "export.cvat.allow_qa_errors=true"
SET_REQUIRE_MARCAJ: Final = "publish.require_source=marcaj"

app = typer.Typer(
    name="vineyard",
    help="Sireț3 Vineyard AI: pre-adnotare, export CVAT, import Marcaj, traseu, măsurători.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(help="Configurația rezolvată.", no_args_is_help=True)
app.add_typer(config_app, name="config")


def _version(value: bool) -> None:
    if value:
        typer.echo(f"vineyard {__version__}")
        raise typer.Exit(0)


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True, help="Versiunea.")] = False,
) -> None:
    """Sireț3 Vineyard AI."""


def _on_annset(opts: CommonOptions, names: tuple[str, ...]) -> None:
    ref = opts.annset or LATEST_MODEL
    dispatch(opts, names, source=source_from_annset_ref(ref), kind=POST_KIND, annset_ref=ref)


@app.command("ingest")
@with_common_options
def ingest(opts: CommonOptions) -> None:
    """Verifică și copiază cele 311 tile-uri, tile_index, straturile in_*."""
    dispatch(opts, ("ingest",), source=Source.MODEL)


@app.command("run")
@with_common_options
def run(
    opts: CommonOptions,
    from_: Annotated[str | None, typer.Option("--from", help="Primul stagiu.")] = None,
    until: Annotated[str | None, typer.Option("--until", help="Ultimul stagiu.")] = None,
) -> None:
    """Percepția completă -> AnnSet(model) (ingest ... qa_previews)."""
    try:
        names = select_stages(PRE_STAGES, from_=from_, until=until)
    except VineyardError as exc:
        raise fail(exc) from exc
    dispatch(opts, names, source=Source.MODEL)


@app.command("all")
@with_common_options
def all_(opts: CommonOptions) -> None:
    """ingest -> ... -> qa_previews -> export-cvat (ZIP-urile de upload)."""
    dispatch(opts, (*PRE_STAGES, "export_cvat"), source=Source.MODEL)


@app.command("bench")
@with_common_options
def bench(opts: CommonOptions) -> None:
    """Doar percepția (ingest ... qa_previews), curată și cronometrată, pe toate tile-urile -> metrics/timings.json."""
    dispatch(replace(opts, force_all=True), PRE_STAGES, source=Source.MODEL)


@app.command("export-cvat")
@with_common_options
def export_cvat(
    opts: CommonOptions,
    allow_qa_errors: Annotated[bool, typer.Option("--allow-qa-errors", help="Exportă și cu erori QA.")] = False,
) -> None:
    """AnnSet(model) -> ZIP-uri de upload validate (<= 85 000 000 B)."""
    _on_annset(opts.with_sets(SET_ALLOW_QA_ERRORS) if allow_qa_errors else opts, ("export_cvat",))


@app.command("import-reference")
@with_common_options
def import_reference(opts: CommonOptions) -> None:
    """05_examples -> AnnSet(reference)."""
    dispatch(opts, ("import_reference",), source=Source.REFERENCE)


@app.command("eval-examples")
@with_common_options
def eval_examples(
    opts: CommonOptions,
    gates: Annotated[bool, typer.Option("--gates", help="Cod != 0 dacă o poartă pică.")] = False,
    baseline: Annotated[Path | None, typer.Option("--baseline", help="JSON de referință pentru regresie.")] = None,
    write_baseline: Annotated[bool, typer.Option("--write-baseline", help="Scrie baseline-ul.")] = False,
) -> None:
    """Metricile oficiale pe cele 2 tile-uri exemplu."""
    extra = [f"eval.enforce_gates={str(gates).lower()}", f"eval.write_baseline={str(write_baseline).lower()}"]
    if baseline is not None:
        extra.append(f"eval.baseline_file={yaml_value(str(baseline.resolve()))}")
    _on_annset(opts.with_sets(*extra), ("evaluate",))


@app.command("qa")
@with_common_options
def qa(opts: CommonOptions) -> None:
    """qa_issues + review_queue + previzualizări pentru un AnnSet."""
    _on_annset(opts, ("qa_previews",))


@app.command("final")
@with_common_options
def final(opts: CommonOptions, files: cli_post.FilesArg, partial: cli_post.PartialOpt = False) -> None:
    """from-marcaj -> post (solver final) -> publish, din exportul Marcaj."""
    run_id, code = cli_post.import_marcaj_run(opts, files, partial)
    if code != 0:
        raise typer.Exit(code)
    post_opts = replace(opts, run_id=None).with_sets(cli_post.SET_FINAL, SET_REQUIRE_MARCAJ)
    _, code = run_pipeline(post_opts, (*cli_post.DERIVE_TO_WEB, "publish"), source=Source.MARCAJ,
                           kind=POST_KIND, annset_ref=run_id)
    raise typer.Exit(code)


@app.command("doctor")
@with_config_options
def doctor(opts: CommonOptions) -> None:
    """Verifică mediul: Python, biblioteci, GDAL/PROJ, torch/MPS, conda, date, disc."""
    from vineyard.doctor import exit_code, render, run_checks

    try:
        cfg = load_cli_config(opts)
    except VineyardError as exc:
        raise fail(exc) from exc
    checks = run_checks(cfg)
    render(checks)
    raise typer.Exit(exit_code(checks))


@config_app.command("show")
@with_config_options
def config_show(
    opts: CommonOptions,
    key: Annotated[str | None, typer.Option("--key", help="Subarbore, ex. rows.detect.")] = None,
    show_hash: Annotated[bool, typer.Option("--hash", help="Afișează și cfg_hash.")] = False,
) -> None:
    """Afișează configurația rezolvată (YAML)."""
    try:
        cfg = load_cli_config(opts)
        data = cfg_subtree(cfg, key) if key else resolved_config_dict(cfg)
        digest = cfg_hash(cfg, [key] if key else list(resolved_config_dict(cfg)))
    except VineyardError as exc:
        raise fail(exc) from exc
    typer.echo(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120).rstrip())
    if show_hash:
        typer.echo(f"# cfg_hash = {digest}")


app.add_typer(cli_post.app)
mount_subapp(app, "nn", "vineyard.nn.cli", "Rețeaua neuronală: pseudolabels, train, infer, ablate, fetch.")
mount_subapp(app, "waste", "vineyard.perception.waste.cli", "Deșeuri: candidați, probe, SAM 3, revizie.")
mount_subapp(app, "cvat", "vineyard.cvat.cli", "CVAT: validate, make-test-zip, roundtrip-examples.")
