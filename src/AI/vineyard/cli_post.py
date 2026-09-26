"""Post-Marcaj commands (merged into the main `vineyard` app): import, derive ... publish, web."""

from __future__ import annotations

import functools
import http.server
from pathlib import Path
from typing import Annotated, Final

import typer

from vineyard.cli_options import (
    CommonOptions,
    dispatch,
    fail,
    load_cli_config,
    run_pipeline,
    with_common_options,
    yaml_list,
)
from vineyard.contracts.enums import Source
from vineyard.errors import ConfigError
from vineyard.pipeline.context import POST_KIND, source_from_annset_ref
from vineyard.pipeline.registry import POST_STAGES

IMPORT_STAGE: Final = "import_marcaj"
DERIVE_TO_WEB: Final[tuple[str, ...]] = POST_STAGES[1:]
DEFAULT_WEB_PORT: Final = 8000
SET_FINAL: Final = "route.solver.final=true"
SET_PARTIAL: Final = "import.require_all_tiles=false"

app = typer.Typer(help="Comenzile post-Marcaj.")
web_app = typer.Typer(help="Pachetul de date web (src/Web/data).", no_args_is_help=True)
app.add_typer(web_app, name="web")

FilesArg = Annotated[list[Path], typer.Argument(help="Export(uri) CVAT 1.1 din Marcaj (.zip sau .xml).")]
PartialOpt = Annotated[bool, typer.Option("--partial", help="Export intermediar: nu cere toate cele 311 imagini.")]
FinalOpt = Annotated[bool, typer.Option("--final", help="Solver cu limita de timp finală (route.solver.final).")]


def _require_annset(opts: CommonOptions) -> str:
    if not opts.annset:
        raise fail(ConfigError("lipsește --annset (run_id, LATEST_MODEL, LATEST_MARCAJ ...)"))
    return opts.annset


def _run_on_annset(opts: CommonOptions, names: tuple[str, ...]) -> None:
    ref = _require_annset(opts)
    dispatch(opts, names, source=source_from_annset_ref(ref), kind=POST_KIND, annset_ref=ref)


def import_sets(files: list[Path], partial: bool) -> tuple[str, ...]:
    """--set overrides carrying the Marcaj files (absolute) and --partial."""
    absolute = [str(f if f.is_absolute() else Path.cwd() / f) for f in files]
    return (f"import.files={yaml_list(absolute)}", *((SET_PARTIAL,) if partial else ()))


def import_marcaj_run(opts: CommonOptions, files: list[Path], partial: bool) -> tuple[str, int]:
    """Run `import_marcaj` in a new marcaj run; returns (run_id, exit code)."""
    ctx, code = run_pipeline(opts.with_sets(*import_sets(files, partial)), (IMPORT_STAGE,), source=Source.MARCAJ)
    return ctx.run_id, code


@app.command("import-marcaj")
@with_common_options
def import_marcaj(opts: CommonOptions, files: FilesArg, partial: PartialOpt = False) -> None:
    """Export CVAT din Marcaj -> AnnSet(marcaj) + qa_issues."""
    _, code = import_marcaj_run(opts, files, partial)
    raise typer.Exit(code)


@app.command("from-marcaj")
@with_common_options
def from_marcaj(opts: CommonOptions, files: FilesArg, partial: PartialOpt = False) -> None:
    """Alias pentru import-marcaj."""
    _, code = import_marcaj_run(opts, files, partial)
    raise typer.Exit(code)


@app.command("derive")
@with_common_options
def derive(opts: CommonOptions) -> None:
    """Rânduri contopite, blocuri, legarea inter-rândurilor."""
    _run_on_annset(opts, ("derive",))


@app.command("passable")
@with_common_options
def passable(opts: CommonOptions) -> None:
    """Domeniul de mers și graful."""
    _run_on_annset(opts, ("passable",))


@app.command("targets")
@with_common_options
def targets(opts: CommonOptions) -> None:
    """Ținte de inspecție."""
    _run_on_annset(opts, ("targets",))


@app.command("route")
@with_common_options
def route(opts: CommonOptions, final: FinalOpt = False) -> None:
    """Traseul (GTSP) + validare."""
    _run_on_annset(opts.with_sets(SET_FINAL) if final else opts, ("route",))


@app.command("measure")
@with_common_options
def measure(opts: CommonOptions) -> None:
    """measurements.csv + measurements.json."""
    _run_on_annset(opts, ("measure",))


@app.command("farms")
@with_common_options
def farms(opts: CommonOptions) -> None:
    """Ferme (blocuri vecine) + clase de drumuri (+ parcele cadastrale AGCC, dacă există instantaneul)."""
    _run_on_annset(opts, ("farms",))


@app.command("post")
@with_common_options
def post(opts: CommonOptions, final: FinalOpt = False) -> None:
    """derive -> passable -> targets -> route -> measure -> farms -> web_bundle."""
    _run_on_annset(opts.with_sets(SET_FINAL) if final else opts, DERIVE_TO_WEB)


@app.command("publish")
@with_common_options
def publish(
    opts: CommonOptions,
    require_source: Annotated[str | None, typer.Option("--require-source", help="model | marcaj")] = None,
) -> None:
    """Copiază și validează route.geojson + measurements.csv în rădăcina repo-ului."""
    extra = (f"publish.require_source={require_source}",) if require_source else ()
    _run_on_annset(opts.with_sets(*extra), ("publish",))


@web_app.command("build")
@with_common_options
def web_build(opts: CommonOptions) -> None:
    """Straturi GeoJSON 4326, panouri JSON, orto, manifest."""
    _run_on_annset(opts, ("web_bundle",))


@web_app.command("serve")
@with_common_options
def web_serve(
    opts: CommonOptions, port: Annotated[int, typer.Option("--port", min=1, max=65535)] = DEFAULT_WEB_PORT
) -> None:
    """Servește local directorul părinte al web.out_dir (src/Web)."""
    root = load_cli_config(opts).web.out_dir.parent
    if not root.is_dir():
        raise fail(ConfigError("directorul web nu există", path=str(root)))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    typer.echo(f"http://127.0.0.1:{port}/  ({root})")
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        server.serve_forever()
