"""Shared CLI options -> RunContext, lazy dispatch to the stage runner, lazy sub-app mounting."""

from __future__ import annotations

import functools
import importlib
import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Final, TypeVar

import typer

from vineyard.config import DEFAULT_CONFIG, AppConfig, load_config
from vineyard.contracts.enums import Source
from vineyard.errors import VineyardError
from vineyard.pipeline import registry
from vineyard.pipeline.context import RunContext, ensure_run_dirs, new_run_context

RUNNER_MODULE: Final = "vineyard.pipeline.runner"
EXIT_NOT_IMPLEMENTED: Final = 3
MSG_NOT_IMPLEMENTED: Final = "comanda nu e implementată încă: {module}"
MSG_ERROR: Final = "eroare: {error}"

F = TypeVar("F", bound=Callable[..., Any])

ConfigOpt = Annotated[
    list[Path] | None,
    typer.Option("--config", "-c", help="YAML suprapus peste configs/default.yaml (repetabil)."),
]
SetOpt = Annotated[
    list[str] | None, typer.Option("--set", help="Suprascrie o cheie: a.b=<valoare YAML> (repetabil).")
]
RunIdOpt = Annotated[str | None, typer.Option("--run-id", help="Id-ul rulării (implicit: generat).")]
TilesOpt = Annotated[list[str] | None, typer.Option("--tiles", help="Glob de tile_id (repetabil).")]
WorkersOpt = Annotated[int | None, typer.Option("--workers", min=1, help="Procese paralele.")]
ForceOpt = Annotated[list[str] | None, typer.Option("--force", help="Recalculează stagiul (repetabil).")]
ForceAllOpt = Annotated[bool, typer.Option("--force-all", help="Ignoră tot cache-ul.")]
AllowFailuresOpt = Annotated[
    bool | None, typer.Option("--allow-failures/--no-allow-failures", help="Cod 0 chiar cu tile-uri eșuate.")
]
AnnsetOpt = Annotated[
    str | None, typer.Option("--annset", help="run_id | LATEST_MODEL | LATEST_MARCAJ | LATEST_REFERENCE | cale.")
]


@dataclass(frozen=True)
class CommonOptions:
    configs: tuple[Path, ...] = ()
    sets: tuple[str, ...] = ()
    run_id: str | None = None
    tiles: tuple[str, ...] = ()
    workers: int | None = None
    force: tuple[str, ...] = ()
    force_all: bool = False
    allow_failures: bool | None = None
    annset: str | None = None

    def with_sets(self, *extra: str) -> CommonOptions:
        return replace(self, sets=(*self.sets, *extra))


_COMMON_PARAMS: Final[tuple[tuple[str, Any, Any], ...]] = (
    ("config", ConfigOpt, None),
    ("set_", SetOpt, None),
    ("run_id", RunIdOpt, None),
    ("tiles", TilesOpt, None),
    ("workers", WorkersOpt, None),
    ("force", ForceOpt, None),
    ("force_all", ForceAllOpt, False),
    ("allow_failures", AllowFailuresOpt, None),
    ("annset", AnnsetOpt, None),
)
_CONFIG_PARAMS: Final = _COMMON_PARAMS[:2]


def _options_from(values: dict[str, Any]) -> CommonOptions:
    return CommonOptions(
        configs=tuple(values.get("config") or ()),
        sets=tuple(values.get("set_") or ()),
        run_id=values.get("run_id"),
        tiles=tuple(values.get("tiles") or ()),
        workers=values.get("workers"),
        force=tuple(values.get("force") or ()),
        force_all=bool(values.get("force_all", False)),
        allow_failures=values.get("allow_failures"),
        annset=values.get("annset"),
    )


def _inject(params: tuple[tuple[str, Any, Any], ...]) -> Callable[[F], F]:
    """Decorator: add `params` as keyword options and pass them to `func` as `opts: CommonOptions`."""

    def decorate(func: F) -> F:
        signature = inspect.signature(func, eval_str=True)  # typer reads Annotated[...] from the signature
        own = [p for p in signature.parameters.values() if p.name != "opts"]
        extra = [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=d, annotation=a) for n, a, d in params]
        merged = [*own, *extra]

        @functools.wraps(func)
        def wrapper(**kwargs: Any) -> Any:
            values = {name: kwargs.pop(name) for name, _, _ in params}
            return func(opts=_options_from(values), **kwargs)

        wrapper.__signature__ = signature.replace(parameters=merged)  # type: ignore[attr-defined]
        wrapper.__annotations__ = {p.name: p.annotation for p in merged}
        return wrapper  # type: ignore[return-value]

    return decorate


with_common_options = _inject(_COMMON_PARAMS)
with_config_options = _inject(_CONFIG_PARAMS)


def echo_err(message: str) -> None:
    typer.echo(message, err=True)


def fail(error: VineyardError) -> typer.Exit:
    echo_err(MSG_ERROR.format(error=error))
    return typer.Exit(error.exit_code)


def load_cli_config(opts: CommonOptions) -> AppConfig:
    return load_config((DEFAULT_CONFIG, *opts.configs), opts.sets)


def yaml_value(value: Any) -> str:
    """--set value text; JSON is valid YAML flow and quotes paths with spaces or '&' safely."""
    return json.dumps(value)


def yaml_list(values: Sequence[Any]) -> str:
    return yaml_value([str(v) for v in values])


def build_context(opts: CommonOptions, *, source: Source, kind: str | None = None,
                  annset_ref: str | None = None) -> RunContext:
    return new_run_context(
        load_cli_config(opts),
        source=source,
        run_id=opts.run_id,
        tiles=opts.tiles,
        force=opts.force,
        force_all=opts.force_all,
        workers=opts.workers,
        allow_failures=opts.allow_failures,
        annset_ref=annset_ref if annset_ref is not None else opts.annset,
        kind=kind,
    )


def missing_modules(names: Sequence[str]) -> tuple[str, ...]:
    """Modules needed to run `names` that do not exist yet (runner first)."""
    needed = [RUNNER_MODULE, *(registry.STAGE_MODULES[n] for n in names)]
    return tuple(m for m in needed if not registry.module_available(m))


def _setup_run_logging(ctx: RunContext) -> None:
    from vineyard.logging_setup import setup_logging

    log_cfg = ctx.cfg.logging
    ensure_run_dirs(ctx.paths)
    setup_logging(level=log_cfg.level, console=log_cfg.console,
                  jsonl_path=ctx.paths.pipeline_log if log_cfg.jsonl else None, run_id=ctx.run_id, tz=log_cfg.tz)


def run_pipeline(opts: CommonOptions, names: Sequence[str], *, source: Source, kind: str | None = None,
                 annset_ref: str | None = None) -> tuple[RunContext, int]:
    """Build the context and run `names` through `vineyard.pipeline.runner.run_stages`."""
    missing = missing_modules(names)
    if missing:
        echo_err(MSG_NOT_IMPLEMENTED.format(module=", ".join(missing)))
        raise typer.Exit(EXIT_NOT_IMPLEMENTED)
    try:
        ctx = build_context(opts, source=source, kind=kind, annset_ref=annset_ref)
        _setup_run_logging(ctx)
        runner = importlib.import_module(RUNNER_MODULE)
        report = runner.run_stages(ctx, tuple(names))
    except VineyardError as exc:
        raise fail(exc) from exc
    return ctx, int(report.exit_code)


def dispatch(opts: CommonOptions, names: Sequence[str], *, source: Source, kind: str | None = None,
             annset_ref: str | None = None) -> None:
    """Run and exit with the run's exit code (0 ok, 1 failures, 2 export blocked, 3 not implemented)."""
    _, code = run_pipeline(opts, names, source=source, kind=kind, annset_ref=annset_ref)
    raise typer.Exit(code)


def _placeholder(parent: typer.Typer, name: str, module: str, help_text: str, reason: str | None) -> None:
    settings = {"allow_extra_args": True, "ignore_unknown_options": True}

    @parent.command(name, help=f"{help_text} (neimplementat: {module})", context_settings=settings)
    def _missing(ctx: typer.Context) -> None:
        echo_err(MSG_NOT_IMPLEMENTED.format(module=module))
        if reason:
            echo_err(f"import eșuat: {reason}")
        raise typer.Exit(EXIT_NOT_IMPLEMENTED)


def mount_subapp(parent: typer.Typer, name: str, module: str, help_text: str) -> bool:
    """Mount `<module>:app` as `name`; a missing or broken module becomes a placeholder (exit 3)."""
    if not registry.module_available(module):
        _placeholder(parent, name, module, help_text, None)
        return False
    try:
        sub = importlib.import_module(module).app
    except Exception as exc:  # noqa: BLE001  (a broken plugin must not break `vineyard --help`)
        _placeholder(parent, name, module, help_text, f"{type(exc).__name__}: {exc}")
        echo_err(f"avertisment: sub-comanda '{name}' nu s-a putut încărca ({type(exc).__name__}: {exc})")
        return False
    parent.add_typer(sub, name=name, help=help_text)
    return True
