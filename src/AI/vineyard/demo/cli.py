"""`vineyard demo tile <tif> --out <dir>`: the web page /analiza runs this on an uploaded tile.

Exit 0 with status.json {state: done} and every web file; exit 1 with status.json {state: error, error: {code,
message}} otherwise (codes: the tile_input ones, `pipeline_failed`). Messages are Romanian, like the rest of the CLI.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Annotated, Final

import typer

from vineyard.demo.export import MaskColour, export_job
from vineyard.demo.job import DEMO_STAGES, base_route_dir, prepare_job, run_job
from vineyard.demo.status import write_status
from vineyard.demo.tile_input import DemoInputError, check_tile

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Pipeline-ul real pe un singur tile încărcat.")

PIPELINE_FAILED: Final = "pipeline_failed"
DEFAULT_MASK_COLOUR: Final = "#D946EF"  # src/Web/frontend/src/theme/rasterColors.json → vegMask (the web passes its own)
DEFAULT_MASK_ALPHA: Final = 0.55
N_STAGES: Final = len(DEMO_STAGES)


def _fail(out: Path, code: str, message: str) -> typer.Exit:
    write_status(out, "error", n_stages=N_STAGES, error=(code, message))
    typer.echo(f"eroare [{code}]: {message}", err=True)
    return typer.Exit(1)


@app.command("tile")
def tile(
    tif: Annotated[Path, typer.Argument(help="GeoTIFF-ul încărcat (un tile Sireț3).")],
    out: Annotated[Path, typer.Option("--out", help="Dosarul jobului (status.json, result.json, straturi).")],
    mask_colour: Annotated[str, typer.Option("--mask-colour", help="Culoarea măștii de vegetație.")] = (
        DEFAULT_MASK_COLOUR),
    mask_alpha: Annotated[float, typer.Option("--mask-alpha", help="Opacitatea măștii (0-1).")] = DEFAULT_MASK_ALPHA,
) -> None:
    """Rulează ingest … assemble izolat pe tile și scrie fișierele paginii /analiza."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_status(out, "running", n_stages=N_STAGES)
    try:
        colour = MaskColour.from_hex(mask_colour, mask_alpha)
        tile_in = check_tile(tif)
    except DemoInputError as exc:
        raise _fail(out, exc.code, exc.message) from exc
    except ValueError as exc:
        raise _fail(out, PIPELINE_FAILED, str(exc)) from exc
    try:
        job = prepare_job(out, tile_in, base_route_dir())
        ctx, code = run_job(job)
        if code != 0:
            raise _fail(out, PIPELINE_FAILED, f"pipeline-ul s-a oprit cu codul {code}")
        export_job(out, ctx.paths, tile_in.tile_id, colour, time.monotonic() - started)
    except typer.Exit as exc:
        if exc.exit_code == 0:
            raise
        raise _fail(out, PIPELINE_FAILED, f"pipeline-ul s-a oprit cu codul {exc.exit_code}") from exc
    except Exception as exc:  # noqa: BLE001  (any crash must reach the page as an error status, with the trace in the log)
        traceback.print_exc()
        raise _fail(out, PIPELINE_FAILED, f"{type(exc).__name__}: {exc}") from exc
    write_status(out, "done", n_stages=N_STAGES, stage=DEMO_STAGES[-1], stage_index=N_STAGES)
    typer.echo(f"gata: {tile_in.tile_id} în {time.monotonic() - started:.1f} s → {out}")
