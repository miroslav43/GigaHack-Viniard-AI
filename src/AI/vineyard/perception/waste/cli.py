"""`vineyard waste ...` sub-app: rebuild the review files of a run, and a calibration report of the rule
filters on a few tiles (default: the 2 example tiles, axes from the reference AnnSet). Messages are Romanian.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Final

import geopandas as gpd
import typer
from shapely.geometry import LineString

from vineyard.annset.io import resolve_run_dir
from vineyard.cli_options import CommonOptions, fail, load_cli_config, with_config_options
from vineyard.config import AppConfig
from vineyard.contracts.enums import Source
from vineyard.errors import StageError, VineyardError
from vineyard.geo.raster import read_tile
from vineyard.geo.tiling import tile_ref
from vineyard.geo.vector_io import read_layer
from vineyard.perception.waste import probe_workflow as pw
from vineyard.perception.waste.filters import reject_counts
from vineyard.perception.waste.types import Candidate
from vineyard.pipeline.atomic import atomic_write_text
from vineyard.pipeline.context import make_run_paths, new_run_context
from vineyard.pipeline.stages.waste import (
    FORBIDDEN_FILE,
    filter_tile,
    local_axes,
    local_forbidden,
    write_review_files,
)
from vineyard.pipeline.tile_cache import load_valid_mask

EXAMPLE_TILES: Final = ("siret3_r021_c012", "siret3_r006_c004")
REFERENCE_AXES: Final = "LATEST_REFERENCE"
ROW_PIECES_FILE: Final = "annset/row_pieces.parquet"
CANDIDATES_FILE: Final = "layers/waste_candidates.parquet"
KEPT: Final = "kept"

app = typer.Typer(help="Deșeuri: revizie și calibrare.", no_args_is_help=True, add_completion=False)

RunOpt = Annotated[str, typer.Option("--run", help="run_id | LATEST_MODEL | cale către rulare.")]
TilesArg = Annotated[
    list[str] | None, typer.Option("--tile", help="tile_id (repetabil; implicit: exemplele).")
]
AxesOpt = Annotated[str, typer.Option("--axes", help="LATEST_REFERENCE (row_pieces) sau o rulare (rows).")]
OutOpt = Annotated[Path | None, typer.Option("--out", help="Scrie raportul Markdown aici.")]
ModelRunOpt = Annotated[str, typer.Option("--run", help="Rularea model (run id | LATEST_MODEL | cale).")]


@app.command("review-html")
@with_config_options
def review_html(run: RunOpt, opts: CommonOptions) -> None:
    """Regenerează qa/waste_candidates.csv, crop-urile și waste_review.html dintr-o rulare."""
    try:
        cfg = load_cli_config(opts)
        run_dir = resolve_run_dir(cfg.paths.work_dir, run)
        layer = read_layer(run_dir / CANDIDATES_FILE, "waste_candidates")
        ctx = new_run_context(cfg, source=Source.MODEL, run_id=run_dir.name, workers=1)
        written = write_review_files(ctx, layer, {"run": run_dir.name, "candidates": len(layer)})
    except (VineyardError, FileNotFoundError) as exc:
        raise fail(exc if isinstance(exc, VineyardError) else StageError(str(exc))) from exc
    typer.echo(f"scris: {written[-1]} ({len(written) - 2} crop-uri)")


def _axes(cfg: AppConfig, axes: str, tile_id: str) -> list[LineString]:
    run_dir = resolve_run_dir(cfg.paths.work_dir, axes)
    if axes == REFERENCE_AXES:
        pieces = read_layer(run_dir / ROW_PIECES_FILE, "row_pieces")
        return list(pieces[pieces["tile_id"] == tile_id].geometry)
    return list(
        local_axes(read_layer(run_dir / "layers" / "rows.parquet", "rows"), tile_ref(tile_id),
                   cfg.waste.axis_margin_m).geometry
    )


def calibrate_tile(
    cfg: AppConfig, tile_id: str, axes: str, forbidden: gpd.GeoDataFrame | None
) -> tuple[Candidate, ...]:
    """Candidates + filter verdicts of one ingested tile (tile_prep cache required)."""
    paths = make_run_paths(cfg, "calibration")
    tile = tile_ref(tile_id)
    rgb = read_tile(paths.tiles_dir / f"{tile_id}.tif")
    valid = load_valid_mask(paths.cache_dir, tile_id)
    return filter_tile(rgb, valid, tile, _axes(cfg, axes, tile_id), local_forbidden(forbidden, tile), cfg)


def calibration_markdown(results: Mapping[str, Sequence[Candidate]], axes: str) -> str:
    reasons = sorted({k for cands in results.values() for k in reject_counts(cands)} - {KEPT})
    head = ["tile", "candidates", *reasons, KEPT]
    lines = [
        f"# Waste calibration (axes: {axes})",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for tile_id, cands in results.items():
        counts = reject_counts(cands)
        cells = [
            tile_id,
            str(len(cands)),
            *(str(counts.get(r, 0)) for r in reasons),
            str(counts.get(KEPT, 0)),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Survivors", ""]
    for cands in results.values():
        lines += [
            f"- {c.cand_key}: {c.colour_class.value}, {c.area_m2:.3f} m², aspect {c.aspect:.2f}, "
            f"HSV {tuple(round(v) for v in c.mean_hsv)}"
            for c in cands
            if not c.rejected
        ]
    return "\n".join(lines) + "\n"


@app.command("calibrate-report")
@with_config_options
def calibrate_report(
    opts: CommonOptions, tile: TilesArg = None, axes: AxesOpt = REFERENCE_AXES, out: OutOpt = None
) -> None:
    """Numără candidații și motivele de respingere pe câteva tile-uri (implicit cele 2 exemple)."""
    try:
        cfg = load_cli_config(opts)
        static = make_run_paths(cfg, "calibration").static_layers_dir / FORBIDDEN_FILE
        forbidden = read_layer(static, "in_forbidden") if static.is_file() else None
        results = {t: calibrate_tile(cfg, t, axes, forbidden) for t in (tile or EXAMPLE_TILES)}
    except VineyardError as exc:
        raise fail(exc) from exc
    text = calibration_markdown(results, axes)
    if out is not None:
        atomic_write_text(out, text)
    typer.echo(text)


@app.command("probe-data")
@with_config_options
def probe_data(
    opts: CommonOptions,
    run: ModelRunOpt = "LATEST_MODEL",
    zip_path: Annotated[Path | None, typer.Option("--zip", help="Extrage întâi UAVVasteDataset.zip (md5).")] = None,
) -> None:
    """Crop-uri pozitive (UAVVaste) și negative (Sireț3) pentru probe -> work/probe/{pos,neg}."""
    try:
        cfg = load_cli_config(opts)
        if zip_path is not None:
            typer.echo(f"UAVVaste: {pw.extract_uavvaste(cfg, zip_path)}")
        stores = pw.build_probe_stores(cfg, resolve_run_dir(cfg.paths.work_dir, run).name)
    except VineyardError as exc:
        raise fail(exc) from exc
    typer.echo(f"pozitive: {stores.n_pos} în {stores.pos}")
    typer.echo(f"negative: {stores.n_neg} în {stores.neg} ({stores.n_tiles} tile-uri)")


@app.command("probe-train")
@with_config_options
def probe_train(
    opts: CommonOptions,
    version: Annotated[str | None, typer.Option("--version", help="Versiunea (implicit waste.probe.version).")] = None,
    store: Annotated[list[Path] | None, typer.Option("--store", help="Crop store (repetabil; implicit pos + neg).")] = None,
) -> None:
    """Embedding-uri OpenCLIP + regresie logistică calibrată -> models/<waste.probe.name>/<versiune>."""
    try:
        cfg = load_cli_config(opts)
        root = pw.probe_root(cfg)
        stores = tuple(store or (root / pw.POS_STORE, root / pw.NEG_STORE))
        model, cv = pw.train_probe_from_stores(cfg, stores, version or cfg.waste.probe.version,
                                               train_cmd=shlex.join(["vineyard", *sys.argv[1:]]))
    except VineyardError as exc:
        raise fail(exc) from exc
    typer.echo(f"probe: {model.name}@{model.version}+{model.sha256[:8]}  ({cv.n_pos} poz / {cv.n_neg} neg)")
    typer.echo(f"AUC {cv.auc:.3f}  AP {cv.average_precision:.3f}  tau* {cv.tau_star:.3f}  prag auto "
               f"{cv.auto_threshold:.3f}  recall {cv.recall_at_threshold:.3f}")
    typer.echo(f"FP la prag: exemple {cv.fp_example_at_threshold}, toate {cv.fp_all_at_threshold}; "
               f"auto: {'da' if cv.auto_enabled else 'nu'} ({cv.auto_reason})")


@app.command("sam3-check")
@with_config_options
def sam3_check(
    opts: CommonOptions,
    tile: TilesArg = None,
    n: Annotated[int, typer.Option("--n", min=1, help="Câte crop-uri 512 px.")] = 3,
) -> None:
    """Încarcă SAM 3 (CPU) și măsoară secunde / crop pe câțiva candidați (implicit exemplele)."""
    try:
        cfg = load_cli_config(opts)
        check = pw.sam3_check(cfg, tuple(tile or EXAMPLE_TILES), n)
    except VineyardError as exc:
        raise fail(exc) from exc
    st = check.status
    typer.echo(f"SAM 3: {st.model_id or '-'} ({st.reason}), încărcare {st.load_s:.1f} s, device {st.device}")
    for key, res in check.results:
        typer.echo(f"  {key}: " + ("fără timp (buget)" if res is None else
                                   f"{res.seconds:.1f} s, scor {res.score:.3f} ({res.prompt or '-'}), "
                                   f"{res.n_negatives} negative"))
    if check.seconds:
        per = check.steady_s_per_crop
        typer.echo(f"s/crop (fără primul): {per:.2f}; buget {cfg.waste.sam3.time_budget_s:.0f} s "
                   f"≈ {cfg.waste.sam3.time_budget_s / per:.0f} crop-uri")
