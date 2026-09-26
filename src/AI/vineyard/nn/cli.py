"""`vineyard nn ...` sub-app (design 03 §6): pseudolabels, train [--smoke] [--variant F], infer, ablate, panels,
fetch, bench. Mounted eagerly by vineyard.cli, so this module never imports torch at import time: the
commands that need it import nn.train / nn.infer inside their body. Messages are Romanian.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Annotated, Final

import typer

from vineyard.annset.io import resolve_run_dir
from vineyard.cli_options import (
    CommonOptions,
    dispatch,
    fail,
    load_cli_config,
    with_common_options,
    with_config_options,
)
from vineyard.config import AppConfig
from vineyard.contracts.enums import Source
from vineyard.errors import VineyardError
from vineyard.nn import workflows
from vineyard.nn.weights import nn_version_string

LATEST_MODEL: Final = "LATEST_MODEL"
SET_NN_ENABLED: Final = "nn.enabled=true"
SET_NO_BAND: Final = ("nn.pseudolabels.ignore_band_px=0", "nn.pseudolabels.edge_band_px=0")
VARIANTS: Final = ("main", "F")

app = typer.Typer(help="Rețeaua neuronală (U-Net canopy de viță).", no_args_is_help=True, add_completion=False)

RunOpt = Annotated[str, typer.Option("--run", help="Rularea model (run id | LATEST_MODEL | cale).")]
ReferenceOpt = Annotated[str | None, typer.Option("--reference", help="AnnSet referință (implicit eval.reference_annset).")]
VariantOpt = Annotated[str, typer.Option("--variant", help="main | F (fără bandă ignore și fără label smoothing).")]
WorkersOpt = Annotated[int, typer.Option("--workers", min=1, help="Procese pentru extragerea patch-urilor.")]


def _cfg(opts: CommonOptions, variant: str = "main") -> AppConfig:
    if variant not in VARIANTS:
        raise typer.BadParameter(f"--variant trebuie să fie unul din {VARIANTS}", param_hint="--variant")
    return load_cli_config(opts.with_sets(*SET_NO_BAND) if variant == "F" else opts)


def _run_id(cfg: AppConfig, run: str) -> str:
    return resolve_run_dir(cfg.paths.work_dir, run).name


@app.command("pseudolabels")
@with_config_options
def pseudolabels(opts: CommonOptions, run: RunOpt = LATEST_MODEL, variant: VariantOpt = "main",
                 workers: WorkersOpt = 1) -> None:
    """Pseudo-etichete pe tile-urile selectate din rularea model -> patch store (512² la 0,05 m)."""
    from vineyard.nn.patch_store import build_store, plan_store, store_stats

    try:
        cfg = _cfg(opts, variant)
        plan = plan_store(cfg, _run_id(cfg, run))
        manifest = build_store(plan, workers=workers)
        stats = store_stats(manifest, plan.store_dir)
    except VineyardError as exc:
        raise fail(exc) from exc
    typer.echo(f"store: {plan.store_dir}")
    typer.echo(f"tile-uri: {stats['n_tiles']}  patch-uri: {stats['n_patches']}  eșuate: {len(manifest.failed)}")


@app.command("train")
@with_config_options
def train(
    opts: CommonOptions,
    run: RunOpt = LATEST_MODEL,
    store: Annotated[Path | None, typer.Option("--store", help="Patch store (implicit: cel al rulării).")] = None,
    smoke: Annotated[bool, typer.Option("--smoke", help="1 epocă, puține batch-uri, versiunea v0-smoke.")] = False,
    variant: VariantOpt = "main",
    reference: ReferenceOpt = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Seed (implicit runtime.seed).")] = None,
) -> None:
    """Antrenează U-Net-ul; early stopping pe scorul oficial (varianta B) al celor 2 tile-uri de referință."""
    try:
        cfg = _cfg(opts, variant)
        res = workflows.train_from_run(cfg, _run_id(cfg, run), store=store, smoke=smoke, variant=variant,
                                       reference=reference, seed=seed, train_cmd=shlex.join(["vineyard", *sys.argv[1:]]))
    except VineyardError as exc:
        raise fail(exc) from exc
    typer.echo(f"model: {nn_version_string(res.card)}  cea mai bună epocă: {res.best_epoch}/{len(res.history)}")
    typer.echo(f"scor validare (B, axe referință): {res.card.metrics['val_score']:.4f}  device: {res.device}")


@app.command("infer")
@with_common_options
def infer(opts: CommonOptions) -> None:
    """Rulează stagiul nn_infer (nn.enabled forțat) pe tile-urile selectate -> cache/nn/<versiune>/canopy_prob."""
    dispatch(opts.with_sets(SET_NN_ENABLED), ("nn_infer",), source=Source.MODEL)


@app.command("ablate")
@with_config_options
def ablate(
    opts: CommonOptions,
    run: RunOpt = LATEST_MODEL,
    reference: ReferenceOpt = None,
    f_version: Annotated[str | None, typer.Option("--f-version", help="Versiunea greutăților F (implicit <v>f).")] = None,
) -> None:
    """Ablația A-F pe cele 2 tile-uri de referință (axe model și referință) + panouri; nu aplică nimic."""
    try:
        cfg = load_cli_config(opts)
        out = workflows.ablate(cfg, _run_id(cfg, run), reference=reference, f_version=f_version)
    except VineyardError as exc:
        raise fail(exc) from exc
    typer.echo(f"raport: {out.json_path}\n        {out.md_path}")
    typer.echo(f"panouri: {len(out.panels)} în {out.panels[0].parent if out.panels else '-'}")
    typer.echo(f"promovat: {out.report.promoted.value} ({out.report.reason})")
    typer.echo(f"aplică cu: --set nn.fusion={out.report.promoted.value}")


@app.command("panels")
@with_config_options
def panels(
    opts: CommonOptions,
    run: RunOpt = LATEST_MODEL,
    tiles: Annotated[list[str] | None, typer.Option("--tile", help="tile_id (repetabil).")] = None,
    n_auto: Annotated[int | None, typer.Option("--auto", help="Câte tile-uri grele din cache-ul NN.")] = None,
) -> None:
    """Panouri RGB | a* | NN | diferență (JPEG) în <rulare>/qa/nn_panels/."""
    try:
        cfg = load_cli_config(opts)
        written = workflows.panels_from_cache(cfg, _run_id(cfg, run), tuple(tiles or ()), n_auto)
    except VineyardError as exc:
        raise fail(exc) from exc
    typer.echo(f"panouri: {len(written)}")
    for path in written:
        typer.echo(f"  {path}")


@app.command("fetch")
@with_config_options
def fetch(
    opts: CommonOptions,
    url: Annotated[str | None, typer.Option("--url", help="URL weights.pt (implicit nn.weights_url).")] = None,
    sha256: Annotated[str | None, typer.Option("--sha256", help="sha256 așteptat (implicit nn.weights_sha256).")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Versiunea (implicit nn.version).")] = None,
    card_url: Annotated[str | None, typer.Option("--card-url", help="URL model_card.json (implicit alături).")] = None,
) -> None:
    """Descarcă greutățile + model card-ul și verifică sha256 înainte de instalare."""
    try:
        cfg = load_cli_config(opts)
        loaded = workflows.fetch(cfg, url=url, sha256=sha256, version=version, card_url=card_url)
    except VineyardError as exc:
        raise fail(exc) from exc
    typer.echo(f"instalat: {nn_version_string(loaded.card)} -> {loaded.weights_path}")


@app.command("bench")
@with_config_options
def bench(
    opts: CommonOptions,
    iters: Annotated[int, typer.Option("--iters", min=1, help="Iterații măsurate.")] = 20,
    batch: Annotated[int | None, typer.Option("--batch", min=1, help="Batch (implicit nn.batch_size).")] = None,
    size: Annotated[int | None, typer.Option("--size", min=32, help="Latura patch-ului (implicit patch_px).")] = None,
) -> None:
    """Secunde / iterație de antrenare (forward + backward + AdamW) pe nn.device, cu date aleatoare."""
    try:
        cfg = load_cli_config(opts)
        res = workflows.bench(cfg, iters=iters, batch=batch, size=size)
    except VineyardError as exc:
        raise fail(exc) from exc
    typer.echo(f"{res.device}: {res.s_per_iter:.4f} s/iterație, {res.img_per_s:.1f} img/s "
               f"(batch {res.batch_size}, {res.size_px}², {res.n_iter} iterații)")
