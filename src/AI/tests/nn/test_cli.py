"""`vineyard nn` sub-app and nn.workflows: fetch (file:// URL), bench, stage dispatch, and — on the real work
dir when its runs exist (`examples`) — ablation + panels end to end with fake constant weights."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from tests.nn.test_train import Tiny, _masked_bce
from vineyard.config import AppConfig, load_config
from vineyard.nn import workflows
from vineyard.nn.cli import app
from vineyard.nn.train import TrainParts
from vineyard.nn.weights import ModelCard, WeightsError, save_weights, weights_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_WORK = Path(os.environ.get("VINEYARD_WORK_DIR", str(PROJECT_ROOT / "work")))
runner = CliRunner()


def _card(version: str = "v1") -> ModelCard:
    return ModelCard(name="vine-unet", version=version, arch="unet", encoder="resnet18", in_gsd_m=0.05,
                     classes=("canopy_prob",), train_data=(), metrics={}, sha256="", created_at="now", train_cmd="t")


def _conv_bytes(bias: float) -> bytes:
    from vineyard.nn.infer import state_dict_bytes

    model = torch.nn.Conv2d(3, 1, 1)
    torch.nn.init.constant_(model.weight, 0.0)
    torch.nn.init.constant_(model.bias, bias)
    return state_dict_bytes(model)


def test_cli_import_is_torch_free() -> None:
    code = "import sys, vineyard.nn.cli, vineyard.nn.workflows; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_help_lists_commands() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("pseudolabels", "train", "infer", "ablate", "panels", "fetch", "bench"):
        assert cmd in res.output


def test_fetch_from_file_url(tmp_path: Path) -> None:
    card = save_weights(b"weights-bytes" * 10, _card(), tmp_path / "release")
    url = (weights_dir(tmp_path / "release", "vine-unet", "v1") / "weights.pt").as_uri()
    models = tmp_path / "models"
    res = runner.invoke(app, ["fetch", "--url", url, "--sha256", card.sha256,
                              "--set", f'paths.models_dir="{models}"'])
    assert res.exit_code == 0, res.output
    assert "vine-unet@v1+" in res.output
    assert (weights_dir(models, "vine-unet", "v1") / "weights.pt").is_file()


def test_fetch_without_url_fails_cleanly() -> None:
    res = runner.invoke(app, ["fetch"])
    assert res.exit_code != 0 and "URL" in res.output
    with pytest.raises(WeightsError):
        workflows.fetch(load_config(), url=None, sha256=None, version=None, card_url=None)


def test_bad_variant_is_rejected() -> None:
    res = runner.invoke(app, ["pseudolabels", "--variant", "G"])
    assert res.exit_code != 0


def test_bench_workflow_with_tiny_parts() -> None:
    cfg = load_config(overrides=["nn.device=cpu"])
    parts = TrainParts(make_model=lambda e, n, w: Tiny(), make_loader=lambda *a: [], make_loss=_masked_bce)
    res = workflows.bench(cfg, iters=2, batch=2, size=32, parts=parts)
    assert res.n_iter == 2 and res.batch_size == 2 and res.size_px == 32 and res.s_per_iter > 0


def test_infer_dispatches_the_stage(tmp_path: Path, tmp_work: Path) -> None:
    import pandas as pd

    pd.DataFrame({"tile_id": ["siret3_r021_c012"]}).to_parquet(tmp_work / "tile_index.parquet")
    res = runner.invoke(app, ["infer", "--run-id", "nn-cli-test", "--set", f'paths.models_dir="{tmp_path}"'])
    assert res.exit_code == 0, res.output  # no weights -> the stage degrades and the run succeeds


# ------------------------------------------------------------------ end to end on the real work dir


def _find_runs() -> tuple[str, str] | None:
    runs = REAL_WORK / "runs"
    if not runs.is_dir():
        return None
    refs = sorted(p.name for p in runs.iterdir() if "-reference-" in p.name and (p / "annset" / "annset.json").is_file())
    models = [p.name for p in (runs / "LATEST_MODEL", runs / "viz42") if (p / "layers" / "rows.parquet").is_file()]
    return (models[0], refs[-1]) if refs and models else None


@pytest.fixture
def mirror(tmp_path: Path) -> Iterator[tuple[AppConfig, str, str]]:
    """A tmp work dir that links the real tiles/cache/layers and the two runs read-only."""
    found = _find_runs()
    if found is None or not (REAL_WORK / "cache" / "veg").is_dir():
        pytest.skip(f"real work dir with a model run and a reference AnnSet missing under {REAL_WORK}")
    model_run, ref_run = found
    work = tmp_path / "work"
    (work / "runs" / model_run / "layers").mkdir(parents=True)
    for name in ("tiles", "layers", "tile_index.parquet"):
        (work / name).symlink_to(REAL_WORK / name)
    (work / "cache").mkdir()
    for sub in ("veg", "valid", "stats", "vis"):  # cache/nn stays private to the test
        (work / "cache" / sub).symlink_to(REAL_WORK / "cache" / sub)
    (work / "runs" / model_run / "layers" / "rows.parquet").symlink_to(
        (REAL_WORK / "runs" / model_run / "layers" / "rows.parquet").resolve())
    (work / "runs" / ref_run).symlink_to((REAL_WORK / "runs" / ref_run).resolve())
    cfg = load_config(overrides=[f'paths.work_dir="{work}"', f'paths.models_dir="{tmp_path / "models"}"',
                                 "nn.device=cpu"], project_root=tmp_path)
    yield cfg, model_run, ref_run


@pytest.mark.examples
@pytest.mark.slow
def test_ablation_and_panels_end_to_end(mirror: tuple[AppConfig, str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    from vineyard.nn import infer

    cfg, model_run, ref_run = mirror
    monkeypatch.setattr(infer, "default_model_factory", lambda enc, n, w: torch.nn.Conv2d(3, n, 1))
    save_weights(_conv_bytes(-5.0), _card("v1"), cfg.paths.models_dir)  # p ~ 0.007: NN finds nothing
    out = workflows.ablate(cfg, model_run, reference=ref_run)
    assert out.report.promoted.value == "A"
    by = {(s.axes, s.variant): s.score for s in out.report.scores}
    assert by[("reference", "B")] == 0.0 and by[("reference", "A")] > 0.8
    assert "F" in out.report.skipped
    assert out.json_path.is_file() and out.md_path.is_file() and len(out.panels) == 2
    assert out.md_path == cfg.paths.project_root / "reports" / "nn_ablation.md"


@pytest.mark.examples
@pytest.mark.slow
def test_panels_need_cached_probabilities(mirror: tuple[AppConfig, str, str]) -> None:
    from vineyard.errors import StageError

    cfg, model_run, _ = mirror
    with pytest.raises(StageError, match="cached"):
        workflows.panels_from_cache(cfg, model_run, ("siret3_r021_c012",), 0)
    probs = {"siret3_r021_c012": np.full((1024, 1024), 0.8, np.float32)}
    paths = workflows.write_panels(cfg, cfg.paths.work_dir / "panels", probs, cfg.paths.work_dir / "cache",
                                   workflows.tile_reader(cfg.paths.work_dir / "tiles"))
    assert [p.name for p in paths] == ["siret3_r021_c012.jpg"]


@pytest.mark.examples
@pytest.mark.slow
def test_panels_from_cache_auto_pick(mirror: tuple[AppConfig, str, str]) -> None:
    from vineyard.nn.probs import prob_png_path, write_prob_png

    cfg, model_run, _ = mirror
    assert workflows.panels_from_cache(cfg, model_run, (), 0) == ()  # nothing cached: warning, no panels
    for tile_id, p in (("siret3_r021_c012", 0.9), ("siret3_r006_c004", 0.1)):
        write_prob_png(prob_png_path(cfg.paths.work_dir / "cache", "v1", "canopy_prob", tile_id),
                       np.full((1024, 1024), p, np.float32))
    paths = workflows.panels_from_cache(cfg, model_run, (), 1)
    assert [p.name for p in paths] == ["siret3_r021_c012.jpg"]  # p = 0.9 everywhere disagrees most with a*


@pytest.mark.examples
@pytest.mark.slow
def test_ablation_without_weights_fails(mirror: tuple[AppConfig, str, str]) -> None:
    from vineyard.errors import StageError

    cfg, model_run, ref_run = mirror
    with pytest.raises(StageError, match="no NN weights"):
        workflows.ablate(cfg, model_run, reference=ref_run)


@pytest.mark.examples
@pytest.mark.slow
def test_train_from_run_with_a_tiny_store(mirror: tuple[AppConfig, str, str], tmp_path: Path) -> None:
    from tests.nn.n2_synth import write_tiny_store
    from vineyard.nn.train import default_parts

    cfg, model_run, ref_run = mirror
    cfg = cfg.model_copy(update={"nn": cfg.nn.model_copy(update={
        "batch_size": 2, "train": cfg.nn.train.model_copy(update={"num_workers": 0, "smoke_max_batches": 1})})})
    store = write_tiny_store(tmp_path / "store")
    parts = TrainParts(make_model=lambda e, n, w: Tiny(), make_loader=default_parts().make_loader,
                       make_loss=_masked_bce)
    from vineyard.errors import StageError

    with pytest.raises(StageError, match="pseudolabels --run"):
        workflows.train_from_run(cfg, model_run, store=tmp_path / "absent", smoke=True, variant="F",
                                 reference=ref_run, seed=None, train_cmd="test", parts=parts)
    res = workflows.train_from_run(cfg, model_run, store=store, smoke=True, variant="main", reference=ref_run,
                                   seed=None, train_cmd="test", parts=parts)
    assert res.card.version == "v0-smoke" and set(res.history[0].val_tiles) == set(cfg.nn.holdout_tiles)
    assert (cfg.paths.work_dir / "runs" / model_run / "metrics" / "nn_train_v0-smoke.json").is_file()


# ------------------------------------------------------------------ command wiring (workflows faked)


@pytest.fixture
def run_dir(tmp_work: Path) -> str:
    (tmp_work / "runs" / "r1").mkdir(parents=True)
    return "r1"


def test_cli_train_reports_the_model(monkeypatch: pytest.MonkeyPatch, run_dir: str) -> None:
    from vineyard.nn.train import TrainResult

    card = ModelCard(**{**_card("v1").__dict__, "sha256": "ab" * 32, "metrics": {"val_score": 0.83}})
    seen: dict[str, object] = {}

    def fake(cfg: AppConfig, run_id: str, **kw: object) -> TrainResult:
        seen.update(kw, run_id=run_id, band=cfg.nn.pseudolabels.ignore_band_px,
                    edge=cfg.nn.pseudolabels.edge_band_px)
        return TrainResult(card, 3, (), True, "mps")

    monkeypatch.setattr(workflows, "train_from_run", fake)
    res = runner.invoke(app, ["train", "--run", run_dir, "--variant", "F", "--smoke"])
    assert res.exit_code == 0, res.output
    assert "vine-unet@v1+abababab" in res.output and "0.8300" in res.output
    assert seen["run_id"] == run_dir and seen["variant"] == "F" and seen["smoke"] is True and seen["band"] == 0
    assert seen["edge"] == 0


def test_cli_ablate_panels_bench_pseudolabels(monkeypatch: pytest.MonkeyPatch, run_dir: str, tmp_path: Path) -> None:
    from vineyard.contracts.enums import FusionVariant
    from vineyard.nn import patch_store
    from vineyard.nn.ablation import AblationReport
    from vineyard.nn.train import BenchResult

    report = AblationReport((), {}, FusionVariant.C, "C wins", "model", ())
    monkeypatch.setattr(workflows, "ablate", lambda cfg, run, **kw: workflows.AblationOutputs(
        report, tmp_path / "a.json", tmp_path / "a.md", (tmp_path / "p.jpg",)))
    monkeypatch.setattr(workflows, "panels_from_cache", lambda cfg, run, tiles, n: (tmp_path / "x.jpg",))
    monkeypatch.setattr(workflows, "bench", lambda cfg, **kw: BenchResult(20, 8, 512, 0.5, 16.0, "mps"))
    monkeypatch.setattr(patch_store, "plan_store", lambda cfg, run_id: type("P", (), {"store_dir": tmp_path})())
    monkeypatch.setattr(patch_store, "build_store", lambda plan, workers: type("M", (), {"failed": ()})())
    monkeypatch.setattr(patch_store, "store_stats", lambda m, d: {"n_tiles": 3, "n_patches": 12})
    out = runner.invoke(app, ["ablate", "--run", run_dir])
    assert out.exit_code == 0 and "nn.fusion=C" in out.output, out.output
    out = runner.invoke(app, ["panels", "--run", run_dir, "--tile", "siret3_r021_c012"])
    assert out.exit_code == 0 and "x.jpg" in out.output, out.output
    out = runner.invoke(app, ["bench"])
    assert out.exit_code == 0 and "0.5000 s/itera" in out.output, out.output
    out = runner.invoke(app, ["pseudolabels", "--run", run_dir])
    assert out.exit_code == 0 and "patch-uri: 12" in out.output, out.output


def test_cli_errors_exit_nonzero(run_dir: str) -> None:
    for args in (["train", "--run", "missing-run"], ["ablate", "--run", "missing-run"],
                 ["panels", "--run", "missing-run"], ["pseudolabels", "--run", "missing-run"]):
        res = runner.invoke(app, args)
        assert res.exit_code != 0 and "run not found" in res.output, (args, res.output)
    res = runner.invoke(app, ["bench", "--set", "nn.device=tpu"])
    assert res.exit_code != 0
