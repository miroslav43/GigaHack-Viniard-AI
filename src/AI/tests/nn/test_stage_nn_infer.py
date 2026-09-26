"""Stage nn_infer: degrades when disabled / without weights, fails on tampered weights, writes contract PNGs
+ keys in the main process, caches, and re-runs when the weights change. Never imports torch at import."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import write_geotiff
from tests.nn.n2_synth import OTHER_TILE, TILE_ID, synth_tile, write_tile_prep_cache
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.nn.probs import PROB_PX, prob_png_path, read_prob_png
from vineyard.nn.weights import WEIGHTS_FILE, ModelCard, WeightsIntegrityError, save_weights, weights_dir
from vineyard.pipeline.cache import read_key
from vineyard.pipeline.context import RunContext, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages import nn_infer

TILES = (TILE_ID, OTHER_TILE)


def _ctx(work: Path, models: Path, *, enabled: bool = True) -> RunContext:
    cfg = load_config(overrides=[f'paths.work_dir="{work}"', f'paths.models_dir="{models}"',
                                 f"nn.enabled={str(enabled).lower()}", "nn.device=cpu"])
    return new_run_context(cfg, source=Source.MODEL, run_id="nn-stage-test", workers=1)


def _prepare_work(work: Path) -> None:
    pd.DataFrame({"tile_id": list(TILES)}).to_parquet(work / "tile_index.parquet")
    for tile_id in TILES:
        s = synth_tile(tile_id)
        write_tile_prep_cache(work / "cache", s)
        write_geotiff(work / "tiles" / f"{tile_id}.tif", tile_id=tile_id, size=s.rgb.shape[0], rgb=s.rgb,
                      compress="DEFLATE")


def _save_fake_weights(models: Path, value: float) -> ModelCard:
    import torch

    from vineyard.nn.infer import state_dict_bytes

    model = torch.nn.Conv2d(3, 1, 1)
    torch.nn.init.constant_(model.weight, 0.0)
    torch.nn.init.constant_(model.bias, value)
    card = ModelCard(name="vine-unet", version="v1", arch="unet", encoder="resnet18", in_gsd_m=0.05,
                     classes=("canopy_prob",), train_data=(), metrics={}, sha256="", created_at="now", train_cmd="t")
    return save_weights(state_dict_bytes(model), card, models)


@pytest.fixture
def conv_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from vineyard.nn import infer

    monkeypatch.setattr(infer, "default_model_factory", lambda enc, n, w: torch.nn.Conv2d(3, n, 1))


def test_import_does_not_load_torch() -> None:
    code = "import sys, vineyard.pipeline.stages.nn_infer; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_registered() -> None:
    spec = load_stage("nn_infer")
    assert spec is nn_infer.STAGE and spec.version == nn_infer.STAGE_VERSION and spec.scope == "global"


def test_disabled_is_skipped_without_files(tmp_work: Path, tmp_path: Path) -> None:
    res = nn_infer.run(_ctx(tmp_work, tmp_path / "models", enabled=False))
    assert (res.n_items, res.n_failed, res.metrics["skipped"]) == (0, 0, 1.0)
    assert not (tmp_work / "cache" / "nn").exists()


def test_missing_weights_degrade_to_classic(tmp_work: Path, tmp_path: Path) -> None:
    _prepare_work(tmp_work)
    res = nn_infer.run(_ctx(tmp_work, tmp_path / "models"))
    assert res.metrics["skipped"] == 1.0 and res.metrics["weights_missing"] == 1.0
    assert not (tmp_work / "cache" / "nn").exists()


def test_tampered_weights_fail_loudly(tmp_work: Path, tmp_path: Path) -> None:
    _prepare_work(tmp_work)
    models = tmp_path / "models"
    _save_fake_weights(models, 0.0)
    path = weights_dir(models, "vine-unet", "v1") / WEIGHTS_FILE
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(WeightsIntegrityError):
        nn_infer.run(_ctx(tmp_work, models))


def test_writes_pngs_keys_and_caches(tmp_work: Path, tmp_path: Path, conv_factory: None) -> None:
    _prepare_work(tmp_work)
    models = tmp_path / "models"
    _save_fake_weights(models, 0.0)  # sigmoid(0) = 0.5 -> 128 everywhere
    ctx = _ctx(tmp_work, models)
    res = nn_infer.run(ctx)
    assert (res.n_items, res.n_cached, res.n_failed) == (2, 0, 0)
    assert res.metrics["s_per_tile"] > 0 and res.metrics["skipped"] == 0.0
    for tile_id in TILES:
        png = prob_png_path(ctx.paths.cache_dir, "v1", "canopy_prob", tile_id)
        arr = read_prob_png(png)
        assert arr.shape == (PROB_PX, PROB_PX) and np.all(arr == 128)
        assert read_key(png)
    again = nn_infer.run(ctx)
    assert (again.n_items, again.n_cached) == (2, 2)


def test_key_changes_with_weights(tmp_work: Path, tmp_path: Path, conv_factory: None) -> None:
    _prepare_work(tmp_work)
    models = tmp_path / "models"
    _save_fake_weights(models, 0.0)
    ctx = _ctx(tmp_work, models)
    nn_infer.run(ctx)
    png = prob_png_path(ctx.paths.cache_dir, "v1", "canopy_prob", TILE_ID)
    first = read_key(png)
    _save_fake_weights(models, 2.0)
    res = nn_infer.run(ctx)
    assert res.n_cached == 0
    assert read_key(png) != first
    assert np.all(read_prob_png(png) == round(255 / (1 + np.exp(-2.0))))


def test_failed_tile_is_recorded(tmp_work: Path, tmp_path: Path, conv_factory: None) -> None:
    _prepare_work(tmp_work)
    (tmp_work / "tiles" / f"{OTHER_TILE}.tif").write_bytes(b"not a tiff")
    models = tmp_path / "models"
    _save_fake_weights(models, 0.0)
    res = nn_infer.run(_ctx(tmp_work, models))
    assert res.n_failed == 1 and res.failed == (OTHER_TILE,)
