"""verify_setup: the run's verifier from config + models dir, degrading L0..L3 (fake CLIP / SAM, real probe files)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from tests.waste.test_probe import EMBED_MODEL, make_data, params
from tests.waste.test_verify import FakeEmbedder, FakeSam
from vineyard.config import AppConfig, load_config
from vineyard.perception.waste.clip_embed import ClipEmbedError
from vineyard.perception.waste.probe import CARD_FILE, PROBE_FILE, ProbeError, save_probe, train_probe
from vineyard.perception.waste.verify import RuleOnlyVerifier
from vineyard.perception.waste.verify_ml import ModelVerifier
from vineyard.perception.waste.verify_setup import find_probe, load_clip, probe_handle, setup_verifier

NAME, VERSION = "waste-probe", "v1"
DIM = 16


@dataclass
class IdEmbedder(FakeEmbedder):
    model_id: str = EMBED_MODEL


def _cfg(models_dir: Path, probe: bool = True, sam: bool = False) -> AppConfig:
    return load_config(overrides=(
        f"paths.models_dir={json.dumps(str(models_dir))}",
        f"waste.probe.enabled={str(probe).lower()}",
        f"waste.sam3.enabled={str(sam).lower()}",
        f"waste.probe.name={NAME}",
        f"waste.probe.version={VERSION}",
    ))


def _reader(tile_id: str) -> np.ndarray:
    return np.zeros((8, 8, 3), np.uint8)


def _save(models_dir: Path) -> None:
    result = train_probe(make_data(4.0), params(), EMBED_MODEL, VERSION)
    save_probe(result, models_dir, NAME, train_data=(), train_cmd="pytest")


def _clip_ok(_cfg: object) -> IdEmbedder:
    return IdEmbedder()


def _clip_fails(_cfg: object) -> IdEmbedder:
    raise ClipEmbedError("OpenCLIP load failed", model="ViT-B-32")


def test_rule_only_when_probe_and_sam_are_disabled(tmp_path: Path) -> None:
    verifier, status = setup_verifier(_cfg(tmp_path, probe=False, sam=False), _reader, clip_loader=_clip_fails)
    assert isinstance(verifier, RuleOnlyVerifier) and status.level == "L3"


def test_missing_probe_ranks_with_clip_only(tmp_path: Path) -> None:
    verifier, status = setup_verifier(_cfg(tmp_path), _reader, clip_loader=_clip_ok)
    assert isinstance(verifier, ModelVerifier) and verifier.probe is None and verifier.embedder is not None
    assert status.level == "L2" and not status.probe and status.clip
    assert f"probe {NAME}@{VERSION} missing" in status.reason and verifier.probe_auto_min is None


def test_trained_probe_gives_level_l1_with_its_threshold(tmp_path: Path) -> None:
    _save(tmp_path)
    verifier, status = setup_verifier(_cfg(tmp_path), _reader, clip_loader=_clip_ok)
    assert status.level == "L1" and status.probe and status.clip
    assert isinstance(verifier, ModelVerifier) and verifier.probe is not None
    assert verifier.probe.version.startswith(f"{NAME}@{VERSION}+")
    assert verifier.probe_auto_min is not None and verifier.probe_auto_min >= 0.9
    emb = np.eye(2, DIM, dtype=np.float32)
    assert verifier.probe.score(emb).shape == (2,)


def test_clip_failure_drops_the_probe_and_names_the_reason(tmp_path: Path) -> None:
    _save(tmp_path)
    verifier, status = setup_verifier(_cfg(tmp_path), _reader, clip_loader=_clip_fails)
    assert status.level == "L3" and not status.probe and not status.clip
    assert "clip unavailable" in status.reason and "probe unusable without CLIP" in status.reason
    assert isinstance(verifier, ModelVerifier) and verifier.probe is None


def test_probe_of_another_clip_model_is_an_error(tmp_path: Path) -> None:
    _save(tmp_path)
    with pytest.raises(ProbeError, match="another CLIP model"):
        setup_verifier(_cfg(tmp_path), _reader, clip_loader=lambda _c: IdEmbedder(model_id="open_clip:other"))


def test_tampered_probe_is_an_error_not_a_degradation(tmp_path: Path) -> None:
    _save(tmp_path)
    card_path = tmp_path / NAME / VERSION / CARD_FILE
    card = json.loads(card_path.read_text())
    card_path.write_text(json.dumps({**card, "sha256": "0" * 64}))
    with pytest.raises(ProbeError, match="sha256"):
        setup_verifier(_cfg(tmp_path), _reader, clip_loader=_clip_ok)


def test_sam_joins_the_probe_for_level_l0(tmp_path: Path) -> None:
    _save(tmp_path)
    mirror = tmp_path / "hf" / "sam3"
    mirror.mkdir(parents=True)
    cfg = _cfg(tmp_path, sam=True)
    verifier, status = setup_verifier(
        cfg, _reader, clip_loader=_clip_ok, sam_loader=lambda _s, _p: FakeSam(),
        sam_lister=lambda _s: ("config.json", "model.safetensors"),
    )
    assert status.level == "L0" and status.sam
    assert isinstance(verifier, ModelVerifier) and verifier.sam is not None


def test_find_probe_and_handle(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert find_probe(cfg.waste.probe, tmp_path) is None
    _save(tmp_path)
    assert (tmp_path / NAME / VERSION / PROBE_FILE).is_file()
    model = find_probe(cfg.waste.probe, tmp_path)
    assert model is not None
    handle = probe_handle(model)
    assert handle.auto_threshold == model.auto_threshold and handle.auto_allowed == model.auto_enabled
    off = probe_handle(replace(model, auto_enabled=False))
    assert not off.auto_allowed and math.isfinite(off.auto_threshold)


def test_load_clip_reports_the_failure() -> None:
    cfg = load_config().waste.probe
    embedder, note = load_clip(cfg, _clip_fails)
    assert embedder is None and note.startswith("clip unavailable")
    ok, note_ok = load_clip(cfg, _clip_ok)
    assert ok is not None and note_ok == ""
