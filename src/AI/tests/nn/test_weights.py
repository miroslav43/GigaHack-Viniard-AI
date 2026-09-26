"""nn.weights: model card + sha256 integrity, fetch with verification, version strings (contract §6)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vineyard.config import load_config
from vineyard.nn.weights import (
    CARD_FILE,
    NO_NN,
    WEIGHTS_FILE,
    DataSource,
    ModelCard,
    WeightsError,
    WeightsIntegrityError,
    WeightsMissingError,
    fetch_weights,
    load_weights,
    nn_version_string,
    read_card,
    resolve_nn_version,
    save_weights,
    sha256_file,
    weights_dir,
    weights_exist,
)

PAYLOAD = b"fake-state-dict-bytes" * 100


def _card(version: str = "v1") -> ModelCard:
    return ModelCard(
        name="vine-unet", version=version, arch="unet", encoder="resnet18", in_gsd_m=0.05,
        classes=("canopy_prob",), train_data=(DataSource("siret3", "https://example.org/d", "challenge"),),
        metrics={"val_score": 0.81}, sha256="", created_at="2026-09-26T03:00:00+03:00",
        train_cmd="vineyard nn train", extra={"seed": 0},
    )


def test_import_is_torch_free() -> None:
    code = "import sys, vineyard.nn.weights; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_save_writes_weights_and_card_with_file_sha(tmp_path: Path) -> None:
    card = save_weights(PAYLOAD, _card(), tmp_path)
    folder = weights_dir(tmp_path, "vine-unet", "v1")
    assert (folder / WEIGHTS_FILE).read_bytes() == PAYLOAD
    assert card.sha256 == sha256_file(folder / WEIGHTS_FILE) == hashlib.sha256(PAYLOAD).hexdigest()
    doc = json.loads((folder / CARD_FILE).read_text(encoding="utf-8"))
    for key in ("name", "version", "arch", "encoder", "in_gsd_m", "classes", "train_data", "metrics",
                "sha256", "created_at", "train_cmd"):
        assert key in doc
    assert doc["train_data"][0] == {"name": "siret3", "url": "https://example.org/d", "licence": "challenge"}
    assert read_card(tmp_path, "vine-unet", "v1") == card
    assert weights_exist(tmp_path, "vine-unet", "v1")


def test_load_verifies_sha(tmp_path: Path) -> None:
    card = save_weights(PAYLOAD, _card(), tmp_path)
    loaded = load_weights(tmp_path, "vine-unet", "v1", card.sha256)
    assert loaded.card == card
    assert loaded.weights_path == weights_dir(tmp_path, "vine-unet", "v1") / WEIGHTS_FILE
    assert load_weights(tmp_path, "vine-unet", "v1", None).card.sha256 == card.sha256


def test_flipping_one_byte_is_detected(tmp_path: Path) -> None:
    save_weights(PAYLOAD, _card(), tmp_path)
    path = weights_dir(tmp_path, "vine-unet", "v1") / WEIGHTS_FILE
    data = bytearray(path.read_bytes())
    data[10] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(WeightsIntegrityError, match="sha256"):
        load_weights(tmp_path, "vine-unet", "v1", None)


def test_expected_sha_mismatch(tmp_path: Path) -> None:
    save_weights(PAYLOAD, _card(), tmp_path)
    with pytest.raises(WeightsIntegrityError, match="expected"):
        load_weights(tmp_path, "vine-unet", "v1", "0" * 64)


def test_missing_weights(tmp_path: Path) -> None:
    assert not weights_exist(tmp_path, "vine-unet", "v1")
    with pytest.raises(WeightsMissingError):
        load_weights(tmp_path, "vine-unet", "v1", None)


def test_bad_card_json(tmp_path: Path) -> None:
    save_weights(PAYLOAD, _card(), tmp_path)
    card_path = weights_dir(tmp_path, "vine-unet", "v1") / CARD_FILE
    doc = json.loads(card_path.read_text(encoding="utf-8"))
    del doc["encoder"]
    card_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(WeightsError, match="encoder"):
        read_card(tmp_path, "vine-unet", "v1")
    card_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(WeightsError, match="JSON"):
        read_card(tmp_path, "vine-unet", "v1")


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden"])
def test_unsafe_names_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(WeightsError):
        weights_dir(tmp_path, bad, "v1")
    with pytest.raises(WeightsError):
        weights_dir(tmp_path, "vine-unet", bad)


def test_version_string() -> None:
    card = _card()
    sha = "1a2b3c4d" + "e" * 56
    stamped = ModelCard(**{**card.__dict__, "sha256": sha})
    assert nn_version_string(stamped) == "vine-unet@v1+1a2b3c4d"
    assert nn_version_string(None) == NO_NN == "none"


def test_resolve_nn_version(tmp_path: Path) -> None:
    cfg = load_config()
    enabled = cfg.nn.model_copy(update={"enabled": True})
    disabled = cfg.nn.model_copy(update={"enabled": False})
    assert resolve_nn_version(disabled, tmp_path) == NO_NN
    assert resolve_nn_version(enabled, tmp_path) == NO_NN
    card = save_weights(PAYLOAD, _card(enabled.version), tmp_path)
    assert resolve_nn_version(enabled, tmp_path) == nn_version_string(card)
    assert resolve_nn_version(disabled, tmp_path) == NO_NN


def _opener(files: dict[str, bytes]):  # noqa: ANN202
    def open_url(url: str) -> bytes:
        if url not in files:
            raise OSError(f"404 {url}")
        return files[url]

    return open_url


def test_fetch_verifies_and_installs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    card = save_weights(PAYLOAD, _card(), src)
    card_bytes = (weights_dir(src, "vine-unet", "v1") / CARD_FILE).read_bytes()
    url = "https://example.org/release/weights.pt"
    files = {url: PAYLOAD, "https://example.org/release/model_card.json": card_bytes}
    dest = tmp_path / "models"
    loaded = fetch_weights(url, dest, "vine-unet", "v1", card.sha256, opener=_opener(files))
    assert loaded.card.sha256 == card.sha256
    assert sha256_file(loaded.weights_path) == card.sha256


def test_fetch_rejects_wrong_sha_and_leaves_nothing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    save_weights(PAYLOAD, _card(), src)
    card_bytes = (weights_dir(src, "vine-unet", "v1") / CARD_FILE).read_bytes()
    url = "https://example.org/release/weights.pt"
    files = {url: PAYLOAD + b"x", "https://example.org/release/model_card.json": card_bytes}
    dest = tmp_path / "models"
    with pytest.raises(WeightsIntegrityError, match="sha256"):
        fetch_weights(url, dest, "vine-unet", "v1", hashlib.sha256(PAYLOAD).hexdigest(), opener=_opener(files))
    assert not weights_exist(dest, "vine-unet", "v1")


def test_fetch_rejects_card_for_other_weights(tmp_path: Path) -> None:
    src = tmp_path / "src"
    save_weights(b"other", _card(), src)
    card_bytes = (weights_dir(src, "vine-unet", "v1") / CARD_FILE).read_bytes()
    url = "https://example.org/w.pt"
    files = {url: PAYLOAD, "https://example.org/model_card.json": card_bytes}
    with pytest.raises(WeightsIntegrityError, match="card"):
        fetch_weights(url, tmp_path / "m", "vine-unet", "v1", hashlib.sha256(PAYLOAD).hexdigest(),
                      opener=_opener(files))


def test_fetch_input_validation(tmp_path: Path) -> None:
    with pytest.raises(WeightsError, match="scheme"):
        fetch_weights("ftp://x/w.pt", tmp_path, "vine-unet", "v1", "a" * 64, opener=_opener({}))
    with pytest.raises(WeightsError, match="sha256"):
        fetch_weights("https://x/w.pt", tmp_path, "vine-unet", "v1", "nothex", opener=_opener({}))
    with pytest.raises(WeightsError, match="download failed"):
        fetch_weights("https://x/w.pt", tmp_path, "vine-unet", "v1", "a" * 64, opener=_opener({}))
