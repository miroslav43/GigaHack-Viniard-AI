"""NN weights on disk (contract §6), torch-free.

`models/<name>/<version>/weights.pt` (a serialised state_dict) + `model_card.json`. The card's sha256
is the sha256 of weights.pt; loading and fetching verify it. `model_version` strings are
`<name>@<version>+<sha256[:8]>`, or "none" when the NN is disabled or has no weights.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from vineyard.errors import VineyardError
from vineyard.pipeline.atomic import atomic_write_bytes, atomic_write_json

if TYPE_CHECKING:
    from vineyard.config import NnConfig

WEIGHTS_FILE: Final = "weights.pt"
CARD_FILE: Final = "model_card.json"
NO_NN: Final = "none"
SMOKE_VERSION: Final = "v0-smoke"  # timing-only weights, never promoted
F_SUFFIX: Final = "f"  # ablation F: trained without ignore band and label smoothing
SHA_PREFIX_LEN: Final = 8
SHA256_HEX_LEN: Final = 64
HASH_CHUNK_BYTES: Final = 1 << 20
FETCH_TIMEOUT_S: Final = 120.0
FETCH_SCHEMES: Final = frozenset({"https", "file"})
_SAFE_SEGMENT: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HEX: Final = re.compile(r"^[0-9a-f]+$")
_REQUIRED: Final = ("name", "version", "arch", "encoder", "in_gsd_m", "classes", "train_data", "metrics",
                    "sha256", "created_at", "train_cmd")

__all__ = [
    "CARD_FILE", "F_SUFFIX", "NO_NN", "SMOKE_VERSION", "WEIGHTS_FILE", "DataSource", "LoadedWeights", "ModelCard", "WeightsError",
    "WeightsIntegrityError", "WeightsMissingError", "fetch_weights", "load_weights", "nn_version_string",
    "read_card", "resolve_nn_version", "save_weights", "sha256_bytes", "sha256_file", "training_version",
    "weights_dir", "weights_exist",
]


class WeightsError(VineyardError):
    """NN weights or model card unusable (bad name, bad card, failed download)."""


class WeightsMissingError(WeightsError):
    """No weights.pt / model_card.json for the requested name and version."""


class WeightsIntegrityError(WeightsError):
    """weights.pt does not match the sha256 of its card or of the configuration."""


@dataclass(frozen=True)
class DataSource:
    name: str
    url: str
    licence: str


@dataclass(frozen=True)
class ModelCard:
    """Contract §6 fields; `extra` carries run details (seed, device, smoke, variant, epochs)."""

    name: str
    version: str
    arch: str
    encoder: str
    in_gsd_m: float
    classes: tuple[str, ...]
    train_data: tuple[DataSource, ...]
    metrics: Mapping[str, float]
    sha256: str
    created_at: str
    train_cmd: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "classes", tuple(self.classes))
        object.__setattr__(self, "train_data", tuple(self.train_data))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ModelCard) and self.to_json_dict() == other.to_json_dict()

    def __hash__(self) -> int:
        return hash(json.dumps(self.to_json_dict(), sort_keys=True, default=str))

    def to_json_dict(self) -> dict[str, Any]:
        doc = {f.name: getattr(self, f.name) for f in fields(self)}
        return {**doc, "classes": list(self.classes), "train_data": [vars(d) for d in self.train_data],
                "metrics": dict(self.metrics), "extra": dict(self.extra)}

    @classmethod
    def from_json_dict(cls, doc: Mapping[str, Any], *, source: str = "") -> ModelCard:
        missing = [k for k in _REQUIRED if k not in doc]
        if missing:
            raise WeightsError("model card lacks required fields", missing=", ".join(missing), source=source)
        try:
            data = tuple(DataSource(str(d["name"]), str(d["url"]), str(d["licence"])) for d in doc["train_data"])
            return cls(
                name=str(doc["name"]), version=str(doc["version"]), arch=str(doc["arch"]),
                encoder=str(doc["encoder"]), in_gsd_m=float(doc["in_gsd_m"]),
                classes=tuple(str(c) for c in doc["classes"]), train_data=data,
                metrics={str(k): float(v) for k, v in dict(doc["metrics"]).items()}, sha256=str(doc["sha256"]),
                created_at=str(doc["created_at"]), train_cmd=str(doc["train_cmd"]), extra=dict(doc.get("extra", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WeightsError("model card has malformed fields", error=f"{type(exc).__name__}: {exc}",
                               source=source) from exc


@dataclass(frozen=True)
class LoadedWeights:
    card: ModelCard
    weights_path: Path


# ------------------------------------------------------------------ paths and hashing


def _segment(value: str, what: str) -> str:
    if not _SAFE_SEGMENT.match(value or ""):
        raise WeightsError(f"unsafe model {what} (letters, digits, . _ - only)", value=value)
    return value


def weights_dir(models_dir: Path, name: str, version: str) -> Path:
    """models/<name>/<version>"""
    return Path(models_dir) / _segment(name, "name") / _segment(version, "version")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha(value: str, what: str) -> str:
    text = (value or "").strip().lower()
    if len(text) != SHA256_HEX_LEN or not _HEX.match(text):
        raise WeightsError(f"{what} is not a sha256 hex digest", value=value)
    return text


# ------------------------------------------------------------------ save / read / load


def save_weights(weights: bytes, card: ModelCard, models_dir: Path) -> ModelCard:
    """Write weights.pt then model_card.json (sha256 filled in from the bytes); returns the stamped card."""
    if not weights:
        raise WeightsError("refusing to save empty weights", name=card.name, version=card.version)
    folder = weights_dir(models_dir, card.name, card.version)
    stamped = replace(card, sha256=sha256_bytes(weights))
    atomic_write_bytes(folder / WEIGHTS_FILE, weights)
    atomic_write_json(folder / CARD_FILE, stamped.to_json_dict())
    return stamped


def _parse_card(raw: bytes, source: str) -> ModelCard:
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeightsError("model card is not valid JSON", source=source, error=str(exc)) from exc
    if not isinstance(doc, dict):
        raise WeightsError("model card is not a JSON object", source=source)
    return ModelCard.from_json_dict(doc, source=source)


def read_card(models_dir: Path, name: str, version: str) -> ModelCard:
    path = weights_dir(models_dir, name, version) / CARD_FILE
    if not path.is_file():
        raise WeightsMissingError("model card missing", path=str(path), hint="vineyard nn train | vineyard nn fetch")
    return _parse_card(path.read_bytes(), str(path))


def weights_exist(models_dir: Path, name: str, version: str) -> bool:
    folder = weights_dir(models_dir, name, version)
    return (folder / WEIGHTS_FILE).is_file() and (folder / CARD_FILE).is_file()


def load_weights(models_dir: Path, name: str, version: str, expected_sha256: str | None) -> LoadedWeights:
    """Card + weights path after checking sha256(weights.pt) == card.sha256 (== expected when given)."""
    card = read_card(models_dir, name, version)
    path = weights_dir(models_dir, name, version) / WEIGHTS_FILE
    if not path.is_file():
        raise WeightsMissingError("weights file missing", path=str(path))
    actual = sha256_file(path)
    if actual != card.sha256:
        raise WeightsIntegrityError("weights sha256 differs from the model card", path=str(path), actual=actual,
                                    card=card.sha256)
    if expected_sha256 is not None and actual != _check_sha(expected_sha256, "nn.weights_sha256"):
        raise WeightsIntegrityError("weights sha256 differs from the expected value", path=str(path),
                                    actual=actual, expected=expected_sha256)
    return LoadedWeights(card=card, weights_path=path)


# ------------------------------------------------------------------ fetch


def _default_opener(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as response:  # noqa: S310  (scheme checked)
        return bytes(response.read())


def _card_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    parent = parts.path.rsplit("/", 1)[0]
    return urllib.parse.urlunsplit(parts._replace(path=f"{parent}/{CARD_FILE}"))


def _download(opener: Callable[[str], bytes], url: str) -> bytes:
    try:
        return opener(url)
    except OSError as exc:  # urllib's URLError / HTTPError are OSErrors
        raise WeightsError("download failed", url=url, error=f"{type(exc).__name__}: {exc}") from exc


def fetch_weights(url: str, models_dir: Path, name: str, version: str, expected_sha256: str, *,
                  card_url: str | None = None, opener: Callable[[str], bytes] | None = None) -> LoadedWeights:
    """Download weights.pt and its model_card.json (sibling URL by default); install only when both the
    weights bytes and the card match `expected_sha256`."""
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme not in FETCH_SCHEMES:
        raise WeightsError("unsupported URL scheme", url=url, allowed=", ".join(sorted(FETCH_SCHEMES)))
    expected = _check_sha(expected_sha256, "expected sha256")
    get = opener or _default_opener
    weights = _download(get, url)
    actual = sha256_bytes(weights)
    if actual != expected:
        raise WeightsIntegrityError("downloaded weights sha256 mismatch", url=url, actual=actual, expected=expected)
    card_src = card_url or _card_url(url)
    card = _parse_card(_download(get, card_src), card_src)
    if card.sha256 != expected:
        raise WeightsIntegrityError("model card belongs to other weights", url=card_src, card=card.sha256,
                                    expected=expected)
    save_weights(weights, replace(card, name=name, version=version), models_dir)
    return load_weights(models_dir, name, version, expected)


# ------------------------------------------------------------------ version strings


def nn_version_string(card: ModelCard | None) -> str:
    """"<name>@<version>+<sha[:8]>" (contract §2.2/§6), "none" without a card."""
    if card is None:
        return NO_NN
    return f"{card.name}@{card.version}+{card.sha256[:SHA_PREFIX_LEN]}"


def resolve_nn_version(cfg: NnConfig, models_dir: Path) -> str:
    """model_version part for the NN: "none" when disabled or when the weights are absent."""
    if not cfg.enabled or not weights_exist(models_dir, cfg.name, cfg.version):
        return NO_NN
    return nn_version_string(read_card(models_dir, cfg.name, cfg.version))


def training_version(version: str, *, smoke: bool, variant: str) -> str:
    """Weights version of a training run: v0-smoke for smoke runs, <version>f for ablation F."""
    if smoke:
        return SMOKE_VERSION
    return version + F_SUFFIX if variant == "F" else version
